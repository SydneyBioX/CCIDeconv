# experiments/run_logo.py
import argparse
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from category_encoders import CatBoostEncoder
from bayes_opt import BayesianOptimization
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import process_data, rf_cv, cast_params, xgbc_cv, xgb_reg, process_test_data
from config import hyperparameter_space_rf_class, hyperparameter_space_xgb_class, hyperparameter_space_reg, gp_params, seed
from train import train_classifier, train_regressor 
from inference import HierarchicalClassifier, HierarchicalRegressor 
from random import random

random.seed(42)
def main(args):
    df = pd.read_csv(args.data_path)
    # remove samples whose separation was influenced mainly by spatial features
    df = df[df['sample']!= 'spe_list_lung_cellchat']
    df = df[df['sample']!= 'spe_list_brainad_cellchat']
    X_train, y_train, y_nuc, y_cyt, ids = process_data(df, exclude_columns=args.exclude_columns.split(','), spatial = False)
    categorical_columns = args.categorical_columns.split(',')
    catboost_encoder = CatBoostEncoder()
    
    # process the test data
    test_df = pd.read_csv(args.test_data_path)
    X_test, y_test, test_ids = process_test_data(test_df) # ensure the test data is processed correctly 
    
    # --- Encode categorical columns ---
    cat_train = catboost_encoder.fit_transform(X_train[categorical_columns], y_train)
    X_train = pd.concat([X_train.drop(columns=categorical_columns).reset_index(drop=True), cat_train.reset_index(drop=True)], axis=1)
    
    cat_test = catboost_encoder.transform(X_test[categorical_columns])
    X_test = pd.concat([X_test.drop(columns=categorical_columns).reset_index(drop=True), cat_test.reset_index(drop=True)], axis=1)
    
    # convert to numpy
    X_train = X_train.to_numpy(dtype=float)
    X_test = X_test.to_numpy(dtype=float)
    
    # --- Inbuilt tuning for classifier ---
    xgbcBO = BayesianOptimization(
            f=lambda **params: rf_cv(**params, X_train= X_train, y_train= y_train),
            pbounds=hyperparameter_space_rf_class,
            random_state=seed,
            verbose=0
        )
    xgbcBO.maximize(init_points=3,n_iter=30)
    rf_int_params = ['n_estimators','max_depth','min_samples_split','min_samples_leaf']
    rf_params = cast_params( xgbcBO.max['params'], int_params=rf_int_params)
    xgbcBO = BayesianOptimization(
        f=lambda **params: xgbc_cv(**params, X_train=X_train, y_train=y_train),
        pbounds=hyperparameter_space_xgb_class,
        random_state=seed,
        verbose=0
    )
    xgbcBO.set_gp_params(**gp_params)
    xgbcBO.maximize(init_points=3,n_iter=30)
    xgb_int_params = ['n_estimators', 'max_depth']
    xgb_params = cast_params( xgbcBO.max['params'], int_params=xgb_int_params)
    # fit the classifier with the best parameters
    classifier_wrapper = train_classifier(X_train, y_train, rf_params, xgb_params)
    
    # --- Fit regressors on positives in training set ---
    mask_train = y_train == 1
    
    xgb_bo = BayesianOptimization(
    f=lambda **params: xgb_reg(**params, X_train_reg= X_train[mask_train,:], y_train_reg = y_cyt.iloc[train_idx,:].to_numpy()[mask_train]),
    pbounds=hyperparameter_space_reg,
    random_state=seed,
    verbose=0
    )
    xgb_bo.set_gp_params(**gp_params)
    xgb_bo.maximize(init_points=3, n_iter=30)
    reg_cyt_params = cast_params(xgb_bo.max['params'], int_params=xgb_int_params)   
    xgb_bo = BayesianOptimization(
    f=lambda **params: xgb_reg(**params, X_train_reg= X_train[mask_train,:], y_train_reg = y_nuc.iloc[train_idx,:].to_numpy()[mask_train]),
    pbounds=hyperparameter_space_reg,
    random_state=seed,
    verbose=0
    )
    xgb_bo.set_gp_params(**gp_params)
    xgb_bo.maximize(init_points=3, n_iter=30)
    reg_nuc_params = cast_params(xgb_bo.max['params'], int_params=xgb_int_params) 
    reg_nuc_wrapper = train_regressor(X_train[mask_train,:], y_nuc.to_numpy()[mask_train], reg_nuc_params, X_val = None, y_val = None)
    # get the predicted values for the test set
    classification_predictions = HierarchicalClassifier(classifier_wrapper).prediction(X_test, test_ids)
    regression_predictions = HierarchicalRegressor(reg_cyt_wrapper, reg_nuc_wrapper).prediction(X_test, test_ids)
    # dump results to file
    # Save
    joblib.dump(classification_predictions, "classification_predictions.joblib")
    joblib.dump(regression_predictions, "regression_predictions.joblib")

            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run single cell experiment")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--categorical_columns", type=str, required=True, 
                        help="Comma-separated list of categorical columns")
    parser.add_argument("--exclude_columns", type=str, required=True, help="Comma-separated list of columns to exclude")
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to test CSV file")
    args = parser.parse_args()
    
    main(args)
    
# example command to run:
# python experiments/run_logo.py \
#     --data_path data/training_data/training_data.csv \
#     --categorical_columns lr_pair,source,target,pathway_name,annotation,ligand.family,ligand.keyword,ligand.secreted_type,ligand.transmembrane,receptor.family,receptor.keyword,receptor.surfaceome_main,receptor.surfaceome_sub,receptor.adhesome,receptor.secreted_type,receptor.transmembrane \
#     --exclude_columns cyt_pval,cyt_pspatial,cyt_P1,sample,cell_pval,cell_P1,tissue,is_neurotransmitter,ligand_location_cellchat,receptor_location_cellchat,ligand_location_hpa,receptor_location_hpa,nuc_pval,nuc_pspatial,nuc_P1,ligand,receptor,labels
#     --test_data_path data/training_data/singlecell_test_data.csv