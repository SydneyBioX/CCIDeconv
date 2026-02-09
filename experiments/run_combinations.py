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
from utils import process_data, rf_cv, cast_params, xgbc_cv, xgb_reg, group_generator
from config import hyperparameter_space_rf_class, hyperparameter_space_xgb_class, hyperparameter_space_reg, gp_params, seed
from train import train_classifier, train_regressor 
from inference import HierarchicalClassifier, HierarchicalRegressor 
import random
import warnings
warnings.filterwarnings("ignore")


random.seed(42)
def main(args):
    """
    run_combinations.py

    Performs validation experiments for each spatial transcriptomics (ST) datasets across multiple combinations of the other datasets.

    Pipeline overview:
    1. Load preprocessed training data.
    2. Encode categorical features using CatBoostEncoder.
    3. Perform Bayesian optimization to tune:
    - Random Forest classifier
    - XGBoost classifier
    - XGBoost regressors (cytoplasmic and nuclear outputs)
    4. Train hierarchical classifier and regressors.
    5. Evaluate on held-out test group.
    6. Save performance metrics and predictions to disk.

    Outputs:
        - data/results_classifier_combinations.joblib
        - data/results_regression_combinations.joblib

    Example usage:
    # example command to run:
    python experiments/run_compartments.py \
         --data_path data/training_data/training_data.csv \
         --categorical_columns lr_pair,source,target,pathway_name,annotation,ligand.family,ligand.keyword,ligand.secreted_type,ligand.transmembrane,receptor.family,receptor.keyword,receptor.surfaceome_main,receptor.surfaceome_sub,receptor.adhesome,receptor.secreted_type,receptor.transmembrane \
         --group_column sample \
         --exclude_columns cyt_pval,cyt_pspatial,cyt_P1,sample,cell_pval,cell_P1,tissue,is_neurotransmitter,ligand_location_cellchat,receptor_location_cellchat,ligand_location_hpa,receptor_location_hpa,nuc_pval,nuc_pspatial,nuc_P1,ligand,receptor,labels

    """
    results_classifier = {}
    results_regression = {}
    categorical_columns = args.categorical_columns.split(',')
    catboost_encoder = CatBoostEncoder()
    df = pd.read_csv(args.data_path)
    all_combinations = group_generator(df)
    groups = df[args.group_column].values
    print("combinations generated")
    X_scaled, y, y_nuc, y_cyt, ids = process_data(df, exclude_columns=args.exclude_columns.split(','), spatial = True)
    # get test data each group
    for test_data in all_combinations.keys(): 
        combinations = all_combinations[test_data]
        test_idx = np.where(groups == test_data)[0]
        X_test_out = X_scaled.iloc[test_idx]
        y_test = y.iloc[test_idx].values.ravel()
        test_ids_out = ids.iloc[test_idx]
        # training with different combinations of the remaining groups
        for i in range (len(combinations)):
            train_idx = np.concatenate([np.where(groups == item)[0] for item in combinations[i]]) 
            X_train = X_scaled.iloc[train_idx]
            y_train = y.iloc[train_idx].values.ravel()
            group_name = f"{test_data}_combo_{i}"
            print(group_name)
            # --- Encode train columns ---
            cat_train = catboost_encoder.fit_transform(X_train[categorical_columns], y_train)
            X_train = pd.concat([X_train.drop(columns=categorical_columns).reset_index(drop=True), cat_train.reset_index(drop=True)], axis=1)
            # copy test
            X_test = X_test_out.copy()
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
            # inference on test set
            test_ids = test_ids_out.copy() # copy so different loops get reset version
            y_pred, class_metrics, roc_table, predictions = HierarchicalClassifier(classifier_wrapper).evaluate(X_test, y_test, test_ids)
            # Save results for this group
            results_classifier[group_name] = {
                'y_pred': y_pred,
                'classification_metrics': class_metrics,
                'roc_table': roc_table,
                'predictions': predictions
            }
        
            # --- Fit regressors on positives in training set ---
            mask_train = y_train == 1
            mask_test = y_pred == 1
            test_ids = test_ids[mask_test]
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
        
            reg_cyt_wrapper = train_regressor(X_train[mask_train,:], y_cyt.iloc[train_idx,:].to_numpy()[mask_train], reg_cyt_params, X_val = X_test[mask_test,:], y_val = y_cyt.iloc[test_idx,:].to_numpy()[mask_test])
            reg_nuc_wrapper = train_regressor(X_train[mask_train,:], y_nuc.iloc[train_idx,:].to_numpy()[mask_train], reg_nuc_params, X_val = X_test[mask_test,:], y_val = y_nuc.iloc[test_idx,:].to_numpy()[mask_test])
        # Inference on test set
            results_regression[group_name] = HierarchicalRegressor(reg_cyt_wrapper, reg_nuc_wrapper).evaluate(X_test[mask_test,:], y_cyt.iloc[test_idx,:].to_numpy()[mask_test].flatten(), y_nuc.iloc[test_idx,:].to_numpy()[mask_test].flatten(), test_ids)
   
    # dump results to file

    # Save
    joblib.dump(results_classifier, "data/results_classifier_combinations.joblib")
    joblib.dump(results_regression, "data/results_regression_combinations.joblib")


            
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run compartment experiment")
    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV file")
    parser.add_argument("--categorical_columns", type=str, required=True, 
                        help="Comma-separated list of categorical columns")
    parser.add_argument("--group_column", type=str, default="sample", help="Column name for groups")
    parser.add_argument("--exclude_columns", type=str, required=True, help="Comma-separated list of columns to exclude")
    args = parser.parse_args()
    
    main(args)
    
# example command to run:
# python experiments/run_compartments.py \
#     --data_path data/training_data/training_data.csv \
#     --categorical_columns lr_pair,source,target,pathway_name,annotation,ligand.family,ligand.keyword,ligand.secreted_type,ligand.transmembrane,receptor.family,receptor.keyword,receptor.surfaceome_main,receptor.surfaceome_sub,receptor.adhesome,receptor.secreted_type,receptor.transmembrane \
#     --group_column sample \
#     --exclude_columns cyt_pval,cyt_pspatial,cyt_P1,sample,cell_pval,cell_P1,tissue,is_neurotransmitter,ligand_location_cellchat,receptor_location_cellchat,ligand_location_hpa,receptor_location_hpa,nuc_pval,nuc_pspatial,nuc_P1,ligand,receptor,labels