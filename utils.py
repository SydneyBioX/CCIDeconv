# utils.py
import itertools
from matplotlib import cm
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler as sc
from sklearn.metrics import classification_report, fbeta_score, f1_score, roc_curve, auc, make_scorer, r2_score, mean_squared_error
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
from bayes_opt import BayesianOptimization
from scipy.stats import gmean

# scaling function
def scale_continuous_by_group(X, groups, cont_cols):
    X_scaled = X.copy()
    groups_array = np.array(groups)
    for g in np.unique(groups_array):
        idx = np.where(groups_array == g)[0] 
        X_scaled.loc[X.iloc[idx].index, cont_cols] = pd.DataFrame(
            sc().fit_transform(X.iloc[idx][cont_cols]), 
            columns=cont_cols, 
            index=X.iloc[idx].index)
    return X_scaled

# processing function. The data of location is alread OHE using the processing.py file. 
def process_data(df, exclude_columns = None, spatial = True):
    # assign labels based on the scores
    # scaling the continuous columns by group (sample)
    groups = df['sample'].values
    if spatial == True: 
        conditions = [
        (df['cyt_score'] == 0) & 
        (df['nuc_score'] == 0), # detected in cell alone
        (df['cyt_score'] > 0) & 
        (df['nuc_score'] > 0), # detected in both cytoplasm and nucleus + cell
        (df['cyt_score'] > 0) & 
        (df['nuc_score'] == 0), # detected in cytoplasm + cell alone
        (df['nuc_score'] > 0) & 
        (df['cyt_score'] == 0), # detected in nucleus + cell alone
        ]
        cont_cols = ['cell_score', 'cell_pspatial', 'nuc_score', 'cyt_score']
    else:
        conditions = [
        (df['cyt_P1'] == 0) & 
        (df['nuc_P1'] == 0), # detected in cell alone
        (df['cyt_P1'] > 0) & 
        (df['nuc_P1'] > 0), # detected in both cytoplasm and nucleus + cell
        (df['cyt_P1'] > 0) & 
        (df['nuc_P1'] == 0), # detected in cytoplasm + cell alone
        (df['nuc_P1'] > 0) & 
        (df['cyt_P1'] == 0), # detected in nucleus + cell alone
        ]
        cont_cols = ['cell_P1', 'nuc_P1', 'cyt_P1']
    labels = [0,1,1,1]
    df['labels'] = np.select(conditions, labels, default=4)
    df = df[df['labels'] != 4] # remove the non significant interactions if any (automatically pval<0.05 is filtered)
    X = df.drop(columns=exclude_columns)
    X_scaled = scale_continuous_by_group(X, groups, cont_cols) 
    if spatial == True:
        y_nuc = X_scaled[['nuc_score']]
        y_cyt = X_scaled[['cyt_score']]
    else:
        y_nuc = X_scaled[['nuc_P1']]
        y_cyt = X_scaled[['cyt_P1']]
    y = df['labels']
    X_scaled = X_scaled.drop(columns=['cyt_score', 'nuc_score', 'cyt_P1', 'nuc_P1', 'labels'], errors = 'ignore') # remove the labels column from the x matrix
    ids = df[['lr_pair', 'source', 'target']]
    return X_scaled, y, y_nuc, y_cyt, ids

def process_test_data(df):    
    # scaling the continuous columns by group (sample)
    X = df
    groups = "test_sample" # assign all to one group for scaling
    cont_cols = ['cell_score']
    X_scaled_test = scale_continuous_by_group(X, groups, cont_cols) 
    ids = df[['lr_pair', 'source', 'target']]
    return X_scaled_test, ids

def process_singlecell(df):
    conditions = [
    (df['cyt_score'] == 0) & 
    (df['nuc_score'] == 0), # detected in cell alone
    (df['cyt_score'] > 0) & 
    (df['nuc_score'] > 0), # detected in both cytoplasm and nucleus + cell
    (df['cyt_score'] > 0) & 
    (df['nuc_score'] == 0), # detected in cytoplasm + cell alone
    (df['nuc_score'] > 0) & 
    (df['cyt_score'] == 0), # detected in nucleus + cell alone
    ]
    labels = [0,1,1,1]
    df['labels'] = np.select(conditions, labels, default=4)
    df = df[df['labels'] != 4] # remove the non significant interactions if any

    # columns to remove from the x_matric
    exclude_columns = ['cell_pspatial',
        'cyt_pval', 'cyt_pspatial', 'cyt_score', 'sample',
        'cell_pval', 'cell_score', 'tissue', 'is_neurotransmitter',
        'ligand_location_cellchat', 'receptor_location_cellchat',
        'ligand_location_hpa', 'receptor_location_hpa',
        'nuc_pval', 'nuc_pspatial', 'nuc_score', 'ligand', 'receptor', 
    ]
    X = df.drop(columns=exclude_columns)
    groups = df['sample'].values
    cont_cols = ['cell_P1', 'nuc_P1', 'cyt_P1']
    X_scaled = scale_continuous_by_group(X, groups, cont_cols) # scaling the continuous columns
    y_nuc = X_scaled[['nuc_P1']]
    y_cyt = X_scaled[['cyt_P1']]

    # Catboost encoding
    catboost_encoder = CatBoostEncoder()
    categorical_columns = ['lr_pair', 'source', 'target',
        'pathway_name', 'annotation',
        'ligand.family', 'ligand.keyword', 'ligand.secreted_type',
        'ligand.transmembrane', 'receptor.family',
        'receptor.keyword', 'receptor.surfaceome_main',
        'receptor.surfaceome_sub', 'receptor.adhesome',
        'receptor.secreted_type', 'receptor.transmembrane']
    labels = ['labels']
    # catboost_encoded = catboost_encoder.fit_transform(X_scaled[categorical_columns], X[labels])
    # X_scaled = X_scaled.drop(categorical_columns, axis = 1)
    # X_scaled = pd.concat([X_scaled.reset_index(drop= True), catboost_encoded.reset_index(drop = True)], axis = 1)
    y = df['labels']
    X_scaled = X_scaled.drop(columns=['cyt_P1', 'nuc_P1']) # remove the labels column from the x matrix
    ids = df[['lr_pair', 'source', 'target']]

# random forest model optimization
def rf_cv(max_samples,n_estimators,max_features, max_depth, min_samples_split, min_samples_leaf, X_train, y_train):
    params = {
        'max_samples': max_samples,
        'max_features':max_features,
        'n_estimators':int(n_estimators),
        'random_state': 42,
        'max_depth': int(max_depth),
        'min_samples_split': int(min_samples_split),
        'min_samples_leaf': int(min_samples_leaf)
    }
    estimator_function = RandomForestClassifier(**params)
    f1_scores = cross_val_score(estimator_function, X_train, y_train, cv=3,
                             scoring='f1_macro', n_jobs=-1)
    return f1_scores.mean()

# xgb classification optimization
def xgbc_cv(max_depth,learning_rate,n_estimators,reg_alpha,colsample_bytree, min_child_weight, gamma, scale_pos_weight, X_train, y_train):
    estimator_function = xgb.XGBClassifier(max_depth=int(max_depth),
                                           colsample_bytree= colsample_bytree,
                                           gamma= gamma,
                                           min_child_weight= int(min_child_weight),
                                           learning_rate= learning_rate,
                                           n_estimators= int(n_estimators),
                                           reg_alpha = reg_alpha,
                                           nthread = -1,
                                           objective='binary:logistic',
                                           scale_pos_weight = scale_pos_weight,
                                           seed = 112)
    f1_scores = cross_val_score(estimator_function, X_train, y_train, cv=3,
                             scoring='f1_macro', n_jobs=-1)
    return f1_scores.mean()

# xgb regression optimization
def xgb_reg(max_depth, learning_rate, n_estimators, reg_alpha, colsample_bytree, min_child_weight, gamma, X_train_reg, y_train_reg):
    max_depth = int(max_depth)
    n_estimators = int(n_estimators)
    estimator_function = xgb.XGBRegressor(
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        reg_alpha=reg_alpha,
        colsample_bytree=colsample_bytree,
        min_child_weight=min_child_weight,
        gamma=gamma,
        objective='reg:squarederror',
        random_state=112,
        n_jobs=-1
    )
    # Fit the estimator
    cv = KFold(n_splits=3, shuffle=True, random_state=112)
    scores = cross_val_score(estimator_function, X_train_reg, y_train_reg, cv=cv,
                             scoring=make_scorer(r2_score))
    return scores.mean()

def classification_metrics(y_test, y_pred, y_proba, test_ids):
    from sklearn.metrics import balanced_accuracy_score
    print(classification_report(y_test, y_pred))
    fpr, tpr, thresholds = roc_curve(y_test, y_proba[:, 1])
    metrics = {
        'accuracy': balanced_accuracy_score(y_test, y_pred),
        'class_report': classification_report(y_test, y_pred),
        'f2':fbeta_score(y_test, y_pred, beta = 2, average = 'binary'),
        'f1': f1_score(y_test, y_pred, average= 'macro'), 
        'roc_auc':  auc(fpr, tpr)
    }
    roc_table = pd.DataFrame({
        'False Positive Rate': fpr,
        'True Positive Rate': tpr,
        'Thresholds': thresholds
    })
    predictions = pd.DataFrame({'pred': y_pred, 'true': y_test})
    predictions = pd.concat([predictions, test_ids.reset_index(drop=True)], axis=1) # add the test ids to the predictions dataframe

    return metrics, roc_table, predictions

def regression_metrics(y_test_reg, y_pred_reg, test_ids):
    metrics = {"R²": r2_score(y_test_reg, y_pred_reg),
               'RMSE': np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)),
               'NRMSE': np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))/np.std(y_test_reg)}
    predictions = pd.DataFrame({'pred': y_pred_reg, 'true': y_test_reg}) # create a dataframe to store the predictions
    predictions = pd.concat([predictions, test_ids.reset_index(drop=True)], axis=1) # add the test ids to the predictions dataframe
    return metrics, predictions

def cast_params(params, int_params=None):
    params = params.copy()
    if int_params is not None:
        for p in int_params:
            if p in params:
                params[p] = int(round(params[p]))
    return params

def test_prediction(y_pred, test_ids):
    predictions = pd.DataFrame({'pred': y_pred}) # create a dataframe to store the predictions
    predictions = pd.concat([predictions, test_ids.reset_index(drop=True)], axis=1) # add the test ids to the predictions dataframe
    return predictions

def group_generator(df):
    datasets = df['sample'].unique()
    all_combinations = {}
    for test_name in datasets: 
        groupnames = [item for item in datasets if item != test_name]
        combination = []
        max_len = min(len(groupnames), 8)  # avoid ValueError
        for k in range(1, max_len + 1):    # k = 1..max_len (9 here)
            combos = list(itertools.combinations(groupnames, k))
            combination.extend(combos)
        all_combinations[test_name] = combination
    return all_combinations

def extract_metrics(results_classifier, results_regression, model_name):
       auc = [results_classifier[group]['classification_metrics']['roc_auc']
              for group in results_classifier.keys()]
       recall = [results_classifier[group]['classification_metrics']['class_report']
              for group in results_classifier.keys()]
       macro_recalls = []
       for report in recall:
              for line in report.splitlines():
                     if line.strip().startswith("macro avg"):
                     # Split the line into parts and take the second number (recall)
                            macro_recall = float(line.split()[3])
                            macro_recalls.append(macro_recall)
       cyt_R2 = [results_regression[group]['cyt'][0]['R²']
              for group in results_regression.keys()]
       nuc_R2 = [results_regression[group]['nuc'][0]['R²']
                     for group in results_regression.keys()]
       cyt_NRMSE = [results_regression[group]['cyt'][0]['NRMSE']
                     for group in results_regression.keys()]
       nuc_NRMSE = [results_regression[group]['nuc'][0]['NRMSE']
                     for group in results_regression.keys()]
       
       composite_metric = [
              gmean([
                     a, 
                     r, 
                     np.mean([c_r2, n_r2]),               # average R² for this row
                     np.mean([1 - c_nrmse, 1 - n_nrmse])  # average "1 - NRMSE" for this row
              ])
              for a, r, c_r2, n_r2, c_nrmse, n_nrmse in zip(
                     auc, macro_recalls, cyt_R2, nuc_R2, cyt_NRMSE, nuc_NRMSE
              )
              ]
       
       metrics = pd.DataFrame({
       "Dataset": results_classifier.keys(),
       "model" : model_name,
       "AUC": auc,
       "macro_recall": macro_recalls,
       "cyt_R2": cyt_R2,
       "nuc_R2": nuc_R2,
       "cyt_NRMSE": cyt_NRMSE,
       "nuc_NRMSE": nuc_NRMSE,
       "composite_metric": composite_metric
       })
       return metrics