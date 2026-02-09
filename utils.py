# utils.py
import itertools
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler as sc
from sklearn.metrics import classification_report, fbeta_score, f1_score, roc_curve, auc, make_scorer, r2_score, mean_squared_error
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from bayes_opt import BayesianOptimization
from scipy.stats import gmean

# scaling function
def scale_continuous_by_group(X, groups, cont_cols):
    """
    Scale continuous columns within each group using standard scaling.

    Args:
        X (pd.DataFrame): Input dataframe containing features to scale.
        groups (array-like): Array or series indicating group membership for each row.
        cont_cols (list of str): List of continuous column names to scale.

    Returns:
        pd.DataFrame: A copy of X with specified continuous columns scaled within each group.
    """
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
    """
    Process input data for classification or regression tasks.

    Labels are assigned based on cytoplasmic and nuclear scores. Continuous columns are scaled
    by sample group. Returns feature matrix, labels, continuous target variables, and ID info.

    Args:
        df (pd.DataFrame): Input dataframe with scores and features.
        exclude_columns (list of str, optional): Columns to exclude from the feature matrix.
        spatial (bool, optional): Whether to use spatial scoring columns (cyt_score/nuc_score)
                                  or non-spatial columns (cyt_P1/nuc_P1).

    Returns:
        tuple: (X_scaled, y, y_nuc, y_cyt, ids)
            X_scaled (pd.DataFrame): Scaled feature matrix.
            y (pd.Series): Classification labels.
            y_nuc (pd.DataFrame): Nuclear scores.
            y_cyt (pd.DataFrame): Cytoplasmic scores.
            ids (pd.DataFrame): Identifier columns ('lr_pair', 'source', 'target').
    """
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
    """
    Process test data by scaling continuous columns for prediction.

    Args:
        df (pd.DataFrame): Test dataframe containing at least 'cell_score'.

    Returns:
        tuple: (X_scaled_test, ids)
            X_scaled_test (pd.DataFrame): Scaled features for prediction.
            ids (pd.DataFrame): Identifier columns ('lr_pair', 'source', 'target').
    """  
    # scaling the continuous columns by group (sample)
    X = df
    groups = "test_sample" # assign all to one group for scaling
    cont_cols = ['cell_score']
    X_scaled_test = scale_continuous_by_group(X, groups, cont_cols) 
    ids = df[['lr_pair', 'source', 'target']]
    return X_scaled_test, ids


# random forest model optimization
def rf_cv(max_samples,n_estimators,max_features, max_depth, min_samples_split, min_samples_leaf, X_train, y_train):
    """
    Evaluate a RandomForestClassifier using cross-validation with specified hyperparameters.

    Args:
        max_samples (float): Fraction of samples to draw for each tree.
        n_estimators (int): Number of trees in the forest.
        max_features (float or int): Max number of features per tree.
        max_depth (int): Maximum depth of each tree.
        min_samples_split (int): Minimum number of samples to split a node.
        min_samples_leaf (int): Minimum number of samples at a leaf node.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.

    Returns:
        float: Mean F1 macro score from 3-fold cross-validation.
    """
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
    """
    Evaluate an XGBoost classifier using cross-validation with specified hyperparameters.

    Args:
        max_depth (int): Maximum tree depth.
        learning_rate (float): Boosting learning rate.
        n_estimators (int): Number of boosting rounds.
        reg_alpha (float): L1 regularization term.
        colsample_bytree (float): Subsample ratio of columns for each tree.
        min_child_weight (int): Minimum sum of instance weight needed in a child.
        gamma (float): Minimum loss reduction required for a split.
        scale_pos_weight (float): Balancing of positive and negative weights.
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.

    Returns:
        float: Mean F1 macro score from 3-fold cross-validation.
    """
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
    """
    Evaluate an XGBoost regressor using cross-validation with specified hyperparameters.

    Args:
        max_depth (int): Maximum tree depth.
        learning_rate (float): Boosting learning rate.
        n_estimators (int): Number of boosting rounds.
        reg_alpha (float): L1 regularization term.
        colsample_bytree (float): Subsample ratio of columns per tree.
        min_child_weight (int): Minimum sum of instance weight needed in a child.
        gamma (float): Minimum loss reduction required for a split.
        X_train_reg (pd.DataFrame): Training features.
        y_train_reg (pd.Series): Target values.

    Returns:
        float: Mean R² score from 3-fold cross-validation.
    """
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

def rf_reg(max_depth, n_estimators,  max_samples, min_samples_leaf, min_samples_split, X_train_reg, y_train_reg):
    """
    Evaluate a RandomForestRegressor using cross-validation with specified hyperparameters.

    Args:
        max_depth (int): Maximum depth of each tree.
        n_estimators (int): Number of trees.
        max_samples (float): Fraction of samples to draw for each tree.
        min_samples_leaf (int): Minimum samples at a leaf node.
        min_samples_split (int): Minimum samples to split a node.
        X_train_reg (pd.DataFrame): Training features.
        y_train_reg (pd.Series): Target values.

    Returns:
        float: Mean R² score from 3-fold cross-validation.
    """
    max_depth = int(max_depth)
    n_estimators = int(n_estimators)
    max_samples = max_samples
    estimator_function = RandomForestRegressor(
        max_depth=max_depth,
        n_estimators=n_estimators,
        max_samples= max_samples,
        min_samples_leaf = int(min_samples_leaf),
        min_samples_split = int(min_samples_split),
        random_state=112
        )
    # Fit the estimator
    estimator_function.fit(X_train_reg,y_train_reg)
    cv = KFold(n_splits=3, shuffle=True, random_state=112)
    scores = cross_val_score(estimator_function, X_train_reg, y_train_reg, cv=cv,
                             scoring=make_scorer(r2_score))
    return scores.mean()

def classification_metrics(y_test, y_pred, y_proba, test_ids):
    """
    Compute classification metrics including F1, F2, ROC AUC, and balanced accuracy.

    Args:
        y_test (pd.Series): True labels.
        y_pred (np.array or pd.Series): Predicted labels.
        y_proba (np.array): Predicted probabilities (n_samples, n_classes).
        test_ids (pd.DataFrame): Identifier columns to include in output.

    Returns:
        tuple: (metrics, roc_table, predictions)
            metrics (dict): Calculated metrics.
            roc_table (pd.DataFrame): FPR, TPR, and thresholds for ROC.
            predictions (pd.DataFrame): Predictions combined with test_ids.
    """

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
    """
    Compute regression metrics including R², RMSE, and NRMSE.

    Args:
        y_test_reg (pd.Series): True target values.
        y_pred_reg (np.array or pd.Series): Predicted target values.
        test_ids (pd.DataFrame): Identifier columns to include in output.

    Returns:
        tuple: (metrics, predictions)
            metrics (dict): Calculated regression metrics.
            predictions (pd.DataFrame): Predictions combined with test_ids.
    """
    metrics = {"R²": r2_score(y_test_reg, y_pred_reg),
               'RMSE': np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)),
               'NRMSE': np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))/np.std(y_test_reg)}
    predictions = pd.DataFrame({'pred': y_pred_reg, 'true': y_test_reg}) # create a dataframe to store the predictions
    predictions = pd.concat([predictions, test_ids.reset_index(drop=True)], axis=1) # add the test ids to the predictions dataframe
    return metrics, predictions

def cast_params(params, int_params=None):
    """
    Convert specified hyperparameters to integers.

    Args:
        params (dict): Dictionary of hyperparameters.
        int_params (list of str, optional): List of parameter names to cast as integers.

    Returns:
        dict: Copy of params with specified keys converted to integers.
    """
    params = params.copy()
    if int_params is not None:
        for p in int_params:
            if p in params:
                params[p] = int(round(params[p]))
    return params

def test_prediction(y_pred, test_ids):
    """
    Combine predicted values with test identifiers into a dataframe.

    Args:
        y_pred (np.array or pd.Series): Predicted values.
        test_ids (pd.DataFrame): Identifier columns.

    Returns:
        pd.DataFrame: Predictions combined with test_ids.
    """
    predictions = pd.DataFrame({'pred': y_pred}) # create a dataframe to store the predictions
    predictions = pd.concat([predictions, test_ids.reset_index(drop=True)], axis=1) # add the test ids to the predictions dataframe
    return predictions

def group_generator(df):
    """
    Generate combinations of training groups for leave-one-group-out evaluation.

    Args:
        df (pd.DataFrame): Input dataframe containing a 'sample' column.

    Returns:
        dict: Dictionary mapping each test group to a list of training group combinations.
    """
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
    """
    Extract and aggregate metrics from classifier and regression results into a summary dataframe.

    Args:
        results_classifier (dict): Dictionary containing classification results per dataset.
        results_regression (dict): Dictionary containing regression results per dataset.
        model_name (str): Name of the model for labeling.

    Returns:
        pd.DataFrame: Summary dataframe including AUC, macro metrics, R², RMSE, NRMSE, and a composite metric.
    """
       auc = [results_classifier[group]['classification_metrics']['roc_auc']
              for group in results_classifier.keys()]
       reports = [results_classifier[group]['classification_metrics']['class_report']
              for group in results_classifier.keys()]
       macro_recalls = []
       macro_precisions = []
       macro_f1s = []
       for report in reports:
              for line in report.splitlines():
                     if line.strip().startswith("macro avg"):
                     # Split the line into parts and take the second number (recall)
                            macro_precision = float(line.split()[2])
                            macro_recall = float(line.split()[3])
                            macro_f1 = float(line.split()[4])
                            macro_precisions.append(macro_precision)
                            macro_recalls.append(macro_recall)
                            macro_f1s.append(macro_f1)
       cyt_R2 = [results_regression[group]['cyt'][0]['R²']
              for group in results_regression.keys()]
       nuc_R2 = [results_regression[group]['nuc'][0]['R²']
                     for group in results_regression.keys()]
       cyt_NRMSE = [results_regression[group]['cyt'][0]['NRMSE']
                     for group in results_regression.keys()]
       nuc_NRMSE = [results_regression[group]['nuc'][0]['NRMSE']
                     for group in results_regression.keys()]
       cyt_RMSE = [results_regression[group]['cyt'][0]['RMSE']
                     for group in results_regression.keys()]
       nuc_RMSE = [results_regression[group]['nuc'][0]['RMSE']
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
       "macro_precision": macro_precisions,
       "macro_recall": macro_recalls,
       "macro_f1": macro_f1s,
       "cyt_R2": cyt_R2,
       "nuc_R2": nuc_R2,
       "cyt_NRMSE": cyt_NRMSE,
       "nuc_NRMSE": nuc_NRMSE,
       "cyt_RMSE": cyt_RMSE,
       'nuc_RMSE': nuc_RMSE,
       "composite_metric": composite_metric
       })
       return metrics