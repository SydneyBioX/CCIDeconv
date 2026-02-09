# train.py
from models import ClassifierWrapper, RegressorWrapper

def train_classifier(X_train, y_train, rf_params, xgb_params):
    """
    Train a classification model using RandomForest and XGBoost in a soft voting ensemble.

    Args:
        X_train (array-like): Feature matrix for training.
        y_train (array-like): Target vector for training.
        rf_params (dict, optional): Parameters for RandomForestClassifier.
        xgb_params (dict, optional): Parameters for XGBClassifier.

    Returns:
        ClassifierWrapper: The trained classifier wrapper containing the fitted VotingClassifier.
    """
    clf = ClassifierWrapper(rf_params=rf_params, xgb_params=xgb_params)
    model_clf = clf.train(X_train, y_train)
    return clf  # return the trained and fit model

def train_regressor(X_train, y_train,xgb_params, X_val=None, y_val=None):
    """
    Train an XGBoost regression model with optional early stopping using a validation set.

    Args:
        X_train (array-like): Feature matrix for training.
        y_train (array-like): Target vector for training.
        xgb_params (dict, optional): Parameters for XGBRegressor.
        X_val (array-like, optional): Feature matrix for validation (for early stopping).
        y_val (array-like, optional): Target vector for validation (for early stopping).

    Returns:
        RegressorWrapper: The trained regressor wrapper containing the fitted XGBRegressor.
    """
    reg = RegressorWrapper(xgb_params)
    model_reg = reg.train(X_train, y_train, X_val=X_val, y_val=y_val)
    return reg  # return the trained and fit model 
