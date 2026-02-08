# train.py
from models import ClassifierWrapper, RegressorWrapper

def train_classifier(X_train, y_train, rf_params, xgb_params):
    clf = ClassifierWrapper(rf_params=rf_params, xgb_params=xgb_params)
    model_clf = clf.train(X_train, y_train)
    return clf  # return the trained and fit model

def train_regressor(X_train, y_train,xgb_params, X_val=None, y_val=None):
    reg = RegressorWrapper(xgb_params)
    model_reg = reg.train(X_train, y_train, X_val=X_val, y_val=y_val)
    return reg  # return the trained and fit model 
