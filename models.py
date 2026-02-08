# models.py
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier, XGBRegressor

class ClassifierWrapper:
    def __init__(self, rf_params=None, xgb_params=None):
        self.rf_params = rf_params
        self.xgb_params = xgb_params
        self.model = None

    def train(self, X_train, y_train):
        rf = RandomForestClassifier(**self.rf_params)
        xgb = XGBClassifier(**self.xgb_params,
                            nthread = -1,
                            seed = 112,
                            objective='binary:logistic')
        self.model = VotingClassifier([('rf', rf), ('xgb', xgb)], voting='soft')
        self.model.fit(X_train, y_train)
        return self.model

class RegressorWrapper:
    def __init__(self, xgb_params):
        self.xgb_params = xgb_params
        self.model = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        params = dict(self.xgb_params)
        params.update({
            "objective": "reg:squarederror",
            "random_state": 112,
            "n_jobs": -1
        })
        eval_set = None
        if X_val is not None and y_val is not None:
            params["early_stopping_rounds"] = 50
            eval_set = [(X_val, y_val)]
        self.model = XGBRegressor(**params)
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        return self.model