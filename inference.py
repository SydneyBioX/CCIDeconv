# inference.py
import numpy as np
from utils import classification_metrics, regression_metrics, test_prediction
class HierarchicalClassifier:
    """
    Handles classification step
    """
    def __init__(self, classifier_wrapper):
        self.classifier_wrapper = classifier_wrapper
        self.y_pred = None
        self.y_proba = None
        self.metrics = None
        self.roc_table = None
        self.predictions = None

    def evaluate(self, X_test, y_test, test_ids):
        """
        Perform prediction and calculate metrics
        """
        self.y_pred = self.classifier_wrapper.model.predict(X_test)
        self.y_proba = self.classifier_wrapper.model.predict_proba(X_test)
        self.metrics, self.roc_table, self.predictions = classification_metrics(
            y_test, self.y_pred, self.y_proba, test_ids
        )
        return self.y_pred, self.metrics, self.roc_table, self.predictions
    def prediction(self, X_test, test_ids):
        """ 
        Perform prediction on new data
        """
        self.y_pred = self.classifier_wrapper.model.predict(X_test)
        self.predictions = test_prediction(self.y_pred, test_ids)
        return self.predictions


class HierarchicalRegressor:
    """
    Handles regression step on predicted positives
    """
    def __init__(self, reg_cyt_wrapper, reg_nuc_wrapper):
        self.reg_cyt_wrapper = reg_cyt_wrapper
        self.reg_nuc_wrapper = reg_nuc_wrapper
        self.results = {}

    def evaluate(self, X_test, cyt_scores_test, nuc_scores_test, test_ids):
        """
        Predict only for samples where y_pred == 1
        """
        self.results = {}

        # Cytoplasm
        y_pred_cyt = self.reg_cyt_wrapper.model.predict(X_test)
        metrics_cyt, predictions_cyt = regression_metrics(cyt_scores_test, y_pred_cyt, test_ids)
        self.results['cyt'] = (metrics_cyt, predictions_cyt)

        # Nucleus
        y_pred_nuc = self.reg_nuc_wrapper.model.predict(X_test)
        metrics_nuc, predictions_nuc = regression_metrics(
            nuc_scores_test, y_pred_nuc, test_ids
        )
        self.results['nuc'] = (metrics_nuc, predictions_nuc)

        return self.results
    
    def prediction(self, X_test, test_ids):
        """ 
        Perform prediction on new data
        """
        y_pred_cyt = self.reg_cyt_wrapper.model.predict(X_test)
        y_pred_nuc = self.reg_nuc_wrapper.model.predict(X_test)
        predictions_cyt = test_prediction(y_pred_cyt, test_ids)
        predictions_nuc = test_prediction(y_pred_nuc, test_ids)
        return predictions_cyt, predictions_nuc

