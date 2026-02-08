# config.py
# Hyperparamter tuning for RF
hyperparameter_space_rf_class =  {
        'max_samples':(0.1,1),
        'max_features':(0.3,1),
        'n_estimators':(150,300),
        'max_depth':(5,30),
        'min_samples_split':(2,10),
        'min_samples_leaf':(2,10)
        }
    
hyperparameter_space_reg = {
    'max_depth': (10, 20),
    'learning_rate': (0.01, 0.1),
    'n_estimators' : (100, 300),
    'reg_alpha': (0, 1),
    'min_child_weight': (1, 3),
    'colsample_bytree': (0.8, 1),
    'gamma' : (0, 3)}
    
hyperparameter_space_xgb_class = {
    'max_depth': (6, 10),
    'learning_rate': (0.01, 0.3),
    'n_estimators' : (100,300),
    'reg_alpha': (0,1),
    'min_child_weight': (1, 20),
    'colsample_bytree': (0.5, 1),
    'gamma' : (0,5),
    'scale_pos_weight': (1, 100)
    }

hyperparameter_space_reg_rf = {
    'max_depth': (5, 30),
    'n_estimators' : (100, 300),
    'max_samples': (0.5,1),
    'min_samples_split': (2, 20),
    'min_samples_leaf': (1, 10)}

gp_params = {"alpha": 1e-10}
seed = 112