import joblib
import os

paths = [
    "data/results_regression.joblib",
    "data/results_regression_combinations.joblib",
    "results_regression.joblib",
    "classification_predictions.joblib",
    "regression_predictions.joblib",
    "data/processed/single_cell/results_P1spatial_removed/xgboost_voting_metrics_P1_trained.csv"
]

for p in paths:
    if os.path.exists(p):
        print(f"File: {p} (size: {os.path.getsize(p):,} bytes)")
        if p.endswith('.joblib'):
            try:
                data = joblib.load(p)
                print(f"  Type: {type(data)}")
                if isinstance(data, dict):
                    print(f"  Keys: {list(data.keys())[:5]}")
                    first_val = data[list(data.keys())[0]]
                    print(f"  First val type: {type(first_val)}")
                    if isinstance(first_val, dict):
                        print(f"    Sub-keys: {list(first_val.keys())}")
                        if 'cyt' in first_val:
                            print(f"    first_val['cyt'] type: {type(first_val['cyt'])}, len: {len(first_val['cyt'])}")
                            if len(first_val['cyt']) > 1 and hasattr(first_val['cyt'][1], 'columns'):
                                print(f"    cyt[1] cols: {list(first_val['cyt'][1].columns)}")
                                print(f"    cyt[1] shape: {first_val['cyt'][1].shape}")
            except Exception as e:
                print(f"  Error loading: {e}")
