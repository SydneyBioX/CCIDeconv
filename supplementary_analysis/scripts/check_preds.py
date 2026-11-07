import joblib
import pandas as pd

reg_pred = joblib.load("regression_predictions.joblib")
print(f"regression_predictions.joblib type: {type(reg_pred)}")
if isinstance(reg_pred, tuple):
    print(f"len: {len(reg_pred)}")
    for i, item in enumerate(reg_pred):
        print(f"item {i} type: {type(item)}")
        if hasattr(item, 'head'):
            print(f"item {i} shape: {item.shape}, cols: {list(item.columns)[:5]}")
            print(item.head(2))

clf_pred = joblib.load("classification_predictions.joblib")
print(f"classification_predictions.joblib shape: {clf_pred.shape}, cols: {list(clf_pred.columns)}")
print(clf_pred.head(2))

# Also check difference between results_regression.joblib and data/results_regression.joblib
r1 = joblib.load("results_regression.joblib")
r2 = joblib.load("data/results_regression.joblib")
print(f"results_regression.joblib keys: {list(r1.keys())}")
print(f"data/results_regression.joblib keys: {list(r2.keys())}")

for k in r1.keys():
    print(f"Dataset {k}: r1 cyt shape: {r1[k]['cyt'][1].shape}, r2 cyt shape: {r2[k]['cyt'][1].shape}")
    print(f"  r1 cyt pred mean: {r1[k]['cyt'][1]['pred'].mean():.4f}, r2 cyt pred mean: {r2[k]['cyt'][1]['pred'].mean():.4f}")
    print(f"  r1 cyt true mean: {r1[k]['cyt'][1]['true'].mean():.4f}, r2 cyt true mean: {r2[k]['cyt'][1]['true'].mean():.4f}")
