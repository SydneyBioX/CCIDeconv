# experiments/run_ablation_study.py
"""
CCIDeconv Feature Ablation Study

Evaluates model dependence on subcellular localization features (CellChatDB and HPA),
ligand/receptor identities, and whole-cell scores across the 9 spatial transcriptomics
datasets using Leave-One-Dataset-Out (LOGO) cross-validation for both Spatial Procedure (SP)
and Single-Cell Procedure (ScP).

Conditions:
- full: Full Model (Baseline) - all features included.
- no_cellchat: Ablation 1A - removes CellChatDB localization features.
- no_hpa: Ablation 1B - removes HPA localization features.
- no_localization: Ablation 1C - removes all localization features.
- no_lr_identity: Ablation 2 - removes lr_pair identity while retaining cell types (source, target), localization, and properties.
- no_whole_cell: Ablation 3 - removes whole-cell communication score (cell_score & cell_pspatial for SP, cell_P1 for ScP).
"""

import argparse
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from category_encoders import CatBoostEncoder
from bayes_opt import BayesianOptimization
import sys
import os
import random
import time
import warnings
warnings.filterwarnings("ignore")

# Robust path resolution
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate from CCIDeconv/supplementary_analysis/scripts/ablation_analysis to CCIDeconv
CCIDECONV_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.append(CCIDECONV_DIR)

from utils import process_data, rf_cv, cast_params, xgbc_cv, xgb_reg
from config import hyperparameter_space_rf_class, hyperparameter_space_xgb_class, hyperparameter_space_reg, gp_params, seed
from train import train_classifier, train_regressor
from inference import HierarchicalClassifier, HierarchicalRegressor

# Calibrated baseline hyperparameters for controlled, fast ablation
CALIBRATED_RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 5,
    'min_samples_leaf': 3,
    'max_features': 0.5,
    'max_samples': 0.8,
    'random_state': 42,
    'n_jobs': -1
}

CALIBRATED_XGBC_PARAMS = {
    'n_estimators': 200,
    'max_depth': 8,
    'learning_rate': 0.1,
    'min_child_weight': 5,
    'gamma': 1.0,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.5,
    'scale_pos_weight': 1.0
}

CALIBRATED_REG_CYT_PARAMS = {
    'n_estimators': 150,
    'max_depth': 12,
    'learning_rate': 0.05,
    'min_child_weight': 2,
    'gamma': 1.0,
    'colsample_bytree': 0.9,
    'reg_alpha': 0.5
}

CALIBRATED_REG_NUC_PARAMS = {
    'n_estimators': 150,
    'max_depth': 12,
    'learning_rate': 0.05,
    'min_child_weight': 2,
    'gamma': 1.0,
    'colsample_bytree': 0.9,
    'reg_alpha': 0.5
}

BASE_EXCLUDE_SP = [
    'cyt_pval', 'cyt_pspatial', 'cyt_P1', 'sample', 'cell_pval', 'cell_P1', 'tissue',
    'is_neurotransmitter', 'ligand_location_cellchat', 'receptor_location_cellchat',
    'ligand_location_hpa', 'receptor_location_hpa', 'nuc_pval', 'nuc_pspatial', 'nuc_P1',
    'ligand', 'receptor', 'labels'
]

BASE_EXCLUDE_SCP = [
    'cyt_pval', 'cyt_pspatial', 'cyt_score', 'sample', 'cell_pval', 'cell_pspatial', 'cell_score',
    'tissue', 'is_neurotransmitter', 'ligand_location_cellchat', 'receptor_location_cellchat',
    'ligand_location_hpa', 'receptor_location_hpa', 'nuc_pval', 'nuc_pspatial', 'nuc_score',
    'ligand', 'receptor', 'labels'
]

ALL_CATEGORICAL_COLUMNS = [
    'lr_pair', 'source', 'target', 'pathway_name', 'annotation',
    'ligand.family', 'ligand.keyword', 'ligand.secreted_type', 'ligand.transmembrane',
    'receptor.family', 'receptor.keyword', 'receptor.surfaceome_main', 'receptor.surfaceome_sub',
    'receptor.adhesome', 'receptor.secreted_type', 'receptor.transmembrane'
]

MOLECULAR_ANNOTATION_COLUMNS = [
    'pathway_name', 'annotation',
    'ligand.family', 'ligand.keyword', 'ligand.secreted_type', 'ligand.transmembrane',
    'receptor.family', 'receptor.keyword', 'receptor.surfaceome_main', 'receptor.surfaceome_sub',
    'receptor.adhesome', 'receptor.secreted_type', 'receptor.transmembrane'
]


def get_condition_columns(condition, df, spatial=True):
    """
    Determine exclude_columns, categorical_columns, and post-scaling drop columns
    for each ablation condition and procedure (SP vs ScP).
    """
    cellchat_loc_cols = [c for c in df.columns if 'location_cellchat_' in c]
    hpa_loc_cols = [c for c in df.columns if 'location_hpa_' in c]
    all_loc_cols = cellchat_loc_cols + hpa_loc_cols

    base_exclude = list(BASE_EXCLUDE_SP if spatial else BASE_EXCLUDE_SCP)
    exclude_cols = [c for c in base_exclude if c 
    in df.columns]
    cat_cols = list(ALL_CATEGORICAL_COLUMNS)
    drop_after_scaling = []

    if condition == 'full':
        pass
    elif condition == 'no_cellchat':
        exclude_cols.extend(cellchat_loc_cols)
    elif condition == 'no_hpa':
        exclude_cols.extend(hpa_loc_cols)
    elif condition == 'no_localization':
        exclude_cols.extend(all_loc_cols)
    elif condition == 'no_lr_identity':
        # Drop lr_pair, explicitly retain source, target and all other categoricals
        exclude_cols.append('lr_pair')
        cat_cols = [c for c in cat_cols if c != 'lr_pair']
    elif condition == 'no_whole_cell':
        # Must be dropped after scaling in process_data to avoid KeyError in scale_continuous_by_group
        if spatial:
            drop_after_scaling = ['cell_score', 'cell_pspatial']
        else:
            drop_after_scaling = ['cell_P1']
    elif condition == 'no_annotations':
        # Ablation 4: Remove biological/molecular annotations, retain locations and core IDs
        exclude_cols.extend(MOLECULAR_ANNOTATION_COLUMNS)
        cat_cols = [c for c in cat_cols if c not in MOLECULAR_ANNOTATION_COLUMNS]
    elif condition == 'no_annotations_no_localization':
        # Ablation 5: Remove both molecular annotations AND all 192 location features
        exclude_cols.extend(all_loc_cols)
        exclude_cols.extend(MOLECULAR_ANNOTATION_COLUMNS)
        cat_cols = [c for c in cat_cols if c not in MOLECULAR_ANNOTATION_COLUMNS]
    else:
        raise ValueError(f"Unknown condition: {condition}")

    return exclude_cols, cat_cols, drop_after_scaling


def run_single_condition(condition, df, procedure='sp', group_column='sample', tune=False, output_dir='supplementary_analysis/results/ablation'):
    """
    Execute Leave-One-Dataset-Out cross-validation for a specific ablation condition and procedure.
    """
    spatial = (procedure.lower() == 'sp')
    proc_label = "SP (Spatial)" if spatial else "ScP (Single-Cell)"

    print(f"\n{'='*75}")
    print(f"RUNNING CONDITION: {condition.upper()} | PROCEDURE: {proc_label}")
    print(f"{'='*75}")

    exclude_cols, cat_cols, drop_after_scaling = get_condition_columns(condition, df, spatial=spatial)
    X_scaled, y, y_nuc, y_cyt, ids = process_data(df, exclude_columns=exclude_cols, spatial=spatial)

    if drop_after_scaling:
        cols_to_drop = [c for c in drop_after_scaling if c in X_scaled.columns]
        X_scaled = X_scaled.drop(columns=cols_to_drop)
        print(f"Dropped whole-cell features: {cols_to_drop}")

    print(f"Processed feature matrix shape: {X_scaled.shape}")
    print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")

    groups = df[group_column].values
    catboost_encoder = CatBoostEncoder()
    logo = LeaveOneGroupOut()

    results_classifier = {}
    results_regression = {}

    start_time = time.time()
    for fold_i, (train_idx, test_idx) in enumerate(logo.split(X_scaled, y, groups)):
        group_name = np.unique(groups[test_idx])[0]
        fold_start = time.time()
        print(f"\n--- Fold {fold_i + 1}/9: Evaluating held-out dataset '{group_name}' ({len(test_idx)} samples) ---")

        X_train, X_test = X_scaled.iloc[train_idx].copy(), X_scaled.iloc[test_idx].copy()
        y_train, y_test = y.iloc[train_idx].values.ravel(), y.iloc[test_idx].values.ravel()
        test_ids = ids.iloc[test_idx].copy()

        # Categorical Encoding
        if len(cat_cols) > 0:
            cat_train = catboost_encoder.fit_transform(X_train[cat_cols], y_train)
            X_train = pd.concat([X_train.drop(columns=cat_cols).reset_index(drop=True), cat_train.reset_index(drop=True)], axis=1)

            cat_test = catboost_encoder.transform(X_test[cat_cols])
            X_test = pd.concat([X_test.drop(columns=cat_cols).reset_index(drop=True), cat_test.reset_index(drop=True)], axis=1)

        X_train_np = X_train.to_numpy(dtype=float)
        X_test_np = X_test.to_numpy(dtype=float)

        if tune:
            print("  Tuning RandomForest classifier via Bayesian Optimization...")
            rf_bo = BayesianOptimization(
                f=lambda **params: rf_cv(**params, X_train=X_train_np, y_train=y_train),
                pbounds=hyperparameter_space_rf_class,
                random_state=seed,
                verbose=0
            )
            rf_bo.maximize(init_points=2, n_iter=8)
            rf_int_params = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']
            rf_params = cast_params(rf_bo.max['params'], int_params=rf_int_params)

            print("  Tuning XGBoost classifier via Bayesian Optimization...")
            xgb_bo = BayesianOptimization(
                f=lambda **params: xgbc_cv(**params, X_train=X_train_np, y_train=y_train),
                pbounds=hyperparameter_space_xgb_class,
                random_state=seed,
                verbose=0
            )
            xgb_bo.set_gp_params(**gp_params)
            xgb_bo.maximize(init_points=2, n_iter=8)
            xgb_int_params = ['n_estimators', 'max_depth']
            xgb_params = cast_params(xgb_bo.max['params'], int_params=xgb_int_params)
        else:
            rf_params = CALIBRATED_RF_PARAMS.copy()
            xgb_params = CALIBRATED_XGBC_PARAMS.copy()

        # Train Classifier Ensemble
        classifier_wrapper = train_classifier(X_train_np, y_train, rf_params, xgb_params)
        y_pred, class_metrics, roc_table, predictions = HierarchicalClassifier(classifier_wrapper).evaluate(X_test_np, y_test, test_ids)

        results_classifier[group_name] = {
            'y_pred': y_pred,
            'classification_metrics': class_metrics,
            'roc_table': roc_table,
            'predictions': predictions
        }
        print(f"  Classification AUC: {class_metrics['roc_auc']:.4f} | Macro F1: {class_metrics['f1']:.4f}")

        # Train Regressors on Positives
        mask_train = (y_train == 1)
        mask_test = (y_pred == 1)
        test_ids_pos = test_ids[mask_test]

        y_train_cyt = y_cyt.iloc[train_idx, :].to_numpy()[mask_train]
        y_train_nuc = y_nuc.iloc[train_idx, :].to_numpy()[mask_train]

        if tune:
            xgb_int_params = ['n_estimators', 'max_depth']
            xgb_reg_cyt_bo = BayesianOptimization(
                f=lambda **params: xgb_reg(**params, X_train_reg=X_train_np[mask_train, :], y_train_reg=y_train_cyt),
                pbounds=hyperparameter_space_reg,
                random_state=seed,
                verbose=0
            )
            xgb_reg_cyt_bo.set_gp_params(**gp_params)
            xgb_reg_cyt_bo.maximize(init_points=2, n_iter=8)
            reg_cyt_params = cast_params(xgb_reg_cyt_bo.max['params'], int_params=xgb_int_params)

            xgb_reg_nuc_bo = BayesianOptimization(
                f=lambda **params: xgb_reg(**params, X_train_reg=X_train_np[mask_train, :], y_train_reg=y_train_nuc),
                pbounds=hyperparameter_space_reg,
                random_state=seed,
                verbose=0
            )
            xgb_reg_nuc_bo.set_gp_params(**gp_params)
            xgb_reg_nuc_bo.maximize(init_points=2, n_iter=8)
            reg_nuc_params = cast_params(xgb_reg_nuc_bo.max['params'], int_params=xgb_int_params)
        else:
            reg_cyt_params = CALIBRATED_REG_CYT_PARAMS.copy()
            reg_nuc_params = CALIBRATED_REG_NUC_PARAMS.copy()

        # Train regressor cleanly on X_train without test-set early stopping
        reg_cyt_wrapper = train_regressor(
            X_train_np[mask_train, :], y_train_cyt, reg_cyt_params,
            X_val=None, y_val=None
        )
        reg_nuc_wrapper = train_regressor(
            X_train_np[mask_train, :], y_train_nuc, reg_nuc_params,
            X_val=None, y_val=None
        )

        if mask_test.sum() >= 2:
            reg_results = HierarchicalRegressor(reg_cyt_wrapper, reg_nuc_wrapper).evaluate(
                X_test_np[mask_test, :],
                y_cyt.iloc[test_idx, :].to_numpy()[mask_test].flatten(),
                y_nuc.iloc[test_idx, :].to_numpy()[mask_test].flatten(),
                test_ids_pos
            )
            cyt_r2_val = reg_results['cyt'][0]['R²']
            nuc_r2_val = reg_results['nuc'][0]['R²']
        else:
            # Fallback for empty/single prediction
            reg_results = {
                'cyt': ({'R²': np.nan, 'RMSE': np.nan, 'NRMSE': np.nan}, pd.DataFrame()),
                'nuc': ({'R²': np.nan, 'RMSE': np.nan, 'NRMSE': np.nan}, pd.DataFrame())
            }
            cyt_r2_val, nuc_r2_val = np.nan, np.nan

        results_regression[group_name] = reg_results

        print(f"  Regression Cyt R²: {cyt_r2_val:.4f} | Nuc R²: {nuc_r2_val:.4f}")
        print(f"  Fold completed in {time.time() - fold_start:.1f}s")

    os.makedirs(output_dir, exist_ok=True)
    proc_suffix = f"_{procedure.lower()}"
    clf_path = os.path.join(output_dir, f"results_classifier{proc_suffix}_{condition}.joblib")
    reg_path = os.path.join(output_dir, f"results_regression{proc_suffix}_{condition}.joblib")

    joblib.dump(results_classifier, clf_path)
    joblib.dump(results_regression, reg_path)

    # For SP, also dump without prefix for backwards compatibility with existing scripts
    if spatial:
        joblib.dump(results_classifier, os.path.join(output_dir, f"results_classifier_{condition}.joblib"))
        joblib.dump(results_regression, os.path.join(output_dir, f"results_regression_{condition}.joblib"))

    print(f"\nCondition '{condition}' ({proc_label}) finished in {time.time() - start_time:.1f}s.")
    print(f"Saved: {clf_path}")
    print(f"Saved: {reg_path}")

    return results_classifier, results_regression


def main():
    parser = argparse.ArgumentParser(description="Run CCIDeconv Feature Ablation Study")
    default_data = os.path.join(CCIDECONV_DIR, "data", "training_data", "training_data.csv")
    default_output = os.path.join(CCIDECONV_DIR, "supplementary_analysis", "results", "ablation")

    parser.add_argument("--data_path", type=str, default=default_data,
                        help="Path to training CSV file")
    parser.add_argument("--condition", type=str, default="all",
                        choices=['all', 'full', 'no_cellchat', 'no_hpa', 'no_localization', 'no_lr_identity', 'no_whole_cell', 'no_annotations', 'no_annotations_no_localization'],
                        help="Ablation condition to run")
    parser.add_argument("--procedure", type=str, default="both",
                        choices=['both', 'sp', 'scp'],
                        help="Procedure to evaluate: sp (spatial), scp (single-cell), or both")
    parser.add_argument("--group_column", type=str, default="sample", help="Group column for LOGO")
    parser.add_argument("--tune", action='store_true', default=False, help="Run Bayesian optimization tuning")
    parser.add_argument("--output_dir", type=str, default=default_output, help="Directory to save results")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)

    conditions_to_run = (
        ['full', 'no_cellchat', 'no_hpa', 'no_localization', 'no_lr_identity', 'no_whole_cell', 'no_annotations', 'no_annotations_no_localization']
        if args.condition == 'all'
        else [args.condition]
    )

    procedures_to_run = (
        ['sp', 'scp'] if args.procedure == 'both'
        else [args.procedure]
    )

    for proc in procedures_to_run:
        for cond in conditions_to_run:
            run_single_condition(
                cond, df,
                procedure=proc,
                group_column=args.group_column,
                tune=args.tune,
                output_dir=args.output_dir
            )


if __name__ == "__main__":
    main()
