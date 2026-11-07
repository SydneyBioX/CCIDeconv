# supplementary_analysis/scripts/random_seed_sensitivity/run_logo.py
"""
run_logo.py - Random Seed Sensitivity Analysis for CCIDeconv LOGO-CV

Evaluates Leave-One-Group-Out (LOGO) cross-validation under multiple random seeds
to assess model robustness and sensitivity to stochastic initialization.
Unlike the original experiments/run_logo.py where seeds (42 and 112) were hardcoded,
this script exposes the random seed across all components:
  1. Python random & NumPy random state
  2. Inner K-Fold cross-validation splits
  3. BayesianOptimization exploration points
  4. Random Forest & XGBoost classifiers and regressors

Outputs (in --out_dir):
  - results_classifier_seed_<seed>.joblib
  - results_regression_seed_<seed>.joblib
  - metrics_seed_<seed>.csv
  - seed_sensitivity_per_dataset.csv
  - seed_sensitivity_summary.csv
"""
import argparse
import os
import random
import sys

import joblib
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from category_encoders import CatBoostEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import KFold, LeaveOneGroupOut, cross_val_score
from xgboost import XGBClassifier, XGBRegressor

# Ensure CCIDeconv root directory is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from config import (
    gp_params,
    hyperparameter_space_reg,
    hyperparameter_space_rf_class,
    hyperparameter_space_xgb_class,
)
from inference import HierarchicalClassifier, HierarchicalRegressor
from utils import cast_params, extract_metrics, process_data


# ----------------------------------------------------------------------
# Seeded Model Wrappers
# ----------------------------------------------------------------------
class SeededClassifierWrapper:
    """VotingClassifier wrapper that propagates the random seed to RF and XGB."""
    def __init__(self, rf_params=None, xgb_params=None, seed=42):
        self.rf_params = rf_params.copy() if rf_params else {}
        self.xgb_params = xgb_params.copy() if xgb_params else {}
        self.seed = seed
        self.model = None

    def train(self, X_train, y_train):
        rf_kwargs = self.rf_params.copy()
        rf_kwargs['random_state'] = self.seed
        rf = RandomForestClassifier(**rf_kwargs)

        xgb_kwargs = self.xgb_params.copy()
        xgb_kwargs.update({
            'nthread': -1,
            'seed': self.seed,
            'random_state': self.seed,
            'objective': 'binary:logistic'
        })
        xgb = XGBClassifier(**xgb_kwargs)
        self.model = VotingClassifier([('rf', rf), ('xgb', xgb)], voting='soft')
        self.model.fit(X_train, y_train)
        return self.model


class SeededRegressorWrapper:
    """XGBRegressor wrapper that propagates the random seed."""
    def __init__(self, xgb_params, seed=42):
        self.xgb_params = xgb_params.copy() if xgb_params else {}
        self.seed = seed
        self.model = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        params = dict(self.xgb_params)
        params.update({
            "objective": "reg:squarederror",
            "random_state": self.seed,
            "seed": self.seed,
            "n_jobs": -1
        })
        eval_set = None
        if X_val is not None and y_val is not None:
            params["early_stopping_rounds"] = 50
            eval_set = [(X_val, y_val)]
        self.model = XGBRegressor(**params)
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        return self.model


# ----------------------------------------------------------------------
# Seeded CV Tuning Functions
# ----------------------------------------------------------------------
def make_rf_cv(seed):
    def rf_cv_seeded(max_samples, n_estimators, max_features, max_depth, min_samples_split, min_samples_leaf, X_train, y_train):
        params = {
            'max_samples': max_samples,
            'max_features': max_features,
            'n_estimators': int(n_estimators),
            'random_state': seed,
            'max_depth': int(max_depth),
            'min_samples_split': int(min_samples_split),
            'min_samples_leaf': int(min_samples_leaf)
        }
        estimator = RandomForestClassifier(**params)
        cv = KFold(n_splits=3, shuffle=True, random_state=seed)
        f1_scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
        return f1_scores.mean()
    return rf_cv_seeded


def make_xgbc_cv(seed):
    def xgbc_cv_seeded(max_depth, learning_rate, n_estimators, reg_alpha, colsample_bytree, min_child_weight, gamma, scale_pos_weight, X_train, y_train):
        estimator = XGBClassifier(
            max_depth=int(max_depth),
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            min_child_weight=int(min_child_weight),
            learning_rate=learning_rate,
            n_estimators=int(n_estimators),
            reg_alpha=reg_alpha,
            nthread=-1,
            objective='binary:logistic',
            scale_pos_weight=scale_pos_weight,
            seed=seed,
            random_state=seed
        )
        cv = KFold(n_splits=3, shuffle=True, random_state=seed)
        f1_scores = cross_val_score(estimator, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
        return f1_scores.mean()
    return xgbc_cv_seeded


def make_xgb_reg(seed):
    def xgb_reg_seeded(max_depth, learning_rate, n_estimators, reg_alpha, colsample_bytree, min_child_weight, gamma, X_train_reg, y_train_reg):
        estimator = XGBRegressor(
            max_depth=int(max_depth),
            learning_rate=learning_rate,
            n_estimators=int(n_estimators),
            reg_alpha=reg_alpha,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            objective='reg:squarederror',
            random_state=seed,
            seed=seed,
            n_jobs=-1
        )
        cv = KFold(n_splits=3, shuffle=True, random_state=seed)
        scores = cross_val_score(estimator, X_train_reg, y_train_reg, cv=cv, scoring=make_scorer(r2_score))
        return scores.mean()
    return xgb_reg_seeded


# ----------------------------------------------------------------------
# Run Single Seed Experiment
# ----------------------------------------------------------------------
def run_seed_experiment(seed_val, args, df):
    csv_path = os.path.join(args.out_dir, f"metrics_seed_{seed_val}.csv")
    clf_path = os.path.join(args.out_dir, f"results_classifier_seed_{seed_val}.joblib")
    reg_path = os.path.join(args.out_dir, f"results_regression_seed_{seed_val}.joblib")
    if os.path.exists(csv_path) and os.path.exists(clf_path) and os.path.exists(reg_path) and not getattr(args, 'force_rerun', False):
        print(f"[Seed {seed_val}] Found existing results in {args.out_dir}. Loading cached metrics...")
        metrics_df = pd.read_csv(csv_path)
        return metrics_df

    print(f"\n{'='*70}\nSTARTING LOGO RUN FOR SEED: {seed_val}\n{'='*70}")
    random.seed(seed_val)
    np.random.seed(seed_val)

    X_scaled, y, y_nuc, y_cyt, ids = process_data(
        df,
        exclude_columns=args.exclude_columns.split(','),
        spatial=args.spatial
    )
    print(f"[Seed {seed_val}] Data processed: X shape = {X_scaled.shape}")

    categorical_columns = args.categorical_columns.split(',')
    groups = df[args.group_column].values
    logo = LeaveOneGroupOut()

    rf_cv_fn = make_rf_cv(seed_val)
    xgbc_cv_fn = make_xgbc_cv(seed_val)
    xgb_reg_fn = make_xgb_reg(seed_val)

    results_classifier = {}
    results_regression = {}

    for train_idx, test_idx in logo.split(X_scaled, y, groups):
        X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx].values.ravel(), y.iloc[test_idx].values.ravel()
        test_ids = ids.iloc[test_idx]
        group_name = np.unique(groups[test_idx])[0]

        # Categorical encoding
        catboost_encoder = CatBoostEncoder(random_state=seed_val)
        cat_train = catboost_encoder.fit_transform(X_train[categorical_columns], y_train)
        X_train = pd.concat([X_train.drop(columns=categorical_columns).reset_index(drop=True), cat_train.reset_index(drop=True)], axis=1)

        cat_test = catboost_encoder.transform(X_test[categorical_columns])
        X_test = pd.concat([X_test.drop(columns=categorical_columns).reset_index(drop=True), cat_test.reset_index(drop=True)], axis=1)

        X_train = X_train.to_numpy(dtype=float)
        X_test = X_test.to_numpy(dtype=float)

        # Tune RF classifier
        rf_bo = BayesianOptimization(
            f=lambda **params: rf_cv_fn(**params, X_train=X_train, y_train=y_train),
            pbounds=hyperparameter_space_rf_class,
            random_state=seed_val,
            verbose=0
        )
        rf_bo.maximize(init_points=args.init_points, n_iter=args.n_iter)
        rf_int_params = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']
        rf_params = cast_params(rf_bo.max['params'], int_params=rf_int_params)

        # Tune XGB classifier
        xgbc_bo = BayesianOptimization(
            f=lambda **params: xgbc_cv_fn(**params, X_train=X_train, y_train=y_train),
            pbounds=hyperparameter_space_xgb_class,
            random_state=seed_val,
            verbose=0
        )
        xgbc_bo.set_gp_params(**gp_params)
        xgbc_bo.maximize(init_points=args.init_points, n_iter=args.n_iter)
        xgb_int_params = ['n_estimators', 'max_depth']
        xgb_params = cast_params(xgbc_bo.max['params'], int_params=xgb_int_params)

        # Train voting classifier
        classifier_wrapper = SeededClassifierWrapper(rf_params, xgb_params, seed=seed_val)
        classifier_wrapper.train(X_train, y_train)

        # Evaluate classifier
        y_pred, class_metrics, roc_table, predictions = HierarchicalClassifier(classifier_wrapper).evaluate(X_test, y_test, test_ids)
        results_classifier[group_name] = {
            'y_pred': y_pred,
            'classification_metrics': class_metrics,
            'roc_table': roc_table,
            'predictions': predictions
        }

        # Train regressors on true positives
        mask_train = y_train == 1
        mask_test = y_pred == 1
        test_ids_pred = test_ids[mask_test]

        y_cyt_train = y_cyt.iloc[train_idx, :].to_numpy()[mask_train]
        y_nuc_train = y_nuc.iloc[train_idx, :].to_numpy()[mask_train]

        reg_cyt_bo = BayesianOptimization(
            f=lambda **params: xgb_reg_fn(**params, X_train_reg=X_train[mask_train, :], y_train_reg=y_cyt_train),
            pbounds=hyperparameter_space_reg,
            random_state=seed_val,
            verbose=0
        )
        reg_cyt_bo.set_gp_params(**gp_params)
        reg_cyt_bo.maximize(init_points=args.init_points, n_iter=args.n_iter)
        reg_cyt_params = cast_params(reg_cyt_bo.max['params'], int_params=xgb_int_params)

        reg_nuc_bo = BayesianOptimization(
            f=lambda **params: xgb_reg_fn(**params, X_train_reg=X_train[mask_train, :], y_train_reg=y_nuc_train),
            pbounds=hyperparameter_space_reg,
            random_state=seed_val,
            verbose=0
        )
        reg_nuc_bo.set_gp_params(**gp_params)
        reg_nuc_bo.maximize(init_points=args.init_points, n_iter=args.n_iter)
        reg_nuc_params = cast_params(reg_nuc_bo.max['params'], int_params=xgb_int_params)

        reg_cyt_wrapper = SeededRegressorWrapper(reg_cyt_params, seed=seed_val)
        reg_cyt_wrapper.train(X_train[mask_train, :], y_cyt_train, X_val=None, y_val=None)

        reg_nuc_wrapper = SeededRegressorWrapper(reg_nuc_params, seed=seed_val)
        reg_nuc_wrapper.train(X_train[mask_train, :], y_nuc_train, X_val=None, y_val=None)

        regressor = HierarchicalRegressor(reg_cyt_wrapper, reg_nuc_wrapper)
        results_regression[group_name] = regressor.evaluate(
            X_test[mask_test, :],
            y_cyt.iloc[test_idx, :].to_numpy()[mask_test].flatten(),
            y_nuc.iloc[test_idx, :].to_numpy()[mask_test].flatten(),
            test_ids_pred
        )
        print(f"[Seed {seed_val}] [{group_name}] Fold completed successfully.")

    # Save per-seed joblib
    os.makedirs(args.out_dir, exist_ok=True)
    clf_path = os.path.join(args.out_dir, f"results_classifier_seed_{seed_val}.joblib")
    reg_path = os.path.join(args.out_dir, f"results_regression_seed_{seed_val}.joblib")
    joblib.dump(results_classifier, clf_path)
    joblib.dump(results_regression, reg_path)

    # Extract metrics table
    metrics_df = extract_metrics(results_classifier, results_regression, model_name=f"seed_{seed_val}")
    metrics_df['Seed'] = seed_val
    metrics_df['avg_R2'] = (metrics_df['cyt_R2'] + metrics_df['nuc_R2']) / 2.0
    metrics_df['avg_NRMSE'] = (metrics_df['cyt_NRMSE'] + metrics_df['nuc_NRMSE']) / 2.0

    csv_path = os.path.join(args.out_dir, f"metrics_seed_{seed_val}.csv")
    metrics_df.to_csv(csv_path, index=False)
    print(f"[Seed {seed_val}] Saved results to:\n  - {clf_path}\n  - {reg_path}\n  - {csv_path}")

    return metrics_df


# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    print(f"Random seed sensitivity experiment initiated for seeds: {seeds}")

    df = pd.read_csv(args.data_path)

    all_metrics = []
    for s in seeds:
        seed_df = run_seed_experiment(s, args, df)
        all_metrics.append(seed_df)

    # Combine into comprehensive dataset-level table
    combined_df = pd.concat(all_metrics, ignore_index=True)
    first_cols = ['Seed', 'Dataset', 'model', 'AUC', 'macro_precision', 'macro_recall', 'macro_f1',
                  'cyt_R2', 'nuc_R2', 'avg_R2', 'cyt_NRMSE', 'nuc_NRMSE', 'avg_NRMSE',
                  'cyt_RMSE', 'nuc_RMSE', 'composite_metric']
    ordered_cols = [c for c in first_cols if c in combined_df.columns] + [c for c in combined_df.columns if c not in first_cols]
    combined_df = combined_df[ordered_cols].sort_values(['Seed', 'Dataset']).reset_index(drop=True)
    combined_path = os.path.join(args.out_dir, "seed_sensitivity_per_dataset.csv")
    combined_df.to_csv(combined_path, index=False)
    print(f"\nSaved combined dataset-level metrics to: {combined_path} ({len(combined_df)} rows)")

    # Build summary comparison table
    summary_rows = []
    numeric_cols = ['AUC', 'macro_f1', 'macro_recall', 'cyt_R2', 'nuc_R2', 'avg_R2', 'cyt_NRMSE', 'nuc_NRMSE', 'avg_NRMSE', 'composite_metric']

    for s in seeds:
        sub = combined_df[combined_df['Seed'] == s]
        row = {'Group': f'Seed {s}', 'n_datasets': len(sub)}
        for col in numeric_cols:
            m = sub[col].mean()
            std = sub[col].std()
            row[f'{col}_mean'] = m
            row[f'{col}_std'] = std
            row[f'{col}_formatted'] = f"{m:.3f} ± {std:.3f}"
        summary_rows.append(row)

    # Overall across all seeds
    row_all = {'Group': f'All Seeds Combined (n={len(seeds)}x9)', 'n_datasets': len(combined_df)}
    for col in numeric_cols:
        m = combined_df[col].mean()
        std = combined_df[col].std()
        row_all[f'{col}_mean'] = m
        row_all[f'{col}_std'] = std
        row_all[f'{col}_formatted'] = f"{m:.3f} ± {std:.3f}"
    summary_rows.append(row_all)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(args.out_dir, "seed_sensitivity_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary comparison table to: {summary_path}")

    print("\nRandom seed sensitivity experiment completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LOGO experiment with random seed sensitivity")
    parser.add_argument("--data_path", type=str, default="data/training_data/training_data.csv", help="Path to CSV file")
    parser.add_argument("--categorical_columns", type=str,
                        default="lr_pair,source,target,pathway_name,annotation,ligand.family,ligand.keyword,ligand.secreted_type,ligand.transmembrane,receptor.family,receptor.keyword,receptor.surfaceome_main,receptor.surfaceome_sub,receptor.adhesome,receptor.secreted_type,receptor.transmembrane",
                        help="Comma-separated list of categorical columns")
    parser.add_argument("--group_column", type=str, default="sample", help="Column name for groups")
    parser.add_argument("--exclude_columns", type=str,
                        default="cyt_pval,cyt_pspatial,cyt_P1,sample,cell_pval,cell_P1,tissue,is_neurotransmitter,ligand_location_cellchat,receptor_location_cellchat,ligand_location_hpa,receptor_location_hpa,nuc_pval,nuc_pspatial,nuc_P1,ligand,receptor,labels",
                        help="Comma-separated list of columns to exclude")
    parser.add_argument("--spatial", action='store_true', default=True, help="Whether to include spatial information (SP procedure)")
    parser.add_argument("--seeds", type=str, default="42,112,123", help="Comma-separated list of random seeds to evaluate")
    parser.add_argument("--init_points", type=int, default=3, help="Bayesian Optimization init_points")
    parser.add_argument("--n_iter", type=int, default=30, help="Bayesian Optimization n_iter")
    parser.add_argument("--force_rerun", action='store_true', default=False, help="Force rerun even if cached results exist on disk")
    parser.add_argument("--out_dir", type=str, default="supplementary_analysis/results/random_seed_sensitivity",
                        help="Directory to save output joblib files and metric CSVs")
    args = parser.parse_args()

    main(args)
