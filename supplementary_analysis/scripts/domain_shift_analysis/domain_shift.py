# supplementary_analysis/scripts/domain_shift_analysis/domain_shift.py
"""
domain_shift.py

Leave-One-Group-Out (LOGO) cross-validation and domain-shift sweep for CCIDeconv.

For every held-out group:
  1. Tune and train the hierarchical classifier and the cyt / nuc regressors on the other groups only.
  2. Evaluate the unshifted held-out group (the honest in-domain baseline, same as run_logo.py).
  3. With those fold models frozen, distort the whole-cell score of the held-out group
     (and optionally the cyt / nuc ground truth), then predict:
       - without matching, and
       - after quantile-matching the shifted whole-cell score back to the TRAINING folds'
         distribution. This step needs no labels, so it is also possible on scRNA-seq.

Run it in a mode that keeps the whole-cell score as a feature (the SP / --spatial command).
The script stops with an error if --x_col is not among the model features.

Outputs (in --out_dir):
    results_classifier.joblib          per-group classifier results (unshifted)
    results_regression.joblib          per-group regression results (unshifted, classifier-predicted positives)
    results_shift_sweep.joblib         {(group, shift_name, rep, matched_x, shifted_y): regressor.evaluate(...) output}
    models/fold_<group>.joblib         fitted fold models, so the sweep can be re-run without retuning
    domain_shift_dataset_level.csv     full dataset-level metrics across all shift conditions
    domain_shift_summary.csv           aggregated mean ± std metrics per shift condition (unmitigated vs matched)
    domain_shift_baseline_unshifted.csv unshifted in-domain baseline performance per dataset
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
from scipy.stats import rankdata
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    precision_score,
    f1_score,
    balanced_accuracy_score
)

# Ensure root CCIDeconv repository is in Python module search path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from utils import process_data, rf_cv, cast_params, xgbc_cv, xgb_reg
from config import hyperparameter_space_rf_class, hyperparameter_space_xgb_class, hyperparameter_space_reg, gp_params, seed
from train import train_classifier, train_regressor
from inference import HierarchicalClassifier, HierarchicalRegressor

random.seed(42)


# ----------------------------------------------------------------------
# Shift helpers
# ----------------------------------------------------------------------
def qmatch(src, ref):
    """Quantile-map src onto the distribution of ref (rank preserving, label free)."""
    src = np.asarray(src, float)
    q = (rankdata(src) - 0.5) / len(src)
    return np.quantile(np.asarray(ref, float), q)


def apply_shift(v, sd_scale=1.0, dmean=0.0, power=1.0, noise=0.0, rng=None):
    """
    Distort a vector in z-score space, then map back to its own mean / SD.
      sd_scale : stretch (>1) or squeeze (<1) around the mean
      dmean    : mean shift, in SD units
      power    : monotone shape change (sign-preserving power on the z-score)
      noise    : Gaussian noise SD, in SD units (this one is NOT invertible by matching)
    """
    v = np.asarray(v, float)
    mu, sd = v.mean(), v.std()
    z = (v - mu) / sd
    z = np.sign(z) * np.abs(z) ** power
    z = z * sd_scale + dmean
    if noise > 0:
        if rng is None:
            rng = np.random.default_rng(42)
        z = z + rng.normal(0.0, noise, len(z))
    return z * sd + mu


def make_shift_grid():
    """One axis at a time, so each curve has a single interpretation."""
    base = dict(sd_scale=1.0, dmean=0.0, power=1.0, noise=0.0)
    grid = [dict(base, name="none")]
    axes = dict(
        sd_scale=(0.6, 0.8, 1.25, 1.5),
        dmean=(-0.5, 0.5),
        power=(0.7, 1.4),
        noise=(0.2, 0.5, 1.0),
    )
    for key, values in axes.items():
        for val in values:
            grid.append(dict(base, **{key: val}, name=f"{key}={val}"))
    return grid


# ----------------------------------------------------------------------
# CSV Metrics Export Helper
# ----------------------------------------------------------------------
def export_domain_shift_metrics(results_shift, results_classifier, results_regression, out_dir, results_classifier_shift=None):
    """
    Export detailed dataset-level, summary, and baseline unshifted metrics to CSV.
    """
    os.makedirs(out_dir, exist_ok=True)

    # 1. Dataset-level shift metrics (Combined Regression + Classification)
    rows_shift = []
    for (group_name, shift_name, rep, matched_x, shifted_y), eval_res in results_shift.items():
        c_m = eval_res['cyt'][0]
        n_m = eval_res['nuc'][0]
        c_r2, c_rmse, c_nrmse = c_m['R²'], c_m['RMSE'], c_m['NRMSE']
        n_r2, n_rmse, n_nrmse = n_m['R²'], n_m['RMSE'], n_m['NRMSE']

        cl_m = results_classifier_shift.get((group_name, shift_name, rep, matched_x), {}) if results_classifier_shift else {}
        auc_val = cl_m.get('auc', np.nan)
        macro_rec = cl_m.get('macro_recall', np.nan)

        rows_shift.append({
            'dataset': group_name,
            'shift_name': shift_name,
            'rep': rep,
            'matched_x': matched_x,
            'shifted_y': shifted_y,
            'auc': auc_val,
            'macro_recall': macro_rec,
            'cyt_R2': c_r2,
            'cyt_RMSE': c_rmse,
            'cyt_NRMSE': c_nrmse,
            'nuc_R2': n_r2,
            'nuc_RMSE': n_rmse,
            'nuc_NRMSE': n_nrmse,
            'avg_R2': np.mean([c_r2, n_r2]),
            'avg_NRMSE': np.mean([c_nrmse, n_nrmse])
        })
    df_shift = pd.DataFrame(rows_shift)
    df_shift_path = os.path.join(out_dir, "domain_shift_dataset_level.csv")
    df_shift.to_csv(df_shift_path, index=False)
    print(f"Saved dataset-level shift metrics to: {df_shift_path} ({len(df_shift)} rows)")

    # 2. Aggregated summary metrics (Regression)
    if not df_shift.empty:
        # Average over reps first (for noise) per dataset, then compute mean and std across datasets
        dataset_avg = df_shift.groupby(
            ['dataset', 'shift_name', 'matched_x', 'shifted_y'], as_index=False
        )[['cyt_R2', 'cyt_RMSE', 'cyt_NRMSE', 'nuc_R2', 'nuc_RMSE', 'nuc_NRMSE', 'avg_R2', 'avg_NRMSE']].mean()

        summary_rows = []
        for (shift_name, matched_x, shifted_y), sub_df in dataset_avg.groupby(
            ['shift_name', 'matched_x', 'shifted_y']
        ):
            n_datasets = len(sub_df)
            row = {
                'shift_name': shift_name,
                'matched_x': matched_x,
                'shifted_y': shifted_y,
                'n_datasets': n_datasets,
                'cyt_R2_mean': sub_df['cyt_R2'].mean(),
                'cyt_R2_std': sub_df['cyt_R2'].std(),
                'cyt_R2_formatted': f"{sub_df['cyt_R2'].mean():.3f} ± {sub_df['cyt_R2'].std():.3f}",
                'cyt_NRMSE_mean': sub_df['cyt_NRMSE'].mean(),
                'cyt_NRMSE_std': sub_df['cyt_NRMSE'].std(),
                'cyt_NRMSE_formatted': f"{sub_df['cyt_NRMSE'].mean():.3f} ± {sub_df['cyt_NRMSE'].std():.3f}",
                'nuc_R2_mean': sub_df['nuc_R2'].mean(),
                'nuc_R2_std': sub_df['nuc_R2'].std(),
                'nuc_R2_formatted': f"{sub_df['nuc_R2'].mean():.3f} ± {sub_df['nuc_R2'].std():.3f}",
                'nuc_NRMSE_mean': sub_df['nuc_NRMSE'].mean(),
                'nuc_NRMSE_std': sub_df['nuc_NRMSE'].std(),
                'nuc_NRMSE_formatted': f"{sub_df['nuc_NRMSE'].mean():.3f} ± {sub_df['nuc_NRMSE'].std():.3f}",
                'avg_R2_mean': sub_df['avg_R2'].mean(),
                'avg_R2_std': sub_df['avg_R2'].std(),
                'avg_R2_formatted': f"{sub_df['avg_R2'].mean():.3f} ± {sub_df['avg_R2'].std():.3f}",
                'avg_NRMSE_mean': sub_df['avg_NRMSE'].mean(),
                'avg_NRMSE_std': sub_df['avg_NRMSE'].std(),
                'avg_NRMSE_formatted': f"{sub_df['avg_NRMSE'].mean():.3f} ± {sub_df['avg_NRMSE'].std():.3f}"
            }
            summary_rows.append(row)
        df_summary = pd.DataFrame(summary_rows)
        df_summary_path = os.path.join(out_dir, "domain_shift_summary.csv")
        df_summary.to_csv(df_summary_path, index=False)
        print(f"Saved domain shift summary metrics to: {df_summary_path} ({len(df_summary)} rows)")

    # 3. Classifier Domain Shift Metrics (AUC & Macro Recall only)
    if results_classifier_shift is not None and len(results_classifier_shift) > 0:
        rows_cl = []
        for (group_name, shift_name, rep, matched_x), m in results_classifier_shift.items():
            rows_cl.append({
                'dataset': group_name,
                'shift_name': shift_name,
                'rep': rep,
                'matched_x': matched_x,
                'auc': m['auc'],
                'macro_recall': m['macro_recall']
            })
        df_cl = pd.DataFrame(rows_cl)
        df_cl_path = os.path.join(out_dir, "domain_shift_classifier_dataset_level.csv")
        df_cl.to_csv(df_cl_path, index=False)
        print(f"Saved classifier dataset-level shift metrics to: {df_cl_path} ({len(df_cl)} rows)")

        # Summary across datasets
        cl_avg = df_cl.groupby(
            ['dataset', 'shift_name', 'matched_x'], as_index=False
        )[['auc', 'macro_recall']].mean()

        cl_summary_rows = []
        for (shift_name, matched_x), sub_df in cl_avg.groupby(['shift_name', 'matched_x']):
            cl_summary_rows.append({
                'shift_name': shift_name,
                'matched_x': matched_x,
                'n_datasets': len(sub_df),
                'auc_mean': sub_df['auc'].mean(),
                'auc_std': sub_df['auc'].std(),
                'auc_formatted': f"{sub_df['auc'].mean():.3f} ± {sub_df['auc'].std():.3f}",
                'macro_recall_mean': sub_df['macro_recall'].mean(),
                'macro_recall_std': sub_df['macro_recall'].std(),
                'macro_recall_formatted': f"{sub_df['macro_recall'].mean():.3f} ± {sub_df['macro_recall'].std():.3f}"
            })
        df_cl_summary = pd.DataFrame(cl_summary_rows)
        df_cl_summary_path = os.path.join(out_dir, "domain_shift_classifier_summary.csv")
        df_cl_summary.to_csv(df_cl_summary_path, index=False)
        print(f"Saved classifier domain shift summary metrics to: {df_cl_summary_path} ({len(df_cl_summary)} rows)")

    # 4. Baseline unshifted performance
    baseline_rows = []
    for group_name in results_classifier.keys():
        cl_res = results_classifier[group_name]['classification_metrics']
        auc = cl_res.get('roc_auc', np.nan)
        report = cl_res.get('class_report', '')
        macro_prec, macro_rec, macro_f1 = np.nan, np.nan, np.nan
        for line in report.splitlines():
            if line.strip().startswith("macro avg"):
                parts = line.split()
                macro_prec = float(parts[2])
                macro_rec = float(parts[3])
                macro_f1 = float(parts[4])

        c_m = results_regression[group_name]['cyt'][0]
        n_m = results_regression[group_name]['nuc'][0]
        c_r2, c_rmse, c_nrmse = c_m['R²'], c_m['RMSE'], c_m['NRMSE']
        n_r2, n_rmse, n_nrmse = n_m['R²'], n_m['RMSE'], n_m['NRMSE']
        baseline_rows.append({
            'dataset': group_name,
            'auc': auc,
            'macro_precision': macro_prec,
            'macro_recall': macro_rec,
            'macro_f1': macro_f1,
            'cyt_R2': c_r2,
            'cyt_RMSE': c_rmse,
            'cyt_NRMSE': c_nrmse,
            'nuc_R2': n_r2,
            'nuc_RMSE': n_rmse,
            'nuc_NRMSE': n_nrmse,
            'avg_R2': np.mean([c_r2, n_r2]),
            'avg_NRMSE': np.mean([c_nrmse, n_nrmse])
        })
    df_baseline = pd.DataFrame(baseline_rows)
    df_base_path = os.path.join(out_dir, "domain_shift_baseline_unshifted.csv")
    df_baseline.to_csv(df_base_path, index=False)
    print(f"Saved baseline unshifted metrics to: {df_base_path} ({len(df_baseline)} datasets)")


# ----------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------
def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    model_dir = os.path.join(args.out_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    # Optional fast path: if user just wants to export CSV from an existing joblib run
    joblib_shift_path = os.path.join(args.out_dir, "results_shift_sweep.joblib")
    joblib_class_path = os.path.join(args.out_dir, "results_classifier.joblib")
    joblib_reg_path = os.path.join(args.out_dir, "results_regression.joblib")

    joblib_class_shift_path = os.path.join(args.out_dir, "results_classifier_shift.joblib")

    if args.export_csv_only:
        if os.path.exists(joblib_shift_path) and os.path.exists(joblib_class_path) and os.path.exists(joblib_reg_path):
            print("Loading pre-computed results from joblib files...")
            results_classifier = joblib.load(joblib_class_path)
            results_regression = joblib.load(joblib_reg_path)
            results_shift = joblib.load(joblib_shift_path)
            results_classifier_shift = joblib.load(joblib_class_shift_path) if os.path.exists(joblib_class_shift_path) else None
            export_domain_shift_metrics(results_shift, results_classifier, results_regression, args.out_dir, results_classifier_shift)
            return
        else:
            print(f"--export_csv_only specified, but required joblib files not found in {args.out_dir}. Proceeding with execution.")

    df = pd.read_csv(args.data_path)
    X_scaled, y, y_nuc, y_cyt, ids = process_data(df, exclude_columns=args.exclude_columns.split(','), spatial=args.spatial)
    print(f"Data processed successfully: X shape = {X_scaled.shape}")
    categorical_columns = args.categorical_columns.split(',')
    groups = df[args.group_column].values
    logo = LeaveOneGroupOut()
    shift_grid = make_shift_grid()

    results_classifier = {}
    results_regression = {}
    results_shift = {}
    results_classifier_shift = {}

    for train_idx, test_idx in logo.split(X_scaled, y, groups):
        X_train, X_test = X_scaled.iloc[train_idx], X_scaled.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx].values.ravel(), y.iloc[test_idx].values.ravel()
        ids_test_all = ids.iloc[test_idx]
        test_ids = ids_test_all
        group_name = np.unique(groups[test_idx])[0]

        # --- Encode categorical columns (fresh encoder per fold, fitted on training groups only) ---
        catboost_encoder = CatBoostEncoder()
        cat_train = catboost_encoder.fit_transform(X_train[categorical_columns], y_train)
        X_train = pd.concat([X_train.drop(columns=categorical_columns).reset_index(drop=True), cat_train.reset_index(drop=True)], axis=1)

        cat_test = catboost_encoder.transform(X_test[categorical_columns])
        X_test = pd.concat([X_test.drop(columns=categorical_columns).reset_index(drop=True), cat_test.reset_index(drop=True)], axis=1)

        # position of the whole-cell score, found BEFORE converting to numpy
        feature_names = list(X_train.columns)
        if args.x_col not in feature_names:
            raise ValueError(
                f"--x_col '{args.x_col}' is not a model feature. Run in the mode that keeps it "
                f"(it is excluded in the non-spatial command). Features: {feature_names}"
            )
        x_idx = feature_names.index(args.x_col)

        X_train = X_train.to_numpy(dtype=float)
        X_test = X_test.to_numpy(dtype=float)

        fold_model_path = os.path.join(model_dir, f"fold_{group_name}.joblib")
        if args.reuse_models and os.path.exists(fold_model_path):
            print(f"[{group_name}] Loading cached fold models from {fold_model_path}...")
            cached = joblib.load(fold_model_path)
            classifier_wrapper = cached['classifier']
            reg_cyt_wrapper = cached['reg_cyt']
            reg_nuc_wrapper = cached['reg_nuc']
        else:
            # --- Tuning for classifier (inner CV on training groups only) ---
            print(f"[{group_name}] Tuning and training Random Forest & XGBoost classifiers...")
            rf_bo = BayesianOptimization(
                f=lambda **params: rf_cv(**params, X_train=X_train, y_train=y_train),
                pbounds=hyperparameter_space_rf_class,
                random_state=seed,
                verbose=0
            )
            rf_bo.maximize(init_points=3, n_iter=30)
            rf_int_params = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']
            rf_params = cast_params(rf_bo.max['params'], int_params=rf_int_params)

            xgbc_bo = BayesianOptimization(
                f=lambda **params: xgbc_cv(**params, X_train=X_train, y_train=y_train),
                pbounds=hyperparameter_space_xgb_class,
                random_state=seed,
                verbose=0
            )
            xgbc_bo.set_gp_params(**gp_params)
            xgbc_bo.maximize(init_points=3, n_iter=30)
            xgb_int_params = ['n_estimators', 'max_depth']
            xgb_params = cast_params(xgbc_bo.max['params'], int_params=xgb_int_params)

            classifier_wrapper = train_classifier(X_train, y_train, rf_params, xgb_params)

        # --- Unshifted evaluation of the classifier (baseline) ---
        y_pred, class_metrics, roc_table, predictions = HierarchicalClassifier(classifier_wrapper).evaluate(X_test, y_test, test_ids)
        results_classifier[group_name] = {
            'y_pred': y_pred,
            'classification_metrics': class_metrics,
            'roc_table': roc_table,
            'predictions': predictions
        }

        # --- Regressors, trained on true positives of the training groups ---
        mask_train = y_train == 1
        mask_test = y_pred == 1
        test_ids_pred = test_ids[mask_test]

        y_cyt_train = y_cyt.iloc[train_idx, :].to_numpy()[mask_train]
        y_nuc_train = y_nuc.iloc[train_idx, :].to_numpy()[mask_train]

        if not (args.reuse_models and os.path.exists(fold_model_path)):
            print(f"[{group_name}] Tuning and training Cytoplasmic & Nuclear regressors...")
            reg_bo = BayesianOptimization(
                f=lambda **params: xgb_reg(**params, X_train_reg=X_train[mask_train, :], y_train_reg=y_cyt_train),
                pbounds=hyperparameter_space_reg,
                random_state=seed,
                verbose=0
            )
            reg_bo.set_gp_params(**gp_params)
            reg_bo.maximize(init_points=3, n_iter=30)
            reg_cyt_params = cast_params(reg_bo.max['params'], int_params=xgb_int_params)

            reg_bo = BayesianOptimization(
                f=lambda **params: xgb_reg(**params, X_train_reg=X_train[mask_train, :], y_train_reg=y_nuc_train),
                pbounds=hyperparameter_space_reg,
                random_state=seed,
                verbose=0
            )
            reg_bo.set_gp_params(**gp_params)
            reg_bo.maximize(init_points=3, n_iter=30)
            reg_nuc_params = cast_params(reg_bo.max['params'], int_params=xgb_int_params)

            reg_cyt_wrapper = train_regressor(X_train[mask_train, :], y_cyt_train, reg_cyt_params, X_val=None, y_val=None)
            reg_nuc_wrapper = train_regressor(X_train[mask_train, :], y_nuc_train, reg_nuc_params, X_val=None, y_val=None)

            # --- Save the fitted fold models so the sweep can be re-run without retuning ---
            joblib.dump(
                dict(classifier=classifier_wrapper, reg_cyt=reg_cyt_wrapper, reg_nuc=reg_nuc_wrapper,
                     feature_names=feature_names, x_idx=x_idx),
                fold_model_path
            )

        regressor = HierarchicalRegressor(reg_cyt_wrapper, reg_nuc_wrapper)

        # --- Unshifted evaluation of the regressors (baseline, same as run_logo.py) ---
        results_regression[group_name] = regressor.evaluate(
            X_test[mask_test, :],
            y_cyt.iloc[test_idx, :].to_numpy()[mask_test].flatten(),
            y_nuc.iloc[test_idx, :].to_numpy()[mask_test].flatten(),
            test_ids_pred
        )

        # ------------------------------------------------------------------
        # Domain-shift sweep: inference only, fold models stay frozen.
        # The regressors are scored on the TRUE positives, so the evaluated cells are
        # identical in every shift condition (the classifier would otherwise change the set).
        # ------------------------------------------------------------------
        pos = y_test == 1
        if pos.sum() < 10:
            print(f"[{group_name}] only {pos.sum()} positives, skipping sweep")
            continue

        ids_pos = ids_test_all[pos]
        yc_true = y_cyt.iloc[test_idx, :].to_numpy()[pos].flatten()
        yn_true = y_nuc.iloc[test_idx, :].to_numpy()[pos].flatten()

        ref_x = X_train[:, x_idx]          # label-free reference: training groups only
        x_test_col = X_test[:, x_idx]      # whole held-out group, shifted BEFORE selecting positives

        for cfg in shift_grid:
            reps = range(args.n_noise_reps) if cfg['noise'] > 0 else range(1)
            for rep in reps:
                rng = np.random.default_rng(int(seed) + rep)
                x_shifted = apply_shift(x_test_col, cfg['sd_scale'], cfg['dmean'], cfg['power'], cfg['noise'], rng)

                for matched_x in (False, True):
                    x_in = qmatch(x_shifted, ref_x) if matched_x else x_shifted
                    X_in = X_test.copy()
                    X_in[:, x_idx] = x_in

                    # Evaluate classifier on shifted X_in (inference only, frozen fold model)
                    y_pred_shift = classifier_wrapper.model.predict(X_in)
                    y_proba_shift = classifier_wrapper.model.predict_proba(X_in)

                    try:
                        auc_val = float(roc_auc_score(y_test, y_proba_shift[:, 1]))
                    except Exception:
                        auc_val = np.nan

                    macro_rec = float(recall_score(y_test, y_pred_shift, average='macro', zero_division=0))

                    results_classifier_shift[(group_name, cfg['name'], rep, matched_x)] = {
                        'auc': auc_val,
                        'macro_recall': macro_rec
                    }

                    for shifted_y in (False, True):
                        if shifted_y and cfg['name'] == "none":
                            continue  # identical to the unshifted-Y case
                        if shifted_y:
                            # same distortion applied to the ground truth (units / scale mismatch);
                            # not available at deployment, so report it as a separate, labelled axis
                            yc = apply_shift(yc_true, cfg['sd_scale'], cfg['dmean'], cfg['power'])
                            yn = apply_shift(yn_true, cfg['sd_scale'], cfg['dmean'], cfg['power'])
                        else:
                            yc, yn = yc_true, yn_true

                        results_shift[(group_name, cfg['name'], rep, matched_x, shifted_y)] = regressor.evaluate(
                            X_in[pos, :], yc, yn, ids_pos
                        )

        print(f"[{group_name}] sweep completed successfully.")

    # Save joblib dumps
    joblib.dump(results_classifier, os.path.join(args.out_dir, "results_classifier.joblib"))
    joblib.dump(results_regression, os.path.join(args.out_dir, "results_regression.joblib"))
    joblib.dump(results_shift, os.path.join(args.out_dir, "results_shift_sweep.joblib"))
    joblib.dump(results_classifier_shift, os.path.join(args.out_dir, "results_classifier_shift.joblib"))

    # Export CSV tables
    export_domain_shift_metrics(results_shift, results_classifier, results_regression, args.out_dir, results_classifier_shift)
    print("\nAll tasks completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LOGO experiment with a domain-shift sweep")
    parser.add_argument("--data_path", type=str, default="data/training_data/training_data.csv", help="Path to CSV file")
    parser.add_argument("--categorical_columns", type=str,
                        default="lr_pair,source,target,pathway_name,annotation,ligand.family,ligand.keyword,ligand.secreted_type,ligand.transmembrane,receptor.family,receptor.keyword,receptor.surfaceome_main,receptor.surfaceome_sub,receptor.adhesome,receptor.secreted_type,receptor.transmembrane",
                        help="Comma-separated list of categorical columns")
    parser.add_argument("--group_column", type=str, default="sample", help="Column name for groups")
    parser.add_argument("--exclude_columns", type=str,
                        default="cyt_pval,cyt_pspatial,cyt_P1,sample,cell_pval,cell_P1,tissue,is_neurotransmitter,ligand_location_cellchat,receptor_location_cellchat,ligand_location_hpa,receptor_location_hpa,nuc_pval,nuc_pspatial,nuc_P1,ligand,receptor,labels",
                        help="Comma-separated list of columns to exclude")
    parser.add_argument("--spatial", action='store_true', default=True, help="Whether to include spatial information (SP procedure)")
    parser.add_argument("--x_col", type=str, default="cell_score", help="Whole-cell score feature to shift")
    parser.add_argument("--n_noise_reps", type=int, default=5, help="Random repeats for noise shifts")
    parser.add_argument("--out_dir", type=str, default="supplementary_analysis/results/domain_shift_analysis",
                        help="Where to write results, fold models, and metric CSVs")
    parser.add_argument("--reuse_models", action='store_true', default=True, help="Reuse existing fold models if found in out_dir/models")
    parser.add_argument("--export_csv_only", action='store_true', help="If results joblib files exist, re-export CSVs without re-running")
    args = parser.parse_args()

    main(args)