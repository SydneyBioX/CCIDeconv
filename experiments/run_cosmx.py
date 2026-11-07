# experiments/run_cosmx.py
import argparse
import os
import sys
import random
import joblib
import pandas as pd
import numpy as np
from category_encoders import CatBoostEncoder
from bayes_opt import BayesianOptimization
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import (
    process_data,
    rf_cv,
    cast_params,
    xgbc_cv,
    xgb_reg,
    scale_continuous_by_group
)
from config import (
    hyperparameter_space_rf_class,
    hyperparameter_space_xgb_class,
    hyperparameter_space_reg,
    gp_params,
    seed
)
from train import train_classifier, train_regressor
from inference import HierarchicalClassifier, HierarchicalRegressor

random.seed(42)
np.random.seed(42)


def align_test_features(X_train_df, X_test_df, categorical_columns):
    """
    Ensure X_test has the exact columns, order, and format as X_train_df.
    Missing numeric/dummy columns in test are filled with 0.
    """
    X_test_aligned = pd.DataFrame(index=X_test_df.index)

    for col in X_train_df.columns:
        if col in X_test_df.columns:
            X_test_aligned[col] = X_test_df[col]
        else:
            if col in categorical_columns:
                X_test_aligned[col] = "unknown"
            else:
                X_test_aligned[col] = 0.0

    return X_test_aligned[X_train_df.columns]


def prepare_cosmx_df(df_test):
    """
    Standardize column names, fill missing location/metadata columns, and generate
    one-hot encoded location dummies matching the training data format.
    """
    from sklearn.preprocessing import MultiLabelBinarizer
    df = df_test.copy()
    if "interaction_name" in df.columns and "lr_pair" not in df.columns:
        df = df.rename(columns={"interaction_name": "lr_pair"})
    if "sample" not in df.columns:
        df["sample"] = "cosmx_lung13"
    if "tissue" not in df.columns:
        df["tissue"] = "Lung"
    if "ligand.location" in df.columns and "ligand_location_cellchat" not in df.columns:
        df["ligand_location_cellchat"] = df["ligand.location"].fillna("unknown")
    if "receptor.location" in df.columns and "receptor_location_cellchat" not in df.columns:
        df["receptor_location_cellchat"] = df["receptor.location"].fillna("unknown")
    if "ligand.location_hpa" in df.columns and "ligand_location_hpa" not in df.columns:
        df["ligand_location_hpa"] = df["ligand.location_hpa"].fillna("unknown")
    if "receptor.location_hpa" in df.columns and "receptor_location_hpa" not in df.columns:
        df["receptor_location_hpa"] = df["receptor.location_hpa"].fillna("unknown")

    # Generate location dummy indicators
    for col, prefix in [
        ('ligand_location_cellchat', 'ligand_location_cellchat_'),
        ('receptor_location_cellchat', 'receptor_location_cellchat_'),
        ('ligand_location_hpa', 'ligand_location_hpa_'),
        ('receptor_location_hpa', 'receptor_location_hpa_')
    ]:
        if col in df.columns:
            items = df[col].astype(str).str.split(';|,').apply(lambda lst: [x.strip() for x in lst if x.strip()])
            mlb = MultiLabelBinarizer()
            dummies = pd.DataFrame(
                mlb.fit_transform(items),
                columns=[f'{prefix}{c}' for c in mlb.classes_],
                index=df.index
            )
            df = pd.concat([df, dummies], axis=1)

    return df


def main(args):
    """
    run_cosmx.py

    Trains CCIDeconv hierarchical classification and regression models on the spatial transcriptomics
    training dataset (using all training groups, no LOGO partitioning) with spatial features,
    and applies the trained model onto CosMx spatial transcriptomics data.
    """
    os.makedirs(args.output_dir, exist_ok=True)
    categorical_columns = [c.strip() for c in args.categorical_columns.split(',') if c.strip()]
    exclude_columns = [c.strip() for c in args.exclude_columns.split(',') if c.strip()]

    print(f"=== 1. Loading and processing training data from: {args.data_path} ===")
    df_train = pd.read_csv(args.data_path)
    X_train_scaled, y_train, y_nuc_train, y_cyt_train, train_ids = process_data(
        df_train,
        exclude_columns=exclude_columns,
        spatial=args.spatial
    )
    print(f"Training feature matrix shape: {X_train_scaled.shape}")
    print(f"Total training interactions: {len(y_train)} (Positive: {np.sum(y_train == 1)}, Negative: {np.sum(y_train == 0)})")

    # --- 2. Load CosMx Test Data ---
    print(f"\n=== 2. Loading CosMx test data from: {args.test_data_path} ===")
    df_test_raw = pd.read_csv(args.test_data_path, low_memory=False)
    print(f"Raw CosMx test shape: {df_test_raw.shape}")

    # Standardize column names and location dummies
    df_test = prepare_cosmx_df(df_test_raw)

    # Optional filtering for significant whole-cell interactions (matching CCIDeconv benchmark pipeline)
    if args.filter_cell_pval and "cell_pval" in df_test.columns:
        n_before = len(df_test)
        df_test = df_test[df_test["cell_pval"] < 0.05].copy().reset_index(drop=True)
        print(f"Filtered for significant whole-cell interactions (cell_pval < 0.05): {len(df_test)} / {n_before} retained.")

    # Check for ground-truth columns in CosMx test data
    has_cyt_score = "cyt_score" in df_test.columns
    has_nuc_score = "nuc_score" in df_test.columns
    has_labels = "labels" in df_test.columns or (has_cyt_score and has_nuc_score)

    y_test = None
    y_cyt_test = None
    y_nuc_test = None

    exclude_columns_test = [c for c in exclude_columns if c in df_test.columns]

    if has_labels or (has_cyt_score and has_nuc_score):
        print("Detected ground-truth target columns in CosMx test data. Evaluation metrics will be calculated.")
        # Process test data using process_data to derive labels and scaled targets
        X_test_scaled, y_test, y_nuc_test, y_cyt_test, test_ids = process_data(
            df_test.copy(),
            exclude_columns=exclude_columns_test,
            spatial=args.spatial
        )
    else:
        print("No ground truth labels/scores found in CosMx data. Running in pure inference mode.")
        test_ids = df_test[['lr_pair', 'source', 'target']].copy() if all(c in df_test.columns for c in ['lr_pair', 'source', 'target']) else pd.DataFrame(index=df_test.index)
        
        # Scale continuous features present in CosMx
        cont_cols = ['cell_score', 'cell_pspatial'] if args.spatial else ['cell_P1']
        cont_cols_present = [c for c in cont_cols if c in df_test.columns]
        
        X_test_scaled = df_test.copy()
        if cont_cols_present:
            scaler = StandardScaler()
            X_test_scaled[cont_cols_present] = scaler.fit_transform(X_test_scaled[cont_cols_present])
        
        drop_cols = [c for c in exclude_columns if c in X_test_scaled.columns]
        X_test_scaled = X_test_scaled.drop(columns=drop_cols, errors='ignore')

    # Align columns to match training feature space
    X_test_scaled = align_test_features(X_train_scaled, X_test_scaled, categorical_columns)
    print(f"Aligned CosMx test feature matrix shape: {X_test_scaled.shape}")

    # --- 3. Encode Categorical Features ---
    print("\n=== 3. Encoding categorical features using CatBoostEncoder ===")
    catboost_encoder = CatBoostEncoder()
    y_train_arr = y_train.values.ravel()
    
    cat_train = catboost_encoder.fit_transform(X_train_scaled[categorical_columns], y_train_arr)
    X_train_enc = pd.concat([
        X_train_scaled.drop(columns=categorical_columns).reset_index(drop=True),
        cat_train.reset_index(drop=True)
    ], axis=1)

    cat_test = catboost_encoder.transform(X_test_scaled[categorical_columns])
    X_test_enc = pd.concat([
        X_test_scaled.drop(columns=categorical_columns).reset_index(drop=True),
        cat_test.reset_index(drop=True)
    ], axis=1)

    X_train_np = X_train_enc.to_numpy(dtype=float)
    X_test_np = X_test_enc.to_numpy(dtype=float)

    # --- 4. Bayesian Optimization & Classifier Training ---
    print("\n=== 4. Hyperparameter tuning & training classifier ===")
    print("Optimizing Random Forest Classifier...")
    rf_bo = BayesianOptimization(
        f=lambda **params: rf_cv(**params, X_train=X_train_np, y_train=y_train_arr),
        pbounds=hyperparameter_space_rf_class,
        random_state=seed,
        verbose=0
    )
    rf_bo.maximize(init_points=args.init_points, n_iter=args.n_iter)
    rf_int_params = ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf']
    rf_params = cast_params(rf_bo.max['params'], int_params=rf_int_params)

    print("Optimizing XGBoost Classifier...")
    xgbc_bo = BayesianOptimization(
        f=lambda **params: xgbc_cv(**params, X_train=X_train_np, y_train=y_train_arr),
        pbounds=hyperparameter_space_xgb_class,
        random_state=seed,
        verbose=0
    )
    xgbc_bo.set_gp_params(**gp_params)
    xgbc_bo.maximize(init_points=args.init_points, n_iter=args.n_iter)
    xgb_int_params = ['n_estimators', 'max_depth']
    xgb_params = cast_params(xgbc_bo.max['params'], int_params=xgb_int_params)

    print("Fitting ensemble VotingClassifier...")
    classifier_wrapper = train_classifier(X_train_np, y_train_arr, rf_params, xgb_params)

    # --- 5. Bayesian Optimization & Regressor Training ---
    print("\n=== 5. Hyperparameter tuning & training regressors ===")
    mask_train = (y_train_arr == 1)
    y_cyt_train_arr = y_cyt_train.to_numpy().flatten()[mask_train]
    y_nuc_train_arr = y_nuc_train.to_numpy().flatten()[mask_train]
    X_train_pos = X_train_np[mask_train, :]

    print(f"Training regressors on {np.sum(mask_train)} positive interactions...")
    print("Optimizing Cytoplasm XGBoost Regressor...")
    xgb_cyt_bo = BayesianOptimization(
        f=lambda **params: xgb_reg(**params, X_train_reg=X_train_pos, y_train_reg=y_cyt_train_arr),
        pbounds=hyperparameter_space_reg,
        random_state=seed,
        verbose=0
    )
    xgb_cyt_bo.set_gp_params(**gp_params)
    xgb_cyt_bo.maximize(init_points=args.init_points, n_iter=args.n_iter)
    reg_cyt_params = cast_params(xgb_cyt_bo.max['params'], int_params=xgb_int_params)
    reg_cyt_wrapper = train_regressor(X_train_pos, y_cyt_train_arr, reg_cyt_params, X_val=None, y_val=None)

    print("Optimizing Nucleus XGBoost Regressor...")
    xgb_nuc_bo = BayesianOptimization(
        f=lambda **params: xgb_reg(**params, X_train_reg=X_train_pos, y_train_reg=y_nuc_train_arr),
        pbounds=hyperparameter_space_reg,
        random_state=seed,
        verbose=0
    )
    xgb_nuc_bo.set_gp_params(**gp_params)
    xgb_nuc_bo.maximize(init_points=args.init_points, n_iter=args.n_iter)
    reg_nuc_params = cast_params(xgb_nuc_bo.max['params'], int_params=xgb_int_params)
    reg_nuc_wrapper = train_regressor(X_train_pos, y_nuc_train_arr, reg_nuc_params, X_val=None, y_val=None)

    # --- 6. Inference and Evaluation on CosMx ---
    print("\n=== 6. Running inference on CosMx dataset ===")
    hier_clf = HierarchicalClassifier(classifier_wrapper)
    hier_reg = HierarchicalRegressor(reg_cyt_wrapper, reg_nuc_wrapper)

    results_classifier = {}
    results_regression = {}

    if y_test is not None:
        y_test_arr = y_test.values.ravel()
        y_pred, class_metrics, roc_table, class_predictions = hier_clf.evaluate(X_test_np, y_test_arr, test_ids)
        results_classifier = {
            'y_pred': y_pred,
            'classification_metrics': class_metrics,
            'roc_table': roc_table,
            'predictions': class_predictions
        }
        print("\n" + "="*50)
        print("  CLASSIFICATION PERFORMANCE ON COSMX")
        print("="*50)
        for metric_name, val in class_metrics.items():
            if metric_name != 'class_report':
                print(f"  {metric_name:<18}: {val:.4f}" if isinstance(val, (int, float)) else f"  {metric_name:<18}: {val}")

        # Save classification metrics to CSV
        clf_metrics_record = {
            k: [v] for k, v in class_metrics.items() if k != 'class_report'
        }
        df_clf_metrics = pd.DataFrame(clf_metrics_record)
        clf_csv_path = os.path.join(args.output_dir, "cosmx_classification_metrics.csv")
        df_clf_metrics.to_csv(clf_csv_path, index=False)
        print(f"\n[Saved] Classification metrics -> {clf_csv_path}")

        # Evaluate regressors on predicted positive interactions
        mask_test = (y_pred == 1)
        if np.sum(mask_test) > 0:
            test_ids_pos = test_ids[mask_test]
            y_cyt_test_pos = y_cyt_test.iloc[mask_test, :].to_numpy().flatten()
            y_nuc_test_pos = y_nuc_test.iloc[mask_test, :].to_numpy().flatten()
            results_regression = hier_reg.evaluate(
                X_test_np[mask_test, :],
                y_cyt_test_pos,
                y_nuc_test_pos,
                test_ids_pos
            )
            print("\n" + "="*50)
            print("  REGRESSION PERFORMANCE ON COSMX (Predicted Positives)")
            print("="*50)
            cyt_m = results_regression.get('cyt', ({}, None))[0]
            nuc_m = results_regression.get('nuc', ({}, None))[0]

            c_r2 = cyt_m.get('R²', np.nan)
            n_r2 = nuc_m.get('R²', np.nan)
            c_rmse = cyt_m.get('RMSE', np.nan)
            n_rmse = nuc_m.get('RMSE', np.nan)
            c_nrmse = cyt_m.get('NRMSE', np.nan)
            n_nrmse = nuc_m.get('NRMSE', np.nan)

            delta_r2 = c_r2 - n_r2 if (c_r2 is not None and n_r2 is not None) else np.nan
            delta_nrmse = n_nrmse - c_nrmse if (n_nrmse is not None and c_nrmse is not None) else np.nan

            print(f"  Cytoplasm R²     : {c_r2:.4f} | RMSE: {c_rmse:.4f} | NRMSE: {c_nrmse:.4f}")
            print(f"  Nucleus R²       : {n_r2:.4f} | RMSE: {n_rmse:.4f} | NRMSE: {n_nrmse:.4f}")
            print(f"  ΔR² (cyt - nuc)  : {delta_r2:.4f}")
            print(f"  ΔNRMSE (nuc - cyt): {delta_nrmse:.4f}")

            # Save regression metrics to CSV
            df_reg_metrics = pd.DataFrame([{
                'Dataset': 'CosMx_Lung13',
                'cyt_R²': c_r2,
                'nuc_R²': n_r2,
                'ΔR²': delta_r2,
                'cyt_RMSE': c_rmse,
                'nuc_RMSE': n_rmse,
                'cyt_NRMSE': c_nrmse,
                'nuc_NRMSE': n_nrmse,
                'ΔNRMSE': delta_nrmse,
                'n_positive_predicted': int(np.sum(mask_test)),
                'n_total_test': len(y_test)
            }])
            reg_csv_path = os.path.join(args.output_dir, "cosmx_regression_metrics.csv")
            df_reg_metrics.to_csv(reg_csv_path, index=False)
            print(f"[Saved] Regression metrics -> {reg_csv_path}")
        else:
            print("No interactions predicted as positive (label=1) in CosMx test set.")
    else:
        # Pure inference without ground truth
        class_predictions = hier_clf.prediction(X_test_np, test_ids)
        y_pred = hier_clf.y_pred
        mask_test = (y_pred == 1)
        
        pred_cyt = np.zeros(len(X_test_np))
        pred_nuc = np.zeros(len(X_test_np))
        
        if np.sum(mask_test) > 0:
            p_cyt, p_nuc = hier_reg.prediction(X_test_np[mask_test, :], test_ids[mask_test])
            pred_cyt[mask_test] = reg_cyt_wrapper.model.predict(X_test_np[mask_test, :])
            pred_nuc[mask_test] = reg_nuc_wrapper.model.predict(X_test_np[mask_test, :])
        
        results_classifier = {'predictions': class_predictions, 'y_pred': y_pred}
        results_regression = {'pred_cyt': pred_cyt, 'pred_nuc': pred_nuc}

    # --- 7. Save Prediction Outputs ---
    print("\n=== 7. Compiling and saving results ===")
    pred_df = test_ids.copy()
    pred_df['predicted_label'] = hier_clf.y_pred
    
    # Predict probabilities for positive class if available
    try:
        proba = classifier_wrapper.model.predict_proba(X_test_np)[:, 1]
        pred_df['predicted_prob'] = proba
    except Exception:
        pass

    pred_cyt_all = np.zeros(len(X_test_np))
    pred_nuc_all = np.zeros(len(X_test_np))
    mask_pos = (hier_clf.y_pred == 1)
    if np.sum(mask_pos) > 0:
        pred_cyt_all[mask_pos] = reg_cyt_wrapper.model.predict(X_test_np[mask_pos, :])
        pred_nuc_all[mask_pos] = reg_nuc_wrapper.model.predict(X_test_np[mask_pos, :])
    pred_df['predicted_cyt_score'] = pred_cyt_all
    pred_df['predicted_nuc_score'] = pred_nuc_all

    if y_test is not None:
        pred_df['true_label'] = y_test.values.ravel()
        pred_df['true_cyt_score'] = y_cyt_test.to_numpy().flatten()
        pred_df['true_nuc_score'] = y_nuc_test.to_numpy().flatten()

    predictions_csv_path = os.path.join(args.output_dir, "cosmx_predictions.csv")
    pred_df.to_csv(predictions_csv_path, index=False)
    print(f"[Saved] Full predictions -> {predictions_csv_path}")

    # Save joblib evaluation bundles
    joblib.dump(results_classifier, os.path.join(args.output_dir, "results_classifier_cosmx.joblib"))
    joblib.dump(results_regression, os.path.join(args.output_dir, "results_regression_cosmx.joblib"))
    print(f"[Saved] Evaluation joblib objects -> {args.output_dir}")

    # Optional model persistence
    if args.save_models:
        model_bundle = {
            'classifier': classifier_wrapper,
            'reg_cyt': reg_cyt_wrapper,
            'reg_nuc': reg_nuc_wrapper,
            'catboost_encoder': catboost_encoder,
            'feature_columns': X_train_scaled.columns.tolist()
        }
        models_path = os.path.join(args.output_dir, "cosmx_trained_models.joblib")
        joblib.dump(model_bundle, models_path)
        print(f"[Saved] Trained model bundle -> {models_path}")

    print("\nExecution completed successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CCIDeconv on spatial ST data and evaluate on CosMx dataset.")
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/training_data/training_data.csv",
        help="Path to training data CSV"
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default="data/cosmx/cosmx_test.csv",
        help="Path to CosMx test CSV file"
    )
    parser.add_argument(
        "--categorical_columns",
        type=str,
        default="lr_pair,source,target,pathway_name,annotation,ligand.family,ligand.keyword,ligand.secreted_type,ligand.transmembrane,receptor.family,receptor.keyword,receptor.surfaceome_main,receptor.surfaceome_sub,receptor.adhesome,receptor.secreted_type,receptor.transmembrane",
        help="Comma-separated list of categorical columns"
    )
    parser.add_argument(
        "--exclude_columns",
        type=str,
        default="cyt_pval,cyt_pspatial,cyt_P1,sample,cell_pval,cell_P1,tissue,is_neurotransmitter,ligand_location_cellchat,receptor_location_cellchat,ligand_location_hpa,receptor_location_hpa,nuc_pval,nuc_pspatial,nuc_P1,ligand,receptor,labels",
        help="Comma-separated list of columns to exclude"
    )
    parser.add_argument(
        "--group_column",
        type=str,
        default="sample",
        help="Column name for groups in training data"
    )
    parser.add_argument(
        "--spatial",
        action="store_true",
        default=True,
        help="Whether to include spatial information (defaults to True)"
    )
    parser.add_argument(
        "--filter_cell_pval",
        action="store_true",
        default=True,
        help="Filter test data for significant whole-cell interactions (cell_pval < 0.05)"
    )
    parser.add_argument(
        "--no_filter_cell_pval",
        action="store_false",
        dest="filter_cell_pval",
        help="Do not filter test data by cell_pval < 0.05"
    )
    parser.add_argument(
        "--init_points",
        type=int,
        default=3,
        help="Number of initial random points for Bayesian Optimization"
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=30,
        help="Number of Bayesian Optimization iterations"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/cosmx/results",
        help="Directory to save evaluation and prediction files"
    )
    parser.add_argument(
        "--save_models",
        action="store_true",
        help="Whether to save the trained model bundle to disk"
    )

    args = parser.parse_args()
    main(args)
