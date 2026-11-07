import os
import joblib
import numpy as np
import pandas as pd
from scipy import stats

def spearman_ci_fisher(rho, n, alpha=0.05):
    """Compute confidence interval for Spearman rho using Fisher z-transform approximation."""
    if np.isnan(rho) or n <= 3:
        return np.nan, np.nan
    # Clamp rho to avoid inf
    r = np.clip(rho, -0.999999, 0.999999)
    z = np.arctanh(r)
    # Standard error for Spearman rho using Fieller et al. approx: se = 1.06 / sqrt(n - 3)
    se = 1.06 / np.sqrt(n - 3)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci_low = np.tanh(z - z_crit * se)
    ci_high = np.tanh(z + z_crit * se)
    return ci_low, ci_high

def main():
    scratch_dir = "/Users/rjay0256/.gemini/antigravity-ide/brain/c1dfd39f-7c93-4072-91e5-563151ee2bec/scratch"
    summary_path = os.path.join(scratch_dir, "dataset_data_quality_metrics.csv")
    if not os.path.exists(summary_path):
        print(f"Waiting for {summary_path}...")
        return

    df_quality = pd.read_csv(summary_path)
    print("Loaded data quality summary:", df_quality.shape)

    # Load model evaluation metrics
    reg_results = joblib.load("data/results_regression.joblib")
    clf_results = joblib.load("data/results_classifier.joblib")

    perf_records = []
    for sample_key in df_quality['sample_key']:
        if sample_key in reg_results:
            cyt_m, cyt_p = reg_results[sample_key]['cyt']
            nuc_m, nuc_p = reg_results[sample_key]['nuc']
            perf_records.append({
                'sample_key': sample_key,
                'cyt_R2': cyt_m['R²'],
                'cyt_RMSE': cyt_m['RMSE'],
                'cyt_NRMSE': cyt_m['NRMSE'],
                'nuc_R2': nuc_m['R²'],
                'nuc_RMSE': nuc_m['RMSE'],
                'nuc_NRMSE': nuc_m['NRMSE'],
                'delta_R2': cyt_m['R²'] - nuc_m['R²'],
                'delta_NRMSE': nuc_m['NRMSE'] - cyt_m['NRMSE'],
                'n_reg_preds': len(cyt_p)
            })
    df_perf = pd.DataFrame(perf_records)
    df_merged = pd.merge(df_quality, df_perf, on='sample_key')
    print("Merged dataset table shape:", df_merged.shape)

    # Save merged dataset table
    merged_path = os.path.join(scratch_dir, "master_dataset_performance_and_quality.csv")
    df_merged.to_csv(merged_path, index=False)
    print(f"Saved master dataset table to {merged_path}")

    # Compute correlation tests
    predictors = [
        # Sparsity
        ('nuc_detected_median', 'Nuclear detected genes (median)', 'Nuclear mRNA Sparsity'),
        ('nuc_zero_frac_median', 'Nuclear zero fraction (median)', 'Nuclear mRNA Sparsity'),
        ('detected_ratio_nuc_to_cyt', 'Detected genes ratio (Nuc / Cyt)', 'Nuclear mRNA Sparsity'),
        ('detected_diff_cyt_minus_nuc', 'Detection deficit (Cyt - Nuc genes)', 'Nuclear mRNA Sparsity'),
        ('zero_frac_diff_nuc_minus_cyt', 'Zero fraction difference (Nuc - Cyt)', 'Nuclear mRNA Sparsity'),
        ('tot_expr_ratio_nuc_to_cyt', 'Total expression ratio (Nuc / Cyt)', 'Nuclear mRNA Sparsity'),
        ('cyt_detected_median', 'Cytoplasmic detected genes (median)', 'Cytoplasmic Sparsity'),
        ('cyt_zero_frac_median', 'Cytoplasmic zero fraction (median)', 'Cytoplasmic Sparsity'),
        
        # Dropout
        ('nuc_matrix_dropout', 'Nuclear matrix dropout rate', 'Technical Dropout'),
        ('nuc_gene_dropout_median', 'Nuclear gene dropout rate (median)', 'Technical Dropout'),
        ('delta_matrix_dropout_nuc_minus_cyt', 'Matrix dropout gap (Nuc - Cyt)', 'Technical Dropout'),
        ('delta_gene_dropout_nuc_minus_cyt', 'Gene dropout gap (Nuc - Cyt)', 'Technical Dropout'),
        ('cyt_matrix_dropout', 'Cytoplasmic matrix dropout rate', 'Technical Dropout'),
        ('cyt_gene_dropout_median', 'Cytoplasmic gene dropout rate (median)', 'Technical Dropout'),
        
        # N/C ratio
        ('gene_nc_ratio_median', 'Gene N/C ratio (median)', 'Nuclear-to-Cytoplasmic Distribution'),
        ('frac_genes_predom_nuc', 'Fraction genes predominantly nuclear', 'Nuclear-to-Cytoplasmic Distribution'),
        ('frac_genes_predom_cyt', 'Fraction genes predominantly cytoplasmic', 'Nuclear-to-Cytoplasmic Distribution'),
        
        # Gene panel coverage
        ('n_genes_panel', 'Total panel gene count', 'Gene-Panel Coverage'),
        ('n_model_genes_det_nuc_1pct', 'Model genes detected in Nuc (>=1% cells)', 'Gene-Panel Coverage'),
        ('model_gene_coverage_ratio_1pct', 'Model gene coverage ratio (Nuc / Cyt)', 'Gene-Panel Coverage'),
        ('model_det_rate_nuc_median', 'Model gene detection rate in Nuc (median)', 'Gene-Panel Coverage'),
        ('delta_model_det_rate', 'Model gene detection gap (Cyt - Nuc)', 'Gene-Panel Coverage'),
        
        # Cell retention
        ('nuc_to_cyt_cell_ratio', 'QC cell retention ratio (Nuc / Cyt cells)', 'QC Filtering / Sparsity')
    ]

    outcomes = [
        ('nuc_R2', 'Nuclear R²'),
        ('nuc_NRMSE', 'Nuclear NRMSE'),
        ('cyt_R2', 'Cytoplasmic R²'),
        ('cyt_NRMSE', 'Cytoplasmic NRMSE'),
        ('delta_R2', 'Performance gap ΔR² (Cyt - Nuc)'),
        ('delta_NRMSE', 'Performance gap ΔNRMSE (Nuc - Cyt)')
    ]

    corr_records = []

    for pred_col, pred_label, category in predictors:
        x_all = df_merged[pred_col].values
        for out_col, out_label in outcomes:
            y_all = df_merged[out_col].values
            mask = np.isfinite(x_all) & np.isfinite(y_all)
            x = x_all[mask]
            y = y_all[mask]
            n_samples = len(x)
            if n_samples >= 4:
                rho, pval = stats.spearmanr(x, y)
                ci_low, ci_high = spearman_ci_fisher(rho, n_samples)
                pearson_r, pearson_pval = stats.pearsonr(x, y)
            else:
                rho, pval, ci_low, ci_high, pearson_r, pearson_pval = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
            corr_records.append({
                'Category': category,
                'Predictor': pred_col,
                'Predictor_Label': pred_label,
                'Outcome': out_col,
                'Outcome_Label': out_label,
                'Spearman_rho': rho,
                'P_value': pval,
                'CI_95_low': ci_low,
                'CI_95_high': ci_high,
                'Pearson_r': pearson_r,
                'Pearson_p': pearson_pval,
                'N': n_samples
            })

    df_corrs = pd.DataFrame(corr_records)
    corr_path = os.path.join(scratch_dir, "spearman_correlations_all.csv")
    df_corrs.to_csv(corr_path, index=False)
    print(f"Saved correlation analysis to {corr_path}")

    # Residuals & Stratified Analysis
    # Load raw_data or training_data to match predictions
    train_df = pd.read_csv("data/training_data/training_data.csv")
    res_records = []
    for sample_key in df_quality['sample_key']:
        label = df_quality.loc[df_quality['sample_key'] == sample_key, 'dataset'].values[0]
        if sample_key in reg_results:
            _, cyt_p = reg_results[sample_key]['cyt']
            _, nuc_p = reg_results[sample_key]['nuc']
            # cyt_p and nuc_p have pred, true, lr_pair, source, target
            for i in range(len(cyt_p)):
                row_c = cyt_p.iloc[i]
                row_n = nuc_p.iloc[i]
                res_records.append({
                    'sample_key': sample_key,
                    'dataset': label,
                    'lr_pair': row_c['lr_pair'],
                    'source': row_c['source'],
                    'target': row_c['target'],
                    'cyt_true': row_c['true'],
                    'cyt_pred': row_c['pred'],
                    'cyt_abs_err': abs(row_c['pred'] - row_c['true']),
                    'cyt_sq_err': (row_c['pred'] - row_c['true'])**2,
                    'nuc_true': row_n['true'],
                    'nuc_pred': row_n['pred'],
                    'nuc_abs_err': abs(row_n['pred'] - row_n['true']),
                    'nuc_sq_err': (row_n['pred'] - row_n['true'])**2,
                })
    df_residuals = pd.DataFrame(res_records)
    
    # Merge with original unscaled training scores and genes
    sub_train = train_df[['sample', 'lr_pair', 'source', 'target', 'cell_score', 'cyt_score', 'nuc_score', 'ligand', 'receptor']].drop_duplicates()
    df_residuals = pd.merge(df_residuals, sub_train, left_on=['sample_key', 'lr_pair', 'source', 'target'], right_on=['sample', 'lr_pair', 'source', 'target'], how='left')
    
    res_path = os.path.join(scratch_dir, "prediction_residuals_detailed.csv")
    df_residuals.to_csv(res_path, index=False)
    print(f"Saved prediction residuals to {res_path} (N={len(df_residuals)})")

if __name__ == "__main__":
    main()
