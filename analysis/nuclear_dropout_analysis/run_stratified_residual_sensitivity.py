import os
import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score, mean_squared_error

def main():
    scratch_dir = "/Users/rjay0256/.gemini/antigravity-ide/brain/c1dfd39f-7c93-4072-91e5-563151ee2bec/scratch"
    res_path = os.path.join(scratch_dir, "prediction_residuals_detailed.csv")
    genes_path = os.path.join(scratch_dir, "gene_level_data_quality_metrics.csv")
    master_path = os.path.join(scratch_dir, "master_dataset_performance_and_quality.csv")

    if not (os.path.exists(res_path) and os.path.exists(genes_path) and os.path.exists(master_path)):
        print("Required files not yet generated.")
        return

    df_res = pd.read_csv(res_path)
    df_genes = pd.read_csv(genes_path)
    df_master = pd.read_csv(master_path)

    print(f"Loaded {len(df_res)} prediction residuals across 9 datasets.")

    # Merge gene-level metrics for ligand and receptor
    # Ligand metrics
    gene_sub = df_genes[['sample_key', 'gene', 'det_rate_cyt', 'det_rate_nuc', 'dropout_rate_cyt', 'dropout_rate_nuc', 'mean_expr_cyt', 'mean_expr_nuc', 'nc_ratio']]
    df_res = pd.merge(df_res, gene_sub.rename(columns={
        'gene': 'ligand',
        'det_rate_cyt': 'ligand_det_cyt',
        'det_rate_nuc': 'ligand_det_nuc',
        'dropout_rate_cyt': 'ligand_dropout_cyt',
        'dropout_rate_nuc': 'ligand_dropout_nuc',
        'mean_expr_cyt': 'ligand_expr_cyt',
        'mean_expr_nuc': 'ligand_expr_nuc',
        'nc_ratio': 'ligand_nc_ratio'
    }), on=['sample_key', 'ligand'], how='left')

    # Receptor metrics
    df_res = pd.merge(df_res, gene_sub.rename(columns={
        'gene': 'receptor',
        'det_rate_cyt': 'receptor_det_cyt',
        'det_rate_nuc': 'receptor_det_nuc',
        'dropout_rate_cyt': 'receptor_dropout_cyt',
        'dropout_rate_nuc': 'receptor_dropout_nuc',
        'mean_expr_cyt': 'receptor_expr_cyt',
        'mean_expr_nuc': 'receptor_expr_nuc',
        'nc_ratio': 'receptor_nc_ratio'
    }), on=['sample_key', 'receptor'], how='left')

    # Limiting partner detection in nucleus and cytoplasm
    df_res['min_lr_det_nuc'] = df_res[['ligand_det_nuc', 'receptor_det_nuc']].min(axis=1)
    df_res['min_lr_det_cyt'] = df_res[['ligand_det_cyt', 'receptor_det_cyt']].min(axis=1)
    df_res['mean_lr_det_nuc'] = df_res[['ligand_det_nuc', 'receptor_det_nuc']].mean(axis=1)
    df_res['mean_lr_det_cyt'] = df_res[['ligand_det_cyt', 'receptor_det_cyt']].mean(axis=1)
    df_res['min_lr_nc_ratio'] = df_res[['ligand_nc_ratio', 'receptor_nc_ratio']].min(axis=1)
    df_res['mean_lr_nc_ratio'] = df_res[['ligand_nc_ratio', 'receptor_nc_ratio']].mean(axis=1)

    # -------------------------------------------------------------
    # 1. Residual / Error Correlation Analysis
    # -------------------------------------------------------------
    valid = df_res.dropna(subset=['min_lr_det_nuc', 'nuc_abs_err', 'min_lr_nc_ratio'])
    corr_det_err, p_det_err = stats.spearmanr(valid['min_lr_det_nuc'], valid['nuc_abs_err'])
    corr_nc_err, p_nc_err = stats.spearmanr(valid['min_lr_nc_ratio'], valid['nuc_abs_err'])
    corr_score_err, p_score_err = stats.spearmanr(valid['nuc_true'], valid['nuc_abs_err'])

    print("\n--- Observation-Level Error Correlations (N={}) ---".format(len(valid)))
    print(f"Nuclear abs error vs Limiting partner nuclear detection rate: rho = {corr_det_err:.4f}, p = {p_det_err:.4e}")
    print(f"Nuclear abs error vs LR N/C ratio: rho = {corr_nc_err:.4f}, p = {p_nc_err:.4e}")
    print(f"Nuclear abs error vs True nuclear score: rho = {corr_score_err:.4f}, p = {p_score_err:.4e}")

    # -------------------------------------------------------------
    # 2. Stratified Analysis by Nuclear Detection Rate
    # -------------------------------------------------------------
    # Divide valid observations into tertiles of min_lr_det_nuc
    valid['det_stratum'] = pd.qcut(valid['min_lr_det_nuc'], q=3, labels=['Low detection', 'Medium detection', 'High detection'])
    
    strata_records = []
    for stratum, group in valid.groupby('det_stratum', observed=False):
        cyt_rmse = np.sqrt(mean_squared_error(group['cyt_true'], group['cyt_pred']))
        nuc_rmse = np.sqrt(mean_squared_error(group['nuc_true'], group['nuc_pred']))
        cyt_nrmse = cyt_rmse / np.std(group['cyt_true']) if np.std(group['cyt_true']) > 0 else np.nan
        nuc_nrmse = nuc_rmse / np.std(group['nuc_true']) if np.std(group['nuc_true']) > 0 else np.nan
        cyt_r2 = r2_score(group['cyt_true'], group['cyt_pred'])
        nuc_r2 = r2_score(group['nuc_true'], group['nuc_pred'])
        
        strata_records.append({
            'Stratum': stratum,
            'N_obs': len(group),
            'Min_det_range': f"{group['min_lr_det_nuc'].min():.4f} - {group['min_lr_det_nuc'].max():.4f}",
            'Cyt_MAE': group['cyt_abs_err'].median(),
            'Nuc_MAE': group['nuc_abs_err'].median(),
            'Cyt_RMSE': cyt_rmse,
            'Nuc_RMSE': nuc_rmse,
            'Cyt_NRMSE': cyt_nrmse,
            'Nuc_NRMSE': nuc_nrmse,
            'Cyt_R2': cyt_r2,
            'Nuc_R2': nuc_r2,
            'Delta_NRMSE': nuc_nrmse - cyt_nrmse,
            'Delta_R2': cyt_r2 - nuc_r2
        })
    df_strata = pd.DataFrame(strata_records)
    df_strata.to_csv(os.path.join(scratch_dir, "stratified_by_detection_quality.csv"), index=False)
    print("\n--- Stratified Analysis by Nuclear Detection Rate ---")
    print(df_strata[['Stratum', 'N_obs', 'Min_det_range', 'Cyt_R2', 'Nuc_R2', 'Cyt_NRMSE', 'Nuc_NRMSE', 'Delta_R2', 'Delta_NRMSE']])

    # -------------------------------------------------------------
    # 3. Sensitivity Analysis
    # -------------------------------------------------------------
    # Sensitivity A: Interaction-level filtering (Exclude interactions where min LR nuclear detection < 1%)
    high_qual_obs = valid[valid['min_lr_det_nuc'] >= 0.01]
    all_cyt_r2 = r2_score(valid['cyt_true'], valid['cyt_pred'])
    all_nuc_r2 = r2_score(valid['nuc_true'], valid['nuc_pred'])
    all_cyt_nrmse = np.sqrt(mean_squared_error(valid['cyt_true'], valid['cyt_pred'])) / np.std(valid['cyt_true'])
    all_nuc_nrmse = np.sqrt(mean_squared_error(valid['nuc_true'], valid['nuc_pred'])) / np.std(valid['nuc_true'])

    hq_cyt_r2 = r2_score(high_qual_obs['cyt_true'], high_qual_obs['cyt_pred'])
    hq_nuc_r2 = r2_score(high_qual_obs['nuc_true'], high_qual_obs['nuc_pred'])
    hq_cyt_nrmse = np.sqrt(mean_squared_error(high_qual_obs['cyt_true'], high_qual_obs['cyt_pred'])) / np.std(high_qual_obs['cyt_true'])
    hq_nuc_nrmse = np.sqrt(mean_squared_error(high_qual_obs['nuc_true'], high_qual_obs['nuc_pred'])) / np.std(high_qual_obs['nuc_true'])

    # Sensitivity B: Dataset-level filtering (Exclude datasets with poorest nuclear retention / sparsity)
    # Median nuclear detected genes cutoff
    med_det_cutoff = df_master['nuc_detected_median'].median()
    hq_datasets = df_master[df_master['nuc_detected_median'] >= med_det_cutoff]
    
    sens_records = [
        {
            'Analysis': 'All Observations (Full evaluation)',
            'N_units': len(valid),
            'Unit': 'Interactions',
            'Cyt_R2': all_cyt_r2,
            'Nuc_R2': all_nuc_r2,
            'Cyt_NRMSE': all_cyt_nrmse,
            'Nuc_NRMSE': all_nuc_nrmse,
            'Delta_R2': all_cyt_r2 - all_nuc_r2,
            'Delta_NRMSE': all_nuc_nrmse - all_cyt_nrmse
        },
        {
            'Analysis': 'Sensitivity 1: High-detection interactions (min LR det >= 1% in Nuc)',
            'N_units': len(high_qual_obs),
            'Unit': 'Interactions',
            'Cyt_R2': hq_cyt_r2,
            'Nuc_R2': hq_nuc_r2,
            'Cyt_NRMSE': hq_cyt_nrmse,
            'Nuc_NRMSE': hq_nuc_nrmse,
            'Delta_R2': hq_cyt_r2 - hq_nuc_r2,
            'Delta_NRMSE': hq_nuc_nrmse - hq_cyt_nrmse
        },
        {
            'Analysis': 'All Datasets (Full LOGO cohort)',
            'N_units': len(df_master),
            'Unit': 'Datasets',
            'Cyt_R2': df_master['cyt_R2'].mean(),
            'Nuc_R2': df_master['nuc_R2'].mean(),
            'Cyt_NRMSE': df_master['cyt_NRMSE'].mean(),
            'Nuc_NRMSE': df_master['nuc_NRMSE'].mean(),
            'Delta_R2': (df_master['cyt_R2'] - df_master['nuc_R2']).mean(),
            'Delta_NRMSE': (df_master['nuc_NRMSE'] - df_master['cyt_NRMSE']).mean()
        },
        {
            'Analysis': 'Sensitivity 2: High-nuclear-detection datasets (above-median Nuc detected genes)',
            'N_units': len(hq_datasets),
            'Unit': 'Datasets',
            'Cyt_R2': hq_datasets['cyt_R2'].mean(),
            'Nuc_R2': hq_datasets['nuc_R2'].mean(),
            'Cyt_NRMSE': hq_datasets['cyt_NRMSE'].mean(),
            'Nuc_NRMSE': hq_datasets['nuc_NRMSE'].mean(),
            'Delta_R2': (hq_datasets['cyt_R2'] - hq_datasets['nuc_R2']).mean(),
            'Delta_NRMSE': (hq_datasets['nuc_NRMSE'] - hq_datasets['cyt_NRMSE']).mean()
        }
    ]
    df_sens = pd.DataFrame(sens_records)
    df_sens.to_csv(os.path.join(scratch_dir, "sensitivity_analysis_summary.csv"), index=False)
    print("\n--- Sensitivity Analysis Summary ---")
    print(df_sens[['Analysis', 'N_units', 'Cyt_R2', 'Nuc_R2', 'Cyt_NRMSE', 'Nuc_NRMSE', 'Delta_R2', 'Delta_NRMSE']])

if __name__ == "__main__":
    main()
