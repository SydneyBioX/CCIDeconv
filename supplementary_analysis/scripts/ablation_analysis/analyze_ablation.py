# analysis/analyze_ablation.py
"""
Analysis and Visualization for CCIDeconv Feature Ablation Study

Loads results from all ablation conditions for both Spatial Procedure (SP) and Single-Cell
Procedure (ScP), computes comprehensive performance metrics across the 9 held-out ST datasets,
generates quantitative comparison tables (mean, std, delta drops), and creates publication-quality
boxplots and barplots.

Ablations:
- Full Model (Baseline)
- Ablation 1A: No CellChatDB Loc
- Ablation 1B: No HPA Loc
- Ablation 1C: No Localization (neither CellChatDB nor HPA)
- Ablation 2: No L-R Identity (with Cell Types)
- Ablation 3: No Whole Cell Score
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gmean

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CCIDECONV_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
sys.path.append(CCIDECONV_DIR)

CONDITION_NAMES = {
    'full': 'Full Model (Baseline)',
    'no_cellchat': 'Ablation 1A (No CellChatDB Loc)',
    'no_hpa': 'Ablation 1B (No HPA Loc)',
    'no_localization': 'Ablation 1C (No Localization)',
    'no_lr_identity': 'Ablation 2 (No L-R Identity, with Cell Types)',
    'no_whole_cell': 'Ablation 3 (No Whole Cell Score)',
    'no_annotations': 'Ablation 4 (No Molecular Annotations)',
    'no_annotations_no_localization': 'Ablation 5 (No Annotations & No Loc)'
}

FEATURE_COUNTS = {
    'sp': {
        'full': 210,
        'no_cellchat': 77,
        'no_hpa': 151,
        'no_localization': 18,
        'no_lr_identity': 209,
        'no_whole_cell': 208,
        'no_annotations': 197,
        'no_annotations_no_localization': 5
    },
    'scp': {
        'full': 209,
        'no_cellchat': 76,
        'no_hpa': 150,
        'no_localization': 17,
        'no_lr_identity': 208,
        'no_whole_cell': 208,
        'no_annotations': 196,
        'no_annotations_no_localization': 4
    }
}

DATASET_LABELS = {
    'spe_list_brainad_cellchat': 'Brain (AD)',
    'spe_list_braingl_cellchat': 'Brain (GL)',
    'spe_list_breast_cancer_cellchat': 'Breast Cancer',
    'spe_list_lung_cellchat': 'Lung',
    'spe_list_lung_high_res_cellchat': 'Lung (High-Res)',
    'spe_list_lymphnode_cellchat': 'Lymph Node',
    'spe_list_ovarian_cellchat': 'Ovarian',
    'spe_list_panc_cellchat': 'Pancreas',
    'spe_list_skin_cellchat': 'Skin'
}


def compute_metrics_for_condition(clf_results, reg_results, condition_key, display_name, procedure):
    """
    Extract all classification, regression, and composite metrics across datasets.
    """
    datasets = list(clf_results.keys())
    records = []

    for ds in datasets:
        clf_data = clf_results[ds]
        reg_data = reg_results.get(ds, {})

        class_metrics = clf_data.get('classification_metrics', {})
        auc = class_metrics.get('roc_auc', np.nan)
        f1 = class_metrics.get('f1', np.nan)
        f2 = class_metrics.get('f2', np.nan)
        accuracy = class_metrics.get('accuracy', np.nan)

        # Parse classification report for macro recall and precision
        report = class_metrics.get('class_report', '')
        macro_recall = np.nan
        macro_precision = np.nan
        for line in report.splitlines():
            if line.strip().startswith("macro avg"):
                parts = line.split()
                macro_precision = float(parts[2])
                macro_recall = float(parts[3])
                break

        cyt_tuple = reg_data.get('cyt', ({}, None))
        nuc_tuple = reg_data.get('nuc', ({}, None))
        cyt_dict = cyt_tuple[0] if isinstance(cyt_tuple, (tuple, list)) else {}
        nuc_dict = nuc_tuple[0] if isinstance(nuc_tuple, (tuple, list)) else {}

        cyt_r2 = cyt_dict.get('R²', np.nan)
        nuc_r2 = nuc_dict.get('R²', np.nan)
        cyt_rmse = cyt_dict.get('RMSE', np.nan)
        nuc_rmse = nuc_dict.get('RMSE', np.nan)
        cyt_nrmse = cyt_dict.get('NRMSE', np.nan)
        nuc_nrmse = nuc_dict.get('NRMSE', np.nan)

        # Composite metric
        # Ensure values in gmean are strictly positive
        avg_r2 = max(0.001, np.nanmean([max(0.0, cyt_r2) if not np.isnan(cyt_r2) else 0.0,
                                       max(0.0, nuc_r2) if not np.isnan(nuc_r2) else 0.0]))
        avg_1_minus_nrmse = max(0.001, np.nanmean([max(0.0, 1.0 - cyt_nrmse) if not np.isnan(cyt_nrmse) else 0.0,
                                                  max(0.0, 1.0 - nuc_nrmse) if not np.isnan(nuc_nrmse) else 0.0]))
        comp_auc = max(0.001, auc if not np.isnan(auc) else 0.5)
        comp_recall = max(0.001, macro_recall if not np.isnan(macro_recall) else 0.5)

        composite = (comp_auc * comp_recall * avg_r2 * avg_1_minus_nrmse) ** 0.25

        records.append({
            'procedure': procedure.upper(),
            'condition_id': condition_key,
            'condition': display_name,
            'dataset_id': ds,
            'dataset': DATASET_LABELS.get(ds, ds),
            'AUC': auc,
            'balanced_accuracy': accuracy,
            'macro_f1': f1,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'f2': f2,
            'cyt_R2': cyt_r2,
            'nuc_R2': nuc_r2,
            'avg_R2': np.nanmean([cyt_r2, nuc_r2]),
            'cyt_RMSE': cyt_rmse,
            'nuc_RMSE': nuc_rmse,
            'cyt_NRMSE': cyt_nrmse,
            'nuc_NRMSE': nuc_nrmse,
            'avg_NRMSE': np.nanmean([cyt_nrmse, nuc_nrmse]),
            'composite_metric': composite
        })

    return pd.DataFrame(records)


def analyze_procedure(procedure, results_dir, output_fig_dir):
    """
    Run analysis and generate summary tables and figures for a given procedure (SP or ScP).
    """
    print(f"\n{'='*80}")
    print(f"ANALYZING PROCEDURE: {procedure.upper()}")
    print(f"{'='*80}")

    dfs = []
    for cond_key, disp_name in CONDITION_NAMES.items():
        # Check files with procedure prefix first, then legacy without prefix
        clf_file_proc = os.path.join(results_dir, f"results_classifier_{procedure}_{cond_key}.joblib")
        reg_file_proc = os.path.join(results_dir, f"results_regression_{procedure}_{cond_key}.joblib")

        clf_file_legacy = os.path.join(results_dir, f"results_classifier_{cond_key}.joblib")
        reg_file_legacy = os.path.join(results_dir, f"results_regression_{cond_key}.joblib")

        if os.path.exists(clf_file_proc) and os.path.exists(reg_file_proc):
            clf_file, reg_file = clf_file_proc, reg_file_proc
        elif procedure.lower() == 'sp' and os.path.exists(clf_file_legacy) and os.path.exists(reg_file_legacy):
            clf_file, reg_file = clf_file_legacy, reg_file_legacy
        else:
            print(f"Warning: Results missing for '{cond_key}' in {procedure.upper()}. Skipping.")
            continue

        clf_res = joblib.load(clf_file)
        reg_res = joblib.load(reg_file)
        cond_df = compute_metrics_for_condition(clf_res, reg_res, cond_key, disp_name, procedure)
        dfs.append(cond_df)

    if not dfs:
        print(f"No results found for procedure '{procedure}'.")
        return None, None

    proc_df = pd.concat(dfs, ignore_index=True)
    per_dataset_path = os.path.join(results_dir, f"ablation_per_dataset_{procedure.lower()}.csv")
    proc_df.to_csv(per_dataset_path, index=False)
    print(f"Saved detailed dataset results to: {per_dataset_path}")

    # Generate Summary Table: Mean +/- Std and Delta Drop relative to Baseline
    metrics_to_summarize = [
        'composite_metric', 'AUC', 'macro_f1', 'macro_recall',
        'cyt_R2', 'nuc_R2', 'avg_R2',
        'cyt_NRMSE', 'nuc_NRMSE', 'avg_NRMSE'
    ]

    summary_rows = []
    baseline_sub = proc_df[proc_df['condition_id'] == 'full']
    if baseline_sub.empty:
        baseline_means = proc_df[metrics_to_summarize].mean()
    else:
        baseline_means = baseline_sub[metrics_to_summarize].mean()

    for cond_key, disp_name in CONDITION_NAMES.items():
        cond_sub = proc_df[proc_df['condition_id'] == cond_key]
        if cond_sub.empty:
            continue

        n_feats = FEATURE_COUNTS.get(procedure.lower(), {}).get(cond_key, 'N/A')
        row = {
            'Procedure': procedure.upper(),
            'Condition': disp_name,
            'Features': n_feats
        }

        for m in metrics_to_summarize:
            mean_val = cond_sub[m].mean()
            std_val = cond_sub[m].std()
            row[f"{m}_mean"] = mean_val
            row[f"{m}_std"] = std_val
            row[f"{m}_formatted"] = f"{mean_val:.3f} ± {std_val:.3f}"

            base_val = baseline_means[m]
            delta = mean_val - base_val
            pct_delta = (delta / base_val) * 100.0 if base_val != 0 else 0.0
            row[f"{m}_pct_drop"] = pct_delta

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(results_dir, f"ablation_summary_table_{procedure.lower()}.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved summary table to: {summary_path}")

    print("\n" + "="*95)
    print(f"ABLATION STUDY SUMMARY TABLE ({procedure.upper()}) (Mean ± Std across 9 Datasets)")
    print("="*95)
    display_cols = ['Condition', 'Features', 'composite_metric_formatted', 'AUC_formatted', 'macro_f1_formatted', 'macro_recall_formatted', 'cyt_R2_formatted', 'nuc_R2_formatted']
    print(summary_df[display_cols].to_string(index=False))

    print("\n" + "="*95)
    print(f"PERCENTAGE CHANGE RELATIVE TO BASELINE (%) ({procedure.upper()})")
    print("="*95)
    pct_cols = ['Condition', 'composite_metric_pct_drop', 'AUC_pct_drop', 'macro_f1_pct_drop', 'macro_recall_pct_drop', 'cyt_R2_pct_drop', 'nuc_R2_pct_drop']
    print(summary_df[pct_cols].to_string(index=False))

    # --- Plotting Visualizations ---
    os.makedirs(output_fig_dir, exist_ok=True)
    sns.set_theme(style="whitegrid", font_scale=1.1)
    condition_order = [CONDITION_NAMES[k] for k in CONDITION_NAMES if k in proc_df['condition_id'].unique()]

    # Palette
    palette = ["#4379B7", "#5BA8A0", "#F2A65A", "#E05D5D", "#8E6FB0", "#C94C4C", "#5C768D", "#D97706"]

    # 1. Composite Metric Boxplot
    plt.figure(figsize=(10, 6))
    ax = sns.boxplot(
        data=proc_df, x='condition', y='composite_metric', hue='condition',
        order=condition_order, palette=palette[:len(condition_order)], width=0.55,
        boxprops=dict(alpha=0.85), fliersize=0, legend=False
    )
    sns.stripplot(
        data=proc_df, x='condition', y='composite_metric',
        order=condition_order, color='black', alpha=0.6, jitter=0.2, size=7
    )
    plt.title(f"Overall Composite Metric Across Ablation Conditions ({procedure.upper()})", fontsize=14, weight='bold', pad=15)
    plt.ylabel("Composite Metric", fontsize=12, weight='bold')
    plt.xlabel("")
    plt.xticks(rotation=25, ha='right', fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_fig_dir, f"composite_metric_boxplot_{procedure.lower()}.png"), dpi=300)
    plt.savefig(os.path.join(output_fig_dir, f"composite_metric_boxplot_{procedure.lower()}.pdf"))
    plt.close()

    # 2. Multi-Metric Comparison (AUC, Macro F1, Cyt R², Nuc R²)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharex=True)
    metric_configs = [
        ('AUC', 'Classification ROC-AUC', axes[0, 0]),
        ('macro_f1', 'Classification Macro F1', axes[0, 1]),
        ('cyt_R2', 'Cytoplasmic R²', axes[1, 0]),
        ('nuc_R2', 'Nuclear R²', axes[1, 1])
    ]

    for m_col, m_title, ax in metric_configs:
        sns.boxplot(
            data=proc_df, x='condition', y=m_col, hue='condition',
            order=condition_order, palette=palette[:len(condition_order)], width=0.55,
            boxprops=dict(alpha=0.85), fliersize=0, ax=ax, legend=False
        )
        sns.stripplot(
            data=proc_df, x='condition', y=m_col,
            order=condition_order, color='black', alpha=0.6, jitter=0.2, size=6, ax=ax
        )
        ax.set_title(m_title, fontsize=13, weight='bold')
        ax.set_ylabel(m_title, fontsize=11)
        ax.set_xlabel("")
        ax.tick_params(axis='x', rotation=25)

    plt.suptitle(f"Model Performance Across Feature Ablation Conditions ({procedure.upper()})", fontsize=16, weight='bold', y=0.99)
    plt.tight_layout()
    plt.savefig(os.path.join(output_fig_dir, f"multimetric_ablation_boxplots_{procedure.lower()}.png"), dpi=300)
    plt.savefig(os.path.join(output_fig_dir, f"multimetric_ablation_boxplots_{procedure.lower()}.pdf"))
    plt.close()

    # 3. Barplot of Performance Drop When Features are Excluded
    pct_summary_df = summary_df[summary_df['Condition'] != 'Full Model (Baseline)'].copy()
    if not pct_summary_df.empty:
        plt.figure(figsize=(11, 6))
        x_indices = np.arange(len(pct_summary_df))
        width = 0.18

        plot_metrics = [
            ('composite_metric_pct_drop', 'Composite Metric', '#4379B7'),
            ('AUC_pct_drop', 'ROC-AUC', '#5BA8A0'),
            ('macro_f1_pct_drop', 'Macro F1', '#F2A65A'),
            ('avg_R2_pct_drop', 'Avg R²', '#E05D5D')
        ]

        for i, (col, label, col_color) in enumerate(plot_metrics):
            offset = (i - 1.5) * width
            plt.bar(x_indices + offset, pct_summary_df[col], width=width, label=label, color=col_color, alpha=0.9)

        plt.axhline(0, color='gray', linestyle='--', linewidth=1)
        plt.xticks(x_indices, pct_summary_df['Condition'], rotation=20, ha='right', fontsize=11)
        plt.ylabel("Relative Change vs. Baseline (%)", fontsize=12, weight='bold')
        plt.title(f"Performance Impact of Feature Ablation vs. Full Model ({procedure.upper()})", fontsize=14, weight='bold', pad=15)
        plt.legend(frameon=True, fontsize=11)
        plt.tight_layout()
        plt.savefig(os.path.join(output_fig_dir, f"ablation_percentage_drop_barplot_{procedure.lower()}.png"), dpi=300)
        plt.savefig(os.path.join(output_fig_dir, f"ablation_percentage_drop_barplot_{procedure.lower()}.pdf"))
        plt.close()

    return proc_df, summary_df


def main():
    results_dir = os.path.join(CCIDECONV_DIR, "supplementary_analysis", "results", "ablation")
    output_fig_dir = os.path.join(results_dir, "figures")

    all_dfs = []
    all_summaries = []

    for proc in ['sp', 'scp']:
        proc_df, summary_df = analyze_procedure(proc, results_dir, output_fig_dir)
        if proc_df is not None:
            all_dfs.append(proc_df)
        if summary_df is not None:
            all_summaries.append(summary_df)

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df.to_csv(os.path.join(results_dir, "ablation_per_dataset_combined.csv"), index=False)

    if all_summaries:
        combined_summary = pd.concat(all_summaries, ignore_index=True)
        combined_summary.to_csv(os.path.join(results_dir, "ablation_summary_table_combined.csv"), index=False)
        print("\n" + "="*95)
        print("COMBINED ABLATION SUMMARY TABLE (SP vs ScP)")
        print("="*95)
        display_cols = ['Procedure', 'Condition', 'Features', 'composite_metric_formatted', 'AUC_formatted', 'macro_f1_formatted', 'cyt_R2_formatted', 'nuc_R2_formatted']
        print(combined_summary[display_cols].to_string(index=False))

    print(f"\nAll publication figures successfully saved to '{output_fig_dir}/' (PNG & PDF)")


if __name__ == "__main__":
    main()
