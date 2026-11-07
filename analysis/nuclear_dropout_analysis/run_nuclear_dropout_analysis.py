#!/usr/bin/env python3
"""
run_nuclear_dropout_analysis.py

Analyzes whether the weaker nuclear-compartment prediction performance across
the nine subcellular spatial transcriptomics (sST) datasets is associated with
underlying data quality metrics (mRNA sparsity, nuclear expression ratio, and zero fraction / detection sparsity).

Key Steps:
1. Reuses existing Leave-One-Group-Out (LOGO-CV) performance metrics from `data/results_regression.joblib`
   and expression data quality metrics from `analysis/nuclear_dropout_analysis/master_dataset_performance_and_quality.csv`.
2. Computes the three specific characteristics for each of the 9 datasets:
   - Nuclear mRNA Sparsity: Median detected genes per nuclear profile/cell
   - Nuclear Expression Ratio: Median log2(N/C) ratio across shared genes/cells
   - Nuclear Zero Fraction / Detection Sparsity: Proportion of zero/undetected expression in the nuclear compartment
3. Calculates Pearson (r, p) and Spearman (rho, p) correlations with Nuclear R², Nuclear NRMSE,
   and Cytoplasmic R²/NRMSE for comparative interpretation.
4. Generates:
   - Main Figure: 3-panel scatter plot (mRNA sparsity vs R², Expression ratio vs R², Zero fraction vs R²)
   - Supplementary Figure: 3-panel scatter plot using Nuclear NRMSE
   - Individual scatter plots for maximum flexibility
   - Summary CSV tables of metrics and statistical test results.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from adjustText import adjust_text

# Set random seed for reproducibility
np.random.seed(42)

# Ensure script runs relative to repo root or current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../.."))
os.chdir(repo_root)

output_dir = os.path.join("analysis", "nuclear_dropout_analysis")
os.makedirs(output_dir, exist_ok=True)

# 1. Load Master Dataset Table
csv_path = os.path.join(output_dir, "master_dataset_performance_and_quality.csv")
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Missing master table: {csv_path}")

df = pd.read_csv(csv_path)

# Ensure log2_nc_ratio is present
df['log2_nc_ratio'] = np.log2(df['gene_nc_ratio_median'])

# Ensure delta columns
df['delta_R2'] = df['cyt_R2'] - df['nuc_R2']
df['delta_NRMSE'] = df['nuc_NRMSE'] - df['cyt_NRMSE']

# Explicitly named columns for the 3 characteristics
df['nuc_mrna_sparsity'] = df['nuc_detected_median']
df['nuc_expression_ratio'] = df['log2_nc_ratio']
df['nuc_zero_fraction_matrix'] = df['nuc_matrix_dropout']
df['nuc_zero_fraction_cell_median'] = df['nuc_zero_frac_median']

# Save clean dataset-level table for presentation
dataset_summary_cols = [
    'dataset', 'n_genes_panel', 'n_cells_cyt', 'n_cells_nuc',
    'nuc_detected_median', 'cyt_detected_median',
    'log2_nc_ratio', 'gene_nc_ratio_median',
    'nuc_matrix_dropout', 'cyt_matrix_dropout',
    'nuc_zero_frac_median', 'cyt_zero_frac_median',
    'nuc_R2', 'cyt_R2', 'delta_R2',
    'nuc_NRMSE', 'cyt_NRMSE', 'delta_NRMSE'
]
df[dataset_summary_cols].to_csv(
    os.path.join(output_dir, "dataset_nine_characteristics_and_performance.csv"),
    index=False
)
print("Saved dataset_nine_characteristics_and_performance.csv")

# 2. Compute Correlation Statistics
characteristics = [
    ('Nuclear mRNA Sparsity\n(Median Detected Genes / Nuclear Cell)', 'nuc_mrna_sparsity', 'nuc_detected_median'),
    ('Nuclear Expression Ratio\n(Median log2(N/C) Ratio across Panel)', 'nuc_expression_ratio', 'log2_nc_ratio'),
    ('Nuclear Zero Fraction / Detection Sparsity\n(Proportion of Zero Entries in Matrix)', 'nuc_zero_fraction_matrix', 'nuc_matrix_dropout')
]

targets = [
    ('Nuclear R²', 'nuc_R2'),
    ('Nuclear NRMSE', 'nuc_NRMSE'),
    ('Cytoplasmic R²', 'cyt_R2'),
    ('Cytoplasmic NRMSE', 'cyt_NRMSE'),
    ('Delta R² (Cyt - Nuc)', 'delta_R2'),
    ('Delta NRMSE (Nuc - Cyt)', 'delta_NRMSE')
]

correlations_list = []
for char_name, char_col, raw_col in characteristics:
    for target_name, target_col in targets:
        pr, pp = stats.pearsonr(df[char_col], df[target_col])
        sr, sp = stats.spearmanr(df[char_col], df[target_col])
        correlations_list.append({
            'Characteristic': char_name.replace('\n', ' '),
            'Variable': char_col,
            'Performance_Metric': target_name,
            'Metric_Variable': target_col,
            'Pearson_r': pr,
            'Pearson_p': pp,
            'Spearman_rho': sr,
            'Spearman_p': sp
        })

corr_df = pd.DataFrame(correlations_list)
corr_df.to_csv(os.path.join(output_dir, "correlation_analysis_results.csv"), index=False)
print("Saved correlation_analysis_results.csv")

# Also print table to console
print("\n" + "="*80)
print("CORRELATION ANALYSIS SUMMARY (N = 9 Datasets)")
print("="*80)
for _, r in corr_df.iterrows():
    print(f"{r['Characteristic'][:35]:35s} vs {r['Performance_Metric']:20s} | "
          f"Pearson r = {r['Pearson_r']:+.4f} (p = {r['Pearson_p']:.4f}) | "
          f"Spearman rho = {r['Spearman_rho']:+.4f} (p = {r['Spearman_p']:.4f})")

# 3. Plotting Setup
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

# Dataset colors
dataset_colors = {
    'Brain (AD)': '#2b5c8f',
    'Brain (GL)': '#3a9254',
    'Breast Cancer': '#d9383a',
    'Lung': '#7b5ea7',
    'Lung (High-Res)': '#8c6239',
    'Lymph Node': '#c95998',
    'Ovarian': '#607274',
    'Pancreas': '#d4a320',
    'Skin': '#17becf'
}

# ----------------------------------------------------
# MAIN FIGURE: 3 Plots for Nuclear R²
# ----------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), sharey=True)
fig.subplots_adjust(wspace=0.16, bottom=0.18, top=0.88, left=0.06, right=0.98)

main_configs = [
    {
        'col': 'nuc_mrna_sparsity',
        'xlabel': 'Nuclear mRNA Sparsity\n(Median Detected Genes per Nuclear Profile)',
        'title': 'mRNA Sparsity vs Nuclear R²',
        'box_x': 0.48, 'box_y': 0.08, 'box_va': 'bottom'
    },
    {
        'col': 'nuc_expression_ratio',
        'xlabel': 'Nuclear Expression Ratio\n(Median log$_2$(N/C) Ratio across Panel)',
        'title': 'Expression Ratio vs Nuclear R²',
        'box_x': 0.05, 'box_y': 0.88, 'box_va': 'top'
    },
    {
        'col': 'nuc_zero_fraction_matrix',
        'xlabel': 'Nuclear Zero Fraction / Detection Sparsity\n(Proportion of Zero Matrix Entries)',
        'title': 'Zero Fraction vs Nuclear R²',
        'box_x': 0.05, 'box_y': 0.88, 'box_va': 'top'
    }
]

for idx, (ax, cfg) in enumerate(zip(axes, main_configs)):
    x_val = df[cfg['col']]
    y_val = df['nuc_R2']
    
    # Regression line with stats
    pr, pp = stats.pearsonr(x_val, y_val)
    sr, sp = stats.spearmanr(x_val, y_val)
    
    # Plot regression line
    sns.regplot(
        x=cfg['col'], y='nuc_R2', data=df, ax=ax,
        color='#475569',
        scatter=False,
        line_kws={'linewidth': 2, 'linestyle': '--', 'alpha': 0.8},
        ci=95
    )
    
    # Scatter points with dataset colors
    texts = []
    for _, row in df.iterrows():
        c = dataset_colors.get(row['dataset'], '#333333')
        ax.scatter(row[cfg['col']], row['nuc_R2'], color=c, s=130, edgecolors='black', linewidth=1.2, zorder=5)
        texts.append(ax.text(row[cfg['col']], row['nuc_R2'], row['dataset'], fontsize=9.5, fontweight='bold', color='#0f172a'))
    
    adjust_text(texts, ax=ax, expand_points=(1.5, 1.5), arrowprops=dict(arrowstyle='->', color='#64748b', lw=0.8, alpha=0.7))
    
    # Formatting
    ax.set_xlabel(cfg['xlabel'], fontsize=11.5, fontweight='bold', labelpad=8)
    if idx == 0:
        ax.set_ylabel('Nuclear Prediction Performance (R²)', fontsize=12.5, fontweight='bold')
    else:
        ax.set_ylabel('')
    
    ax.set_title(cfg['title'], fontsize=13.5, fontweight='bold', pad=12)
    ax.grid(True, linestyle=':', alpha=0.5, color='#cbd5e1')
    ax.set_ylim(0.42, 1.08)
    
    # Text box with stats
    stats_text = (
        f"Pearson $r = {pr:+.3f}$ ($P = {pp:.3f}$)\n"
        f"Spearman $\\rho = {sr:+.3f}$ ($P = {sp:.3f}$)"
    )
    ax.text(
        cfg['box_x'], cfg['box_y'], stats_text,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight='medium',
        verticalalignment=cfg['box_va'],
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='#94a3b8', alpha=0.95)
    )

fig.suptitle('Association between Subcellular Data Characteristics and Nuclear Prediction Accuracy (R²)', 
             fontsize=15, fontweight='bold', y=0.98)

main_fig_png = os.path.join(output_dir, "main_figure_nuclear_characteristics_vs_r2.png")
main_fig_pdf = os.path.join(output_dir, "main_figure_nuclear_characteristics_vs_r2.pdf")
fig.savefig(main_fig_png, dpi=300, bbox_inches='tight')
fig.savefig(main_fig_pdf, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved {main_fig_png} and .pdf")


# ----------------------------------------------------
# SUPPLEMENTARY FIGURE: 3 Plots for Nuclear NRMSE
# ----------------------------------------------------
fig_supp, axes_supp = plt.subplots(1, 3, figsize=(18, 5.8), sharey=True)
fig_supp.subplots_adjust(wspace=0.16, bottom=0.18, top=0.88, left=0.06, right=0.98)

supp_configs = [
    {
        'col': 'nuc_mrna_sparsity',
        'xlabel': 'Nuclear mRNA Sparsity\n(Median Detected Genes per Nuclear Profile)',
        'title': 'mRNA Sparsity vs Nuclear NRMSE',
        'box_x': 0.48, 'box_y': 0.88, 'box_va': 'top'
    },
    {
        'col': 'nuc_expression_ratio',
        'xlabel': 'Nuclear Expression Ratio\n(Median log$_2$(N/C) Ratio across Panel)',
        'title': 'Expression Ratio vs Nuclear NRMSE',
        'box_x': 0.05, 'box_y': 0.12, 'box_va': 'bottom'
    },
    {
        'col': 'nuc_zero_fraction_matrix',
        'xlabel': 'Nuclear Zero Fraction / Detection Sparsity\n(Proportion of Zero Matrix Entries)',
        'title': 'Zero Fraction vs Nuclear NRMSE',
        'box_x': 0.05, 'box_y': 0.12, 'box_va': 'bottom'
    }
]

for idx, (ax, cfg) in enumerate(zip(axes_supp, supp_configs)):
    x_val = df[cfg['col']]
    y_val = df['nuc_NRMSE']
    
    # Regression line with stats
    pr, pp = stats.pearsonr(x_val, y_val)
    sr, sp = stats.spearmanr(x_val, y_val)
    
    # Plot regression line
    sns.regplot(
        x=cfg['col'], y='nuc_NRMSE', data=df, ax=ax,
        color='#475569',
        scatter=False,
        line_kws={'linewidth': 2, 'linestyle': '--', 'alpha': 0.8},
        ci=95
    )
    
    # Scatter points with dataset colors
    texts = []
    for _, row in df.iterrows():
        c = dataset_colors.get(row['dataset'], '#333333')
        ax.scatter(row[cfg['col']], row['nuc_NRMSE'], color=c, s=130, edgecolors='black', linewidth=1.2, zorder=5)
        texts.append(ax.text(row[cfg['col']], row['nuc_NRMSE'], row['dataset'], fontsize=9.5, fontweight='bold', color='#0f172a'))
    
    adjust_text(texts, ax=ax, expand_points=(1.5, 1.5), arrowprops=dict(arrowstyle='->', color='#64748b', lw=0.8, alpha=0.7))
    
    # Formatting
    ax.set_xlabel(cfg['xlabel'], fontsize=11.5, fontweight='bold', labelpad=8)
    if idx == 0:
        ax.set_ylabel('Nuclear Error Rate (NRMSE)', fontsize=12.5, fontweight='bold')
    else:
        ax.set_ylabel('')
    
    ax.set_title(cfg['title'], fontsize=13.5, fontweight='bold', pad=12)
    ax.grid(True, linestyle=':', alpha=0.5, color='#cbd5e1')
    ax.set_ylim(0.02, 0.78)
    
    # Text box with stats
    stats_text = (
        f"Pearson $r = {pr:+.3f}$ ($P = {pp:.3f}$)\n"
        f"Spearman $\\rho = {sr:+.3f}$ ($P = {sp:.3f}$)"
    )
    ax.text(
        cfg['box_x'], cfg['box_y'], stats_text,
        transform=ax.transAxes,
        fontsize=10.5,
        fontweight='medium',
        verticalalignment=cfg['box_va'],
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='#94a3b8', alpha=0.95)
    )

fig_supp.suptitle('Supplementary: Association between Subcellular Data Characteristics and Nuclear Error Rate (NRMSE)', 
                  fontsize=15, fontweight='bold', y=0.98)

supp_fig_png = os.path.join(output_dir, "supplementary_figure_nuclear_characteristics_vs_nrmse.png")
supp_fig_pdf = os.path.join(output_dir, "supplementary_figure_nuclear_characteristics_vs_nrmse.pdf")
fig_supp.savefig(supp_fig_png, dpi=300, bbox_inches='tight')
fig_supp.savefig(supp_fig_pdf, dpi=300, bbox_inches='tight')
plt.close(fig_supp)
print(f"Saved {supp_fig_png} and .pdf")


# ----------------------------------------------------
# INDIVIDUAL SCATTER PLOTS (For granular presentation)
# ----------------------------------------------------
indiv_configs = [
    ('sparsity_vs_r2', 'nuc_mrna_sparsity', 'nuc_R2', 
     'Nuclear mRNA Sparsity\n(Median Detected Genes per Nuclear Profile)', 'Nuclear R²', 
     'Nuclear mRNA Sparsity vs Nuclear R²'),
    ('expr_ratio_vs_r2', 'nuc_expression_ratio', 'nuc_R2', 
     'Nuclear Expression Ratio\n(Median log$_2$(N/C) Ratio across Panel)', 'Nuclear R²', 
     'Nuclear Expression Ratio vs Nuclear R²'),
    ('zero_frac_vs_r2', 'nuc_zero_fraction_matrix', 'nuc_R2', 
     'Nuclear Zero Fraction / Detection Sparsity\n(Proportion of Zero Matrix Entries)', 'Nuclear R²', 
     'Nuclear Zero Fraction vs Nuclear R²'),
    ('sparsity_vs_nrmse', 'nuc_mrna_sparsity', 'nuc_NRMSE', 
     'Nuclear mRNA Sparsity\n(Median Detected Genes per Nuclear Profile)', 'Nuclear NRMSE', 
     'Nuclear mRNA Sparsity vs Nuclear NRMSE'),
    ('expr_ratio_vs_nrmse', 'nuc_expression_ratio', 'nuc_NRMSE', 
     'Nuclear Expression Ratio\n(Median log$_2$(N/C) Ratio across Panel)', 'Nuclear NRMSE', 
     'Nuclear Expression Ratio vs Nuclear NRMSE'),
    ('zero_frac_vs_nrmse', 'nuc_zero_fraction_matrix', 'nuc_NRMSE', 
     'Nuclear Zero Fraction / Detection Sparsity\n(Proportion of Zero Matrix Entries)', 'Nuclear NRMSE', 
     'Nuclear Zero Fraction vs Nuclear NRMSE')
]

for tag, x_col, y_col, xlab, ylab, title in indiv_configs:
    fig_ind, ax_ind = plt.subplots(figsize=(6.5, 5.5))
    x_val = df[x_col]
    y_val = df[y_col]
    pr, pp = stats.pearsonr(x_val, y_val)
    sr, sp = stats.spearmanr(x_val, y_val)
    
    sns.regplot(
        x=x_col, y=y_col, data=df, ax=ax_ind,
        color='#475569',
        scatter=False,
        line_kws={'linewidth': 2, 'linestyle': '--', 'alpha': 0.8},
        ci=95
    )
    texts = []
    for _, row in df.iterrows():
        c = dataset_colors.get(row['dataset'], '#333333')
        ax_ind.scatter(row[x_col], row[y_col], color=c, s=130, edgecolors='black', linewidth=1.2, zorder=5)
        texts.append(ax_ind.text(row[x_col], row[y_col], row['dataset'], fontsize=9.5, fontweight='bold', color='#1e293b'))
    
    adjust_text(texts, ax=ax_ind, arrowprops=dict(arrowstyle='->', color='#64748b', lw=0.8, alpha=0.8))
    
    ax_ind.set_xlabel(xlab, fontsize=11, fontweight='bold', labelpad=8)
    ax_ind.set_ylabel(ylab, fontsize=12, fontweight='bold')
    ax_ind.set_title(title, fontsize=13, fontweight='bold', pad=12)
    ax_ind.grid(True, linestyle=':', alpha=0.5, color='#cbd5e1')
    
    loc_y = 0.08 if 'R²' in ylab else 0.92
    valign = 'bottom' if 'R²' in ylab else 'top'
    stats_text = (
        f"Pearson $r = {pr:+.3f}$ ($P = {pp:.3f}$)\n"
        f"Spearman $\\rho = {sr:+.3f}$ ($P = {sp:.3f}$)"
    )
    ax_ind.text(
        0.05, loc_y, stats_text,
        transform=ax_ind.transAxes,
        fontsize=10.5,
        fontweight='medium',
        verticalalignment=valign,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8fafc', edgecolor='#cbd5e1', alpha=0.92)
    )
    
    p_out = os.path.join(output_dir, f"plot_{tag}.png")
    fig_ind.savefig(p_out, dpi=300, bbox_inches='tight')
    plt.close(fig_ind)

print("Generated all individual plots successfully.")
