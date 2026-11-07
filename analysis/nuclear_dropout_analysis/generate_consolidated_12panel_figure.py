#!/usr/bin/env python3
"""
generate_consolidated_12panel_figure.py

Generates the single consolidated 12-panel figure (3 rows x 4 columns)
correlating the 3 nuclear measurement characteristics with all 4 prediction-performance metrics:
- Nuclear R²
- Nuclear NRMSE
- Cytoplasmic R²
- Cytoplasmic NRMSE

Across all nine sST datasets evaluated under Leave-One-Group-Out cross-validation (n = 9).
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from adjustText import adjust_text

# Set random seed
np.random.seed(42)

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(current_dir, "../.."))
os.chdir(repo_root)

output_dir = os.path.join("analysis", "nuclear_dropout_analysis")
os.makedirs(output_dir, exist_ok=True)

# 1. Load data
csv_path = os.path.join(output_dir, "master_dataset_performance_and_quality.csv")
df = pd.read_csv(csv_path)
df['log2_nc_ratio'] = np.log2(df['gene_nc_ratio_median'])

# Distinct color palette for the 9 datasets
dataset_colors = {
    'Brain (AD)': '#2b5c8f',
    'Brain (GL)': '#3a9254',
    'Breast Cancer': '#d9383a',
    'Lung': '#7b5ea7',
    'Lung (High-Res)': '#8c6239',
    'Lymph Node': '#c95998',
    'Ovarian': '#506575',
    'Pancreas': '#d4a320',
    'Skin': '#17becf'
}

# The 3 nuclear characteristics (rows)
rows_config = [
    {
        'name': 'Nuclear mRNA Sparsity',
        'sub': '(Median Detected Genes / Nuclear Profile)',
        'col': 'nuc_detected_median',
        'xlim': (10, 138),
        'box_pos': {
            'R2': {'x': 0.46, 'y': 0.08, 'va': 'bottom'},
            'NRMSE': {'x': 0.46, 'y': 0.88, 'va': 'top'}
        }
    },
    {
        'name': 'Nuclear:Cytoplasmic Expression Ratio',
        'sub': '(Median log$_2$(N/C) Ratio across Panel)',
        'col': 'log2_nc_ratio',
        'xlim': (-1.08, 0.58),
        'box_pos': {
            'R2': {'x': 0.05, 'y': 0.88, 'va': 'top'},
            'NRMSE': {'x': 0.05, 'y': 0.12, 'va': 'bottom'}
        }
    },
    {
        'name': 'Nuclear Zero Fraction / Detection Sparsity',
        'sub': '(Proportion of Zero Matrix Entries)',
        'col': 'nuc_matrix_dropout',
        'xlim': (0.865, 0.995),
        'box_pos': {
            'R2': {'x': 0.05, 'y': 0.88, 'va': 'top'},
            'NRMSE': {'x': 0.05, 'y': 0.12, 'va': 'bottom'}
        }
    }
]

# The 4 performance metrics (columns)
cols_config = [
    {
        'name': 'Nuclear R²',
        'col': 'nuc_R2',
        'type': 'R2',
        'ylim': (0.42, 1.08),
        'ylabel': 'Nuclear R²\n(Higher = Superior Accuracy)'
    },
    {
        'name': 'Nuclear NRMSE',
        'col': 'nuc_NRMSE',
        'type': 'NRMSE',
        'ylim': (0.02, 0.78),
        'ylabel': 'Nuclear NRMSE\n(Lower = Superior Accuracy)'
    },
    {
        'name': 'Cytoplasmic R²',
        'col': 'cyt_R2',
        'type': 'R2',
        'ylim': (0.42, 1.08),
        'ylabel': 'Cytoplasmic R²\n(Higher = Superior Accuracy)'
    },
    {
        'name': 'Cytoplasmic NRMSE',
        'col': 'cyt_NRMSE',
        'type': 'NRMSE',
        'ylim': (0.02, 0.78),
        'ylabel': 'Cytoplasmic NRMSE\n(Lower = Superior Accuracy)'
    }
]

# Set styling
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.1
plt.rcParams['xtick.major.width'] = 1.1
plt.rcParams['ytick.major.width'] = 1.1

fig, axes = plt.subplots(3, 4, figsize=(24, 16.5))
fig.subplots_adjust(hspace=0.28, wspace=0.22, top=0.93, bottom=0.06, left=0.06, right=0.98)

# Title for the whole figure
fig.suptitle(
    'Consolidated Evaluation of Nuclear Measurement Characteristics vs. Prediction Performance (n = 9 sST Datasets)',
    fontsize=18, fontweight='bold', y=0.97
)

for r_idx, r_cfg in enumerate(rows_config):
    for c_idx, c_cfg in enumerate(cols_config):
        ax = axes[r_idx, c_idx]
        x_data = df[r_cfg['col']]
        y_data = df[c_cfg['col']]
        
        # Pearson and Spearman
        pr, pp = stats.pearsonr(x_data, y_data)
        sr, sp = stats.spearmanr(x_data, y_data)
        
        # Linear regression trendline with 95% CI band
        sns.regplot(
            x=r_cfg['col'], y=c_cfg['col'], data=df, ax=ax,
            color='#475569', scatter=False,
            line_kws={'linewidth': 1.8, 'linestyle': '--', 'alpha': 0.85},
            ci=95
        )
        
        # Scatter points with dataset colors
        texts = []
        for _, row in df.iterrows():
            c = dataset_colors.get(row['dataset'], '#333333')
            ax.scatter(
                row[r_cfg['col']], row[c_cfg['col']],
                color=c, s=110, edgecolors='black', linewidth=1.1, zorder=5
            )
            texts.append(
                ax.text(
                    row[r_cfg['col']], row[c_cfg['col']], row['dataset'],
                    fontsize=8.5, fontweight='bold', color='#0f172a'
                )
            )
        
        # Collision avoidance for point labels
        adjust_text(
            texts, ax=ax, expand_points=(1.4, 1.4),
            arrowprops=dict(arrowstyle='->', color='#64748b', lw=0.7, alpha=0.7)
        )
        
        # X & Y limits
        ax.set_xlim(r_cfg['xlim'])
        ax.set_ylim(c_cfg['ylim'])
        
        # Grid lines
        ax.grid(True, linestyle=':', alpha=0.5, color='#cbd5e1')
        
        # Labels
        ax.set_xlabel(f"{r_cfg['name']}\n{r_cfg['sub']}", fontsize=10.5, fontweight='bold', labelpad=6)
        ax.set_ylabel(c_cfg['ylabel'], fontsize=11, fontweight='bold', labelpad=6)
        
        # Panel Title
        ax.set_title(f"{r_cfg['name']}\nvs {c_cfg['name']}", fontsize=11.5, fontweight='bold', pad=8)
        
        # Stats annotation box
        box_meta = r_cfg['box_pos'][c_cfg['type']]
        stats_text = (
            f"Pearson $r = {pr:+.3f}$ ($P = {pp:.3f}$)\n"
            f"Spearman $\\rho = {sr:+.3f}$ ($P = {sp:.3f}$)"
        )
        ax.text(
            box_meta['x'], box_meta['y'], stats_text,
            transform=ax.transAxes,
            fontsize=9.5,
            fontweight='medium',
            verticalalignment=box_meta['va'],
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8fafc', edgecolor='#94a3b8', alpha=0.92)
        )

# Save figure in output_dir
fig_png = os.path.join(output_dir, "figure_consolidated_12panel_nuclear_characteristics_vs_performance.png")
fig_pdf = os.path.join(output_dir, "figure_consolidated_12panel_nuclear_characteristics_vs_performance.pdf")
fig.savefig(fig_png, dpi=300, bbox_inches='tight')
fig.savefig(fig_pdf, dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"Saved {fig_png} and .pdf")

# Also copy to manuscript revisions figures directory
ms_fig_dir = "/Users/rjay0256/Library/CloudStorage/OneDrive-TheUniversityofSydney(Students)/University of Sydney/PhD projects/main/Cell Cell Communication/Manuscript/Briefings in Bioinformatics/revisions/figures"
if os.path.exists(ms_fig_dir):
    import shutil
    shutil.copy(fig_png, os.path.join(ms_fig_dir, "figure_consolidated_12panel_nuclear_characteristics_vs_performance.png"))
    shutil.copy(fig_pdf, os.path.join(ms_fig_dir, "figure_consolidated_12panel_nuclear_characteristics_vs_performance.pdf"))
    print(f"Mirrored to {ms_fig_dir}")
