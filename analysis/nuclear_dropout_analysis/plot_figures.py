import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

def make_publication_figure():
    scratch_dir = "/Users/rjay0256/.gemini/antigravity-ide/brain/c1dfd39f-7c93-4072-91e5-563151ee2bec/scratch"
    summary_path = os.path.join(scratch_dir, "master_dataset_performance_and_quality.csv")
    subcells_path = os.path.join(scratch_dir, "cell_level_subsample_metrics.csv")
    genes_path = os.path.join(scratch_dir, "gene_level_data_quality_metrics.csv")
    strata_path = os.path.join(scratch_dir, "stratified_by_detection_quality.csv")
    
    df_master = pd.read_csv(summary_path)
    df_cells = pd.read_csv(subcells_path)
    df_genes = pd.read_csv(genes_path)
    df_strata = pd.read_csv(strata_path)

    # Clean dataset labels
    label_map = {
        'Brain (AD)': 'Brain (AD)',
        'Brain (GL)': 'Brain (GL)',
        'Breast Cancer': 'Breast',
        'Lung': 'Lung',
        'Lung (High-Res)': 'Lung (HR)',
        'Lymph Node': 'Lymph Node',
        'Ovarian': 'Ovarian',
        'Pancreas': 'Pancreas',
        'Skin': 'Skin'
    }
    df_master['short_name'] = df_master['dataset'].map(label_map)
    df_cells['short_name'] = df_cells['dataset'].map(label_map)
    df_genes['short_name'] = df_genes['dataset'].map(label_map)

    # Set overall visual style
    plt.rcParams['font.sans-serif'] = 'Helvetica', 'Arial', 'DejaVu Sans'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.linewidth'] = 0.8
    plt.rcParams['axes.edgecolor'] = '#333333'
    plt.rcParams['xtick.major.width'] = 0.8
    plt.rcParams['ytick.major.width'] = 0.8

    # Colors
    c_cyt = '#2b83ba'  # Rich teal/blue
    c_nuc = '#d7191c'  # Crimson/coral red
    c_gap = '#7b3294'  # Purple for gap

    fig = plt.figure(figsize=(16, 11), dpi=300)
    gs = fig.add_gridspec(2, 3, hspace=0.36, wspace=0.28, left=0.06, right=0.96, top=0.94, bottom=0.07)

    # -------------------------------------------------------------
    # Panel A: Cellular mRNA Sparsity: Detected Genes per Cell
    # -------------------------------------------------------------
    ax_a = fig.add_subplot(gs[0, 0])
    order = df_master.sort_values('cyt_detected_median', ascending=False)['short_name'].tolist()
    
    sns.boxplot(
        data=df_cells,
        x='short_name',
        y='detected_genes',
        hue='compartment',
        palette={'Cytoplasm': c_cyt, 'Nucleus': c_nuc},
        order=order,
        ax=ax_a,
        fliersize=0.5,
        linewidth=0.7,
        width=0.65
    )
    ax_a.set_title("A. Cellular mRNA Sparsity: Detected Genes", fontsize=11, fontweight='bold', loc='left', pad=8)
    ax_a.set_xlabel("")
    ax_a.set_ylabel("Detected Genes per Cell", fontsize=10)
    ax_a.tick_params(axis='x', rotation=40)
    ax_a.grid(axis='y', linestyle='--', alpha=0.3)
    ax_a.legend(title="", frameon=True, facecolor='white', framealpha=0.9, fontsize=9, loc='upper right')

    overall_cyt_med = df_master['cyt_detected_median'].median()
    overall_nuc_med = df_master['nuc_detected_median'].median()
    ax_a.text(0.04, 0.92, f"Cohort Medians:\nCyt = {overall_cyt_med:.0f} genes\nNuc = {overall_nuc_med:.0f} genes\n(Ratio: {overall_nuc_med/overall_cyt_med:.2f})",
              transform=ax_a.transAxes, fontsize=8.5, verticalalignment='top',
              bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='lightgray', alpha=0.9))

    # -------------------------------------------------------------
    # Panel B: Gene-Level Technical Dropout Rate
    # -------------------------------------------------------------
    ax_b = fig.add_subplot(gs[0, 1])
    sns.violinplot(
        data=df_genes,
        x='short_name',
        y='dropout_rate_nuc',
        color=c_nuc,
        order=order,
        ax=ax_b,
        inner=None,
        alpha=0.3,
        density_norm='width'
    )
    for idx, name in enumerate(order):
        row = df_master[df_master['short_name'] == name].iloc[0]
        ax_b.scatter(idx - 0.15, row['cyt_gene_dropout_median'], color=c_cyt, s=40, zorder=4, edgecolor='black', linewidth=0.5, label='Cyt median' if idx==0 else "")
        ax_b.scatter(idx + 0.15, row['nuc_gene_dropout_median'], color=c_nuc, s=40, zorder=4, edgecolor='black', linewidth=0.5, label='Nuc median' if idx==0 else "")
        ax_b.plot([idx - 0.15, idx + 0.15], [row['cyt_gene_dropout_median'], row['nuc_gene_dropout_median']], color='#666666', linestyle='-', linewidth=1.2, zorder=3)

    ax_b.set_title("B. Gene-Level Technical Dropout Rate", fontsize=11, fontweight='bold', loc='left', pad=8)
    ax_b.set_xlabel("")
    ax_b.set_ylabel("Gene Dropout Rate (Fraction Undetected Cells)", fontsize=10)
    ax_b.set_ylim(-0.02, 1.05)
    ax_b.tick_params(axis='x', rotation=40)
    ax_b.grid(axis='y', linestyle='--', alpha=0.3)
    ax_b.legend(title="", frameon=True, facecolor='white', framealpha=0.9, fontsize=9, loc='lower left')

    # -------------------------------------------------------------
    # Panel C: Distribution of Nuclear / Cytoplasmic Expression Ratios
    # -------------------------------------------------------------
    ax_c = fig.add_subplot(gs[0, 2])
    palette_kde = sns.color_palette("tab10", len(order))
    for i, name in enumerate(order):
        sub_g = df_genes[df_genes['short_name'] == name]
        sns.kdeplot(sub_g['log2_nc_ratio'], ax=ax_c, label=name, linewidth=1.3, alpha=0.8, color=palette_kde[i])
    
    ax_c.axvline(0, color='black', linestyle='--', linewidth=1.2, label='Balanced (N/C=1)')
    ax_c.set_title("C. Nuclear-to-Cytoplasmic Expression Imbalance", fontsize=11, fontweight='bold', loc='left', pad=8)
    ax_c.set_xlabel(r"$\log_2(\text{Nuclear / Cytoplasmic Expression})$", fontsize=10)
    ax_c.set_ylabel("Density across Panel Genes", fontsize=10)
    ax_c.set_xlim(-5.5, 3.5)
    ax_c.grid(linestyle='--', alpha=0.3)
    ax_c.legend(title="Dataset", bbox_to_anchor=(1.02, 1.02), loc='upper left', frameon=False, fontsize=8)

    # -------------------------------------------------------------
    # Panel D: Nuclear R² versus Technical Dropout Gap
    # -------------------------------------------------------------
    ax_d = fig.add_subplot(gs[1, 0])
    x_d = df_master['delta_gene_dropout_nuc_minus_cyt'] * 100
    y_d = df_master['nuc_R2']
    rho_d, p_d = stats.spearmanr(x_d, y_d)
    r_d, pr_d = stats.pearsonr(x_d, y_d)
    
    sns.regplot(x=x_d, y=y_d, ax=ax_d, color=c_nuc, scatter=False,
                line_kws={'linewidth': 1.5, 'linestyle': '-'})
    ax_d.scatter(x_d, y_d, color=c_nuc, s=65, edgecolor='black', linewidth=0.8, zorder=4)
    for _, r in df_master.iterrows():
        ax_d.annotate(r['short_name'], (r['delta_gene_dropout_nuc_minus_cyt'] * 100, r['nuc_R2']),
                      xytext=(5, 4), textcoords='offset points', fontsize=8.5)

    ax_d.set_title(r"D. Nuclear $R^2$ vs Technical Dropout Gap", fontsize=11, fontweight='bold', loc='left', pad=8)
    ax_d.set_xlabel(r"Dropout Excess in Nucleus: $(\text{Dropout}_{\text{nuc}} - \text{Dropout}_{\text{cyt}})\ (\%)$", fontsize=10)
    ax_d.set_ylabel(r"Nuclear Prediction $R^2$", fontsize=10)
    ax_d.grid(linestyle='--', alpha=0.3)
    ax_d.text(0.05, 0.12, f"Spearman $\\rho = {rho_d:.2f}$ ($P = {p_d:.3f}$)\nPearson $r = {r_d:.2f}$ ($P = {pr_d:.3f}$)\n$n = 9$ datasets",
              transform=ax_d.transAxes, fontsize=9,
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='lightgray', alpha=0.9))

    # -------------------------------------------------------------
    # Panel E: Performance Gap (ΔNRMSE) vs Model Gene Coverage Ratio
    # -------------------------------------------------------------
    ax_e = fig.add_subplot(gs[1, 1])
    x_e = df_master['model_gene_coverage_ratio_1pct']
    y_e = df_master['delta_NRMSE']
    rho_e, p_e = stats.spearmanr(x_e, y_e)
    r_e, pr_e = stats.pearsonr(x_e, y_e)
    
    sns.regplot(x=x_e, y=y_e, ax=ax_e, color=c_gap, scatter=False,
                line_kws={'linewidth': 1.5, 'linestyle': '-'})
    ax_e.scatter(x_e, y_e, color=c_gap, s=65, edgecolor='black', linewidth=0.8, zorder=4)
    for _, r in df_master.iterrows():
        ax_e.annotate(r['short_name'], (r['model_gene_coverage_ratio_1pct'], r['delta_NRMSE']),
                      xytext=(5, 4), textcoords='offset points', fontsize=8.5)

    ax_e.set_title("E. Performance Gap vs Model Feature Coverage", fontsize=11, fontweight='bold', loc='left', pad=8)
    ax_e.set_xlabel(r"Model Gene Coverage Ratio (Nuc / Cyt at $\geq$1% cells)", fontsize=10)
    ax_e.set_ylabel(r"Performance Gap $\Delta$NRMSE ($\text{NRMSE}_{\text{nuc}} - \text{NRMSE}_{\text{cyt}}$)", fontsize=10)
    ax_e.axhline(0, color='gray', linestyle=':', linewidth=1)
    ax_e.grid(linestyle='--', alpha=0.3)
    ax_e.text(0.48, 0.84, f"Spearman $\\rho = {rho_e:.2f}$ ($P = {p_e:.3f}$)\nPearson $r = {r_e:.2f}$ ($P = {pr_e:.3f}$)\n$n = 9$ datasets",
              transform=ax_e.transAxes, fontsize=9,
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='lightgray', alpha=0.9))

    # -------------------------------------------------------------
    # Panel F: Stratified Model Performance across Nuclear Detection Strata
    # -------------------------------------------------------------
    ax_f = fig.add_subplot(gs[1, 2])
    # Show Cyt vs Nuc R² across the three detection tertiles
    x_indices = np.arange(len(df_strata))
    width = 0.35
    
    rects1 = ax_f.bar(x_indices - width/2, df_strata['Cyt_R2'], width, label='Cytoplasm', color=c_cyt, edgecolor='black', linewidth=0.7)
    rects2 = ax_f.bar(x_indices + width/2, df_strata['Nuc_R2'], width, label='Nucleus', color=c_nuc, edgecolor='black', linewidth=0.7)

    # Annotate delta R² above each pair
    for i, row in df_strata.iterrows():
        gap = row['Delta_R2']
        h = max(row['Cyt_R2'], row['Nuc_R2']) + 0.03
        symbol = f"$\\Delta R^2 = {gap:+.3f}$"
        if i == 2:
            symbol += "\n(No gap)"
        ax_f.text(i, h, symbol, ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#222222')

    ax_f.set_title("F. Performance Gap Vanishes in High-Detection Strata", fontsize=11, fontweight='bold', loc='left', pad=8)
    ax_f.set_xticks(x_indices)
    ax_f.set_xticklabels(['Low Detection\n(min det < 5%)', 'Medium Detection\n(5% - 8.8%)', 'High Detection\n(min det > 8.8%)'], fontsize=9)
    ax_f.set_ylabel(r"Prediction Accuracy ($R^2$)", fontsize=10)
    ax_f.set_ylim(0, 1.05)
    ax_f.grid(axis='y', linestyle='--', alpha=0.3)
    ax_f.legend(loc='lower right', frameon=True, facecolor='white', framealpha=0.9, fontsize=9)

    # Save outputs
    out_fig = os.path.join(scratch_dir, "supplementary_figure_nuclear_investigation.png")
    out_pdf = os.path.join(scratch_dir, "supplementary_figure_nuclear_investigation.pdf")
    plt.savefig(out_fig, dpi=300, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    plt.close()

    # Also copy to repo figures/
    os.system(f"cp '{out_fig}' figures/supplementary_figure_nuclear_investigation.png")
    os.system(f"cp '{out_pdf}' figures/supplementary_figure_nuclear_investigation.pdf")
    print(f"Generated publication figure: {out_fig} and {out_pdf}")

if __name__ == "__main__":
    make_publication_figure()
