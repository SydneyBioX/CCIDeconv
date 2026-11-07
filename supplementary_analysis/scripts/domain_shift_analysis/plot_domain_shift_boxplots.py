# supplementary_analysis/scripts/domain_shift_analysis/plot_domain_shift_boxplots.py
"""
plot_domain_shift_boxplots.py

Generates publication-quality boxplots of evaluation metrics from domain_shift_dataset_level.csv.
Plots:
1. Unified Boxplot: Metrics on the x-axis, values on the y-axis, stratified by Quantile-Matching status.
2. Two-Panel Boxplot: Split into Goodness-of-Fit (R²) and Normalized Error (NRMSE) for clean scaling.
3. Shift-Condition Faceted Boxplot: Detailed performance breakdown across specific shift types.
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "supplementary_analysis", "results", "domain_shift_analysis")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

CSV_PATH = os.path.join(RESULTS_DIR, "domain_shift_dataset_level.csv")


def prepare_tidy_data(df, shifted_y_filter=False):
    """
    Filter and melt dataframe so metrics are in a single column for x-axis boxplots.
    """
    if shifted_y_filter is not None:
        sub_df = df[df['shifted_y'] == shifted_y_filter].copy()
    else:
        sub_df = df.copy()

    # Map column names to clean publication-ready labels
    metric_mapping = {
        'cyt_R2': 'Cytoplasm R²',
        'nuc_R2': 'Nuclear R²',
        'avg_R2': 'Average R²',
        'cyt_NRMSE': 'Cytoplasm NRMSE',
        'nuc_NRMSE': 'Nuclear NRMSE',
        'avg_NRMSE': 'Average NRMSE'
    }

    sub_df['Mitigation'] = sub_df['matched_x'].map({
        False: 'Unmitigated (Raw Shift)',
        True: 'Quantile-Matched (qmatch)'
    })

    tidy_df = sub_df.melt(
        id_vars=['dataset', 'shift_name', 'rep', 'matched_x', 'shifted_y', 'Mitigation'],
        value_vars=list(metric_mapping.keys()),
        var_name='metric_raw',
        value_name='Value'
    )
    tidy_df['Metric'] = tidy_df['metric_raw'].map(metric_mapping)
    tidy_df['Metric_Type'] = tidy_df['metric_raw'].apply(lambda x: 'R² (Goodness of Fit)' if 'R2' in x else 'NRMSE (Normalized Error)')
    return tidy_df


def set_publication_style():
    sns.set_theme(style="ticks")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 14
    })


def plot_unified_boxplot(tidy_df):
    """
    Plot with all metrics along the x-axis and their values on the y-axis.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=(12, 6.5))

    palette = {'Unmitigated (Raw Shift)': '#E74C3C', 'Quantile-Matched (qmatch)': '#2ECC71'}

    # Draw Boxplot
    sns.boxplot(
        data=tidy_df,
        x='Metric',
        y='Value',
        hue='Mitigation',
        palette=palette,
        boxprops=dict(alpha=0.75, edgecolor='black', linewidth=1.2),
        medianprops=dict(color='black', linewidth=1.8),
        whiskerprops=dict(color='black', linewidth=1.2),
        capprops=dict(color='black', linewidth=1.2),
        showfliers=False,
        ax=ax
    )

    # Overlay jittered data points
    sns.stripplot(
        data=tidy_df,
        x='Metric',
        y='Value',
        hue='Mitigation',
        dodge=True,
        alpha=0.25,
        jitter=0.2,
        size=4,
        palette=palette,
        edgecolor='black',
        linewidth=0.4,
        ax=ax
    )

    # Fix duplicate legend from stripplot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2], title="Mitigation Regime", frameon=True, loc="upper right")

    ax.set_title("CCIDeconv Domain Shift Evaluation: Performance Metrics Distribution Across 9 Spatial Datasets", pad=15, fontweight='bold')
    ax.set_xlabel("Evaluation Metric", fontweight='bold', labelpad=10)
    ax.set_ylabel("Metric Value", fontweight='bold', labelpad=10)
    ax.axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    ax.grid(axis='y', linestyle=':', alpha=0.6)

    plt.tight_layout()
    png_path = os.path.join(FIGURES_DIR, "domain_shift_metrics_unified_boxplot.png")
    pdf_path = os.path.join(FIGURES_DIR, "domain_shift_metrics_unified_boxplot.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved unified boxplot to: {png_path}")
    return png_path


def plot_two_panel_boxplot(tidy_df):
    """
    Split into 2 panels: R² (higher is better) and NRMSE (lower is better).
    """
    set_publication_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    palette = {'Unmitigated (Raw Shift)': '#E74C3C', 'Quantile-Matched (qmatch)': '#2ECC71'}

    # Panel 1: R² Metrics
    r2_df = tidy_df[tidy_df['Metric_Type'] == 'R² (Goodness of Fit)']
    sns.boxplot(
        data=r2_df,
        x='Metric',
        y='Value',
        hue='Mitigation',
        palette=palette,
        boxprops=dict(alpha=0.8, edgecolor='black', linewidth=1.2),
        medianprops=dict(color='black', linewidth=1.8),
        showfliers=False,
        ax=axes[0]
    )
    sns.stripplot(
        data=r2_df,
        x='Metric',
        y='Value',
        hue='Mitigation',
        dodge=True,
        alpha=0.3,
        jitter=0.2,
        size=4,
        palette=palette,
        edgecolor='black',
        linewidth=0.3,
        ax=axes[0]
    )
    axes[0].set_title("A. Goodness-of-Fit Metrics (R²: Higher is Better)", fontweight='bold', pad=12)
    axes[0].set_xlabel("R² Metric", fontweight='bold')
    axes[0].set_ylabel("Coefficient of Determination (R²)", fontweight='bold')
    axes[0].set_ylim(-0.4, 1.05)
    axes[0].axhline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
    axes[0].grid(axis='y', linestyle=':', alpha=0.6)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles[:2], labels[:2], title="Mitigation", loc="lower left", frameon=True)

    # Panel 2: NRMSE Metrics
    nrmse_df = tidy_df[tidy_df['Metric_Type'] == 'NRMSE (Normalized Error)']
    sns.boxplot(
        data=nrmse_df,
        x='Metric',
        y='Value',
        hue='Mitigation',
        palette=palette,
        boxprops=dict(alpha=0.8, edgecolor='black', linewidth=1.2),
        medianprops=dict(color='black', linewidth=1.8),
        showfliers=False,
        ax=axes[1]
    )
    sns.stripplot(
        data=nrmse_df,
        x='Metric',
        y='Value',
        hue='Mitigation',
        dodge=True,
        alpha=0.3,
        jitter=0.2,
        size=4,
        palette=palette,
        edgecolor='black',
        linewidth=0.3,
        ax=axes[1]
    )
    axes[1].set_title("B. Normalized Error Metrics (NRMSE: Lower is Better)", fontweight='bold', pad=12)
    axes[1].set_xlabel("NRMSE Metric", fontweight='bold')
    axes[1].set_ylabel("Normalized RMSE (RMSE / Std)", fontweight='bold')
    axes[1].set_ylim(0.0, 1.35)
    axes[1].grid(axis='y', linestyle=':', alpha=0.6)
    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(handles[:2], labels[:2], title="Mitigation", loc="upper left", frameon=True)

    plt.tight_layout()
    png_path = os.path.join(FIGURES_DIR, "domain_shift_metrics_two_panel_boxplot.png")
    pdf_path = os.path.join(FIGURES_DIR, "domain_shift_metrics_two_panel_boxplot.pdf")
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    print(f"Saved two-panel boxplot to: {png_path}")
    return png_path


def main():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Cannot find {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} records from {CSV_PATH}")

    # Standard realistic biological evaluation (shifted_y == False)
    tidy_df = prepare_tidy_data(df, shifted_y_filter=False)

    plot_unified_boxplot(tidy_df)
    plot_two_panel_boxplot(tidy_df)


if __name__ == "__main__":
    main()
