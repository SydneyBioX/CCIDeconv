#!/usr/bin/env python3
"""
nuclear_compartment_analysis.py

Comprehensive investigation of nuclear compartment prediction performance
in CCIDeconv in response to Reviewer Comment 2.

This script performs dedicated, dataset-level correlation analyses across all
n = 9 held-out datasets for the 4 distinct evaluation settings:
1. Spatial LOGO-CV (spatial_logo): Model trained with spatial features (SP procedure; Table 1)
2. Single LOGO-CV (single_logo): Model trained without spatial features (ScP procedure; Supplementary Table 4)
3. Spatial Robustness (spatial_robustness): Robustness across 255 combinations with spatial features (SP procedure; Supplementary Table 3)
4. Single Robustness (single_robustness): Robustness across 255 combinations without spatial features (ScP procedure)

All 4 analyses are computed separately and never pooled together.
Outputs are saved in: supplementary_analysis/results/nuclear_compartment_analysis
"""

import os
import sys
import subprocess
import argparse
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


def get_paths(repo_root=None):
    if repo_root is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(script_dir, "..", ".."))

    server_root = os.path.abspath(os.path.join(repo_root, ".."))
    data_dir = os.path.join(server_root, "data")
    processed_dir = os.path.join(data_dir, "processed")
    datasets_dir = os.path.join(processed_dir, "datasets")
    
    output_dir = os.path.join(
        repo_root, "supplementary_analysis", "results", "nuclear_compartment_analysis"
    )
    os.makedirs(output_dir, exist_ok=True)

    # 1. Spatial LOGO-CV
    spatial_logo_path = os.path.join(
        processed_dir, "plots_metrics", "comparison_baselearners", "xgboost_voting_metrics.csv"
    )
    if not os.path.exists(spatial_logo_path):
        alt_path = os.path.join(processed_dir, "xgboost_voting_metrics.csv")
        if os.path.exists(alt_path):
            spatial_logo_path = alt_path

    # 2. Single LOGO-CV (without spatial)
    single_logo_path = os.path.join(
        processed_dir, "single_cell", "results_P1spatial_removed", "xgboost_voting_metrics_P1_trained.csv"
    )

    # 3 & 4. Master robustness combinations table
    robustness_master_path = os.path.join(
        processed_dir, "metrics_all_combinations_trainings_10032025.csv"
    )

    return {
        "repo_root": repo_root,
        "data_dir": data_dir,
        "processed_dir": processed_dir,
        "datasets_dir": datasets_dir,
        "output_dir": output_dir,
        "spatial_logo_path": spatial_logo_path,
        "single_logo_path": single_logo_path,
        "robustness_master_path": robustness_master_path,
    }


def compute_or_load_predictors(paths, force_recompute=False):
    """
    Computes or loads the dataset-level biological/technical predictors.
    Uses cached CSV if available unless force_recompute is True.
    """
    cached_csv = os.path.join(paths["output_dir"], "single_cell_technical_predictors.csv")
    temp_cache = "/tmp/spe_list_predictors.csv"

    if not force_recompute:
        if os.path.exists(cached_csv):
            print(f"Loading cached predictors from: {cached_csv}")
            df = pd.read_csv(cached_csv)
            if "N_C_ratio" not in df.columns and "N_C_ratio_gt0" in df.columns:
                df["N_C_ratio"] = df["N_C_ratio_gt0"]
            return df
        elif os.path.exists(temp_cache):
            print(f"Loading predictors from temporary cache: {temp_cache}")
            df = pd.read_csv(temp_cache)
            if "N_C_ratio" not in df.columns and "N_C_ratio_gt0" in df.columns:
                df["N_C_ratio"] = df["N_C_ratio_gt0"]
            df.to_csv(cached_csv, index=False)
            return df

    print("Computing dataset-level biological/technical predictors from RDS files using R...")
    r_script = f"""
    suppressPackageStartupMessages(library(SingleCellExperiment))
    suppressPackageStartupMessages(library(Matrix))

    datasets_dir <- '{paths["datasets_dir"]}'

    files <- list(
      spe_list_brainad_cellchat = 'spe_list_brainad.rds',
      spe_list_braingl_cellchat = 'spe_list_braingl.rds',
      spe_list_breast_cancer_cellchat = 'spe_list_breast_cancer.rds',
      spe_list_lung_cellchat = 'spe_list_lung.rds',
      spe_list_lung_high_res_cellchat = 'spe_list_lung_high_res.rds',
      spe_list_lymphnode_cellchat = 'spe_list_lymphnode.rds',
      spe_list_ovarian_cellchat = 'spe_list_ovarian.rds',
      spe_list_panc_cellchat = 'spe_list_panc.rds',
      spe_list_skin_cellchat = 'spe_list_skin.rds'
    )

    res <- data.frame()

    for (nm in names(files)) {{
      f <- file.path(datasets_dir, files[[nm]])
      cat('Processing', nm, 'from', files[[nm]], '...\\n')
      obj_list <- readRDS(f)
      
      c_obj <- obj_list[['cyt']]
      n_obj <- obj_list[['nuc']]
      
      c_cnts <- counts(c_obj)
      n_cnts <- counts(n_obj)
      
      # A. Nuclear mRNA sparsity
      c_det <- colSums(c_cnts > 0)
      n_det <- colSums(n_cnts > 0)
      med_c_det <- median(c_det)
      med_n_det <- median(n_det)
      det_ratio <- med_n_det / med_c_det
      
      # B. Gene zero fraction (technical dropout)
      c_zero <- 1 - (rowSums(c_cnts > 0) / ncol(c_cnts))
      n_zero <- 1 - (rowSums(n_cnts > 0) / ncol(n_cnts))
      med_c_zero <- median(c_zero)
      med_n_zero <- median(n_zero)
      gap <- med_n_zero - med_c_zero
      
      # C. Nuclear-cytoplasmic expression balance (N/C ratio)
      c_sum <- rowSums(c_cnts)
      n_sum <- rowSums(n_cnts)
      valid <- c_sum > 0
      nc_ratio <- median(n_sum[valid] / c_sum[valid])
      
      # D. Panel size 
      panel_size <- nrow(c_cnts)
      
      res <- rbind(res, data.frame(
        Dataset = nm,
        nuclear_detected_genes = med_n_det,
        cytoplasmic_detected_genes = med_c_det,
        nuclear_detection_ratio = det_ratio,
        nuclear_zero_fraction = med_n_zero,
        cytoplasmic_zero_fraction = med_c_zero,
        dropout_gap = gap,
        N_C_ratio = nc_ratio,
        panel_gene_count = panel_size,
        n_cells_cyt = ncol(c_cnts),
        n_cells_nuc = ncol(n_cnts)
      ))
      
      rm(obj_list, c_obj, n_obj, c_cnts, n_cnts, c_det, n_det, c_zero, n_zero, c_sum, n_sum)
      gc(verbose = FALSE)
    }}

    write.csv(res, '{cached_csv}', row.names = FALSE)
    cat('Successfully wrote predictors to: {cached_csv}\\n')
    """
    
    env = os.environ.copy()
    env["R_MAX_VSIZE"] = "100Gb"
    cmd = ["Rscript", "-e", r_script]
    subprocess.run(cmd, env=env, check=True)
    df = pd.read_csv(cached_csv)
    if "N_C_ratio" not in df.columns and "N_C_ratio_gt0" in df.columns:
        df["N_C_ratio"] = df["N_C_ratio_gt0"]
    return df


def load_logo_metrics(csv_path):
    """
    Loads LOGO-CV prediction metrics and computes contrasts:
    ΔR² = cyt_R² - nuc_R²
    ΔNRMSE = nuc_NRMSE - cyt_NRMSE
    """
    df = pd.read_csv(csv_path)
    cols = ["Dataset", "cyt_R²", "cyt_RMSE", "cyt_NRMSE", "nuc_R²", "nuc_RMSE", "nuc_NRMSE"]
    sub = df[cols].copy()
    sub["ΔR²"] = sub["cyt_R²"] - sub["nuc_R²"]
    sub["ΔNRMSE"] = sub["nuc_NRMSE"] - sub["cyt_NRMSE"]
    return sub


def load_robustness_metrics(master_csv_path, condition_name):
    """
    Loads robustness combinations filtered by condition ('with_spatial' or 'without_spatial'),
    aggregates across training combinations using the median for each of the n = 9 held-out datasets (test_name),
    and calculates contrasts:
    ΔR² = cyt_R² - nuc_R²
    ΔNRMSE = nuc_NRMSE - cyt_NRMSE
    """
    df = pd.read_csv(master_csv_path)
    sub = df[df["condition"] == condition_name].copy()
    
    # Aggregate across combinations per held-out dataset
    grouped = sub.groupby("test_name").agg({
        "cyt_R²": "median",
        "cyt_RMSE": "median",
        "cyt_NRMSE": "median",
        "nuc_R²": "median",
        "nuc_RMSE": "median",
        "nuc_NRMSE": "median",
        "condition": "count"
    }).reset_index()

    grouped.rename(columns={"test_name": "Dataset", "condition": "n_combinations"}, inplace=True)
    grouped["ΔR²"] = grouped["cyt_R²"] - grouped["nuc_R²"]
    grouped["ΔNRMSE"] = grouped["nuc_NRMSE"] - grouped["cyt_NRMSE"]
    return grouped


def run_correlations(data_df, metric_mapping, predictor_mapping, analysis_name):
    """
    Computes both Spearman and Pearson correlations for all metric-predictor pairs.
    """
    records = []
    n = len(data_df)

    for metric_col, metric_label in metric_mapping.items():
        for pred_col, pred_label in predictor_mapping.items():
            x = data_df[pred_col].values
            y = data_df[metric_col].values

            # Spearman
            rho, p_spearman = stats.spearmanr(y, x)
            # Pearson
            r, p_pearson = stats.pearsonr(y, x)

            dir_spearman = "positive" if rho > 0 else ("negative" if rho < 0 else "neutral")
            dir_pearson = "positive" if r > 0 else ("negative" if r < 0 else "neutral")

            records.append({
                "analysis": analysis_name,
                "performance_metric": metric_label,
                "predictor": pred_label,
                "method": "Spearman",
                "correlation": float(rho),
                "p_value": float(p_spearman),
                "n": n,
                "direction": dir_spearman
            })
            records.append({
                "analysis": analysis_name,
                "performance_metric": metric_label,
                "predictor": pred_label,
                "method": "Pearson",
                "correlation": float(r),
                "p_value": float(p_pearson),
                "n": n,
                "direction": dir_pearson
            })

    return pd.DataFrame(records)


def plot_single_heatmap(corr_df, title, output_png, output_pdf, method="Spearman"):
    """
    Plots a single publication-ready heatmap.
    """
    sub_df = corr_df[corr_df["method"] == method].copy()

    row_order = ["Nuclear R²", "Nuclear NRMSE", "ΔR²", "ΔNRMSE"]
    col_order = [
        "Nuclear detected genes",
        "Nuclear zero-count fraction",
        "N/C expression ratio",
        "Nuclear–cytoplasmic dropout gap",
        "Nuclear detection ratio",
        "Total panel gene count"
    ]

    corr_matrix = sub_df.pivot(index="performance_metric", columns="predictor", values="correlation").reindex(index=row_order, columns=col_order)
    pval_matrix = sub_df.pivot(index="performance_metric", columns="predictor", values="p_value").reindex(index=row_order, columns=col_order)

    annot_matrix = np.empty(corr_matrix.shape, dtype=object)
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            val = corr_matrix.iloc[i, j]
            pval = pval_matrix.iloc[i, j]
            annot_matrix[i, j] = f"{val:+.2f}\n(p={pval:.2f})" if not np.isnan(val) else "N/A"

    plt.figure(figsize=(10.5, 5.2), dpi=300)
    sns.set_theme(style="white", font_scale=0.95)

    ax = sns.heatmap(
        corr_matrix,
        annot=annot_matrix,
        fmt="",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        center=0.0,
        cbar_kws={
            "label": f"{method} Correlation Coefficient ({'ρ' if method == 'Spearman' else 'r'})",
            "shrink": 0.85,
            "ticks": [-1.0, -0.5, 0.0, 0.5, 1.0]
        },
        linewidths=1.2,
        linecolor="white",
        square=False
    )

    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    ax.set_xlabel("Dataset-Level Biological / Technical Predictor", fontsize=11, fontweight="bold", labelpad=10)
    ax.set_ylabel("Prediction Performance Metric", fontsize=11, fontweight="bold", labelpad=10)

    wrapped_cols = [
        "Nuclear\ndetected genes",
        "Nuclear\nzero-count fraction",
        "N/C expression\nratio",
        "Nuclear–cytoplasmic\ndropout gap",
        "Nuclear\ndetection ratio",
        "Total panel\ngene count"
    ]
    ax.set_xticklabels(wrapped_cols, rotation=0, ha="center", fontsize=9.5)
    ax.set_yticklabels(row_order, rotation=0, fontsize=10, fontweight="medium")

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_pdf, bbox_inches="tight")
    plt.close()


def plot_combined_4panel_heatmaps(dfs_dict, output_png, output_pdf, method="Spearman"):
    """
    Plots a comprehensive 4-panel figure comparing all 4 datasets:
    A. Spatial LOGO-CV
    B. Single LOGO-CV
    C. Spatial Robustness
    D. Single Robustness
    """
    row_order = ["Nuclear R²", "Nuclear NRMSE", "ΔR²", "ΔNRMSE"]
    col_order = [
        "Nuclear detected genes",
        "Nuclear zero-count fraction",
        "N/C expression ratio",
        "Nuclear–cytoplasmic dropout gap",
        "Nuclear detection ratio",
        "Total panel gene count"
    ]
    wrapped_cols = [
        "Nuclear\ndetected genes",
        "Nuclear\nzero-count fraction",
        "N/C expression\nratio",
        "Nuclear–cytoplasmic\ndropout gap",
        "Nuclear\ndetection ratio",
        "Total panel\ngene count"
    ]

    def get_matrices(df):
        sub = df[df["method"] == method].copy()
        c = sub.pivot(index="performance_metric", columns="predictor", values="correlation").reindex(index=row_order, columns=col_order)
        p = sub.pivot(index="performance_metric", columns="predictor", values="p_value").reindex(index=row_order, columns=col_order)
        annot = np.empty(c.shape, dtype=object)
        for i in range(c.shape[0]):
            for j in range(c.shape[1]):
                val = c.iloc[i, j]
                pval = p.iloc[i, j]
                annot[i, j] = f"{val:+.2f}\n(p={pval:.2f})"
        return c, annot

    fig, axes = plt.subplots(4, 1, figsize=(11.0, 18.0), dpi=300)
    sns.set_theme(style="white", font_scale=0.95)

    panels = [
        ("spatial_logo", "A. Spatial LOGO-CV (Table 1; SP procedure)", axes[0], "Spatial LOGO-CV"),
        ("single_logo", "B. Single LOGO-CV (Supp. Table 4; ScP procedure)", axes[1], "Single LOGO-CV"),
        ("spatial_robustness", "C. Spatial Robustness (Supp. Table 3; SP procedure; median-aggregated)", axes[2], "Spatial Robustness"),
        ("single_robustness", "D. Single Robustness (ScP procedure; median-aggregated)", axes[3], "Single Robustness")
    ]

    for key, title, ax, y_lbl in panels:
        c, a = get_matrices(dfs_dict[key])
        sns.heatmap(
            c,
            annot=a,
            fmt="",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            center=0.0,
            cbar=True,
            cbar_kws={"label": f"{method} ρ", "shrink": 0.85, "ticks": [-1.0, -0.5, 0.0, 0.5, 1.0]},
            linewidths=1.2,
            linecolor="white",
            ax=ax
        )
        ax.set_title(title, fontsize=12.5, fontweight="bold", loc="left", pad=10)
        ax.set_ylabel(y_lbl, fontsize=10.0, fontweight="bold")
        ax.set_xticklabels(wrapped_cols, rotation=0, ha="center", fontsize=9.5)
        ax.set_yticklabels(row_order, rotation=0, fontsize=10)
        ax.set_xlabel("")

    axes[3].set_xlabel("Dataset-Level Biological / Technical Predictor", fontsize=11, fontweight="bold", labelpad=10)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.savefig(output_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved 4-panel heatmap to: {output_png} and {output_pdf}")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive investigation of nuclear prediction performance in CCIDeconv across 4 evaluation settings."
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="Force recomputation of single-cell technical predictors from RDS files via R."
    )
    args = parser.parse_args()

    paths = get_paths()
    print("=== CCIDeconv Nuclear Compartment Performance Analysis (4 Settings) ===")
    print(f"Output directory: {paths['output_dir']}")

    # 1. Load / compute predictors
    predictors_df = compute_or_load_predictors(paths, force_recompute=args.force_recompute)
    print("\nDataset Technical Predictors (n = 9):")
    print(predictors_df[["Dataset", "panel_gene_count", "nuclear_detected_genes", "cytoplasmic_detected_genes", "nuclear_detection_ratio", "dropout_gap", "N_C_ratio"]])

    # 2. Load the 4 evaluation settings
    print("\nLoading 4 performance datasets:")
    # Setting 1: Spatial LOGO-CV
    print("  1. Spatial LOGO-CV from:", paths["spatial_logo_path"])
    df_spatial_logo = load_logo_metrics(paths["spatial_logo_path"])

    # Setting 2: Single LOGO-CV
    print("  2. Single LOGO-CV from:", paths["single_logo_path"])
    df_single_logo = load_logo_metrics(paths["single_logo_path"])

    # Setting 3: Spatial Robustness
    print("  3. Spatial Robustness from master table (condition == 'with_spatial')")
    df_spatial_robustness = load_robustness_metrics(paths["robustness_master_path"], condition_name="with_spatial")

    # Setting 4: Single Robustness
    print("  4. Single Robustness from master table (condition == 'without_spatial')")
    df_single_robustness = load_robustness_metrics(paths["robustness_master_path"], condition_name="without_spatial")

    # 3. Create dataset-level tables
    dataset_cols = [
        "Dataset",
        "nuclear_detected_genes",
        "cytoplasmic_detected_genes",
        "nuclear_detection_ratio",
        "nuclear_zero_fraction",
        "cytoplasmic_zero_fraction",
        "dropout_gap",
        "N_C_ratio",
        "panel_gene_count"
    ]
    perf_cols = [
        "Dataset",
        "cyt_R²",
        "nuc_R²",
        "ΔR²",
        "cyt_NRMSE",
        "nuc_NRMSE",
        "ΔNRMSE"
    ]

    table_data = {
        "spatial_logo": pd.merge(predictors_df[dataset_cols], df_spatial_logo[perf_cols], on="Dataset"),
        "single_logo": pd.merge(predictors_df[dataset_cols], df_single_logo[perf_cols], on="Dataset"),
        "spatial_robustness": pd.merge(predictors_df[dataset_cols], df_spatial_robustness[perf_cols], on="Dataset"),
        "single_robustness": pd.merge(predictors_df[dataset_cols], df_single_robustness[perf_cols], on="Dataset"),
    }

    # Save individual dataset-level tables
    for key, df_table in table_data.items():
        out_p = os.path.join(paths["output_dir"], f"dataset_level_{key}.csv")
        df_table.to_csv(out_p, index=False)
        print(f"Saved dataset table for {key} to: {out_p}")

    # Aliases for convenience / original prompt specifications
    table_data["spatial_logo"].to_csv(os.path.join(paths["output_dir"], "dataset_level_predictors_and_performance.csv"), index=False)
    table_data["single_logo"].to_csv(os.path.join(paths["output_dir"], "dataset_level_without_spatial_performance.csv"), index=False)
    table_data["spatial_robustness"].to_csv(os.path.join(paths["output_dir"], "dataset_level_robustness_performance.csv"), index=False)

    # 4. Correlation mappings
    metric_mapping = {
        "nuc_R²": "Nuclear R²",
        "nuc_NRMSE": "Nuclear NRMSE",
        "ΔR²": "ΔR²",
        "ΔNRMSE": "ΔNRMSE"
    }
    predictor_mapping = {
        "nuclear_detected_genes": "Nuclear detected genes",
        "nuclear_zero_fraction": "Nuclear zero-count fraction",
        "N_C_ratio": "N/C expression ratio",
        "dropout_gap": "Nuclear–cytoplasmic dropout gap",
        "nuclear_detection_ratio": "Nuclear detection ratio",
        "panel_gene_count": "Total panel gene count"
    }

    # 5. Run correlations for all 4 settings
    corr_dfs = {}
    for key, df_table in table_data.items():
        corr_dfs[key] = run_correlations(df_table, metric_mapping, predictor_mapping, analysis_name=key)

    # Combined correlation results table
    combined_corr_df = pd.concat(list(corr_dfs.values()), ignore_index=True)
    corr_csv_path = os.path.join(paths["output_dir"], "correlation_analysis_results.csv")
    combined_corr_df.to_csv(corr_csv_path, index=False)
    print(f"\nSaved master correlation results (all 4 settings) to: {corr_csv_path}")

    # 6. Generate heatmaps
    for key, c_df in corr_dfs.items():
        title_map = {
            "spatial_logo": "Spatial LOGO-CV (Table 1; SP): Predictors vs Prediction Performance (n = 9)",
            "single_logo": "Single LOGO-CV (Supp. Table 4; ScP): Predictors vs Prediction Performance (n = 9)",
            "spatial_robustness": "Spatial Robustness (Supp. Table 3; SP): Predictors vs Performance (n = 9)",
            "single_robustness": "Single Robustness (ScP): Predictors vs Performance (n = 9)"
        }
        png_p = os.path.join(paths["output_dir"], f"heatmap_{key}.png")
        pdf_p = os.path.join(paths["output_dir"], f"heatmap_{key}.pdf")
        plot_single_heatmap(c_df, title_map[key], png_p, pdf_p, method="Spearman")

    # Also save primary and robustness standard names
    plot_single_heatmap(corr_dfs["spatial_logo"], "Primary LOGO-CV With Spatial Features (n = 9)",
                        os.path.join(paths["output_dir"], "primary_correlation_heatmap.png"),
                        os.path.join(paths["output_dir"], "primary_correlation_heatmap.pdf"))
    plot_single_heatmap(corr_dfs["spatial_robustness"], "Robustness Analysis With Spatial Features (n = 9)",
                        os.path.join(paths["output_dir"], "robustness_correlation_heatmap.png"),
                        os.path.join(paths["output_dir"], "robustness_correlation_heatmap.pdf"))

    # Master 4-panel publication figure
    plot_combined_4panel_heatmaps(
        corr_dfs,
        output_png=os.path.join(paths["output_dir"], "combined_correlation_heatmap.png"),
        output_pdf=os.path.join(paths["output_dir"], "combined_correlation_heatmap.pdf"),
        method="Spearman"
    )

    print("\n=== Analysis Complete ===")
    print(f"All files generated in: {paths['output_dir']}")


if __name__ == "__main__":
    main()
