#!/usr/bin/env python3
"""
supplementary_analysis/scripts/overlap_test/run_quantitative_overlap.py

Comprehensive Quantitative Overlap and Enrichment Analysis for CCIDeconv across ALL THREE Contexts:
1. SP (Spatial Procedure LOGO-CV, 9 datasets)
2. ScP (Single-cell Procedure LOGO-CV without spatial coordinates, 9 datasets)
3. Real Single-Cell Data (scRNA-seq non-small cell lung cancer)

Statistical Protocol:
- Permutations: 10,000 iterations per test.
- Dataset-stratified sampling within held-out datasets to preserve candidate counts and cell-type cluster structures.
- Strict exclusion: 146,726 HPA records removed; CellChatDB excluded.
- Evaluates both:
  * Strict Tier 1 Physical Localization (direct experimental assay)
  * Total Biological Support (Tier 1 Exp + Tier 2 Mechanistic/Pathway)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    df['ligand_tier1'] = df['Ligand Experimental Localization'].str.strip().str.upper() == 'YES'
    df['receptor_tier1'] = df['Receptor Experimental Localization'].str.strip().str.upper() == 'YES'
    df['is_tier1_hit'] = df['ligand_tier1'] | df['receptor_tier1']
    df['is_total_supported'] = df['Validation Classification'] != 'No Independent Evidence'

    df_cyt = df[df['Predicted Compartment'] == 'Cytoplasm'].copy().reset_index(drop=True)
    df_nuc = df[df['Predicted Compartment'] == 'Nucleus'].copy().reset_index(drop=True)
    return df, df_cyt, df_nuc


def run_permutation_test(comp_df, hit_column='is_tier1_hit', top_n=20, n_permutations=10000, random_seed=42):
    np.random.seed(random_seed)

    datasets = comp_df['dataset_id'].unique()
    observed_mask = comp_df['Rank'] <= top_n
    observed_hits = comp_df.loc[observed_mask, hit_column].sum()
    n_observed = observed_mask.sum()
    observed_prop = observed_hits / n_observed if n_observed > 0 else 0.0

    dataset_pools = {}
    for d in datasets:
        sub = comp_df[comp_df['dataset_id'] == d]
        arr = sub[hit_column].values.astype(int)
        k_d = min(top_n, len(arr))
        dataset_pools[d] = (arr, k_d)

    null_hits = np.empty(n_permutations, dtype=int)
    for i in range(n_permutations):
        tot = 0
        for d, (arr, k_d) in dataset_pools.items():
            sampled = np.random.choice(arr, size=k_d, replace=False)
            tot += sampled.sum()
        null_hits[i] = tot

    null_props = null_hits / n_observed
    null_mean = float(np.mean(null_hits))
    null_mean_prop = float(np.mean(null_props))
    null_std = float(np.std(null_hits))
    pi_low = float(np.percentile(null_hits, 2.5))
    pi_high = float(np.percentile(null_hits, 97.5))

    fold_enrichment = observed_prop / null_mean_prop if null_mean_prop > 0 else np.nan
    empirical_p = (np.sum(null_hits >= observed_hits) + 1.0) / (n_permutations + 1.0)

    return {
        'n_predictions': n_observed,
        'observed_hits': int(observed_hits),
        'observed_prop': observed_prop,
        'null_mean_hits': null_mean,
        'null_mean_prop': null_mean_prop,
        'null_std': null_std,
        'pi_hits_low': pi_low,
        'pi_hits_high': pi_high,
        'fold_enrichment': fold_enrichment,
        'empirical_p_value': empirical_p,
        'null_distribution': null_hits
    }


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, "../../.."))

    sp_path = os.path.join(base_dir, "supplementary_analysis/results/validation/logo_validation_all_lr_pairs.csv")
    scp_path = os.path.join(base_dir, "supplementary_analysis/results/validation/scp_validation_all_lr_pairs.csv")
    sc_path = os.path.join(base_dir, "supplementary_analysis/results/validation/single_cell_validation_all_lr_pairs.csv")

    out_dir = os.path.join(base_dir, "supplementary_analysis/results/overlap_test")
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 85)
    print("LOADING THREE VALIDATION DATASETS (SP, ScP, REAL SINGLE-CELL)")
    print("=" * 85)
    sp_df, sp_cyt, sp_nuc = load_dataset(sp_path)
    scp_df, scp_cyt, scp_nuc = load_dataset(scp_path)
    sc_df, sc_cyt, sc_nuc = load_dataset(sc_path)

    print(f"1. SP (Spatial LOGO-CV):           {len(sp_df):,} total rows (Cyt: {len(sp_cyt):,}, Nuc: {len(sp_nuc):,})")
    print(f"2. ScP (Single-cell Proc LOGO-CV): {len(scp_df):,} total rows (Cyt: {len(scp_cyt):,}, Nuc: {len(scp_nuc):,})")
    print(f"3. Real Single-Cell Data (scRNA):  {len(sc_df):,} total rows (Cyt: {len(sc_cyt):,}, Nuc: {len(sc_nuc):,})")

    tests = [
        # --- SP (Spatial LOGO-CV) ---
        ('SP (Spatial LOGO-CV)', 'Top 20 per dataset', 'Cytoplasm', sp_cyt, 'is_tier1_hit', 'Tier 1 Experimental Localization'),
        ('SP (Spatial LOGO-CV)', 'Top 20 per dataset', 'Nucleus', sp_nuc, 'is_tier1_hit', 'Tier 1 Experimental Localization'),
        ('SP (Spatial LOGO-CV)', 'Top 20 per dataset', 'Nucleus', sp_nuc, 'is_total_supported', 'Total Biological Support (Tier 1 + 2)'),
        ('SP (Spatial LOGO-CV)', 'Top 20 per dataset', 'Cytoplasm', sp_cyt, 'is_total_supported', 'Total Biological Support (Tier 1 + 2)'),
        ('SP (Spatial LOGO-CV)', 'Top 10 per dataset', 'Cytoplasm', sp_cyt, 'is_tier1_hit', 'Tier 1 Experimental Localization'),
        ('SP (Spatial LOGO-CV)', 'Top 10 per dataset', 'Nucleus', sp_nuc, 'is_total_supported', 'Total Biological Support (Tier 1 + 2)'),

        # --- ScP (Single-cell Procedure LOGO-CV) ---
        ('ScP (Single-cell Proc LOGO-CV)', 'Top 20 per dataset', 'Cytoplasm', scp_cyt, 'is_tier1_hit', 'Tier 1 Experimental Localization'),
        ('ScP (Single-cell Proc LOGO-CV)', 'Top 20 per dataset', 'Nucleus', scp_nuc, 'is_tier1_hit', 'Tier 1 Experimental Localization'),
        ('ScP (Single-cell Proc LOGO-CV)', 'Top 20 per dataset', 'Nucleus', scp_nuc, 'is_total_supported', 'Total Biological Support (Tier 1 + 2)'),
        ('ScP (Single-cell Proc LOGO-CV)', 'Top 20 per dataset', 'Cytoplasm', scp_cyt, 'is_total_supported', 'Total Biological Support (Tier 1 + 2)'),
        ('ScP (Single-cell Proc LOGO-CV)', 'Top 10 per dataset', 'Cytoplasm', scp_cyt, 'is_tier1_hit', 'Tier 1 Experimental Localization'),
        ('ScP (Single-cell Proc LOGO-CV)', 'Top 10 per dataset', 'Nucleus', scp_nuc, 'is_total_supported', 'Total Biological Support (Tier 1 + 2)'),

        # --- Real Single-Cell Data (scRNA-seq Lung) ---
        ('Real Single-Cell Data (scRNA-seq)', 'Top 20', 'Cytoplasm', sc_cyt, 'is_tier1_hit', 'Tier 1 Experimental Localization'),
        ('Real Single-Cell Data (scRNA-seq)', 'Top 20', 'Nucleus', sc_nuc, 'is_tier1_hit', 'Tier 1 Experimental Localization'),
        ('Real Single-Cell Data (scRNA-seq)', 'Top 20', 'Nucleus', sc_nuc, 'is_total_supported', 'Total Biological Support (Tier 1 + 2)'),
        ('Real Single-Cell Data (scRNA-seq)', 'Top 20', 'Cytoplasm', sc_cyt, 'is_total_supported', 'Total Biological Support (Tier 1 + 2)')
    ]

    summary_rows = []
    plot_dict = {}

    print("\nRunning 10,000 Dataset-Stratified Permutations for all contexts...")
    for context, pset, comp, df_in, hit_col, desc in tests:
        top_n = 10 if 'Top 10' in pset else 20
        res = run_permutation_test(df_in, hit_column=hit_col, top_n=top_n, n_permutations=10000)

        summary_rows.append({
            'Benchmark Context': context,
            'Prediction Set': pset,
            'Target Compartment': comp,
            'Evidence Scope': desc,
            'N Predictions': res['n_predictions'],
            'Observed Hits (N)': res['observed_hits'],
            'Observed Overlap (%)': round(res['observed_prop'] * 100.0, 2),
            'Null Mean Hits': round(res['null_mean_hits'], 2),
            'Expected Overlap (%)': round(res['null_mean_prop'] * 100.0, 2),
            '95% Permutation Interval': f"[{res['pi_hits_low']:.0f}, {res['pi_hits_high']:.0f}]",
            'Fold Enrichment': round(res['fold_enrichment'], 3),
            'Empirical P-value': f"{res['empirical_p_value']:.4e}" if res['empirical_p_value'] < 1e-3 else f"{res['empirical_p_value']:.4f}"
        })

        tag = f"{context}_{comp}_{hit_col}_{pset}"
        plot_dict[tag] = (context, comp, desc, pset, res)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "overlap_enrichment_summary_table.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved complete three-way overlap summary table to:\n  • {summary_path}")

    # Plot 3x2 Grid for the Key Benchmarks
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    panels = [
        (axes[0, 0], 'SP (Spatial LOGO-CV)_Cytoplasm_is_tier1_hit_Top 20 per dataset', '#4A90E2'),
        (axes[0, 1], 'SP (Spatial LOGO-CV)_Nucleus_is_total_supported_Top 20 per dataset', '#BD10E0'),
        (axes[1, 0], 'ScP (Single-cell Proc LOGO-CV)_Cytoplasm_is_tier1_hit_Top 20 per dataset', '#50E3C2'),
        (axes[1, 1], 'ScP (Single-cell Proc LOGO-CV)_Nucleus_is_total_supported_Top 20 per dataset', '#9013FE'),
        (axes[2, 0], 'Real Single-Cell Data (scRNA-seq)_Cytoplasm_is_tier1_hit_Top 20', '#F5A623'),
        (axes[2, 1], 'Real Single-Cell Data (scRNA-seq)_Nucleus_is_total_supported_Top 20', '#E91E63'),
    ]

    for ax, tag, color in panels:
        context, comp, desc, pset, res = plot_dict[tag]
        sns.histplot(res['null_distribution'], bins=25, ax=ax, color=color, kde=True, stat="density", alpha=0.6)
        ax.axvline(res['observed_hits'], color="#D0021B", linestyle="--", linewidth=2.5,
                   label=f"Observed: {res['observed_hits']}/{res['n_predictions']} ({res['observed_prop']*100:.1f}%)")
        ax.axvline(res['null_mean_hits'], color="#333333", linestyle=":", linewidth=2,
                   label=f"Null Mean: {res['null_mean_hits']:.1f} ({res['null_mean_prop']*100:.1f}%)")
        ax.axvspan(res['pi_hits_low'], res['pi_hits_high'], color=color, alpha=0.15, label="95% Permutation Int.")
        p_str = f"{res['empirical_p_value']:.4e}" if res['empirical_p_value'] < 1e-3 else f"{res['empirical_p_value']:.4f}"
        ax.set_title(f"{context} — {comp}\n{desc} ({pset})\nFold = {res['fold_enrichment']:.2f}x, P = {p_str}",
                     fontsize=10, fontweight="bold", pad=6)
        ax.set_xlabel("Number of Supported Predictions", fontsize=9)
        ax.set_ylabel("Permutation Density", fontsize=9)
        ax.legend(loc="upper left", frameon=True, fontsize=8)

    plt.tight_layout()
    png_path = os.path.join(fig_dir, "permutation_null_distributions_three_way.png")
    pdf_path = os.path.join(fig_dir, "permutation_null_distributions_three_way.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Saved three-way publication figures to:\n  • {png_path}\n  • {pdf_path}")

    print("\n" + "=" * 90)
    print("THREE-WAY QUANTITATIVE OVERLAP RESULTS:")
    print("=" * 90)
    print(summary_df[['Benchmark Context', 'Target Compartment', 'Prediction Set', 'Evidence Scope', 'Observed Overlap (%)', 'Expected Overlap (%)', 'Fold Enrichment', 'Empirical P-value']].to_string(index=False))


if __name__ == "__main__":
    main()
