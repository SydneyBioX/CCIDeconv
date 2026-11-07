#!/usr/bin/env python3
"""
supplementary_analysis/scripts/validation/run_three_way_validation.py

Complete Independent External Biological Validation across ALL THREE Contexts:
1. SP (Spatial Procedure): 9 spatial transcriptomics datasets with spatial features (LOGO-CV)
2. ScP (Single-cell Procedure): 9 datasets trained without spatial features (LOGO-CV)
3. Real Single-Cell Data: External scRNA-seq non-small cell lung cancer test dataset

Outputs:
- logo_validation_all_lr_pairs.csv (SP)
- scp_validation_all_lr_pairs.csv (ScP)
- single_cell_validation_all_lr_pairs.csv (Real Single-Cell)
- validation_summary_three_way.csv (SP vs. ScP vs. Real Single-Cell)
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np

# Ensure validation script directory is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from external_validator import ExternalValidator
from run_full_validation_all_pairs import (
    load_lr_mapping,
    build_single_cell_dataframe,
    build_logo_dataframe,
    evaluate_dataframe,
    DATASET_LABELS
)


def compute_three_way_summary(sp_df, scp_df, sc_df):
    summary_rows = []
    datasets = [
        ('SP (Spatial LOGO-CV)', sp_df),
        ('ScP (Single-cell Procedure LOGO-CV)', scp_df),
        ('Real Single-Cell Data (scRNA-seq Lung)', sc_df)
    ]

    for ds_label, df in datasets:
        for subset_name, mask in [('Top 20', df['Is Top 20'] == 'Yes'), ('All', slice(None))]:
            sub_df = df[mask] if isinstance(mask, pd.Series) else df.copy()

            for comp in ['Cytoplasm', 'Nucleus', 'Combined']:
                comp_df = sub_df if comp == 'Combined' else sub_df[sub_df['Predicted Compartment'] == comp]
                n_tot = len(comp_df)
                if n_tot == 0:
                    continue
                n_t1 = (comp_df['Validation Classification'] == 'Independent Experimental Localization').sum()
                n_t2 = (comp_df['Validation Classification'] == 'Independent Biological / Mechanistic Support').sum()
                n_none = (comp_df['Validation Classification'] == 'No Independent Evidence').sum()
                n_supp = n_t1 + n_t2

                pct_t1 = (n_t1 / n_tot) * 100.0
                pct_t2 = (n_t2 / n_tot) * 100.0
                pct_none = (n_none / n_tot) * 100.0
                pct_supp = (n_supp / n_tot) * 100.0

                summary_rows.append({
                    'Benchmark Context': ds_label,
                    'Evaluation Subset': subset_name,
                    'Predicted Compartment': comp,
                    'Total Predictions (N)': n_tot,
                    'Tier 1 (Experimental) N': n_t1,
                    'Tier 1 (%)': round(pct_t1, 1),
                    'Tier 2 (Mechanistic) N': n_t2,
                    'Tier 2 (%)': round(pct_t2, 1),
                    'No Independent Evidence N': n_none,
                    'No Independent Evidence (%)': round(pct_none, 1),
                    'Total Supported N': n_supp,
                    'Total Supported Fraction': f"{n_supp}/{n_tot}",
                    'Total Supported (%)': round(pct_supp, 1)
                })

    return pd.DataFrame(summary_rows)


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(script_dir, "../../.."))
    cache_dir = os.path.join(base_dir, "supplementary_analysis/results/validation/cache")
    comp_path = os.path.join(base_dir, "data/references/human_compartment_knowledge.tsv")
    out_dir = os.path.join(base_dir, "supplementary_analysis/results/validation")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 90)
    print("INDEPENDENT EXTERNAL VALIDATION: THREE-WAY EVALUATION (SP, ScP, REAL SINGLE-CELL)")
    print("=" * 90)

    validator = ExternalValidator(cache_dir=cache_dir, compartments_path=comp_path)
    lr_map = load_lr_mapping(os.path.join(base_dir, "data/training_data/training_data.csv"))

    # 1. SP (Spatial Procedure LOGO-CV)
    print("\n[1/3] Loading / Evaluating SP (Spatial Procedure LOGO-CV, 9 datasets)...")
    sp_out_file = os.path.join(out_dir, "logo_validation_all_lr_pairs.csv")
    if os.path.exists(sp_out_file):
        print(f"Loading existing SP validation from: {sp_out_file}")
        sp_val = pd.read_csv(sp_out_file)
    else:
        sp_df = build_logo_dataframe(lr_map, reg_path=os.path.join(base_dir, "data/results_regression.joblib"))
        sp_val = evaluate_dataframe(sp_df, validator)
        sp_val.to_csv(sp_out_file, index=False)

    # 2. ScP (Single-cell Procedure LOGO-CV without spatial features)
    print("\n[2/3] Evaluating ScP (Single-cell Procedure LOGO-CV without spatial features, 9 datasets)...")
    scp_reg_path = os.path.join(base_dir, "supplementary_analysis/results/cascading_metrics/results_regression_scp.joblib")
    scp_df = build_logo_dataframe(lr_map, reg_path=scp_reg_path)
    scp_df['Dataset'] = scp_df['Dataset'] + " (ScP)"
    print(f"Total ScP predictions: {len(scp_df)} (Top 20: {(scp_df['Is Top 20']).sum()})")
    scp_val = evaluate_dataframe(scp_df, validator)
    scp_out_file = os.path.join(out_dir, "scp_validation_all_lr_pairs.csv")
    scp_val.to_csv(scp_out_file, index=False)
    print(f"-> Saved ScP validation to: {scp_out_file}")

    # 3. Real Single-Cell Data (scRNA-seq Lung Cancer)
    print("\n[3/3] Loading / Evaluating Real Single-Cell Data (scRNA-seq Lung Cancer)...")
    sc_out_file = os.path.join(out_dir, "single_cell_validation_all_lr_pairs.csv")
    if os.path.exists(sc_out_file):
        print(f"Loading existing Real Single-Cell validation from: {sc_out_file}")
        sc_val = pd.read_csv(sc_out_file)
    else:
        sc_df = build_single_cell_dataframe(
            lr_map,
            cyt_path=os.path.join(base_dir, "../data/results/predictions/scatter_data_cyt_single_cell.csv"),
            nuc_path=os.path.join(base_dir, "../data/results/predictions/scatter_data_nuc_single_cell.csv")
        )
        sc_val = evaluate_dataframe(sc_df, validator)
        sc_val.to_csv(sc_out_file, index=False)

    # 4. Generate Three-Way Summary Table
    print("\nGenerating Three-Way Summary Table (SP vs. ScP vs. Real Single-Cell)...")
    three_way_df = compute_three_way_summary(sp_val, scp_val, sc_val)
    three_way_out = os.path.join(out_dir, "validation_summary_three_way.csv")
    three_way_df.to_csv(three_way_out, index=False)
    print(f"-> Saved Three-Way Summary Table to: {three_way_out}")

    print("\n" + "=" * 90)
    print("THREE-WAY EXTERNAL VALIDATION SUMMARY (TOP 20 vs. ALL)")
    print("=" * 90)
    print(three_way_df[['Benchmark Context', 'Evaluation Subset', 'Predicted Compartment', 'Total Predictions (N)', 'Tier 1 (%)', 'Tier 2 (%)', 'Total Supported (%)']].to_string(index=False))


if __name__ == "__main__":
    main()
