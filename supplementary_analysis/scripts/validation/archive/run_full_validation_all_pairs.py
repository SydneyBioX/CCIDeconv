# supplementary_analysis/scripts/validation_analysis/run_full_validation_all_pairs.py
"""
Complete Independent External Biological Validation for ALL LR pairs
in Single Cell and LOGO (sST) datasets.

This script validates all 1,126 single-cell predictions and all 3,988 LOGO-CV predictions
against UniProtKB (reviewed Swiss-Prot), non-HPA COMPARTMENTS (score >= 3 or experimental),
and Reactome Pathway Knowledgebase reactions.

Strict Methodological Criteria:
1. Strict exclusion of HPA records (146,726 removed) to avoid training feature overlap.
2. Strict exclusion of CellChatDB to ensure complete independence from training priors.
3. Mutually exclusive evidence tiers:
   - Tier 1: Independent Experimental Localization
   - Tier 2: Independent Biological / Mechanistic Support
   - No qualifying independent evidence identified
4. Pre-computes validation across all unique (ligand, receptor, compartment) tuples
   and maps onto all candidate predictions.
5. Exports:
   - Single Cell full validation CSV (1,126 rows)
   - LOGO full validation CSV (3,988 rows)
   - Summary CSV comparing Top 20 vs. All across Single Cell and LOGO
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


def load_lr_mapping(data_path="data/training_data/training_data.csv"):
    if not os.path.exists(data_path):
        alt_paths = [
            "../data/training_data/training_data.csv",
            "/Users/rjay0256/Library/CloudStorage/OneDrive-TheUniversityofSydney(Students)/University of Sydney/PhD projects/main/Cell Cell Communication/server/cell_cell_commuication/data/training_data/training_data.csv"
        ]
        for p in alt_paths:
            if os.path.exists(p):
                data_path = p
                break
    df_raw = pd.read_csv(data_path)
    lr_map = df_raw[['lr_pair', 'ligand', 'receptor']].drop_duplicates().set_index('lr_pair').to_dict(orient='index')
    return lr_map


def build_single_cell_dataframe(lr_map, cyt_path=None, nuc_path=None):
    if cyt_path is None:
        cyt_path = "../data/results/predictions/scatter_data_cyt_single_cell.csv"
        if not os.path.exists(cyt_path):
            cyt_path = "data/results/predictions/scatter_data_cyt_single_cell.csv"
    if nuc_path is None:
        nuc_path = "../data/results/predictions/scatter_data_nuc_single_cell.csv"
        if not os.path.exists(nuc_path):
            nuc_path = "data/results/predictions/scatter_data_nuc_single_cell.csv"

    cyt_df = pd.read_csv(cyt_path)
    nuc_df = pd.read_csv(nuc_path)

    cyt_df['Predicted Compartment'] = 'Cytoplasm'
    nuc_df['Predicted Compartment'] = 'Nucleus'

    rows = []
    for comp_df, comp in [(cyt_df, 'Cytoplasm'), (nuc_df, 'Nucleus')]:
        sorted_df = comp_df.sort_values(by='pred', ascending=False).reset_index(drop=True)
        sorted_df['Rank'] = range(1, len(sorted_df) + 1)
        sorted_df['Is Top 20'] = sorted_df['Rank'] <= 20
        sorted_df['Dataset'] = 'scRNA-seq Lung Cancer'
        sorted_df['dataset_id'] = 'single_cell_lung'
        sorted_df['Ligand'] = sorted_df['lr_pair'].apply(lambda x: lr_map.get(x, {}).get('ligand', x.split('_')[0]))
        sorted_df['Receptor'] = sorted_df['lr_pair'].apply(lambda x: lr_map.get(x, {}).get('receptor', x.split('_')[-1]))
        sorted_df['CCIDeconv Score'] = sorted_df['pred']
        sorted_df['Source Cell Type'] = sorted_df['source']
        sorted_df['Target Cell Type'] = sorted_df['target']
        sorted_df['Interaction / LR Pair'] = sorted_df['lr_pair']
        rows.append(sorted_df)

    sc_full = pd.concat(rows, ignore_index=True)
    return sc_full


def build_logo_dataframe(lr_map, reg_path="data/results_regression.joblib"):
    if not os.path.exists(reg_path):
        alt = "../data/results_regression.joblib"
        if os.path.exists(alt):
            reg_path = alt

    reg_results = joblib.load(reg_path)
    all_rows = []

    for ds_id in reg_results.keys():
        ds_name = DATASET_LABELS.get(ds_id, ds_id)
        cyt_preds = reg_results[ds_id]['cyt'][1].copy()
        nuc_preds = reg_results[ds_id]['nuc'][1].copy()

        for df_p, comp in [(cyt_preds, 'Cytoplasm'), (nuc_preds, 'Nucleus')]:
            df_p['Predicted Compartment'] = comp
            sorted_p = df_p.sort_values(by='pred', ascending=False).reset_index(drop=True)
            sorted_p['Rank'] = range(1, len(sorted_p) + 1)
            sorted_p['Is Top 20'] = sorted_p['Rank'] <= 20
            sorted_p['Dataset'] = ds_name
            sorted_p['dataset_id'] = ds_id
            sorted_p['Ligand'] = sorted_p['lr_pair'].apply(lambda x: lr_map.get(x, {}).get('ligand', x.split('_')[0]))
            sorted_p['Receptor'] = sorted_p['lr_pair'].apply(lambda x: lr_map.get(x, {}).get('receptor', x.split('_')[-1]))
            sorted_p['CCIDeconv Score'] = sorted_p['pred']
            sorted_p['Source Cell Type'] = sorted_p['source']
            sorted_p['Target Cell Type'] = sorted_p['target']
            sorted_p['Interaction / LR Pair'] = sorted_p['lr_pair']
            all_rows.append(sorted_p)

    logo_full = pd.concat(all_rows, ignore_index=True)
    return logo_full


def evaluate_dataframe(df, validator):
    """
    Evaluate interactions using an in-memory memoized cache of (ligand, receptor, compartment).
    """
    unique_interactions = df[['Ligand', 'Receptor', 'Predicted Compartment']].drop_duplicates()
    print(f"Total unique (ligand, receptor, compartment) tuples to evaluate: {len(unique_interactions)}")

    cache = {}
    for _, row in unique_interactions.iterrows():
        lig = row['Ligand']
        rec = row['Receptor']
        comp = row['Predicted Compartment']
        key = (lig, rec, comp)
        cache[key] = validator.validate_interaction(lig, rec, comp)

    print("Finished evaluating unique interactions. Mapping results to all prediction rows...")

    evaluated_records = []
    for _, row in df.iterrows():
        lig = row['Ligand']
        rec = row['Receptor']
        comp = row['Predicted Compartment']
        eval_res = cache[(lig, rec, comp)]
        p_item = eval_res['primary_item']
        is_supported = (eval_res['val_class'] != 'No Independent Evidence')

        record = {
            'Rank': row['Rank'],
            'Is Top 20': 'Yes' if row['Is Top 20'] else 'No',
            'Dataset': row['Dataset'],
            'dataset_id': row['dataset_id'],
            'Interaction / LR Pair': row['Interaction / LR Pair'],
            'Ligand': lig,
            'Receptor': rec,
            'Source Cell Type': row['Source Cell Type'],
            'Target Cell Type': row['Target Cell Type'],
            'Predicted Compartment': comp,
            'CCIDeconv Score': row['CCIDeconv Score'],
            'Ligand Experimental Localization': eval_res['ligand_exp'],
            'Receptor Experimental Localization': eval_res['receptor_exp'],
            'Ligand Biological / Pathway Support': eval_res['ligand_path'],
            'Receptor Biological / Pathway Support': eval_res['receptor_path'],
            'Independent Evidence Component': eval_res['evidence_comp'],
            'Evidence Tier': eval_res['evidence_tier'],
            'Validation Classification': eval_res['val_class'],
            'Concordance / Supported': 'Yes' if is_supported else 'No',
            'Database Source': eval_res['db_source'],
            'Primary Evidence Type': p_item['evidence_type'],
            'Exact Annotation / Reaction Description': p_item['description'],
            'Primary Accession / Identifier': p_item['accession'],
            'Evidence Code': p_item['evidence_code'],
            'PubMed / Reference': p_item['pubmed_id'],
            'Independent Evidence Citation': eval_res['citation'],
            'HPA Evidence Excluded': 'Yes (146,726 HPA records removed)',
            'CellChatDB Excluded': 'Yes (Not queried / independent)'
        }
        evaluated_records.append(record)

    return pd.DataFrame(evaluated_records)


def compute_summary_table(sc_val_df, logo_val_df):
    """
    Compute structured summary table split by Top 20 and All across Single Cell and LOGO.
    """
    summary_rows = []

    datasets = [
        ('Single Cell', sc_val_df),
        ('LOGO', logo_val_df)
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
                    'Dataset Context': ds_label,
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
    # Go up 3 levels from supplementary_analysis/scripts/validation_analysis to CCIDeconv root
    base_dir = os.path.abspath(os.path.join(script_dir, "../../.."))
    cache_dir = os.path.join(base_dir, "supplementary_analysis/results/validation/cache")
    comp_path = os.path.join(base_dir, "data/references/human_compartment_knowledge.tsv")
    out_dir = os.path.join(base_dir, "supplementary_analysis/results/validation")
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("INDEPENDENT EXTERNAL VALIDATION: ALL LR PAIRS (SINGLE CELL & LOGO)")
    print("=" * 80)

    validator = ExternalValidator(cache_dir=cache_dir, compartments_path=comp_path)
    lr_map = load_lr_mapping(os.path.join(base_dir, "data/training_data/training_data.csv"))

    # 1. Single Cell
    print("\n[1/3] Preparing Single Cell dataset...")
    sc_df = build_single_cell_dataframe(
        lr_map,
        cyt_path=os.path.join(base_dir, "../data/results/predictions/scatter_data_cyt_single_cell.csv"),
        nuc_path=os.path.join(base_dir, "../data/results/predictions/scatter_data_nuc_single_cell.csv")
    )
    print(f"Total Single Cell predictions: {len(sc_df)} (Top 20: {(sc_df['Is Top 20']).sum()}, All: {len(sc_df)})")
    sc_val = evaluate_dataframe(sc_df, validator)
    sc_out_file = os.path.join(out_dir, "single_cell_validation_all_lr_pairs.csv")
    sc_val.to_csv(sc_out_file, index=False)
    print(f"-> Saved Single Cell validation to: {sc_out_file}")

    # 2. LOGO
    print("\n[2/3] Preparing LOGO dataset (9 sST datasets)...")
    logo_df = build_logo_dataframe(lr_map, reg_path=os.path.join(base_dir, "data/results_regression.joblib"))
    print(f"Total LOGO predictions: {len(logo_df)} (Top 20: {(logo_df['Is Top 20']).sum()}, All: {len(logo_df)})")
    logo_val = evaluate_dataframe(logo_df, validator)
    logo_out_file = os.path.join(out_dir, "logo_validation_all_lr_pairs.csv")
    logo_val.to_csv(logo_out_file, index=False)
    print(f"-> Saved LOGO validation to: {logo_out_file}")

    # 3. Summary Table
    print("\n[3/3] Generating summary table split by Top 20 and All...")
    summary_df = compute_summary_table(sc_val, logo_val)
    summary_out_file = os.path.join(out_dir, "validation_summary_top20_vs_all.csv")
    summary_df.to_csv(summary_out_file, index=False)
    print(f"-> Saved Summary Table to: {summary_out_file}")

    print("\n" + "=" * 90)
    print("SUMMARY RESULTS: TOP 20 vs. ALL PREDICTIONS")
    print("=" * 90)
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
