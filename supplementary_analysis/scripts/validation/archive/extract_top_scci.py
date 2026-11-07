# analysis/extract_top_scci.py
"""
Extract top 20 cytoplasmic and top 20 nuclear sCCI predictions per dataset from CCIDeconv.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

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


def extract_top_scci(clf_path="results_classifier.joblib",
                     reg_path="results_regression.joblib",
                     data_path="data/training_data/training_data.csv",
                     output_dir="data/validation_results",
                     top_n=20):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading predictions from {clf_path} and {reg_path}...")
    clf_results = joblib.load(clf_path)
    reg_results = joblib.load(reg_path)

    print(f"Loading L-R gene mapping from {data_path}...")
    df_raw = pd.read_csv(data_path)
    lr_map = df_raw[['lr_pair', 'ligand', 'receptor']].drop_duplicates().set_index('lr_pair').to_dict(orient='index')

    all_candidates = []

    for ds_id in clf_results.keys():
        ds_name = DATASET_LABELS.get(ds_id, ds_id)
        print(f"\nProcessing dataset: {ds_name} ({ds_id})")

        # Regression predictions for predicted positive interactions
        cyt_preds = reg_results[ds_id]['cyt'][1].copy()
        nuc_preds = reg_results[ds_id]['nuc'][1].copy()

        # Add gene symbols
        for df_p in [cyt_preds, nuc_preds]:
            df_p['ligand'] = df_p['lr_pair'].apply(lambda x: lr_map.get(x, {}).get('ligand', x.split('_')[0]))
            df_p['receptor'] = df_p['lr_pair'].apply(lambda x: lr_map.get(x, {}).get('receptor', x.split('_')[-1]))
            df_p['dataset_id'] = ds_id
            df_p['dataset'] = ds_name

        # Top 20 Cytoplasm
        cyt_sorted = cyt_preds.sort_values(by='pred', ascending=False).reset_index(drop=True)
        top_cyt = cyt_sorted.head(top_n).copy()
        top_cyt['predicted_compartment'] = 'Cytoplasm'
        top_cyt['ccideconv_score'] = top_cyt['pred']
        top_cyt['rank'] = range(1, len(top_cyt) + 1)

        # Top 20 Nucleus
        nuc_sorted = nuc_preds.sort_values(by='pred', ascending=False).reset_index(drop=True)
        top_nuc = nuc_sorted.head(top_n).copy()
        top_nuc['predicted_compartment'] = 'Nucleus'
        top_nuc['ccideconv_score'] = top_nuc['pred']
        top_nuc['rank'] = range(1, len(top_nuc) + 1)

        print(f"  Extracted top {len(top_cyt)} cytoplasmic predictions (score range: {top_cyt['pred'].min():.3f} to {top_cyt['pred'].max():.3f})")
        print(f"  Extracted top {len(top_nuc)} nuclear predictions (score range: {top_nuc['pred'].min():.3f} to {top_nuc['pred'].max():.3f})")

        all_candidates.extend([top_cyt, top_nuc])

    candidates_df = pd.concat(all_candidates, ignore_index=True)
    out_file = os.path.join(output_dir, "top_scci_candidates.csv")
    candidates_df.to_csv(out_file, index=False)
    print(f"\nTotal candidate predictions extracted: {len(candidates_df)}")
    print(f"Saved to: {out_file}")

    return candidates_df


if __name__ == "__main__":
    extract_top_scci()
