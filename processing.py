# processing.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MultiLabelBinarizer

# feature engineering functions
def encoding_location_total(df):
    """
    Encode the location column as a binary variable
    """
    # split the location column into ligand and receptor location for both cellchat and hpa
    df = df.copy()
    df[['ligand_location_cellchat', 'receptor_location_cellchat']] = df['ligand.location_receptor.location'].str.split('_', n=1, expand=True)
    df['ligand_location_cellchat'] = df['ligand_location_cellchat'].str.strip()
    df['receptor_location_cellchat'] = df['receptor_location_cellchat'].str.strip()
    df[['ligand_location_hpa', 'receptor_location_hpa']] = df['ligand_location_hpa_receptor_location_hpa'].str.split('_', n=1, expand=True)
    df['ligand_location_hpa'] = df['ligand_location_hpa'].str.strip()
    df['receptor_location_hpa'] = df['receptor_location_hpa'].str.strip()
    df.drop(columns = ["ligand.location_receptor.location", "ligand_location_hpa_receptor_location_hpa"], inplace=True)

    return df # call process_df from utils.py for encoding the categorical variables and scaling the numerical variables

def ohe_location(df, separated = True):
    """
    One-hot encode the location columns
    """
    df = df.copy()
    df = encoding_location_total(df)
    if separated:
        ### ----------- CELLCHAT LOCATION ------------ ###
        ligands = df['ligand_location_cellchat'].str.split(',').apply(lambda lst: [x.strip() for x in lst])
        mlb = MultiLabelBinarizer()
        ligand_dummies = pd.DataFrame(
            mlb.fit_transform(ligands),
            columns=[f'ligand_location_cellchat_{loc}' for loc in mlb.classes_],
            index=df.index
        )
        receptors = df['receptor_location_cellchat'].str.split(',').apply(lambda lst: [x.strip() for x in lst])
        receptor_dummies = pd.DataFrame(
            mlb.fit_transform(receptors),
            columns=[f'receptor_location_cellchat_{loc}' for loc in mlb.classes_],
            index=df.index
        )
        ### ----------- HPA LOCATION ------------ ###
        ligands_hpa = df['ligand_location_hpa'].str.split(',').apply(lambda lst: [x.strip() for x in lst])
        mlb = MultiLabelBinarizer()
        ligand_dummies_hpa = pd.DataFrame(
            mlb.fit_transform(ligands_hpa),
            columns=[f'ligand_location_hpa_{loc}' for loc in mlb.classes_],
            index=df.index
        )
        receptors_hpa = df['receptor_location_hpa'].str.split(',').apply(lambda lst: [x.strip() for x in lst])
        receptor_dummies_hpa = pd.DataFrame(
            mlb.fit_transform(receptors_hpa),
            columns=[f'receptor_location_hpa_{loc}' for loc in mlb.classes_],
            index=df.index
        )
        df = pd.concat(
            [df, ligand_dummies, receptor_dummies, ligand_dummies_hpa, receptor_dummies_hpa],
            axis=1
        )
    else:
        # one-hot encode the location columns for both cellchat and hpa
        encoded_columns = ['ligand_location_cellchat','receptor_location_cellchat', 'ligand_location_hpa', 'receptor_location_hpa']
        encoder = OneHotEncoder(sparse_output=False)
        one_hot_encoded = encoder.fit_transform(df[encoded_columns])
        one_hot_df = pd.DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(encoded_columns))
        df = pd.concat([df.drop(encoded_columns, axis=1), one_hot_df], axis=1)
    return df

# feature engineering approaches. 
# Model A: separate the location features and Ohe, and load the data to the model.
data = pd.read_csv('data/training_data/raw_data.csv')
ohe_df = ohe_location(data, separated = False)
ohe_df.to_csv('data/training_data/df_modelA.csv', index = False)
# Model B: separate the location features and catboost is done while training
sep_df = encoding_location_total(data)
sep_df.to_csv('data/training_data/df_modelB.csv', index = False)
# Model C: No encoding besides catboost (raw_data combined model), load the data directly to the model. - saved in the data dir from processsing.qmd
data.to_csv('data/training_data/df_modelC.csv', index = False)
# Model D: OHE the location features, and load the data to the model. (final model)
ohe_sep_df = ohe_location(data, separated = True)
ohe_sep_df.to_csv('data/training_data/training_data.csv', index = False)