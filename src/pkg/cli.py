from pipeline import evaluate_strategies
import numpy as np
import pandas as pd

microbiome_path = "./src/resources/microbiome.csv"
serum_lipo_path = "./src/resources/serum_lipo.csv"
metadata_path = "./src/resources/metadata.csv"



def load_data(metadata_path, microbiome_path, serum_lipo_path):
    metadata = pd.read_csv(metadata_path)
    microbiome = pd.read_csv(microbiome_path)
    merged_data = metadata.merge(microbiome, on='SampleID', how='inner')
    metadata_cols = [c for c in metadata.columns]
    microbiome_cols = [c for c in microbiome.columns if c != "SampleID"]
    X = merged_data.drop(columns=metadata_cols)
    y = merged_data['PATGROUPFINAL_C']
    # change y to binary - 0 for class 8, 1 for classes 1-7
    y_binary = (y != '8').astype(int)
    return X, y_binary

X, y = load_data(metadata_path, microbiome_path, serum_lipo_path)  
strategies = {
    "No Resampling": "none",
    "SMOTE": "smote",
    "Dirichlet MLE with thresholding": "Dirichlet_MLE_thresholding",
    "Dirichlet MLE": "Dirichlet_MLE",
    "Dirichlet MoM with thresholding": "Dirichlet_MoM_thresholding",
    "Dirichlet MoM": "Dirichlet_MoM"
}
table = evaluate_strategies(X, y, strategies, k_features=200, random_state=42)
print(table)