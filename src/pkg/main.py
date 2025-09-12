import numpy as np
import pandas as pd
import os
os.environ["SCIPY_ARRAY_API"] = "1"
from pipeline import run_pipeline_cross_val, run_pipeline_with_test

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
    "SMOTE with thresholding": "smote_thresholding",
    "SMOTE preserve_zero_pattern": "smote_preserve_zero_pattern",
    "SMOTE with min-positive vector": "smote_min_positive",
    "SMOTE only": "smote_only",
    "Dirichlet MLE with thresholding": "Dirichlet_MLE_thresholding",
    "Dirichlet MLE": "Dirichlet_MLE",
    "Dirichlet MoM with thresholding": "Dirichlet_MoM_thresholding",
    "Dirichlet MoM": "Dirichlet_MoM",
    "Random Oversampling": "resample_random_samples",
}

print("Evaluation Results:")

# table = run_pipeline_cross_val(X, y, strategies, k_features=200, random_state=42, test_healthy_ratio=0.90)
table = run_pipeline_with_test(X, y, strategies, k_features=200, random_state=30, ratio=0.8, test_healthy_ratio=0.90, eval=False)
print(table)