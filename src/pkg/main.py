import numpy as np
import pandas as pd
import os
os.environ["SCIPY_ARRAY_API"] = "1"
from pipeline import run_pipeline_cross_val, run_pipeline_with_test
from pkg.globals import *

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

if __name__ == "__main__":

    microbiome_path = "./src/resources/microbiome.csv"
    serum_lipo_path = "./src/resources/serum_lipo.csv"
    metadata_path = "./src/resources/metadata.csv"

    X, y = load_data(metadata_path, microbiome_path, serum_lipo_path)  

    strategies = STRATEGIES

    print("Evaluation Results:")

    # table = run_pipeline_cross_val(X, y, strategies, k_features=200, random_state=42, test_healthy_ratio=0.90)
    # table = run_pipeline_with_test(X, y, strategies, k_features=200, random_state=42, train_test_ratio=0.8, test_healthy_ratio=0.90, sampling_ratio=1, eval=False)
    # print(table)


    all_tables = []
    for random_state in RANDOM_STATES:
        print(f"Random State: {random_state}")
        table = run_pipeline_with_test(X, y, strategies, k_features=200, random_state=random_state, train_test_ratio=0.8, test_healthy_ratio=0.90, sampling_ratio=1, eval=False)
        print(table)
        all_tables.append(table)

    final_table = pd.concat(all_tables).groupby(level=GROUP_INDEX_NAME).agg(["mean", "std"]).sort_values(("AUPR", "mean"), ascending=False) # mean and std over different random states
    print("Final averaged results over different random states:")
    print(final_table)
    final_table.to_csv("./final_evaluation_results.csv")