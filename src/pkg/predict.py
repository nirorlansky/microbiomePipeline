import os
import sys
import numpy as np
import pandas as pd

this_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(this_dir, ".."))
sys.path.append(project_root)

from globals import *
from main import load_data
from pipeline import make_pipeline

def run_and_export_best(
    metadata_path: str,
    microbiome_labeled_path: str,
    serum_lipo_path: str,
    microbiome_unlabeled_path: str,
    best_strategy_name: str,
    output_path: str = "output.csv",
    k_features: int = 200,
    random_state: int = 42,
    eval: bool = False,
):
    """
    Train the chosen pipeline on ALL labeled data, then predict on unlabeled microbiome.csv
    and export predictions to output.csv.

    Args:
        metadata_path: CSV with labels and metadata.
        microbiome_labeled_path: CSV with labeled microbiome data (for training).
        serum_lipo_path: CSV with serum/lipids data (if required by load_data).
        microbiome_unlabeled_path: CSV with unlabeled microbiome samples (for prediction).
        best_strategy_name: Key in `strategies` dict, already selected by you.
        output_path: File path for the exported predictions (default 'output.csv').
    """

    # Load labeled data for training
    X, y = load_data(
        metadata_path=metadata_path,
        microbiome_path=microbiome_labeled_path,
        serum_lipo_path=serum_lipo_path,
    )

    # Build and train pipeline using the best strategy
    best_sampler = STRATEGIES[best_strategy_name]
    pipe = make_pipeline(
        k_features=k_features,
        sampler=best_sampler,
        random_state=random_state,
        feature_selection=False,
        eval=eval,
    )
    pipe.fit(X, y)

    if not hasattr(pipe, "predict_proba"):
        raise ValueError(f"Pipeline for '{best_strategy_name}' does not support predict_proba().")

    # Load unlabeled microbiome data
    df_new = pd.read_csv(microbiome_unlabeled_path)
    if "SampleID" not in df_new.columns:
        raise ValueError("Unlabeled microbiome CSV must contain 'SampleID' column.")

    ids = df_new["SampleID"].astype(str).values
    X_new = df_new.drop(columns=["SampleID"])

    # Soft column alignment with training data
    if hasattr(X, "columns"):
        train_cols = list(X.columns)
        for c in train_cols:
            if c not in X_new.columns:
                X_new[c] = 0
        extra_cols = [c for c in X_new.columns if c not in train_cols]
        if extra_cols:
            X_new = X_new.drop(columns=extra_cols)
        X_new = X_new[train_cols]

    # Predict probabilities for positive class (sick=1)
    prob_sick = pipe.predict_proba(X_new)[:, 1]

    # Export predictions to CSV in the required format
    out_df = pd.DataFrame({
        "ID": ids,
        "Probability": np.round(prob_sick, 2),
    })
    out_df.to_csv(output_path, index=False)

    print(f"Predictions written to {output_path}")

if __name__ == "__main__":
    run_and_export_best(
        metadata_path="./src/resources/metadata.csv",
        microbiome_labeled_path="./src/resources/microbiome.csv",  # training data with labels
        serum_lipo_path="./src/resources/serum_lipo.csv",
        microbiome_unlabeled_path="./test/microbiome.csv",  # unlabeled data to predict
        best_strategy_name="Random Oversampling",  
        output_path="output.csv",
        k_features=200,
        random_state=42,
        eval=False,
    )
