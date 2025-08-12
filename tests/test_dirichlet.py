# prepare_microbiome_dataset.py
# Build X_train (100 microbiome features, each row sums to 10000) and y_train (H/S)
# Steps:
# 1) Merge microbiome.csv and metadata.csv on SampleID
# 2) Keep all microbiome feature columns; drop all metadata except PATGROUPFINAL_C
# 3) Map PATGROUPFINAL_C: 8 -> 'H', anything else -> 'S'
# 4) y_train is that mapped column
# 5) Feature selection: top 100 features by variance (simple heuristic)
# 6) Scale rows so each row sums to 10000
# 7) Return/save X_train and y_train

import numpy as np
import pandas as pd
from pathlib import Path
from src.samplers.dirichlet_new import balance_healthy_samples, balance_with_dirichlet_mle


def prepare_microbiome_dataset(
    microbiome_csv: str,
    metadata_csv: str,
    k_features: int = 100,
    scale_total: float = 10000.0,
):
    """
    Returns:
        X_train: np.ndarray of shape (n_samples, k_features), each row sums ~ scale_total
        y_train: np.ndarray of shape (n_samples,), values 'H' or 'S'
        selected_features: list[str] of chosen microbiome feature names (length k_features)
        sample_ids: np.ndarray of SampleID values aligned with rows in X_train/y_train
    """

    # --- Read CSVs ---
    df_micro = pd.read_csv(microbiome_csv)
    df_meta = pd.read_csv(metadata_csv)

    # --- Normalize column names (trim spaces) ---
    df_micro.columns = df_micro.columns.str.strip()
    df_meta.columns = df_meta.columns.str.strip()

    # --- Basic checks ---
    if "SampleID" not in df_micro.columns or "SampleID" not in df_meta.columns:
        raise ValueError("SampleID column must exist in both CSV files.")
    if "PATGROUPFINAL_C" not in df_meta.columns:
        raise ValueError("Expected 'PATGROUPFINAL_C' in metadata.csv.")

    # --- Identify microbiome feature columns (all except SampleID) ---
    micro_features = [c for c in df_micro.columns if c != "SampleID"]

    # --- Make microbiome numeric (coerce) ---
    df_micro_num = df_micro.copy()
    for c in micro_features:
        df_micro_num[c] = pd.to_numeric(df_micro_num[c], errors="coerce")

    # --- Merge on SampleID ---
    df = df_micro_num.merge(
        df_meta[["SampleID", "PATGROUPFINAL_C"]],
        on="SampleID",
        how="inner",
    )

    # --- Map PATGROUPFINAL_C -> labels ---
    def map_group(val):
        try:
            return "H" if float(val) == 8 else "S"
        except Exception:
            return "H" if str(val).strip() == "8" else "S"

    df["label"] = df["PATGROUPFINAL_C"].apply(map_group)

    # --- Build y_train ---
    y_train = df["label"].astype(str).to_numpy()

    # --- Build X (microbiome only) ---
    X_all = df[micro_features].to_numpy(dtype=float)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    # --- Feature selection: top-k by variance ---
    k = min(k_features, X_all.shape[1])
    variances = X_all.var(axis=0)
    top_idx = np.argsort(variances)[-k:]
    top_idx.sort()  # keep original ordering ascending by index

    selected_features = [micro_features[i] for i in top_idx]
    X_sel = X_all[:, top_idx]

    # --- Scale rows so each row sums to scale_total ---
    row_sums = X_sel.sum(axis=1, keepdims=True)
    safe_sums = np.where(row_sums <= 0, 1.0, row_sums)  # avoid divide-by-zero
    X_scaled = (X_sel / safe_sums) * scale_total
    X_scaled = X_scaled.astype(np.float64)

    # --- SampleIDs aligned with rows ---
    sample_ids = df["SampleID"].to_numpy()

    return X_scaled, y_train, selected_features, sample_ids


if __name__ == "__main__":
    # --- Example usage ---
    MICROBIOME_CSV = "src/resources/microbiome.csv"
    METADATA_CSV = "src/resources/metadata.csv"

    X_train, y_train, features, sample_ids = prepare_microbiome_dataset(
        MICROBIOME_CSV,
        METADATA_CSV,
        k_features=100,
        scale_total=1.0,
    )


print("balance_healthy_samples (MoM)\n")
X_bal_mom, y_bal_mom =  balance_healthy_samples(
    X_train,
    y_train,
    target_ratio=9.0,
    seed=42,
    use_dynamic_eps=True,
    orders_below=3,
    fallback=1e-12,
)

# Save MoM-balanced outputs
pd.DataFrame(X_bal_mom, columns=features).to_csv("data_files/X_balanced_mom.csv", index=False)
pd.DataFrame({"y_train": y_bal_mom}).to_csv("data_files/y_balanced_mom.csv", index=False)

print("\n\nbalance_with_dirichlet_mle (MLE)\n")
X_bal_mle, y_bal_mle, synth_samples, info = balance_with_dirichlet_mle(
    X_train,
    y_train,
    target_ratio=9.0,
    seed=42,
    use_dynamic_eps=False,
    orders_below=3,
    fallback=1e-12,
)

# Save MLE-balanced outputs
pd.DataFrame(X_bal_mle, columns=features).to_csv("data_files/X_balanced_mle.csv", index=False)
pd.DataFrame({"y_train": y_bal_mle}).to_csv("data_files/y_balanced_mle.csv", index=False)

print("\nSaved:")
print("  data_files/X_balanced_mom.csv")
print("  data_files/y_balanced_mom.csv")
print("  data_files/X_balanced_mle.csv")
print("  data_files/y_balanced_mle.csv")