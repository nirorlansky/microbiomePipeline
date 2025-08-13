from imblearn.over_sampling import SMOTE
import numpy as np

def smote_to_minority_share(X_train, y_train, minority_share=0.9, random_state=42, k_neighbors=None):
    """
    Runs SMOTE so that after resampling, class 0 will represent `minority_share` of the dataset.
    Prints how many samples were actually added.
    
    Parameters:
    - X_train: Feature matrix (numeric values)
    - y_train: Corresponding labels (0/1), same order as X_train
    - minority_share: Target fraction of minority class (0) after resampling (e.g., 0.1 for 10%)
    - random_state: Seed for reproducibility
    - k_neighbors: Number of nearest neighbors for SMOTE. If None, will be chosen automatically.
    
    Returns:
    - X_res, y_res: Resampled feature matrix and labels
    """
    y = np.asarray(y_train)
    
    # Count class 0 (minority) and class 1 (majority) before resampling
    n_min_before = np.sum(y == 0)
    n_maj = np.sum(y == 1)

    # Target ratio for SMOTE: minority/majority
    ratio = minority_share / (1.0 - minority_share)

    # Auto-adjust k_neighbors if not provided
    if k_neighbors is None:
        k_neighbors = max(1, min(5, n_min_before - 1))

    # Run SMOTE oversampling
    smote = SMOTE(
        sampling_strategy=ratio,
        k_neighbors=k_neighbors,
        random_state=random_state
    )
    X_res, y_res = smote.fit_resample(X_train, y_train)

    # Count how many new samples were added to class 0
    n_min_after = np.sum(y_res == 0)
    print(f"Added {n_min_after - n_min_before} samples to class 0 (total now: {n_min_after})")

    return X_res, y_res
