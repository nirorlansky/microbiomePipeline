import numpy as np
import pandas as pd
from pkg.globals import HEALTHY, SICK

def inflate_test_healthy_ratio(splits, y, target_healthy_ratio=0.90):
    """
    Duplicate existing healthy (label==0) samples in each test fold to reach
    approximately the desired class mix in test: target_healthy_ratio healthy and
    (1 - target_healthy_ratio) sick. Training folds are not changed.
    The function never deletes samples; it only appends duplicates of healthy ones.
    If the current ratio already meets/exceeds the target or a class is missing,
    the test fold is left unchanged.

    Parameters
    ----------
    splits : list[tuple[np.ndarray, np.ndarray]]
        Each item is (train_idx, test_idx) as produced by a CV splitter.
    y : array-like (pd.Series or np.ndarray)
        Binary labels aligned with X/y indices. Healthy must be 0 and sick 1.
    target_healthy_ratio : float, default=0.90
        Desired fraction of healthy in the test fold.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray]]
        New splits where only test_idx may be longer due to appended duplicates.
    """
    # Coerce labels to a NumPy array without copying when possible
    y_arr = y.values if hasattr(y, "values") else np.asarray(y)
    new_splits = []

    # From H / (H + S) = r  =>  H = r/(1 - r) * S
    ratio_factor = target_healthy_ratio / (1.0 - target_healthy_ratio)

    for train_idx, test_idx in splits:
        y_test = y_arr[test_idx]
        healthy_mask = (y_test == 0)  # 0 = healthy
        sick_mask = ~healthy_mask     # assumes binary labels {0,1}

        nH = int(healthy_mask.sum())
        nS = int(sick_mask.sum())

        # If one of the classes is absent, or already meeting the target, keep as is
        if nS == 0 or nH == 0:
            new_splits.append((train_idx, test_idx))
            continue

        target_nH = int(np.ceil(ratio_factor * nS))
        if nH >= target_nH:
            new_splits.append((train_idx, test_idx))
            continue

        # Number of additional healthy samples needed (by duplication)
        need = target_nH - nH
        healthy_indices = test_idx[healthy_mask]

        # Distribute duplicates as evenly as possible across the existing healthy samples
        base = need // nH
        extra = need % nH
        reps = np.full(nH, base, dtype=int)
        if extra:
            reps[:extra] += 1  # deterministic tie-breaking

        duplicates = np.repeat(healthy_indices, reps)

        # Preserve original order and append duplicates at the end
        new_test_idx = np.concatenate([test_idx, duplicates])
        new_splits.append((train_idx, new_test_idx))
    print(f"[INFLATE TEST] with cross_val target healthy ratio: {target_healthy_ratio:.2f} | Original H/S: {nH}/{nS} | Target H/S: {target_nH}/{nS} | Added healthy: {len(duplicates)} | Final H/S: {nH + len(duplicates)}/{nS}")

    return new_splits


def inflate_test_healthy_ratio_no_cross_val(X_test, y_test, target_healthy_ratio=0.9):
    """
    Deterministically duplicate existing healthy (label==0) samples in the test set
    to reach target_healthy_ratio. No randomness: distribute duplicates as evenly
    as possible across healthy samples; if there's a remainder, the first samples
    get +1 (at most one extra over others).
    """
    # Preserve pandas types if provided
    is_df = hasattr(X_test, "values") and hasattr(X_test, "columns")
    is_series = hasattr(y_test, "values") and hasattr(y_test, "name")

    # Coerce to NumPy arrays (without copying when possible)
    X_arr = X_test.values if hasattr(X_test, "values") else np.asarray(X_test)
    y_arr = y_test.values if hasattr(y_test, "values") else np.asarray(y_test)

    # Build healthy/sick masks
    healthy_mask = (y_arr == 0)  # 0 = healthy
    sick_mask = ~healthy_mask     # assumes binary labels {0,1}

    nH = int(healthy_mask.sum())
    nS = int(sick_mask.sum())

    # If one of the classes is absent, or already meets target, return as-is
    if nS == 0 or nH == 0:
        return X_test, y_test

    # From H / (H + S) = r  =>  H = r/(1 - r) * S
    ratio_factor = target_healthy_ratio / (1.0 - target_healthy_ratio)
    target_nH = int(np.ceil(ratio_factor * nS))
    if nH >= target_nH:
        return X_test, y_test

    # How many additional healthy samples are needed?
    need = target_nH - nH
    healthy_indices = np.where(healthy_mask)[0]  # positions within current test set

    # Deterministic, even distribution of duplicates across existing healthy samples
    base = need // nH
    extra = need % nH
    reps = np.full(nH, base, dtype=int)
    if extra:
        reps[:extra] += 1  # first 'extra' healthy samples receive +1 duplicate

    # Build the duplicate index vector in original order, appended at the end
    duplicates_pos = np.repeat(healthy_indices, reps)

    # Append duplicates to X and y (preserving original order first)
    X_new = np.vstack([X_arr, X_arr[duplicates_pos]])
    y_new = np.hstack([y_arr, y_arr[duplicates_pos]])

    # (Optional) restore pandas types to keep downstream pipelines happy
    if is_df:
        X_new = pd.DataFrame(X_new, columns=X_test.columns)
    if is_series:
        y_new = pd.Series(y_new, name=y_test.name)

    print(f"[INFLATE TEST] no cross_val Original H/S: {nH}/{nS} | Target H/S: {target_nH}/{nS} | Added healthy: {len(duplicates_pos)} | Final H/S: {nH + len(duplicates_pos)}/{nS}")

    return X_new, y_new

