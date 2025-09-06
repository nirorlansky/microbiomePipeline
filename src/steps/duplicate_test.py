import numpy as np
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

    return new_splits


def inflate_test_healthy_ratio_no_cross_val(X_test, y_test, target_healthy_ratio=0.9, random_state=42):
    """
    Duplicate existing healthy (label==0) samples in test set to reach target_healthy_ratio.
    """
    # Coerce labels to a NumPy array without copying when possible
    y_arr = y_test.values if hasattr(y_test, "values") else np.asarray(y_test)
    X_arr = X_test.values if hasattr(X_test, "values") else np.asarray(X_test)

    healthy_mask = (y_arr == 0)  # 0 = healthy
    sick_mask = ~healthy_mask     # assumes binary labels {0,1}

    nH = int(healthy_mask.sum())
    nS = int(sick_mask.sum())

    # If one of the classes is absent, or already meeting the target, keep as is
    if nS == 0 or nH == 0:
        return X_test, y_test

    # From H / (H + S) = r  =>  H = r/(1 - r) * S
    ratio_factor = target_healthy_ratio / (1.0 - target_healthy_ratio)
    target_nH = int(np.ceil(ratio_factor * nS))
    if nH >= target_nH:
        return X_test, y_test

    # Number of additional healthy samples needed (by duplication)
    need = target_nH - nH
    healthy_indices = np.where(healthy_mask)[0]

    np.random.seed(random_state)
    chosen_indices = np.random.choice(healthy_indices, size=need, replace=True)

    X_to_add = X_arr[chosen_indices]
    y_to_add = y_arr[chosen_indices]

    X_new = np.vstack([X_arr, X_to_add])
    y_new = np.hstack([y_arr, y_to_add])

    return X_new, y_new