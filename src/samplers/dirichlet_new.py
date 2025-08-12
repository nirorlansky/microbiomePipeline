import numpy as np
import dirichlet

# ---------- helpers ----------

def replace_zeros_with_dynamic_eps(X, orders_below=3, fallback=1e-12):
    """
    Input X is proportions (each row ~1). Replace zeros per feature using a dynamic epsilon:
      eps_vec[j]  = min positive in column j   (returned as-is)
      eps_repl[j] = eps_vec[j] * 10^(-orders_below)  (used for replacement)
    Then renormalize rows to sum to 1.

    Returns:
      X2: normalized proportions (rows sum ~1) after zero-replacement
      eps_vec: per-feature min positive (>0) values (NOT scaled)
    """
    pos = np.where(X > 0, X, np.inf)
    col_min_pos = pos.min(axis=0)
    has_col_pos = np.isfinite(col_min_pos)

    # global min positive if any
    global_min_pos = col_min_pos[has_col_pos].min() if np.any(has_col_pos) else np.inf

    # store original per-feature min-positive (as requested)
    eps_vec = col_min_pos.copy()
    eps_vec[~has_col_pos] = global_min_pos if np.isfinite(global_min_pos) else fallback

    eps_repl = eps_vec * (10.0 ** (-orders_below))
    X2 = np.where(X <= 0, eps_repl, X)

    rs = X2.sum(axis=1, keepdims=True)
    rs = np.clip(rs, 1e-12, None)  # guard division
    X2 = X2 / rs
    return X2, eps_vec

def zero_below_eps_and_renormalize(synth_prop, eps_vec):
    """
    Zero-out synth_prop[:, j] where synth_prop[:, j] < eps_vec[j],
    then renormalize each row to sum to 1. If a row becomes all zeros, fallback to original row.
    """
    sp = np.where(synth_prop < eps_vec, 0.0, synth_prop)
    rs = sp.sum(axis=1, keepdims=True)
    nz = (rs.squeeze() > 0)
    if np.any(nz):
        sp[nz] = sp[nz] / rs[nz, :]
    if np.any(~nz):
        sp[~nz] = synth_prop[~nz]
    return sp

# ---------- MoM ----------

def balance_healthy_samples(
    X_train,
    y_train,
    target_ratio=9.0,
    seed=42,
    use_dynamic_eps=False,
    orders_below=3,
    fallback=1e-12,
):
    """
    MoM: fit Dirichlet on Healthy (on proportions), sample, optional zero-below-eps, append.

    Assumes each row of X_train already sums to ~1.
    use_dynamic_eps=True:
      - Before fitting: dynamic per-feature eps replacement + renormalize
      - After sampling: zero-out values below ORIGINAL eps per feature + renormalize
    """
    rng = np.random.default_rng(seed)

    healthy_count = int(np.sum(y_train == 'H'))
    sick_count    = int(np.sum(y_train == 'S'))
    target_healthy = int(target_ratio * sick_count)
    samples_to_add = max(0, target_healthy - healthy_count)
    if samples_to_add == 0:
        print("No samples to add; already at or above target ratio.")
        return X_train, y_train

    # Healthy proportions (rows should already sum to 1; renorm just in case)
    healthy_data = X_train[y_train == 'H']
    rs = healthy_data.sum(axis=1, keepdims=True)
    rs = np.clip(rs, 1e-12, None)
    healthy_prop = healthy_data / rs

    eps_vec = None
    if use_dynamic_eps:
        healthy_prop, eps_vec = replace_zeros_with_dynamic_eps(
            healthy_prop, orders_below=orders_below, fallback=fallback
        )

    # MoM on proportions
    mu  = healthy_prop.mean(axis=0)
    var = np.clip(healthy_prop.var(axis=0), 1e-12, None)
    alpha0_est = (mu * (1 - mu) / var) - 1
    mask = np.isfinite(alpha0_est) & (alpha0_est > 0)
    if not np.any(mask):
        raise ValueError("MoM failed: no positive/finite alpha0 estimates on proportions. Try MLE.")
    alpha0   = float(np.mean(alpha0_est[mask]))
    alpha_vec = mu * alpha0

    # Sample proportions
    synthetic_prop = rng.dirichlet(alpha_vec, size=samples_to_add)

    # Optional: zero below per-feature ORIGINAL eps, then renormalize
    if use_dynamic_eps and eps_vec is not None:
        synthetic_prop = zero_below_eps_and_renormalize(synthetic_prop, eps_vec)

    # Append (rows all sum to 1)
    X_balanced = np.vstack([X_train, synthetic_prop])
    y_balanced = np.hstack([y_train, np.full(samples_to_add, 'H', dtype=y_train.dtype)])

    print(f"Original healthy: {healthy_count} | sick: {sick_count}")
    print(f"Added synthetic healthy: {samples_to_add}")
    print(f"Balanced healthy: {np.sum(y_balanced == 'H')} | sick: {np.sum(y_balanced == 'S')}")
    print(f"Shapes: X={X_balanced.shape}, y={y_balanced.shape}")
    return X_balanced, y_balanced

# ---------- MLE ----------

def balance_with_dirichlet_mle(
    X_train,
    y_train,
    target_ratio=9.0,
    seed=42,
    use_dynamic_eps=False,
    orders_below=3,
    fallback=1e-12,
):
    """
    MLE: fit Dirichlet on Healthy (on proportions), sample, optional zero-below-eps, append.

    Assumes each row of X_train already sums to ~1.
    use_dynamic_eps=True:
      - Before fitting: dynamic per-feature eps replacement + renormalize
      - After sampling: zero-out values below ORIGINAL eps per feature + renormalize
    use_dynamic_eps=False:
      - Minimal static clip to avoid log(0), then renormalize
    """
    rng = np.random.default_rng(seed)

    healthy_count = int(np.sum(y_train == 'H'))
    sick_count    = int(np.sum(y_train == 'S'))
    target_healthy = int(target_ratio * sick_count)
    samples_to_add = max(0, target_healthy - healthy_count)
    if samples_to_add == 0:
        print("No samples to add; already at or above target ratio.")
        return X_train, y_train, np.array([]), {
            "healthy_count": healthy_count,
            "sick_count": sick_count,
            "target_healthy": target_healthy,
            "alpha": None,
            "eps_vec_min_positive": None
        }

    # Healthy proportions (renorm just in case)
    healthy_data = X_train[y_train == 'H']
    rs = healthy_data.sum(axis=1, keepdims=True)
    rs = np.clip(rs, 1e-12, None)
    healthy_prop = healthy_data / rs

    eps_vec = None
    if use_dynamic_eps:
        healthy_prop, eps_vec = replace_zeros_with_dynamic_eps(
            healthy_prop, orders_below=orders_below, fallback=fallback
        )
    else:
        # Minimal static clip just to avoid log(0) in MLE
        healthy_prop = np.clip(healthy_prop, fallback, None)
        healthy_prop = healthy_prop / healthy_prop.sum(axis=1, keepdims=True)

    # Fit alpha via MLE
    alpha_hat = dirichlet.mle(healthy_prop)

    # Sample proportions
    synthetic_prop = rng.dirichlet(alpha_hat, size=samples_to_add)

    # Optional: zero below per-feature ORIGINAL eps, then renormalize
    if use_dynamic_eps and eps_vec is not None:
        synthetic_prop = zero_below_eps_and_renormalize(synthetic_prop, eps_vec)

    # Append (rows all sum to 1)
    X_balanced = np.vstack([X_train, synthetic_prop])
    y_balanced = np.hstack([y_train, np.full(samples_to_add, 'H', dtype=y_train.dtype)])

    info = {
        "healthy_count": int(np.sum(y_balanced == 'H')),
        "sick_count": int(np.sum(y_balanced == 'S')),
        "target_healthy": target_healthy,
        "alpha": alpha_hat,
        "eps_vec_min_positive": eps_vec  # None if use_dynamic_eps=False
    }
    print(f"Original healthy: {healthy_count} | sick: {sick_count}")
    print(f"Added synthetic healthy: {samples_to_add}")     
    print(f"Balanced healthy: {info['healthy_count']} | sick: {info['sick_count']}")
    print(f"Shapes: X={X_balanced.shape}, y={y_balanced.shape}")

    
    return X_balanced, y_balanced, synthetic_prop, info

