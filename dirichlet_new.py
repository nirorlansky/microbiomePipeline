import numpy as np
import dirichlet
import pandas as pd

# ====== Input ======
# X_train: numpy array of shape (n_samples, n_features) - each row sums to 1
# y_train: numpy array of shape (n_samples,) - values are 'H' (Healthy) or 'S' (Sick)
# Below is an example dataset for demonstration purposes

# np.random.seed(42)
# n_features = 5
# X_train = np.random.dirichlet(alpha=[2, 5, 3, 4, 6], size=1000)
# y_train = np.array(['H'] * 100 + ['S'] * 900)  # 100 healthy, 900 sick

def balance_healthy_samples(X_train, y_train, target_ratio=9.0, seed=42):
    """
    Balance by adding synthetic Healthy samples drawn from a Dirichlet
    fitted via Method of Moments (MoM).
    """
    rng = np.random.default_rng(seed)

    # ====== Step 1: Count healthy and sick samples ======
    healthy_count = int(np.sum(y_train == 'H'))
    sick_count = int(np.sum(y_train == 'S'))

    # ====== Step 2: Target Healthy count and how many to add (ensure ints) ======
    target_healthy = int(target_ratio * sick_count)  # cast to int
    samples_to_add = max(0, target_healthy - healthy_count)  # ensure non-negative int

    if samples_to_add == 0:
        # Already at or above the target ratio
        print("No samples to add; already at or above target ratio.")
        return X_train, y_train

    # ====== Step 3: Extract only the healthy samples ======
    healthy_data = X_train[y_train == 'H']

    # ====== Step 4: Compute mean (mu) and variance (Var) for each feature ======
    mu = healthy_data.mean(axis=0)
    var = healthy_data.var(axis=0)
    # Protect against zero-variance columns
    var = np.clip(var, 1e-12, None)

    # ====== Step 5: Estimate alpha0 using Method of Moments ======
    alpha0_estimates = (mu * (1 - mu) / var) - 1
    mask = np.isfinite(alpha0_estimates) & (alpha0_estimates > 0)
    if not np.any(mask):
        raise ValueError("MoM failed: no positive/finite alpha0 estimates. Consider MLE fallback.")
    alpha0 = float(np.mean(alpha0_estimates[mask]))

    # ====== Step 6: Compute alpha vector ======
    alpha_vector = mu * alpha0
    print(f"Estimated alpha (MoM): {alpha_vector}")

    # ====== Step 7: Generate synthetic healthy samples (size must be int) ======
    synthetic_healthy = rng.dirichlet(alpha_vector, size=samples_to_add)

    # ====== Step 8: Append synthetic samples to the dataset ======
    X_balanced = np.vstack([X_train, synthetic_healthy])
    y_balanced = np.hstack([y_train, np.array(['H'] * samples_to_add, dtype=y_train.dtype)])

    # ====== Step 9: Report ======
    print(f"Original healthy samples: {healthy_count}")
    print(f"Original sick samples: {sick_count}")
    print(f"Synthetic healthy samples added: {samples_to_add}")
    print(f"Balanced healthy samples: {np.sum(y_balanced == 'H')}")
    print(f"Balanced sick samples: {np.sum(y_balanced == 'S')}")
    print(f"Shapes: X={X_balanced.shape}, y={y_balanced.shape}")

    return X_balanced, y_balanced



def balance_with_dirichlet_mle(X_train, y_train, target_ratio=9.0, seed=42):
    """
    X_train: (n_samples, n_features) each row sums to 1 (compositional)
    y_train: (n_samples,) labels 'H' (healthy) or 'S' (sick)
    target_ratio: desired Healthy:Sick ratio after augmentation (e.g., 9.0 means H_final = 9 * S_count)
    """
    rng = np.random.default_rng(seed)

    # --- Count classes ---
    healthy_count = np.sum(y_train == 'H')
    sick_count = np.sum(y_train == 'S')

    # --- Compute how many healthy samples to add to reach target_ratio ---
    target_healthy = int(target_ratio * sick_count)
    samples_to_add = max(0, target_healthy - healthy_count)
    if samples_to_add == 0:
        # Already at or above the target ratio
        return X_train, y_train, np.array([]), {
            "healthy_count": healthy_count,
            "sick_count": sick_count,
            "target_healthy": target_healthy,
            "alpha": None
        }

    # --- Extract healthy data only ---
    healthy_data = X_train[y_train == 'H']

    # --- Safety: clip tiny zeros and renormalize (MLE needs strictly > 0 and rows sum to 1) ---
    eps = 1e-12
    healthy_data = np.clip(healthy_data, eps, None)
    healthy_data = healthy_data / healthy_data.sum(axis=1, keepdims=True)

    # --- MLE for Dirichlet parameters on the healthy group ---
    # dirichlet.mle returns the alpha vector
    alpha_hat = dirichlet.mle(healthy_data)

    # --- Generate synthetic healthy samples using the estimated alpha ---
    synthetic_healthy = rng.dirichlet(alpha_hat, size=samples_to_add)

    # --- Append back to dataset ---
    X_balanced = np.vstack([X_train, synthetic_healthy])
    y_balanced = np.hstack([y_train, np.full(samples_to_add, 'H', dtype=y_train.dtype)])

    info = {
        "healthy_count": int(np.sum(y_balanced == 'H')),
        "sick_count": int(np.sum(y_balanced == 'S')),
        "target_healthy": target_healthy,
        "alpha": alpha_hat
    }

    print("Final healthy count:", info["healthy_count"])
    print("Final sick count:", info["sick_count"])
    print("Target healthy:", info["target_healthy"])
    print("Estimated alpha (MLE):", info["alpha"])

    return X_balanced, y_balanced, synthetic_healthy, info


