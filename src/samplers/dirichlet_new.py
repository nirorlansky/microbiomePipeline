import numpy as np
import dirichlet
from imblearn.over_sampling.base import BaseOverSampler
import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from pkg.globals import *
from samplers.evaluation import eval_healthy_and_synthetic

class DirichletSampler(BaseOverSampler):

    def __init__(self, 
                 sampling_strategy=0.9, 
                 random_state=42, 
                 jitter=0.00, 
                 method="mle",
                 use_dynamic_eps=False,
                 orders_below=3,
                 fallback=1e-12,
                 scale_factor=0.3,
                 eval=False,
                 method_string=""):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.jitter = jitter
        self.method = method
        self.use_dynamic_eps = use_dynamic_eps
        self.orders_below = orders_below
        self.fallback = fallback
        self.eval = eval 
        self.method_string = method_string
        self.scale_factor = scale_factor 

    def _fit_resample(self, X, y):
        return self.balance_healthy_dirichlet(
            X, y,
            method=self.method ,  # or "mom"
            target_ratio=self.sampling_strategy*10,  # target ratio of Healthy to Sick
            seed=self.random_state,
            use_dynamic_eps=self.use_dynamic_eps,
            orders_below=self.orders_below,
            fallback=self.fallback,
            scale_factor=self.scale_factor,
        )

    # ---------- helpers ----------

    def replace_zeros_with_dynamic_eps(self, X, orders_below=3, fallback=1e-12):
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

        rs = X2.sum(axis=1, keepdims=True) # sum of each row
        rs = np.clip(rs, fallback, None)  # avoid divide-by-zero
        X2 = X2 / rs # renormalize rows to sum to 1
        return X2, eps_vec

    def zero_below_eps_and_renormalize(self, synth_prop, eps_vec):
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

    # ---------- unified balancer ----------

    def balance_healthy_dirichlet(self,
        X_train,
        y_train,
        method="mle",           # "mle" or "mom"
        target_ratio=9.0,
        seed=42,
        use_dynamic_eps=False,
        orders_below=3,
        fallback=1e-12,
        scale_factor=0.3,      # scale concentration by this factor to increase dispersion 
    ):
        """
        Balance by adding synthetic Healthy samples drawn from a Dirichlet fitted on the Healthy group.
        Assumes each row of X_train already sums to ~1.

        method:
        - "mle": estimate alpha via maximum likelihood (ericsuh/dirichlet)
        - "mom": estimate alpha via method-of-moments

        use_dynamic_eps=True:
        - Before fitting: replace zeros with dynamic per-feature eps and renormalize
        - After sampling: zero-out values below ORIGINAL per-feature min-positive (eps_vec) and renormalize

        Returns:
        X_balanced: np.ndarray (n_samples + added, n_features)
        y_balanced: np.ndarray (n_samples + added,)
        synth_prop: np.ndarray of synthetic Healthy samples (proportions)
        info: dict with counts, target, alpha, and eps_vec (if used)
        """
        rng = np.random.default_rng(seed)

        healthy_count = int(np.sum(y_train == HEALTHY))
        sick_count    = int(np.sum(y_train == SICK))
        target_healthy = int(target_ratio * sick_count)
        samples_to_add = max(0, target_healthy - healthy_count)
        if samples_to_add == 0:
            print("No samples to add; already at or above target ratio.")
            return X_train, y_train

        # Healthy proportions (normalize just in case)
        healthy_data = X_train[y_train == HEALTHY]
        if healthy_data.shape[0] == 0:
            print("No healthy samples in this fold; skipping oversampling.")
            return X_train, y_train

        # rs = healthy_data.sum(axis=1, keepdims=True)
        # rs = np.clip(rs, 1e-12, None)
        # healthy_prop = healthy_data / rs # should already be 1
        healthy_prop = healthy_data #####

        eps_vec = None
        if use_dynamic_eps:
            healthy_prop, eps_vec = self.replace_zeros_with_dynamic_eps(
                healthy_prop, orders_below=orders_below, fallback=fallback
            )
        elif method.lower() == "mle":
            # Minimal static clip just to avoid log(0) in MLE
            healthy_prop = np.clip(healthy_prop, fallback, None)
            healthy_prop = healthy_prop / healthy_prop.sum(axis=1, keepdims=True)

        method_l = method.lower()
        if method_l == "mom":
            # Method of Moments on proportions
            # --- MoM (fixed) ---
            mu  = healthy_prop.mean(axis=0)                       # mean per feature
            var = healthy_prop.var(axis=0, ddof=1)                # sample variance
            var = np.clip(var, fallback, None)                    # avoid zeros
            num = float(np.sum(mu * (1.0 - mu)))
            den = float(np.sum(var))
            alpha0 = max(num / den - 1.0, 1e-3)                   # guard lower bound
            alpha_vec = mu * alpha0

        elif method_l == "mle":
            # Maximum Likelihood on proportions
            alpha_vec = dirichlet.mle(healthy_prop)
        else:
            raise ValueError("method must be 'mle' or 'mom'.")

        # Scale concentration to increase dispersion but keep the mean (mu stays the same)
        tau = float(getattr(self, "tau", scale_factor))  
        alpha_vec = np.maximum(1e-8, tau) * alpha_vec  # keep alphas positive

        # Sample synthetic proportions
        synth_prop = rng.dirichlet(alpha_vec, size=samples_to_add)


        # Optional post-process: zero below ORIGINAL per-feature eps, then renormalize
        if use_dynamic_eps and eps_vec is not None:
            synth_prop = self.zero_below_eps_and_renormalize(synth_prop, eps_vec)

        if self.eval:
            eval_healthy_and_synthetic(
                healthy_data, synth_prop, eps_vec=eps_vec, method_string=self.method_string
            )

        # Append (rows all sum to 1)
        X_balanced = np.vstack([X_train, synth_prop])
        y_balanced = np.hstack([y_train, np.full(samples_to_add, HEALTHY, dtype=y_train.dtype)])

        info = {
            "method": method_l,
            "healthy_count": int(np.sum(y_balanced == HEALTHY)),
            "sick_count": int(np.sum(y_balanced == SICK)),
            "target_healthy": target_healthy,
            "alpha": alpha_vec,
            "eps_vec_min_positive": eps_vec  # None if use_dynamic_eps=False
        }

        print(f"[{method_l.upper()}] Added synthetic healthy: {samples_to_add} | Final H/S: {info['healthy_count']}/{info['sick_count']}")
        return X_balanced, y_balanced

