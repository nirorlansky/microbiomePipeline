import numpy as np
import dirichlet
from math import ceil
from imblearn.over_sampling.base import BaseOverSampler
import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from pkg.globals import *
from samplers.evaluation import eval_healthy_and_synthetic

class DirichletSampler(BaseOverSampler):

    def __init__(self, 
                 sampling_strategy=9,           # ← integer factor k (≥1)
                 random_state=42, 
                 jitter=0.00, 
                 method="mle",
                 use_dynamic_eps=False,
                 orders_below=3,
                 fallback=1e-12,
                 scale_factor=0.1,
                 eval=False,
                 method_string=""):

        # accept integer-only factor and map to p=k/(1+k) for BaseOverSampler
        if not (isinstance(sampling_strategy, int) and sampling_strategy >= 1):
            raise ValueError(f"'sampling_strategy' must be an integer factor ≥ 1, got {sampling_strategy}")
        self.factor = int(sampling_strategy)
        p = self.factor / (1.0 + self.factor)

        super().__init__(sampling_strategy=p)
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
            method=self.method,                 # or "mom"
            target_ratio=self.sampling_strategy,  # kept for API compatibility (ignored internally)
            seed=self.random_state,
            use_dynamic_eps=self.use_dynamic_eps,
            orders_below=self.orders_below,
            fallback=self.fallback,
            scale_factor=self.scale_factor,
        )

    # ---------- helpers ----------

    def replace_zeros_with_dynamic_eps(self, X, orders_below=3, fallback=1e-12):
        pos = np.where(X > 0, X, np.inf)
        col_min_pos = pos.min(axis=0)
        has_col_pos = np.isfinite(col_min_pos)
        global_min_pos = col_min_pos[has_col_pos].min() if np.any(has_col_pos) else np.inf
        eps_vec = col_min_pos.copy()
        eps_vec[~has_col_pos] = global_min_pos if np.isfinite(global_min_pos) else fallback
        eps_repl = eps_vec * (10.0 ** (-orders_below))
        X2 = np.where(X <= 0, eps_repl, X)
        rs = X2.sum(axis=1, keepdims=True)
        rs = np.clip(rs, fallback, None)
        X2 = X2 / rs
        return X2, eps_vec

    def zero_below_eps_and_renormalize(self, synth_prop, eps_vec):
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
        target_ratio=9.0,       # kept for signature compatibility; not used for target calc
        seed=42,
        use_dynamic_eps=False,
        orders_below=3,
        fallback=1e-12,
        scale_factor=0.3,      # scale concentration by this factor to increase dispersion 
    ):
        """
        Balance by adding synthetic Healthy samples drawn from a Dirichlet fitted on the Healthy group.
        Assumes each row of X_train already sums to ~1.
        """
        rng = np.random.default_rng(seed)

        healthy_count = int(np.sum(y_train == HEALTHY))
        sick_count    = int(np.sum(y_train == SICK))

        # ---- change: integer factor k ----
        target_healthy = int(ceil(self.factor * sick_count))
        samples_to_add = max(0, target_healthy - healthy_count)
        if samples_to_add == 0:
            print("No samples to add; already at or above target factor.")
            return X_train, y_train

        healthy_data = X_train[y_train == HEALTHY]
        if healthy_data.shape[0] == 0:
            print("No healthy samples in this fold; skipping oversampling.")
            return X_train, y_train

        healthy_prop = healthy_data  ##### (assumes rows sum to 1)

        eps_vec = None
        if use_dynamic_eps:
            healthy_prop, eps_vec = self.replace_zeros_with_dynamic_eps(
                healthy_prop, orders_below=orders_below, fallback=fallback
            )
        elif method.lower() == "mle":
            healthy_prop = np.clip(healthy_prop, fallback, None)
            healthy_prop = healthy_prop / healthy_prop.sum(axis=1, keepdims=True)

        method_l = method.lower()
        if method_l == "mom":
            mu  = healthy_prop.mean(axis=0)
            var = healthy_prop.var(axis=0, ddof=1)
            var = np.clip(var, fallback, None)
            num = float(np.sum(mu * (1.0 - mu)))
            den = float(np.sum(var))
            alpha0 = max(num / den - 1.0, 1e-3)
            alpha_vec = mu * alpha0
        elif method_l == "mle":
            alpha_vec = dirichlet.mle(healthy_prop)
        else:
            raise ValueError("method must be 'mle' or 'mom'.")

        tau = float(getattr(self, "tau", scale_factor))
        alpha_vec = np.maximum(1e-8, tau) * alpha_vec  # keep alphas positive

        synth_prop = rng.dirichlet(alpha_vec, size=samples_to_add)

        if use_dynamic_eps and eps_vec is not None:
            synth_prop = self.zero_below_eps_and_renormalize(synth_prop, eps_vec)

        if self.eval:
            eval_healthy_and_synthetic(
                healthy_data, synth_prop, eps_vec=eps_vec, method_string=self.method_string, fraction=self.factor
            )

        X_balanced = np.vstack([X_train, synth_prop])
        y_balanced = np.hstack([y_train, np.full(samples_to_add, HEALTHY, dtype=y_train.dtype)])

        info = {
            "method": method_l,
            "healthy_count": int(np.sum(y_balanced == HEALTHY)),
            "sick_count": int(np.sum(y_balanced == SICK)),
            "target_healthy": target_healthy,
            "alpha": alpha_vec,
            "eps_vec_min_positive": eps_vec
        }

        print(f"[{method_l.upper()}] Added synthetic healthy: {samples_to_add} | Final H/S: {info['healthy_count']}/{info['sick_count']}")
        return X_balanced, y_balanced
