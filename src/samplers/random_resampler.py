import numpy as np
import pandas as pd
from math import ceil
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling.base import BaseOverSampler
from pkg.globals import *


class ResampleSamples(BaseOverSampler):

    def __init__(self, sampling_strategy=9, 
                 random_state=42, 
                 minority_label=HEALTHY):

        # support integer factor k (≥1) only; map to p=k/(1+k) for BaseOverSampler
        if not (isinstance(sampling_strategy, int) and sampling_strategy >= 1):
            raise ValueError(f"'sampling_strategy' must be an integer factor ≥ 1, got {sampling_strategy}")
        self.factor = int(sampling_strategy)  # keep the integer factor for internal target calc
        p = self.factor / (1.0 + self.factor)

        super().__init__(sampling_strategy=p)
        self.random_state = random_state
        self.minority_label = minority_label
    
    def _fit_resample(self, X, y):
        # randomly duplicate minority class samples until it reaches the desired minority share

        return self.random_oversample_to_minority_share(
            X, y,
            random_state=self.random_state,
            minority_label=self.minority_label
        )

    # ---------- helpers ----------
    def random_oversample_to_minority_share(
        self,
        X,
        y,
        random_state: int = 42,
        minority_label: int = HEALTHY
    ):
        """
        Randomly oversample class `minority_label` until it reaches the desired share.
        Here, the desired share is driven by integer factor k stored in self.factor,
        i.e., target Healthy count H_target = ceil(k * S), where S = current majority count.
        receive X, y (binary) as numpy arrays
        """
        
        # randomly choose samples from minority class and add them to X, y until minority class reaches desired share

        sick_count = int(np.sum(y == SICK))
        if sick_count == 0:
            return X, y

        # Use the integer factor: H_target = ceil(k * S)
        H_target = int(ceil(self.factor * sick_count))

        healthy_count = int(np.sum(y == minority_label))
        samples_to_add = H_target - healthy_count
        if samples_to_add <= 0:
            print(f"[RANDOM] No need to add samples (already >= {self.factor}x majority).")
            return X, y

        rng = np.random.default_rng(random_state)
        minority_indices = np.where(y == minority_label)[0]
        chosen_indices = rng.choice(minority_indices, size=int(samples_to_add), replace=True)

        X_to_add = X[chosen_indices]
        y_to_add = y[chosen_indices]
        X_balanced = np.vstack([X, X_to_add])
        y_balanced = np.hstack([y, y_to_add])

        H_final = int(np.sum(y_balanced == minority_label))
        S_final = int(np.sum(y_balanced != minority_label))
        print(f"[RANDOM] Added synthetic healthy: {samples_to_add} | Final H/S: {H_final}/{S_final} (~{H_final/(H_final+S_final):.1%} healthy)")
        return X_balanced, y_balanced
