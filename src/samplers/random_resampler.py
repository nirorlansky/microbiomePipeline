import numpy as np
import pandas as pd
from math import ceil
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling.base import BaseOverSampler
from pkg.globals import *


class ResampleSamples(BaseOverSampler):

    def __init__(self, sampling_strategy=0.9, 
                 random_state=42, 
                 minority_label=HEALTHY):

        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.minority_label = minority_label
    
    def _fit_resample(self, X, y):
        # randomly duplicate minority class samples until it reaches the desired minority share

        return self.random_oversample_to_minority_share(
            X, y,
            target_ratio=self.sampling_strategy,
            random_state=self.random_state,
            minority_label=self.minority_label
        )
    # ---------- helpers ----------
    def random_oversample_to_minority_share(
        self,
        X,
        y,
        target_ratio: float = 0.9,
        random_state: int = 42,
        minority_label: int = HEALTHY
    ):
        """
        Randomly oversample class `minority_label` until it reaches `minority_share` of the dataset.
        receive X, y (binary) as numpy arrays
        """
        
        # randomly choose samples from minority class and add them to X, y until minority class reaches desired share

        sick_count = int(np.sum(y == SICK))
        if sick_count == 0 or target_ratio <= 0:
            return X, y

        H_target = target_ratio * sick_count

        healthy_count = int(np.sum(y == minority_label))
        samples_to_add = H_target - healthy_count
        if samples_to_add <= 0:
            print(f"[RANDOM] No need to add samples (already >= target share {target_ratio:.2f}).")
            return X, y

        rng = np.random.default_rng(random_state)
        minority_indices = np.where(y == minority_label)[0]
        chosen_indices = rng.choice(minority_indices, size=samples_to_add, replace=True)

        X_to_add = X[chosen_indices]
        y_to_add = y[chosen_indices]
        X_balanced = np.vstack([X, X_to_add])
        y_balanced = np.hstack([y, y_to_add])

        H_final = int(np.sum(y_balanced == minority_label))
        S_final = int(np.sum(y_balanced != minority_label))
        print(f"[RANDOM] Added synthetic healthy: {samples_to_add} | Final H/S: {H_final}/{S_final} (~{H_final/(H_final+S_final):.1%} healthy)")
        return X_balanced, y_balanced

