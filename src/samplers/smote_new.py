import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling.base import BaseOverSampler
from pkg.globals import *


class SmoteSampler(BaseOverSampler):

    def __init__(self, 
                minority_share: float = 0.9,
                random_state: int = 42,
                k_neighbors: int | None = None,
                threshold: float | None = None,
                preserve_zero_pattern: bool = False,
                use_feature_eps: bool = False,
                minority_label: int = HEALTHY
                ):
        super().__init__(sampling_strategy=minority_share) # target minority share
        self.minority_share = minority_share
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.threshold = threshold
        self.preserve_zero_pattern = preserve_zero_pattern
        self.use_feature_eps = use_feature_eps
        self.minority_label = minority_label

    def _fit_resample(self, X, y):
        return self.smote_to_minority_share(
            X, y,
            minority_share=self.minority_share,
            random_state=self.random_state,
            k_neighbors=self.k_neighbors,
            threshold=self.threshold,
            preserve_zero_pattern=self.preserve_zero_pattern,
            use_feature_eps=self.use_feature_eps,
            minority_label=self.minority_label
        )


    def smote_to_minority_share(
        self,
        X_train,
        y_train,
        minority_share: float = 0.9,
        random_state: int = 42,
        k_neighbors: int | None = None,
        threshold: float | None = None,
        preserve_zero_pattern: bool = False,
        use_feature_eps: bool = False,
        minority_label: int = 0
    ):
        """
        Oversample class `minority_label` until it reaches `minority_share` of the dataset.

        Modes (mutually exclusive):
        1) Regular SMOTE
        2) Regular SMOTE + global threshold
        3) Custom SMOTE with preserve_zero_pattern (no threshold)
        4) Regular SMOTE + per-feature epsilon (no threshold)
        """

        # --- Convert y_train to numeric 0/1 ---
        if hasattr(y_train, "to_numpy"):  # pandas
            y = y_train.to_numpy()
        else:  # already numpy
            y = np.array(y_train)

        # if y is strings or objects -> convert to int or encode
        if y.dtype.kind in {"U", "S", "O"}:
            try:
                y = y.astype(int)
            except ValueError:
                y = LabelEncoder().fit_transform(y)

        # Validations of parameters - cannot combine certain modes
        if preserve_zero_pattern and (threshold is not None and threshold > 0):
            raise ValueError("threshold cannot be used when preserve_zero_pattern=True.")
        if use_feature_eps and (threshold is not None and threshold > 0):
            raise ValueError("threshold cannot be used when use_feature_eps=True.")
        if use_feature_eps and preserve_zero_pattern:
            raise ValueError("use_feature_eps cannot be combined with preserve_zero_pattern=True.")

        rng = np.random.default_rng(random_state)

        # count how many healthy and sick samples, compute how many to add
        minority_mask = (y == minority_label)
        n_min = int(minority_mask.sum())
        majority_mask = ~minority_mask
        n_maj = int(majority_mask.sum())
        ratio = minority_share / minority_share
        target_min = int(np.ceil(ratio * n_maj))
        n_to_add = max(0, target_min - n_min)
        if n_to_add == 0:
            print(f"No samples to add (minority '{minority_label}' count: {n_min}, target: {target_min}). Returning original data.")
            X_out = X_train.copy() if hasattr(X_train, "copy") else np.array(X_train, copy=True)
            return X_out, y.copy()

        # Determine k_neighbors- if not set, use min(5, n_min-1)
        if k_neighbors is None:
            k_neighbors = max(1, min(5, n_min - 1))
        else:
            k_neighbors = min(k_neighbors, max(1, n_min - 1))

        # Handle per-feature eps
        eps_vec = None
        if use_feature_eps:
            if hasattr(X_train, "to_numpy"):
                A = X_train.to_numpy(copy=False)
            else:
                A = np.asarray(X_train)
            pos_mask = (A > 0)
            with np.errstate(invalid="ignore"):
                col_min_pos = np.where(pos_mask, A, np.inf).min(axis=0) # min positive per column
            col_has_pos = pos_mask.any(axis=0) # which columns have any positive values
            eps_vec = np.where(col_has_pos, col_min_pos, 0.0) # set eps=0 for columns with no positive values- it's not suppose to happen, but just in case

        # MODE 3: custom preserve_zero_pattern
        if preserve_zero_pattern:
            X_min = X_train.loc[minority_mask] if hasattr(X_train, "loc") else X_train[minority_mask] # x_min = samples of minority class
            Xm = X_min.to_numpy(copy=False) if hasattr(X_min, "to_numpy") else np.asarray(X_min) # convert to numpy (if it isn't already)
            # find k nearest neighbors of each minority sample (in minority set)
            nn = NearestNeighbors(n_neighbors=k_neighbors + 1) # +1 for self
            nn.fit(Xm)
            distances, indices = nn.kneighbors(Xm) # 2 vectors: distances from each sample to its neighbors, and their indices
            distances, indices = distances[:, 1:], indices[:, 1:] # remove self (first column)

            zero_mask_min = (Xm == 0) # 1 if a feature is zero (in a minority sample)

            def pick_neighbor_for_base(i):
                neigh_rows = indices[i]
                neigh_dists = distances[i]
                base_zero = zero_mask_min[i]
                neigh_zeros = zero_mask_min[neigh_rows]
                overlaps = (neigh_zeros & base_zero).sum(axis=1)
                best = np.where(overlaps == overlaps.max())[0]
                if len(best) == 1:
                    return neigh_rows[best[0]]
                return neigh_rows[best[np.argmin(neigh_dists[best])]]

            new_samples = np.empty((n_to_add, Xm.shape[1]), dtype=Xm.dtype) # matrix to hold the new samples
            base_seq = np.tile(np.arange(n_min), int(np.ceil(n_to_add / n_min)))[:n_to_add] # which minority samples to use as bases for new samples
            for t, i in enumerate(base_seq): # for each new sample to create
                j = pick_neighbor_for_base(i)
                delta = rng.random()
                new_samples[t] = Xm[i] + delta * (Xm[j] - Xm[i])

            if hasattr(X_train, "loc"):
                X_new = pd.DataFrame(new_samples, columns=X_train.columns)
                X_res = pd.concat([X_train, X_new], ignore_index=True)
            else:
                X_res = np.vstack([X_train, new_samples])
            y_res = np.concatenate([y, np.full(n_to_add, minority_label, dtype=y.dtype)])

            print(
                f"Added {n_to_add} samples to class {minority_label} (before: {n_min}, after: {n_min + n_to_add}); "
                f"k={k_neighbors}, preserve_zero_pattern=True"
            )
            return X_res, y_res

        # MODES 1/2/4: regular SMOTE
        smote = SMOTE(
            sampling_strategy=ratio,
            k_neighbors=k_neighbors,
            random_state=random_state
        )
        X_res, y_res = smote.fit_resample(X_train, y)

        if threshold is not None and threshold > 0 and not use_feature_eps:
            arr = X_res.to_numpy(copy=False) if hasattr(X_res, "to_numpy") else X_res
            arr[arr < threshold] = 0
            print(
                f"Added {n_to_add} samples to class {minority_label} (before: {n_min}, after: {n_min + n_to_add}); "
                f"k={k_neighbors}, threshold={threshold}"
            )
            return X_res, y_res

        if use_feature_eps:
            arr = X_res.to_numpy(copy=False) if hasattr(X_res, "to_numpy") else X_res
            mask = arr < eps_vec
            arr[mask] = 0
            print(
                f"Added {n_to_add} samples to class {minority_label} (before: {n_min}, after: {n_min + n_to_add}); "
                f"k={k_neighbors}, applied per-feature eps zeroing"
            )
            return X_res, y_res

        print(
            f"Added {n_to_add} samples to class {minority_label} (before: {n_min}, after: {n_min + n_to_add}); "
            f"k={k_neighbors}, regular SMOTE"
        )
        return X_res, y_res