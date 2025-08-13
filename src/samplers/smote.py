import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling.base import BaseOverSampler
import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))
from pkg.globals import *

class SmoteOverSampler(BaseOverSampler):
    """
    Oversample a chosen class (default=0) to reach a desired share of the dataset.

    Modes (mutually exclusive):
      1) Regular SMOTE                               -> threshold is None/0, use_feature_eps=False, preserve_zero_pattern=False
      2) Regular SMOTE + global threshold            -> threshold > 0
      3) Custom SMOTE with preserve_zero_pattern     -> preserve_zero_pattern=True (no threshold / no eps)
      4) Regular SMOTE + per-feature epsilon         -> use_feature_eps=True (no threshold / no preserve)

    Parameters
    ----------
    sampling_strategy : float
        Desired final share of the chosen minority_label in the whole dataset (e.g., 0.9 for 90%).
    random_state : int
        RNG seed.
    k_neighbors : int or None
        k for neighbor search (SMOTE and custom mode). Auto-chosen if None.
    threshold : float or None
        If >0, zero out values < threshold after regular SMOTE (mode 2).
    preserve_zero_pattern : bool
        If True, use custom neighbor pick that maximizes shared zero positions (mode 3).
    use_feature_eps : bool
        If True, compute per-feature epsilon (min positive >0 before oversampling) and zero out values < eps_j after regular SMOTE (mode 4).
    minority_label : int
        Which class to oversample towards the target share.
    """

    def __init__(
        self,
        sampling_strategy: float = 0.9,
        random_state: int = 42,
        k_neighbors: int | None = 5,
        threshold: float | None = None,
        preserve_zero_pattern: bool = False,
        use_feature_eps: bool = False,
        minority_label: int = 0,
    ):
        super().__init__(sampling_strategy=sampling_strategy)
        self.random_state = random_state
        self.k_neighbors = k_neighbors
        self.threshold = threshold
        self.preserve_zero_pattern = preserve_zero_pattern
        self.use_feature_eps = use_feature_eps
        self.minority_label = minority_label

    # --------------------------- utils ---------------------------
    @staticmethod
    def _to_numpy(x):
        return x.to_numpy(copy=False) if hasattr(x, "to_numpy") else np.asarray(x)

    @staticmethod
    def _take_rows(X, mask):
        return X.loc[mask] if hasattr(X, "loc") else X[mask]

    @staticmethod
    def _concat_rows(X, X_new):
        if hasattr(X, "loc"):
            return pd.concat([X, pd.DataFrame(X_new, columns=X.columns)], ignore_index=True)
        return np.vstack([X, X_new])

    @staticmethod
    def _inplace_zero_lt(X, thresh):
        if hasattr(X, "to_numpy"):
            arr = X.to_numpy(copy=False)
            arr[arr < thresh] = 0
        else:
            X[X < thresh] = 0

    @staticmethod
    def _inplace_zero_mask(X, mask):
        if hasattr(X, "to_numpy"):
            arr = X.to_numpy(copy=False)
            arr[mask] = 0
        else:
            X[mask] = 0

    # ---------------------- core resampling ----------------------
    def _fit_resample(self, X, y):
        rng = np.random.default_rng(self.random_state)

        # y -> numeric (handle "0"/"1" strings etc.)
        y = self._to_numpy(y)
        if y.dtype.kind in {"U", "S", "O"}:
            try:
                y = y.astype(int)
            except ValueError:
                y = LabelEncoder().fit_transform(y)

        # exclusivity checks
        if self.preserve_zero_pattern and (self.threshold is not None and self.threshold > 0):
            raise ValueError("threshold cannot be used when preserve_zero_pattern=True.")
        if self.use_feature_eps and (self.threshold is not None and self.threshold > 0):
            raise ValueError("threshold cannot be used when use_feature_eps=True.")
        if self.use_feature_eps and self.preserve_zero_pattern:
            raise ValueError("use_feature_eps cannot be combined with preserve_zero_pattern=True.")

        # counts & targets w.r.t. chosen minority_label
        minority_mask = (y == self.minority_label)
        n_min = int(minority_mask.sum())
        n_maj = int((~minority_mask).sum())

        ratio = float(self.sampling_strategy) / (1.0 - float(self.sampling_strategy))
        target_min = int(np.ceil(ratio * n_maj))
        n_to_add = max(0, target_min - n_min)

        if n_to_add == 0:
            print(f"No samples to add (minority '{self.minority_label}' count: {n_min}, target: {target_min}). Returning original data.")
            X_out = X.copy() if hasattr(X, "copy") else np.array(X, copy=True)
            return X_out, y.copy()

        # choose k
        if self.k_neighbors is None:
            k = max(1, min(5, n_min - 1))
        else:
            k = min(self.k_neighbors, max(1, n_min - 1))

        # (4) per-feature eps (computed BEFORE adding samples)
        eps_vec = None
        if self.use_feature_eps:
            A = self._to_numpy(X)
            pos_mask = (A > 0)
            with np.errstate(invalid="ignore"):
                col_min_pos = np.where(pos_mask, A, np.inf).min(axis=0)
            col_has_pos = pos_mask.any(axis=0)
            eps_vec = np.where(col_has_pos, col_min_pos, 0.0)

        # ---------------- mode 3: custom neighbor by zero-pattern ----------------
        if self.preserve_zero_pattern:
            X_min = self._take_rows(X, minority_mask)
            Xm = self._to_numpy(X_min)

            if Xm.shape[0] < 2:
                print(f"preserve_zero_pattern=True but minority '{self.minority_label}' has <2 samples (n_min={Xm.shape[0]}). Returning original data.")
                X_out = X.copy() if hasattr(X, "copy") else np.array(X, copy=True)
                return X_out, y.copy()

            nn = NearestNeighbors(n_neighbors=k + 1)  # +1 to include self, we'll drop it
            nn.fit(Xm)
            distances, indices = nn.kneighbors(Xm)
            distances, indices = distances[:, 1:], indices[:, 1:]  # drop self
            zero_mask_min = (Xm == 0)

            def pick_neighbor_for_base(i: int) -> int:
                neigh_rows = indices[i]
                neigh_dists = distances[i]
                base_zero = zero_mask_min[i]
                neigh_zeros = zero_mask_min[neigh_rows]
                overlaps = (neigh_zeros & base_zero).sum(axis=1)
                best = np.where(overlaps == overlaps.max())[0]
                if len(best) == 1:
                    return neigh_rows[best[0]]
                return neigh_rows[best[np.argmin(neigh_dists[best])]]

            new_samples = np.empty((n_to_add, Xm.shape[1]), dtype=Xm.dtype)
            base_seq = np.tile(np.arange(n_min), int(np.ceil(n_to_add / n_min)))[:n_to_add]
            for t, i in enumerate(base_seq):
                j = pick_neighbor_for_base(i)
                delta = rng.random()
                new_samples[t] = Xm[i] + delta * (Xm[j] - Xm[i])

            X_res = self._concat_rows(X, new_samples)
            y_res = np.concatenate([y, np.full(n_to_add, self.minority_label, dtype=y.dtype)])

            print(
                f"Added {n_to_add} samples to class {self.minority_label} (before: {n_min}, after: {n_min + n_to_add}); "
                f"k={k}, preserve_zero_pattern=True"
            )
            return X_res, y_res

        # ---------------- modes 1/2/4: regular SMOTE ----------------
        # Force SMOTE to oversample the chosen class to 'target_min' (dict form)
        smote = SMOTE(
            sampling_strategy={self.minority_label: target_min},
            k_neighbors=k,
            random_state=self.random_state
        )
        X_res, y_res = smote.fit_resample(X, y)

        # (2) global threshold
        if self.threshold is not None and self.threshold > 0 and not self.use_feature_eps:
            self._inplace_zero_lt(X_res, self.threshold)
            print(
                f"Added {n_to_add} samples to class {self.minority_label} (before: {n_min}, after: {n_min + n_to_add}); "
                f"k={k}, threshold={self.threshold}"
            )
            return X_res, y_res

        # (4) per-feature epsilon
        if self.use_feature_eps:
            arr = self._to_numpy(X_res)
            mask = arr < eps_vec
            self._inplace_zero_mask(X_res, mask)
            print(
                f"Added {n_to_add} samples to class {self.minority_label} (before: {n_min}, after: {n_min + n_to_add}); "
                f"k={k}, applied per-feature eps zeroing"
            )
            return X_res, y_res

        # (1) plain regular SMOTE
        print(
            f"Added {n_to_add} samples to class {self.minority_label} (before: {n_min}, after: {n_min + n_to_add}); "
            f"k={k}, regular SMOTE"
        )
        return X_res, y_res
