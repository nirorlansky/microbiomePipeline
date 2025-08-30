import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class AddRemainder(BaseEstimator, TransformerMixin):
    """Append 1 - sum(row) so each sample sums to 1 after selection."""
    def __init__(self, clip=True):
        self.clip = clip
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        rem = 1.0 - X.sum(axis=1, keepdims=True) # for each row, remainder to sum to 1
        # if any rem < 0, raise error
        if np.any(rem < 0):
            raise ValueError("Some rows sum to >1, cannot add remainder column.")
        if np.any(rem > 1):
            raise ValueError("Some rows sum to <1, cannot add remainder column.")
        if np.any(rem == 1):
            print("Warning: Some rows sum to 0, remainder column will be 1.")
        return np.hstack([X, rem])