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
        rem = 1.0 - X.sum(axis=1, keepdims=True)
        if self.clip: rem = np.clip(rem, 0.0, 1.0)
        return np.hstack([X, rem])