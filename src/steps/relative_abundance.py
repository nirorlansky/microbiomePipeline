import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class RelativeAbundance(BaseEstimator, TransformerMixin):
    def __init__(self, eps=1e-12):
        self.eps = eps
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        s = X.sum(axis=1, keepdims=True) # sum of each row
        s[s == 0] = self.eps # where sum is zero, set to eps to avoid division by zero
        return X / s