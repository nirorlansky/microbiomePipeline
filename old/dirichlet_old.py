import numpy as np
from scipy.stats import dirichlet
from collections import Counter
from imblearn.base import BaseSampler 

class dirichletOverSampler(BaseSampler):
    """
    Oversample minority class using Dirichlet distribution.
    This is a custom implementation that fits the Dirichlet parameters on the minority class.
    """
    _parameter_constraints = {}  # Add this line
    _sampling_type = 'over-sampling'  # <-- Add this line



    def __init__(self, sampling_strategy='auto', random_state=None):
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.minority_label = 8

    def closure(self, X, eps=1e-9):
        """מוסיף pseudocount קטן ומנרמל שכל שורה תסתכם ל-1."""
        X = np.asarray(X, dtype=float)
        X = X + eps
        row_sums = X.sum(axis=1, keepdims=True)
        return X / row_sums
    
    def fit_dirichlet_on_minority(self, X_train, y_train, eps=1e-9, scale=1.0):
        """
        fit לפרמטרי דיריכלה (alpha) על דגימות minority ב-train.
        scale>1 מחדד (יותר ריכוז סביב הממוצע), scale<1 מרכך.
        """
        X_min = X_train[y_train == '8']
        X_min = self.closure(X_min, eps=eps)
        mean = np.mean(X_min, axis=0)
        var = np.var(X_min, axis=0)
        mean = np.clip(mean, eps, 1)
        var = np.clip(var, eps, None)
        s = (mean * (1 - mean) / var - 1)
        alpha_hat = mean * s
        alpha_hat = np.clip(alpha_hat, eps, None) * scale
        return alpha_hat
    
    def sample_dirichlet(self, alpha, n_samples):
        return dirichlet(alpha).rvs(size=n_samples, random_state=self.random_state)
    
    def oversample_minority_with_dirichlet_train_only(
        self, X_train, y_train, test_size=0.2, random_state=42, 
        target_count="match_majority", eps=1e-9, scale=1.0
    ):

        # כמה דגימות צריך להוסיף?
        counts = Counter(y_train)
        min_count = counts['8']
        n_to_add = 1200 - min_count

        # fit לדיריכלה על minority ב-train בלבד
        alpha = self.fit_dirichlet_on_minority(X_train, y_train, eps=eps, scale=scale)

        # דגימה קומפוזיציונית חדשה
        X_new = self.sample_dirichlet(alpha, n_to_add)
        y_new = np.full(n_to_add, '8')

        # מאחדים רק ל-train
        X_train_bal = np.vstack([X_train, X_new])
        y_train_bal = np.hstack([y_train, y_new])

        return X_train_bal, y_train_bal
    
    def _fit_resample(self, X, y):
    # Use your oversample_minority_with_dirichlet_train_only logic
        X_res, y_res = self.oversample_minority_with_dirichlet_train_only(X, y)
        return X_res, y_res
