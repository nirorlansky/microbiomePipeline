HEALTHY = 0
SICK = 1
#
SCORES = {
        "roc_auc": "roc_auc",
        "aupr": "average_precision",  # FIXED: use built-in string scorer
        "recall": "recall",
        "acc": "accuracy",
    }

GROUP_INDEX_NAME = "strategy"

STRATEGIES =  {
        "No Resampling": "none",
        "SMOTE with thresholding": "smote_thresholding",
        "SMOTE preserve_zero_pattern": "smote_preserve_zero_pattern",
        "SMOTE with min-positive vector": "smote_min_positive",
        "SMOTE only": "smote_only",
        "Dirichlet MLE with thresholding": "Dirichlet_MLE_thresholding",
        "Dirichlet MLE": "Dirichlet_MLE",
        "Dirichlet MoM with thresholding": "Dirichlet_MoM_thresholding",
        "Dirichlet MoM": "Dirichlet_MoM",
        "Random Oversampling": "resample_random_samples",
    }

RANDOM_STATES = list(range(1, 51))  # list of random states to use for evaluation from 1-100