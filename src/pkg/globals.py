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
RANDOM_STATES = [0, 42, 7, 100, 2024]