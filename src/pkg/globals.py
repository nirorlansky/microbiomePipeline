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
RANDOM_STATES = list(range(1, 101))  # list of random states to use for evaluation from 1-100