import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, average_precision_score
from sklearn.linear_model import LogisticRegression


import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, RandomOverSampler
from imblearn.base import FunctionSampler
from sklearn.base import clone  
from steps.relative_abundance import RelativeAbundance
from steps.remainder_col import AddRemainder
from samplers.dirichlet_new import DirichletSampler
from samplers.smote import SmoteOverSampler


identity_sampler = FunctionSampler(func=lambda X, y: (X, y))  # "no resampling" baseline

def make_pipeline(k_features=200, sampler="none", random_state=42, model=None):
    if model is None:
        # Choose a deterministic solver; lbfgs is deterministic.
        model = LogisticRegression(max_iter=500)

    # Choose sampler object
    if sampler == "none":
        samp = identity_sampler
    elif sampler == "smote_thresholding":
        samp = SmoteOverSampler(threshold = 0.01)
    elif sampler == "smote_preserve_zero_pattern":
        samp = SmoteOverSampler(preserve_zero_pattern=True)
    elif sampler == "smote_min_positive":
        samp = SmoteOverSampler(use_feature_eps=True)
    elif sampler == "smote_only":
        samp = SmoteOverSampler()

    elif sampler == "Dirichlet_MLE_thresholding":
        samp = DirichletSampler(
            method="mle",
            use_dynamic_eps=True,
        )
    elif sampler == "Dirichlet_MLE":
        samp = DirichletSampler(
            method="mle",
            use_dynamic_eps=False,
        )
    elif sampler == "Dirichlet_MoM_thresholding":
        samp = DirichletSampler(
            method="mom",
            use_dynamic_eps=True,
        )
    elif sampler == "Dirichlet_MoM":
        samp = DirichletSampler(
            method="mom",
            use_dynamic_eps=False,
        )
    else:
        raise ValueError(f"Unknown sampler: {sampler}")

    pipe = ImbPipeline([
        ("rel",     RelativeAbundance()),
        ("select",  SelectKBest(mutual_info_classif, k=k_features, 
                                # set MI randomness for determinism
                                # mutual_info_classif(random_state=...) exists in sklearn
                                )),
        ("remainder", AddRemainder()),
        ("sampler", samp),     # runs only on train folds
        ("clf",     model),
    ])
    # If using mutual_info_classif with randomness, set via set_params:
    # pipe.set_params(select__score_func=partial(mutual_info_classif, random_state=random_state))
    return pipe


def make_splits(X, y, n_splits=5, random_state=42):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(cv.split(X, y))  # materialize once, reuse everywhere

# ---------- Evaluate multiple strategies on the same splits ----------
def evaluate_strategies(X, y, strategies, k_features=200, random_state=42):
    splits = make_splits(X, y, n_splits=5, random_state=random_state)

    scorers = {
        "roc_auc": "roc_auc",
        "aupr": "average_precision",  # FIXED: use built-in string scorer
        "f1": "f1",
        "acc": "accuracy",
    }

    rows = []
    for name, sampler_key in strategies.items():
        pipe = make_pipeline(k_features=k_features, sampler=sampler_key, random_state=random_state)

        # Use the same splits for everyone
        scores = cross_validate(
            estimator=pipe,
            X=X, y=y,
            scoring=scorers,
            cv=splits,              # <--- identical folds
            n_jobs=-1,
            return_train_score=False
        )
        summary = {m: scores[f"test_{m}"].mean() for m in ["roc_auc","aupr","f1","acc"]}
        summary.update({"strategy": name})
        rows.append(summary)

    return pd.DataFrame(rows).set_index("strategy").sort_values("roc_auc", ascending=False)
