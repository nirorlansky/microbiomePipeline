import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, average_precision_score
from functools import partial
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split


import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN, RandomOverSampler
from imblearn.base import FunctionSampler
from sklearn.base import clone  
from steps.relative_abundance import RelativeAbundance
from steps.remainder_col import AddRemainder
from samplers.dirichlet_new import DirichletSampler
from samplers.random_resampler import ResampleSamples
from samplers.smote_new import SmoteSampler
from sklearn.ensemble import RandomForestClassifier
from pkg.globals import *

import numpy as np
from steps.duplicate_test import inflate_test_healthy_ratio, inflate_test_healthy_ratio_no_cross_val

identity_sampler = FunctionSampler(func=lambda X, y: (X, y))  # "no resampling" baseline

def make_pipeline(k_features=200, sampler="none", random_state=42, model=None, feature_selection=True, eval=False, sampling_ratio=0.5):
    if model is None:
        # Choose a deterministic solver- random forest with fixed random state
        model = RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1)

    # Choose sampler object
    if sampler == "none":
        samp = identity_sampler
    elif sampler == "resample_random_samples":
        samp = ResampleSamples(sampling_strategy=sampling_ratio)
    elif sampler == "smote_thresholding":
        samp = SmoteSampler(minority_share = sampling_ratio, threshold = 0.01, eval=eval, method_string="SMOTE with thresholding")
    elif sampler == "smote_preserve_zero_pattern":
        samp = SmoteSampler(minority_share = sampling_ratio, preserve_zero_pattern=True, eval=eval, method_string="SMOTE with zero pattern")
    elif sampler == "smote_min_positive":
        samp = SmoteSampler(minority_share = sampling_ratio, use_feature_eps=True, method_string="SMOTE with min-positive vector", eval=eval)
    elif sampler == "smote_only":
        samp = SmoteSampler(minority_share = sampling_ratio, eval=eval, method_string="SMOTE only")

    elif sampler == "Dirichlet_MLE_thresholding":
        samp = DirichletSampler(
            sampling_strategy=sampling_ratio,
            method="mle",
            use_dynamic_eps=True,
            eval=eval,
            method_string="Dirichlet_MLE_thresholding", 
        )
    elif sampler == "Dirichlet_MLE":
        samp = DirichletSampler(
            sampling_strategy=sampling_ratio,
            method="mle",
            use_dynamic_eps=False,
            eval=eval,
            method_string="Dirichlet_MLE"
        )
    elif sampler == "Dirichlet_MoM_thresholding":
        samp = DirichletSampler(
            sampling_strategy=sampling_ratio,
            method="mom",
            use_dynamic_eps=True,
            eval=eval,
            method_string="Dirichlet_MoM_thresholding"
        )
    elif sampler == "Dirichlet_MoM":
        samp = DirichletSampler(
            sampling_strategy=sampling_ratio,
            method="mom",
            use_dynamic_eps=False,
            eval=eval,
            method_string="Dirichlet_MoM"
        )
    else:
        raise ValueError(f"Unknown sampler: {sampler}")

    if not feature_selection:
        pipe = ImbPipeline([
        ("rel",     RelativeAbundance()),
        ("sampler", samp),     # runs only on train folds
        ("clf",     model),
    ])
    else:
        pipe = ImbPipeline([
            ("rel",     RelativeAbundance()),
            ("select",  SelectKBest(score_func=partial(mutual_info_classif, random_state=random_state), k=k_features
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
# Output: list of (train_index, test_index) tuples.
# Each train_index and test_index are arrays of row indices from X and y.
# Their lengths add up to the total number of samples, with no overlap.


# ---------- Evaluate multiple strategies on the same splits ----------
def run_pipeline_cross_val(X, y, strategies, k_features=200, random_state=42, test_healthy_ratio=None):
    splits = make_splits(X, y, n_splits=5, random_state=random_state)

    if test_healthy_ratio is not None: # inflate the healthy smaples in the tests
        splits = inflate_test_healthy_ratio(splits, y, target_healthy_ratio=test_healthy_ratio)

    rows = []
    for name, sampler_key in strategies.items():
        pipe = make_pipeline(k_features=k_features, sampler=sampler_key, random_state=random_state, feature_selection=False, sampling_strategy=0.5)

        # Use the same splits for everyone
        scores = cross_validate(
            estimator=pipe,
            X=X, y=y,
            scoring=SCORES,
            cv=splits,              # <--- identical folds
            n_jobs=-1,
            return_train_score=False
        )
        summary = {m: scores[f"test_{m}"].mean() for m in ["roc_auc","aupr","recall","acc"]}
        summary.update({"strategy": name})
        rows.append(summary)

    return pd.DataFrame(rows).set_index("strategy").sort_values("aupr", ascending=False)


def run_pipeline_with_test(X, y, strategies, k_features=200, random_state=42,train_test_ratio=0.8, sampling_ratio = 1, test_healthy_ratio=None, eval=False):
    """
    Evaluate different sampling strategies on the given dataset.
    this function evaluate samplers with only test and train, no cross validation
    """    
    #split to train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_test_ratio, stratify=y, random_state=random_state)
    if test_healthy_ratio is not None:
        X_test, y_test = inflate_test_healthy_ratio_no_cross_val(X_test, y_test, target_healthy_ratio=test_healthy_ratio)
        
    results = []
    for name, sampler in strategies.items():
        pipe = make_pipeline(k_features=k_features, sampler=sampler, random_state=random_state, feature_selection=False, eval=eval, sampling_ratio=sampling_ratio)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None # predict probabilities for ROC AUC and AUPR

        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
        aupr = average_precision_score(y_test, y_proba) if y_proba is not None else np.nan
        recall = recall_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        results.append({
            GROUP_INDEX_NAME: name,
            "ROC AUC": roc_auc,
            "AUPR": aupr,
            "Recall": recall,
            "Accuracy": acc
        })
    return pd.DataFrame(results).set_index(GROUP_INDEX_NAME).sort_values("AUPR", ascending=False)