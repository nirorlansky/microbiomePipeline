from pipeline import make_pipeline
from steps.duplicate_test import inflate_test_healthy_ratio_no_cross_val
from globals import *
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, recall_score, accuracy_score

def evaluate_samplers(X, y, strategies, k_features=200, random_state=42,ratio=0.8,test_healthy_ratio=None):
    """
    Evaluate different sampling strategies on the given dataset.
    this function evaluate samplers with only test and train, no cross validation
    """    
    #split to train and test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-ratio, stratify=y, random_state=random_state)
    if test_healthy_ratio is not None:
        X_test, y_test = inflate_test_healthy_ratio_no_cross_val(X_test, y_test, target_healthy_ratio=test_healthy_ratio, random_state=random_state)
        # print train and test healthy/sick ratio
        print(f"Train H/S: {int(np.sum(y_train == HEALTHY))}/{int(np.sum(y_train != HEALTHY))} | Test H/S: {int(np.sum(y_test == HEALTHY))}/{int(np.sum(y_test != HEALTHY))}")

    results = []
    for name, sampler in strategies.items():
        print(f"Evaluating strategy: {name}")
        pipe = make_pipeline(k_features=k_features, sampler=sampler, random_state=random_state, feature_selection=False)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else np.nan
        aupr = average_precision_score(y_test, y_proba) if y_proba is not None else np.nan
        recall = recall_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        results.append({
            "Strategy": name,
            "ROC AUC": roc_auc,
            "AUPR": aupr,
            "Recall": recall,
            "Accuracy": acc
        })
    return pd.DataFrame(results).set_index("Strategy").sort_values("AUPR", ascending=False)

