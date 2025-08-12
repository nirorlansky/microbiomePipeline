from pipeline import evaluate_strategies

X, y = ...  # your data (X numeric, non-negative for RA), y labels
strategies = {
    "No Resampling": "none",
    "SMOTE": "smote",
    "BorderlineSMOTE": "bsmote",
    "ADASYN": "adasyn",
    "RandomOverSampler": "ros",
}
table = evaluate_strategies(X, y, strategies, k_features=200, random_state=42)
print(table)