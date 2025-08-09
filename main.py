
import pandas as pd
from preprocess import preprocess_data
from imblearn.pipeline import Pipeline  # <-- Use imblearn's Pipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
import sklearn

# Import dirichletOverSampler if available in your project or define it here
from dirichlet import dirichletOverSampler  # Adjust the import path as needed



import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn import set_config
from imblearn.base import BaseSampler 

from imblearn.base import SamplerMixin
set_config(transform_output="pandas")  # כל ה-Transformers יחזירו DataFrame עם שמות עמודות

def _to_dataframe(X, y=None):
    # הופך את X ל-DataFrame ושומר y בעמודה בשם target (אם קיים)
    if hasattr(X, "to_numpy") and hasattr(X, "columns"):
        df = X.copy()
    else:
        X = np.asarray(X)
        cols = [f"f{i}" for i in range(X.shape[1])]
        df = pd.DataFrame(X, columns=cols)
    if y is not None:
        s = y if isinstance(y, (pd.Series, pd.DataFrame)) else pd.Series(y, name="target")
        df["target"] = np.asarray(s).ravel()
    return df

class SnapshotSampler(BaseSampler):
    """
    שלב 'צילום' שלא משנה את המידע אלא רק שומר אותו ל-CSV, ואז מחזיר (X,y) כמו שהם.
    """
    _parameter_constraints = {}  # <-- Add this line
    _sampling_type = 'over-sampling'  # <-- Add this line


    def __init__(self, name, out_dir="snapshots", index=False, float_fmt=None, sampling_strategy='auto'):
        self.name = name
        self.out_dir = out_dir
        self.index = index
        self.float_fmt = float_fmt
        self.sampling_strategy = sampling_strategy  # <-- Add this line
        self._counter = 0

    def _fit_resample(self, X, y):
        os.makedirs(self.out_dir, exist_ok=True)
        self._counter += 1
        fname = os.path.join(self.out_dir, f"{self._counter:02d}{self.name}.csv")

        df = _to_dataframe(X, y)
        df.to_csv(fname, index=self.index, float_format=self.float_fmt)
        # מחזיר ללא שינוי
        return X, y



def main():
    # Preprocess the data
    columnTransformer, data = preprocess_data(metadata="data_files\metadata.csv",
                                     microbiome="data_files\microbiome.csv",
                                     serum_lipo="data_files\serum_lipo.csv")
    
    X = data.drop(columns=['PATGROUPFINAL_C'])
    y = data['PATGROUPFINAL_C']
    # change y to binary - 0 for class 8, 1 for classes 1-7
    y_binary = (y != '8').astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Save to CSV

    smote = SMOTE(sampling_strategy={'8': 1000})
    sampler = dirichletOverSampler()

    pipeline_smote = Pipeline([
        ('preprocessing', columnTransformer),
        ("snap_after_preprocessing", SnapshotSampler("after_modifications")),
        ('feature_selection', SelectKBest(score_func=f_classif, k=100)),  # mrmr\bourta
        ("snap_after_feature_select", SnapshotSampler("after_feature_selection")),
        ('smote', smote),
        ("snap_after_oversample", SnapshotSampler("after_oversample")),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipeline_dirichle = Pipeline([
        ('preprocessing', columnTransformer),
        ('feature_selection', SelectKBest(score_func=f_classif, k=100)),  # mrmr\bourta
        ("dirichlet_oversample", sampler),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42))
    ])


    # Train the model
    pipeline_smote.fit(X_train, y_train)
    pipeline_dirichle.fit(X_train, y_train)

    # Example: Predict on the same data
    predictions_smote = pipeline_smote.predict(X_test)
    print("Predictions with smote:", predictions_smote)
    predictions_dirichle = pipeline_dirichle.predict(X_test)
    print("Predictions with Dirichlet Oversampling:", predictions_dirichle)     

    # You can add code here for making predictions or evaluating the model

if __name__ == "__main__":
    main()