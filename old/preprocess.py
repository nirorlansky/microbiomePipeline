import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression   # דוגמה; אפשר להחליף לכל מודל
from typing import List, Tuple

CONST_VAL  = 0.5  
CLR_PSEUDOCOUNT =0.5
const_cols = ["SMOKE", "GENDER"]      # שתי העמודות שמתמלאות ב-0.5
median_cols = ["AGE"]
center_col = ["CENTER_C"]


# ---- 1) טרנספורם CLR עבור מיקרוביום ----
class CLRTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, pseudocount: float = 0.5):
        self.pseudocount = pseudocount
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X = X + self.pseudocount              # למנוע אפסים
        row_sums = X.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        P = X / row_sums
        logP = np.log(P)
        clr = logP - logP.mean(axis=1, keepdims=True)
        return clr

# ---- 2) טרנספורם Log + Z-score עבור Lipo ----
class LogZScoreTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, add_one: bool = True):
        self.add_one = add_one
        self.scaler_ = StandardScaler(with_mean=True, with_std=True)
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        X = np.log1p(np.maximum(X, 0)) if self.add_one else np.log(X)
        self.scaler_.fit(X)                   # ה-Scaler לומד מה-Train בלבד
        return self
    def transform(self, X):
        X = np.asarray(X, dtype=float)
        X = np.log1p(np.maximum(X, 0)) if self.add_one else np.log(X)
        return self.scaler_.transform(X)


def load_data(metadata_path, microbiome_path, serum_lipo_path):
    metadata = pd.read_csv(metadata_path)
    microbiome = pd.read_csv(microbiome_path)
    serum_lipo = pd.read_csv(serum_lipo_path)
    microbiome_cols = [c for c in microbiome.columns if c != "SampleID"]
    lipo_cols       = [c for c in serum_lipo.columns if c != "SampleID"]
    return metadata, microbiome, serum_lipo, microbiome_cols, lipo_cols

def merge_datasets(metadata, microbiome, serum_lipo):
    merged_data = metadata.merge(microbiome, on='SampleID', how='inner')
    merged_data = merged_data.merge(serum_lipo, on='SampleID', how='inner')
    return merged_data


def preprocess_data(metadata, microbiome, serum_lipo):
    # Load data
    metadata, microbiome, serum_lipo, microbiome_cols, lipo_cols = load_data(metadata, microbiome, serum_lipo)
    merged_data = merge_datasets(metadata, microbiome, serum_lipo)
    data = merged_data.drop(columns=['DDS', 'pa_work_2cl', 'SampleID'], errors='ignore')

    return ColumnTransformer(
        transformers=[
            ("meta_const",  SimpleImputer(strategy="constant", fill_value=CONST_VAL), const_cols),
            ("meta_median", SimpleImputer(strategy="median"), median_cols),
            ("center_onehot", OneHotEncoder(drop=None, sparse_output=False), center_col),  # <-- add this line
            ("microbiome",  Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
                ("clr", CLRTransformer(pseudocount=CLR_PSEUDOCOUNT)),
            ]), microbiome_cols),
            ("lipo", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("logz", LogZScoreTransformer(add_one=True)),
            ]), lipo_cols),
        ],
        remainder="passthrough"
    ), data