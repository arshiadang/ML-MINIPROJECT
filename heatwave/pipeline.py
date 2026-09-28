import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from . import INPUT_COLUMNS
from .calendar import doy365
from .labeling import BASE_THRESHOLDS

NUMERIC_FEATURES = ["max_temp", "departure", "doy_sin", "doy_cos", "lat", "lon"]


class FeatureBuilder(TransformerMixin, BaseEstimator):
    def fit(self, X, y=None):
        self.transform(X)
        self.n_features_in_ = len(INPUT_COLUMNS)
        self.feature_names_in_ = np.array(INPUT_COLUMNS, dtype=object)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a DataFrame with named raw columns")
        missing = set(INPUT_COLUMNS) - set(X.columns)
        if missing:
            raise ValueError(f"Missing raw columns: {sorted(missing)}")
        out = X[INPUT_COLUMNS].copy()
        out["date"] = pd.to_datetime(out.date)
        if not out.date.dt.month.between(3, 6).all():
            raise ValueError("Prediction dates must be March 1 through June 30")
        if not out.region_type.isin(BASE_THRESHOLDS).all():
            raise ValueError("Unknown region_type")
        if not np.isfinite(out[["max_temp", "normal_temp", "lat", "lon"]].to_numpy(dtype=float)).all():
            raise ValueError("Numeric inputs must be finite")
        out["departure"] = out.max_temp - out.normal_temp
        out["doy365"] = doy365(out.date)
        angle = 2 * np.pi * out.doy365 / 365
        out["doy_sin"], out["doy_cos"] = np.sin(angle), np.cos(angle)
        return out


def make_pipeline(k=5, metric="euclidean", weights="uniform"):
    preprocessing = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC_FEATURES),
        ("region", OneHotEncoder(categories=[["plains", "coastal", "hilly"]], handle_unknown="error", sparse_output=False), ["region_type"]),
    ])
    return Pipeline([("features", FeatureBuilder()), ("preprocessing", preprocessing), ("knn", KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights, n_jobs=1))])
