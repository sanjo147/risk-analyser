"""sklearn preprocessing pipeline (fitted on training data only)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .schema import CATEGORICAL_FEATURES, NUMERIC_FEATURES


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Winsorise numeric columns to [q_low, q_high] learned on the training data (outlier handling)."""

    def __init__(self, q_low: float = 0.005, q_high: float = 0.995):
        self.q_low = q_low
        self.q_high = q_high

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.low_ = np.nanquantile(X, self.q_low, axis=0)
        self.high_ = np.nanquantile(X, self.q_high, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.low_, self.high_)  # NaN passes through untouched

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features)


def make_preprocessor(scale: bool = True) -> ColumnTransformer:
    num_steps = [("clip", QuantileClipper()), ("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50, sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", Pipeline(num_steps), NUMERIC_FEATURES), ("cat", cat, CATEGORICAL_FEATURES)],
        verbose_feature_names_out=False,
    )


def transformed_to_original(pre: ColumnTransformer) -> list[str]:
    """For every transformed column, the original model feature it came from."""
    names = list(pre.get_feature_names_out())
    out = []
    for n in names:
        if n in NUMERIC_FEATURES:
            out.append(n)
        else:
            out.append(next(c for c in CATEGORICAL_FEATURES if n.startswith(c + "_")))
    return out


def as_frame(pre: ColumnTransformer, X: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(pre.transform(X), columns=pre.get_feature_names_out(), index=X.index)
