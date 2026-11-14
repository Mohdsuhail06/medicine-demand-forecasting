"""Candidate models: two simple baselines and three machine-learning models.

Baselines matter: an ML model is only worth deploying if it clearly beats them.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from demand_forecast.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS


class SeasonalNaive(BaseEstimator, RegressorMixin):
    """Predict the demand of the same weekday two weeks earlier (the newest known value)."""

    def fit(self, X: pd.DataFrame, y=None) -> SeasonalNaive:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["lag_14"].to_numpy(dtype=float)


class MovingAverage(BaseEstimator, RegressorMixin):
    """Predict the average daily demand of the last 28 known days."""

    def fit(self, X: pd.DataFrame, y=None) -> MovingAverage:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return X["roll_mean_28"].to_numpy(dtype=float)


class NonNegative(BaseEstimator, RegressorMixin):
    """Wrap a regressor so it never predicts negative demand."""

    def __init__(self, estimator):
        self.estimator = estimator

    def fit(self, X: pd.DataFrame, y) -> NonNegative:
        self.estimator_ = clone(self.estimator).fit(X[FEATURE_COLUMNS], y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(self.estimator_.predict(X[FEATURE_COLUMNS]), 0, None)


def _ridge():
    one_hot = CATEGORICAL_FEATURES + ["day_of_week", "month"]
    numeric = [c for c in FEATURE_COLUMNS if c not in one_hot]
    prep = ColumnTransformer(
        [
            ("onehot", OneHotEncoder(handle_unknown="ignore"), one_hot),
            ("scale", StandardScaler(), numeric),
        ]
    )
    # Demand is multiplicative (size x season x weekday), so fit on log scale.
    model = TransformedTargetRegressor(
        regressor=make_pipeline(prep, Ridge(alpha=1.0)), func=np.log1p, inverse_func=np.expm1
    )
    return NonNegative(model)


def _random_forest():
    return NonNegative(
        RandomForestRegressor(
            n_estimators=120, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=0
        )
    )


def _gradient_boosting():
    # Poisson loss suits count data (units dispensed); ids are treated as categories.
    return NonNegative(
        HistGradientBoostingRegressor(
            loss="poisson",
            learning_rate=0.05,
            max_iter=400,
            min_samples_leaf=40,
            l2_regularization=1.0,
            categorical_features=[FEATURE_COLUMNS.index(c) for c in CATEGORICAL_FEATURES],
            random_state=0,
        )
    )


MODEL_FACTORIES: dict[str, Callable[[], BaseEstimator]] = {
    "Seasonal naive": SeasonalNaive,
    "Moving average (28d)": MovingAverage,
    "Ridge regression": _ridge,
    "Random forest": _random_forest,
    "Gradient boosting": _gradient_boosting,
}
