"""Honest evaluation for time series: rolling-origin backtesting.

A random train/test split would let the model "see the future". Instead we stand at a
forecast origin, train only on what was known before it, forecast the next 14 days,
then move the origin forward and repeat.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from demand_forecast.features import FEATURE_COLUMNS, HORIZON


def metrics(actual, predicted) -> dict[str, float]:
    """Forecast accuracy.

    mae  - mean absolute error, in units
    rmse - root mean squared error, in units (punishes big misses)
    wape - weighted absolute percentage error: total |error| / total demand, in %
    bias - total over-forecast (+) or under-forecast (-) as % of demand
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    total = actual.sum()
    if total == 0:
        raise ValueError("Actual demand sums to zero; percentage metrics are undefined")
    err = predicted - actual
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "wape": float(np.abs(err).sum() / total * 100),
        "bias": float(err.sum() / total * 100),
    }


def forecast_origins(first: dt.date, last: dt.date, step: int = HORIZON) -> list[dt.date]:
    """Origins every `step` days such that each 14-day window fits inside [first, last]."""
    origins = []
    origin = first
    while origin + dt.timedelta(days=HORIZON - 1) <= last:
        origins.append(origin)
        origin += dt.timedelta(days=step)
    return origins


def backtest(
    feats: pd.DataFrame,
    model_factory: Callable,
    origins: Sequence[dt.date],
) -> pd.DataFrame:
    """Train a fresh model at each origin and forecast the following 14 days.

    Returns one row per forecast with the actual units, the prediction and the origin.
    """
    results = []
    for origin in origins:
        train = feats[feats["date"] < origin]
        window_end = origin + dt.timedelta(days=HORIZON)
        test = feats[(feats["date"] >= origin) & (feats["date"] < window_end)]
        if train.empty or test.empty:
            continue
        X_train = train[FEATURE_COLUMNS].copy()
        X_train.attrs["max_date"] = train["date"].max()
        model = model_factory().fit(X_train, train["units"])
        out = test[["date", "hospital_id", "medicine_id", "category", "units"]].copy()
        out["prediction"] = model.predict(test[FEATURE_COLUMNS])
        out["origin"] = origin
        results.append(out)
    return pd.concat(results, ignore_index=True)


def compare_models(
    feats: pd.DataFrame, factories: dict[str, Callable], origins: Sequence[dt.date]
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Backtest every model on the same origins. Returns a score table and all predictions."""
    rows, predictions = [], {}
    for name, factory in factories.items():
        preds = backtest(feats, factory, origins)
        predictions[name] = preds
        rows.append({"model": name, **metrics(preds["units"], preds["prediction"])})
    table = pd.DataFrame(rows).sort_values("wape").reset_index(drop=True)
    return table, predictions
