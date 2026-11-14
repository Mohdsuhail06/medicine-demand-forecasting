"""Inference: load the trained model bundle and forecast the next 14 days for one series."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from demand_forecast.features import FEATURE_COLUMNS, HISTORY_FEATURES, HORIZON, make_features
from demand_forecast.inventory import reorder_recommendation
from demand_forecast.uncertainty import UncertaintyModel


@dataclass
class ModelBundle:
    """Everything needed to make forecasts later: the model plus its metadata."""

    name: str
    model: Any
    trained_until: dt.date
    uncertainty: UncertaintyModel  # forecast error over the lead time, from validation
    metrics: dict = field(default_factory=dict)
    feature_columns: list[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, path: Path | str) -> ModelBundle:
        """Load a bundle. Only load files you created yourself: joblib can run code."""
        bundle = joblib.load(path)
        if not isinstance(bundle, cls):
            raise TypeError(f"{path} does not contain a ModelBundle")
        return bundle


def forecast_series(
    bundle: ModelBundle, history: pd.DataFrame, hospital_id: str, medicine_id: str
) -> pd.DataFrame:
    """Forecast daily units for the 14 days after the last date in `history`.

    Raises:
        KeyError: if there is no history for this hospital/medicine.
        ValueError: if the history is too short to build the features.
    """
    series = history[
        (history["hospital_id"] == hospital_id) & (history["medicine_id"] == medicine_id)
    ]
    if series.empty:
        raise KeyError(f"No history for hospital '{hospital_id}' and medicine '{medicine_id}'")

    last = max(series["date"])
    future = pd.DataFrame(
        {
            "date": [last + dt.timedelta(days=d) for d in range(1, HORIZON + 1)],
            "hospital_id": hospital_id,
            "medicine_id": medicine_id,
            "category": series["category"].iloc[0],
            "units": np.nan,
        }
    )
    combined = pd.concat([series, future], ignore_index=True)
    feats = make_features(combined, dropna=False)
    rows = feats[feats["date"] > last]
    if rows[HISTORY_FEATURES].isna().any().any():
        raise ValueError("Not enough history to forecast: at least 70 days are needed")

    prediction = bundle.model.predict(rows[bundle.feature_columns])
    return pd.DataFrame(
        {"date": rows["date"].tolist(), "forecast_units": np.clip(prediction, 0, None)}
    )


def recommend(
    bundle: ModelBundle,
    history: pd.DataFrame,
    hospital_id: str,
    medicine_id: str,
    current_stock: float,
    lead_time_days: int = 7,
    service_level: float = 0.95,
) -> dict:
    """Forecast the next 14 days and recommend how much to order for the given lead time."""
    if not 1 <= lead_time_days <= HORIZON:
        raise ValueError(f"lead_time_days must be between 1 and {HORIZON}")
    fc = forecast_series(bundle, history, hospital_id, medicine_id)
    std = bundle.uncertainty.lead_time_std(hospital_id, medicine_id, lead_time_days)
    rec = reorder_recommendation(
        fc["forecast_units"].iloc[:lead_time_days].tolist(), current_stock, std, service_level
    )
    return {
        "hospital_id": hospital_id,
        "medicine_id": medicine_id,
        "model": bundle.name,
        "forecast": [
            {"date": d.isoformat(), "units": round(float(u), 1)}
            for d, u in zip(fc["date"], fc["forecast_units"], strict=True)
        ],
        "recommendation": rec.to_dict(),
    }
