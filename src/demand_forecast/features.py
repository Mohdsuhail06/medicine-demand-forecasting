"""Feature engineering for direct 14-day-ahead forecasting.

To forecast day t we may only use demand known on day t - HORIZON (the forecast origin).
So every lag is >= HORIZON days and every rolling window ends HORIZON days before t.
With that rule one model can forecast any day 1..14 days ahead without leaking the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from demand_forecast.data import HOSPITALS, MEDICINES, is_ramadan

HORIZON = 14
LAGS = (14, 21, 28, 35)
WINDOWS = (7, 28, 56)

_HOSPITAL_CODE = {h.hospital_id: i for i, h in enumerate(HOSPITALS)}
_MEDICINE_CODE = {m.medicine_id: i for i, m in enumerate(MEDICINES)}
_CATEGORIES = sorted({m.category for m in MEDICINES})
_CATEGORY_CODE = {c: i for i, c in enumerate(_CATEGORIES)}

CATEGORICAL_FEATURES = ["hospital_code", "medicine_code", "category_code"]
CALENDAR_FEATURES = ["day_of_week", "is_weekend", "month", "doy_sin", "doy_cos", "is_ramadan"]
HISTORY_FEATURES = (
    [f"lag_{k}" for k in LAGS] + [f"roll_mean_{w}" for w in WINDOWS] + ["roll_std_28"]
)
FEATURE_COLUMNS = CATEGORICAL_FEATURES + CALENDAR_FEATURES + HISTORY_FEATURES


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add day-of-week, weekend, month, cyclical day-of-year and Ramadan flags."""
    ts = pd.to_datetime(df["date"])
    out = df.copy()
    out["day_of_week"] = ts.dt.day_of_week
    out["is_weekend"] = ts.dt.day_of_week.isin([4, 5]).astype(int)
    out["month"] = ts.dt.month
    angle = 2 * np.pi * ts.dt.day_of_year / 365.25
    out["doy_sin"] = np.sin(angle)
    out["doy_cos"] = np.cos(angle)
    out["is_ramadan"] = [int(is_ramadan(d)) for d in ts.dt.date]
    return out


def add_codes(df: pd.DataFrame) -> pd.DataFrame:
    """Encode hospital, medicine and category as stable integer codes.

    Raises:
        KeyError: for a hospital or medicine the model does not know.
    """
    out = df.copy()
    unknown_h = set(out["hospital_id"]) - _HOSPITAL_CODE.keys()
    unknown_m = set(out["medicine_id"]) - _MEDICINE_CODE.keys()
    if unknown_h or unknown_m:
        raise KeyError(f"Unknown ids: {sorted(unknown_h | unknown_m)}")
    out["hospital_code"] = out["hospital_id"].map(_HOSPITAL_CODE)
    out["medicine_code"] = out["medicine_id"].map(_MEDICINE_CODE)
    out["category_code"] = out["category"].map(_CATEGORY_CODE)
    return out


def make_features(df: pd.DataFrame, dropna: bool = True) -> pd.DataFrame:
    """Build the model's input table from raw daily demand.

    Args:
        df: columns date, hospital_id, medicine_id, category, units.
        dropna: drop rows that do not yet have enough history for every feature.
    """
    out = df.sort_values(["hospital_id", "medicine_id", "date"]).reset_index(drop=True)
    out = add_codes(add_calendar_features(out))
    grouped = out.groupby(["hospital_id", "medicine_id"], sort=False)["units"]

    for k in LAGS:
        out[f"lag_{k}"] = grouped.shift(k)

    known = grouped.shift(HORIZON)  # the newest value known at the forecast origin
    series_key = [out["hospital_id"], out["medicine_id"]]
    for w in WINDOWS:
        out[f"roll_mean_{w}"] = known.groupby(series_key).transform(
            lambda s, w=w: s.rolling(w).mean()
        )
    out["roll_std_28"] = known.groupby(series_key).transform(lambda s: s.rolling(28).std())

    if dropna:
        out = out.dropna(subset=HISTORY_FEATURES).reset_index(drop=True)
    return out
