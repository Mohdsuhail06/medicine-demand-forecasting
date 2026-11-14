"""Tests for feature engineering. The key property: no information from the future leaks in."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from demand_forecast.features import FEATURE_COLUMNS, HORIZON, make_features


def one_series(values: list[int], start=dt.date(2025, 1, 1)) -> pd.DataFrame:
    dates = [start + dt.timedelta(days=i) for i in range(len(values))]
    return pd.DataFrame(
        {
            "date": dates,
            "hospital_id": "H01",
            "medicine_id": "AMOXICILLIN",
            "category": "Antibiotic",
            "units": values,
        }
    )


def test_horizon_is_two_weeks():
    assert HORIZON == 14


def test_output_has_all_feature_columns_and_target(small_demand):
    feats = make_features(small_demand)

    assert set(FEATURE_COLUMNS) <= set(feats.columns)
    assert "units" in feats.columns


def test_lag_features_look_exactly_horizon_days_back():
    df = one_series(list(range(100)))

    feats = make_features(df).set_index("date")

    day = dt.date(2025, 1, 1) + dt.timedelta(days=80)  # value on that day is 80
    assert feats.loc[day, "lag_14"] == 80 - 14
    assert feats.loc[day, "lag_21"] == 80 - 21
    assert feats.loc[day, "lag_28"] == 80 - 28


def test_rolling_mean_only_uses_data_known_at_forecast_time():
    df = one_series(list(range(100)))

    feats = make_features(df).set_index("date")

    day = dt.date(2025, 1, 1) + dt.timedelta(days=90)
    # 7-day window ending 14 days before the target day: values 70..76
    assert feats.loc[day, "roll_mean_7"] == pytest.approx(np.mean(range(70, 77)))


def test_changing_the_last_14_days_does_not_change_features():
    """If a feature changed when recent (unknown) values change, it would be leaking."""
    values = list(np.random.default_rng(0).integers(10, 50, size=120))
    base = make_features(one_series(values)).iloc[-1]

    tampered_values = values[:-14] + [999] * 14
    tampered = make_features(one_series(tampered_values)).iloc[-1]

    pd.testing.assert_series_equal(
        base[FEATURE_COLUMNS].astype(float),
        tampered[FEATURE_COLUMNS].astype(float),
        check_names=False,
    )


def test_rows_without_enough_history_are_dropped():
    df = one_series(list(range(100)))

    feats = make_features(df)

    # the longest lookback is 14 (horizon) + 56 (longest window) - 1 days
    assert feats["date"].min() == dt.date(2025, 1, 1) + dt.timedelta(days=69)


def test_series_are_not_mixed_up(small_demand):
    feats = make_features(small_demand)
    one = feats[(feats.hospital_id == "H01") & (feats.medicine_id == feats.medicine_id.iloc[0])]
    raw = small_demand.set_index(["hospital_id", "medicine_id", "date"])["units"]

    row = one.iloc[10]
    expected = raw.loc[("H01", row.medicine_id, row.date - dt.timedelta(days=14))]
    assert row.lag_14 == expected


def test_calendar_features():
    df = one_series([1] * 120)
    feats = make_features(df).set_index("date")

    friday = dt.date(2025, 3, 14)
    assert feats.loc[friday, "day_of_week"] == 4
    assert feats.loc[friday, "is_weekend"] == 1
    assert feats.loc[friday, "is_ramadan"] == 1
    assert -1 <= feats.loc[friday, "doy_sin"] <= 1
