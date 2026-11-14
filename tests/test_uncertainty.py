"""Tests for the forecast-uncertainty model behind the safety stock.

The safety stock needs the uncertainty of the *total* demand over the lead time. Daily
errors that move together (a level shift, an outbreak) add up faster than the textbook
sqrt(lead time) rule assumes, so the factor is measured from backtest errors.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from demand_forecast.uncertainty import (
    UncertaintyModel,
    fit_uncertainty,
    lead_time_errors,
    safety_stock_coverage,
)


def make_preds(errors_by_origin: list[list[float]], level: float = 100.0) -> pd.DataFrame:
    """Backtest-shaped predictions for one series; one inner list of daily errors per origin."""
    rows = []
    for i, errors in enumerate(errors_by_origin):
        origin = dt.date(2025, 1, 1) + dt.timedelta(days=14 * i)
        for d, err in enumerate(errors):
            rows.append(
                {
                    "date": origin + dt.timedelta(days=d),
                    "hospital_id": "H01",
                    "medicine_id": "AMOXICILLIN",
                    "units": level,
                    "prediction": level + err,
                    "origin": origin,
                }
            )
    return pd.DataFrame(rows)


def test_lead_time_errors_are_cumulative_within_each_origin():
    preds = make_preds([[1, 2, 3], [-1, -1, -1]])

    le = lead_time_errors(preds)

    first = le[le["origin"] == dt.date(2025, 1, 1)].sort_values("lead_time")
    assert first["error_sum"].tolist() == [1, 3, 6]
    assert first["actual_sum"].tolist() == [100, 200, 300]


def test_independent_daily_errors_follow_the_square_root_rule():
    rng = np.random.default_rng(0)
    preds = make_preds(rng.normal(0, 2, size=(4000, 14)).tolist())

    model = fit_uncertainty(preds)

    assert model.series_std[("H01", "AMOXICILLIN")] == pytest.approx(2.0, rel=0.05)
    assert model.factors[13] == pytest.approx(1.0, abs=0.08)


def test_errors_that_move_together_get_a_larger_factor():
    rng = np.random.default_rng(0)
    shifts = rng.normal(0, 2, size=4000)  # the same error on every day of an origin
    preds = make_preds([[s] * 14 for s in shifts])

    model = fit_uncertainty(preds)

    # fully correlated errors add up linearly: factor = L / sqrt(L) = sqrt(L)
    assert model.factors[13] == pytest.approx(np.sqrt(14), rel=0.08)


def test_bias_is_part_of_the_uncertainty():
    preds = make_preds([[3.0] * 14 for _ in range(50)])  # always 3 units too high

    model = fit_uncertainty(preds)

    assert model.series_std[("H01", "AMOXICILLIN")] == pytest.approx(3.0)


def test_lead_time_std_uses_series_value_or_default():
    model = UncertaintyModel(series_std={("H01", "X"): 2.0}, default_std=5.0, factors=[1.0] * 14)

    assert model.lead_time_std("H01", "X", 4) == pytest.approx(2.0 * 2)
    assert model.lead_time_std("H09", "Y", 4) == pytest.approx(5.0 * 2)


def test_lead_time_must_be_between_1_and_14():
    model = UncertaintyModel(series_std={}, default_std=1.0, factors=[1.0] * 14)

    with pytest.raises(ValueError, match="lead_time"):
        model.lead_time_std("H01", "X", 15)


def test_calibrated_safety_stock_reaches_its_service_level():
    rng = np.random.default_rng(1)
    shifts = rng.normal(0, 3, size=(3000, 1))
    noise = rng.normal(0, 2, size=(3000, 14))
    train_preds = make_preds((shifts + noise).tolist())
    shifts2 = rng.normal(0, 3, size=(3000, 1))
    noise2 = rng.normal(0, 2, size=(3000, 14))
    test_preds = make_preds((shifts2 + noise2).tolist())

    model = fit_uncertainty(train_preds)
    coverage = safety_stock_coverage(test_preds, model, service_level=0.95)

    for lead_time in (3, 7, 14):
        assert coverage[lead_time] == pytest.approx(0.95, abs=0.02)
