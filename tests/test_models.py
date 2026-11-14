"""Tests for the candidate models and their evaluation."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from demand_forecast.evaluate import backtest, forecast_origins, metrics
from demand_forecast.features import FEATURE_COLUMNS, make_features
from demand_forecast.models import MODEL_FACTORIES, MovingAverage, SeasonalNaive


@pytest.fixture(scope="module")
def feats(small_demand):
    return make_features(small_demand)


class TestMetrics:
    def test_perfect_forecast_has_zero_error(self):
        m = metrics(np.array([10, 20]), np.array([10, 20]))

        assert m == {"mae": 0.0, "rmse": 0.0, "wape": 0.0, "bias": 0.0}

    def test_known_values(self):
        # errors: +10 and -20 -> MAE 15, RMSE sqrt(250), WAPE 30/100, bias -10/100
        m = metrics(np.array([40, 60]), np.array([50, 40]))

        assert m["mae"] == pytest.approx(15.0)
        assert m["rmse"] == pytest.approx(np.sqrt(250))
        assert m["wape"] == pytest.approx(30.0)
        assert m["bias"] == pytest.approx(-10.0)

    def test_all_zero_actuals_cannot_be_scored_in_percent(self):
        with pytest.raises(ValueError, match="zero"):
            metrics(np.array([0, 0]), np.array([1, 1]))


class TestBaselines:
    def test_seasonal_naive_repeats_the_value_from_two_weeks_ago(self, feats):
        pred = (
            SeasonalNaive()
            .fit(feats[FEATURE_COLUMNS], feats["units"])
            .predict(feats[FEATURE_COLUMNS])
        )

        np.testing.assert_array_equal(pred, feats["lag_14"].to_numpy())

    def test_moving_average_uses_the_28_day_mean(self, feats):
        pred = (
            MovingAverage()
            .fit(feats[FEATURE_COLUMNS], feats["units"])
            .predict(feats[FEATURE_COLUMNS])
        )

        np.testing.assert_allclose(pred, feats["roll_mean_28"].to_numpy())


def test_all_models_return_non_negative_predictions(feats):
    X, y = feats[FEATURE_COLUMNS], feats["units"]

    for name, factory in MODEL_FACTORIES.items():
        pred = factory().fit(X, y).predict(X)
        assert (pred >= 0).all(), name
        assert len(pred) == len(X)


def test_forecast_origins_are_spaced_by_the_horizon():
    origins = forecast_origins(dt.date(2025, 1, 1), dt.date(2025, 2, 28))

    assert origins[0] == dt.date(2025, 1, 1)
    assert all((b - a).days == 14 for a, b in zip(origins, origins[1:], strict=False))
    assert origins[-1] + dt.timedelta(days=13) <= dt.date(2025, 2, 28)


def test_backtest_never_trains_on_data_from_the_test_window(feats):
    seen: list[tuple[dt.date, dt.date]] = []

    class Spy(SeasonalNaive):
        def fit(self, X, y):
            seen.append(X.attrs["max_date"])
            return self

    origins = [dt.date(2025, 1, 15), dt.date(2025, 1, 29)]
    backtest(feats, Spy, origins)

    assert seen == [dt.date(2025, 1, 14), dt.date(2025, 1, 28)]


def test_backtest_returns_one_prediction_per_test_row(feats):
    origins = [dt.date(2025, 1, 15), dt.date(2025, 1, 29)]

    preds = backtest(feats, SeasonalNaive, origins)

    expected_rows = 2 * 3 * 14 * 2  # hospitals x medicines x days x origins
    assert len(preds) == expected_rows
    assert {"date", "hospital_id", "medicine_id", "units", "prediction", "origin"} <= set(
        preds.columns
    )


def test_gradient_boosting_beats_the_naive_baseline(feats):
    origins = forecast_origins(dt.date(2024, 11, 1), dt.date(2025, 2, 28))
    naive = backtest(feats, MODEL_FACTORIES["Seasonal naive"], origins)
    gbm = backtest(feats, MODEL_FACTORIES["Gradient boosting"], origins)

    naive_wape = metrics(naive["units"], naive["prediction"])["wape"]
    gbm_wape = metrics(gbm["units"], gbm["prediction"])["wape"]
    assert gbm_wape < naive_wape


def test_model_list_contains_baselines_and_ml_models():
    assert set(MODEL_FACTORIES) == {
        "Seasonal naive",
        "Moving average (28d)",
        "Ridge regression",
        "Random forest",
        "Gradient boosting",
    }


def test_predictions_are_in_a_dataframe_aligned_with_inputs(feats):
    origins = [dt.date(2025, 2, 1)]
    preds = backtest(feats, MovingAverage, origins)

    merged = preds.merge(feats, on=["date", "hospital_id", "medicine_id"], suffixes=("", "_f"))
    assert (merged["units"] == merged["units_f"]).all()
    assert isinstance(preds, pd.DataFrame)
