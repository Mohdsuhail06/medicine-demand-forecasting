"""Tests for the training pipeline, the saved model bundle and forecasting with it."""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

import demand_forecast.evaluate as ev
from demand_forecast.evaluate import backtest
from demand_forecast.features import make_features
from demand_forecast.forecast import ModelBundle, forecast_series, recommend
from demand_forecast.models import MODEL_FACTORIES, SeasonalNaive
from demand_forecast.train import split_periods, train
from demand_forecast.uncertainty import UncertaintyModel

FAST_MODELS = {k: MODEL_FACTORIES[k] for k in ("Seasonal naive", "Gradient boosting")}


@pytest.fixture(scope="module")
def trained(small_demand, tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    result = train(small_demand, out / "models", out / "reports", factories=FAST_MODELS)
    return result, out


def test_split_periods_puts_holdout_at_the_end():
    periods = split_periods(dt.date(2025, 12, 31), holdout_days=92, validation_days=183)

    assert periods.holdout_end == dt.date(2025, 12, 31)
    assert periods.holdout_start == dt.date(2025, 10, 1)
    assert periods.validation_end == dt.date(2025, 9, 30)
    assert periods.validation_start == dt.date(2025, 4, 1)


def test_best_model_is_chosen_on_validation_not_holdout(trained):
    result, _ = trained

    assert result.best_model == result.validation_scores.iloc[0]["model"]
    assert result.best_model == "Gradient boosting"


def test_reports_and_model_files_are_written(trained):
    _, out = trained

    assert (out / "models" / "model.joblib").exists()
    files = {p.name for p in (out / "reports").iterdir()}
    assert {
        "metrics.json",
        "RESULTS.md",
        "model_comparison.png",
        "forecast_example.png",
        "feature_importance.png",
        "error_by_category.png",
    } <= files


def test_metrics_json_contains_both_evaluation_stages(trained):
    _, out = trained

    data = json.loads((out / "reports" / "metrics.json").read_text(encoding="utf-8"))

    assert data["best_model"] == "Gradient boosting"
    assert {row["model"] for row in data["validation"]} == set(FAST_MODELS)
    assert {row["model"] for row in data["holdout"]} == set(FAST_MODELS)
    assert data["data"]["series"] == 6
    assert set(data["safety_stock_coverage_95"]) == {"3", "7", "14"}
    assert all(0 <= v <= 1 for v in data["safety_stock_coverage_95"].values())


def test_bundle_round_trip(trained):
    _, out = trained

    bundle = ModelBundle.load(out / "models" / "model.joblib")

    assert bundle.name == "Gradient boosting"
    assert bundle.trained_until == dt.date(2025, 2, 28)
    assert bundle.uncertainty.series_std[("H01", "AMOXICILLIN")] > 0
    assert len(bundle.uncertainty.factors) == 14


def test_forecast_covers_the_next_14_days(trained, small_demand):
    _, out = trained
    bundle = ModelBundle.load(out / "models" / "model.joblib")

    fc = forecast_series(bundle, small_demand, "H01", "AMOXICILLIN")

    assert list(fc.columns) == ["date", "forecast_units"]
    assert fc["date"].tolist() == [dt.date(2025, 3, 1) + dt.timedelta(d) for d in range(14)]
    assert (fc["forecast_units"] >= 0).all()


def test_forecast_is_close_to_recent_demand_level(trained, small_demand):
    _, out = trained
    bundle = ModelBundle.load(out / "models" / "model.joblib")
    series = small_demand[
        (small_demand.hospital_id == "H01") & (small_demand.medicine_id == "PARACETAMOL")
    ]
    recent_mean = series.tail(28)["units"].mean()

    fc = forecast_series(bundle, small_demand, "H01", "PARACETAMOL")

    assert fc["forecast_units"].mean() == pytest.approx(recent_mean, rel=0.35)


def test_unknown_series_raises_key_error(trained, small_demand):
    _, out = trained
    bundle = ModelBundle.load(out / "models" / "model.joblib")

    with pytest.raises(KeyError, match="H09"):
        forecast_series(bundle, small_demand, "H09", "AMOXICILLIN")


def test_recommend_combines_forecast_and_reorder_logic(trained, small_demand):
    _, out = trained
    bundle = ModelBundle.load(out / "models" / "model.joblib")

    result = recommend(
        bundle, small_demand, "H01", "AMOXICILLIN", current_stock=0, lead_time_days=7
    )

    assert len(result["forecast"]) == 14
    rec = result["recommendation"]
    assert rec["status"] == "Urgent order"
    assert rec["order_quantity"] > 0


def test_lead_time_longer_than_horizon_is_rejected(trained, small_demand):
    _, out = trained
    bundle = ModelBundle.load(out / "models" / "model.joblib")

    with pytest.raises(ValueError, match="lead_time_days"):
        recommend(bundle, small_demand, "H01", "AMOXICILLIN", current_stock=0, lead_time_days=30)


def test_training_on_too_little_data_fails_clearly():
    tiny = pd.DataFrame(
        {
            "date": [dt.date(2025, 1, 1) + dt.timedelta(d) for d in range(60)],
            "hospital_id": "H01",
            "medicine_id": "AMOXICILLIN",
            "category": "Antibiotic",
            "units": 5,
        }
    )

    with pytest.raises(ValueError, match="history"):
        train(tiny, "unused", "unused", factories=FAST_MODELS)


class GoodOnlyInSummer(SeasonalNaive):
    """Stand-in model; the test below makes it perfect in May-Oct and useless otherwise."""


def test_selection_ignores_the_holdout(small_demand, tmp_path, monkeypatch):
    """A model that wins on validation must be chosen even if it loses on the holdout."""

    original = ev.backtest

    def backtest_with_truth(feats, factory, origins):
        preds = original(feats, factory, origins)
        if factory is GoodOnlyInSummer:
            summer = pd.to_datetime(preds["date"]).dt.month.between(5, 10)
            preds["prediction"] = np.where(summer, preds["units"], 0.0)
        return preds

    monkeypatch.setattr(ev, "backtest", backtest_with_truth)
    factories = {
        "Summer only": GoodOnlyInSummer,
        "Gradient boosting": FAST_MODELS["Gradient boosting"],
    }

    result = train(small_demand, tmp_path / "m", tmp_path / "r", factories=factories)

    hold = result.holdout_scores.set_index("model")["wape"]
    assert hold["Summer only"] > hold["Gradient boosting"]  # loses on the holdout...
    assert result.best_model == "Summer only"  # ...but was chosen on validation


def test_serving_gives_the_same_forecast_as_the_backtest(small_demand):
    """Train/serve consistency: same model + same history -> same numbers."""
    origin = dt.date(2025, 1, 15)
    feats = make_features(small_demand)
    backtested = backtest(feats, SeasonalNaive, [origin])
    expected = (
        backtested[(backtested.hospital_id == "H01") & (backtested.medicine_id == "AMOXICILLIN")]
        .sort_values("date")["prediction"]
        .to_numpy()
    )
    bundle = ModelBundle(
        name="naive",
        model=SeasonalNaive(),
        trained_until=origin - dt.timedelta(days=1),
        uncertainty=UncertaintyModel({}, 1.0, [1.0] * 14),
    )
    history = small_demand[small_demand["date"] < origin]

    served = forecast_series(bundle, history, "H01", "AMOXICILLIN")

    np.testing.assert_allclose(served["forecast_units"].to_numpy(), expected)
