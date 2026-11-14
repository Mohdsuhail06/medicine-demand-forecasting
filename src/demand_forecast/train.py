"""Training pipeline: compare models, pick the best on validation, confirm on a holdout,
train the final model on all data and write the model file plus a results report.

    |------ training ------|---- validation (model selection) ----|---- holdout (final check) ----|
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

from demand_forecast.evaluate import compare_models, forecast_origins, metrics  # noqa: E402
from demand_forecast.features import FEATURE_COLUMNS, HORIZON, MAX_LOOKBACK, make_features  # noqa: E402
from demand_forecast.forecast import ModelBundle  # noqa: E402
from demand_forecast.models import MODEL_FACTORIES  # noqa: E402
from demand_forecast.uncertainty import fit_uncertainty, safety_stock_coverage  # noqa: E402

MIN_TRAINING_DAYS = 56


@dataclass(frozen=True)
class Periods:
    validation_start: dt.date
    validation_end: dt.date
    holdout_start: dt.date
    holdout_end: dt.date


@dataclass
class TrainResult:
    best_model: str
    validation_scores: pd.DataFrame
    holdout_scores: pd.DataFrame
    bundle_path: Path


def split_periods(
    last_date: dt.date, holdout_days: int = 92, validation_days: int = 183
) -> Periods:
    """Holdout = the last `holdout_days`; validation = the `validation_days` before it."""
    holdout_start = last_date - dt.timedelta(days=holdout_days - 1)
    validation_end = holdout_start - dt.timedelta(days=1)
    validation_start = validation_end - dt.timedelta(days=validation_days - 1)
    return Periods(validation_start, validation_end, holdout_start, last_date)


def _round_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.round({"mae": 2, "rmse": 2, "wape": 2, "bias": 2})


def _to_markdown(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    divider = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def _plot_comparison(scores: pd.DataFrame, best: str, path: Path) -> None:
    data = scores.sort_values("wape", ascending=False)
    colors = ["#1565c0" if m == best else "#90a4ae" for m in data["model"]]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(data["model"], data["wape"], color=colors)
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_xlabel("WAPE on holdout (%) - lower is better")
    ax.set_title("Model comparison (holdout period)")
    ax.set_xlim(0, data["wape"].max() * 1.18)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_examples(preds: pd.DataFrame, path: Path) -> None:
    keys = preds.groupby(["hospital_id", "medicine_id"])["units"].mean().sort_values()
    picks = [keys.index[-1], keys.index[len(keys) // 2]]
    fig, axes = plt.subplots(len(picks), 1, figsize=(11, 3.6 * len(picks)), sharex=True)
    for ax, (hospital, medicine) in zip(axes, picks, strict=True):
        s = preds[(preds.hospital_id == hospital) & (preds.medicine_id == medicine)]
        s = s.sort_values("date")
        ax.plot(s["date"], s["units"], color="#455a64", linewidth=1.2, label="Actual")
        ax.plot(s["date"], s["prediction"], color="#e53935", linewidth=1.8, label="Forecast")
        ax.set_title(f"{hospital} - {medicine}")
        ax.set_ylabel("units / day")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")
    fig.suptitle("Forecast vs actual on the holdout period (14-day-ahead forecasts)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_importance(model, feats: pd.DataFrame, path: Path) -> None:
    sample = feats.sample(min(len(feats), 3000), random_state=0)
    result = permutation_importance(
        model,
        sample[FEATURE_COLUMNS],
        sample["units"],
        n_repeats=3,
        random_state=0,
        scoring="neg_mean_absolute_error",
    )
    imp = pd.Series(result.importances_mean, index=FEATURE_COLUMNS).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(imp.index, imp.values, color="#43a047")
    ax.set_xlabel("Increase in MAE when the feature is shuffled (units)")
    ax.set_title("Which features matter most? (permutation importance)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_by_category(by_cat: pd.DataFrame, path: Path) -> None:
    data = by_cat.sort_values("wape")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(data["category"], data["wape"], color="#fb8c00")
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_xlabel("WAPE on holdout (%)")
    ax.set_title("Forecast error by medicine category (best model)")
    ax.set_xlim(0, data["wape"].max() * 1.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def train(
    demand: pd.DataFrame,
    models_dir: Path | str,
    reports_dir: Path | str,
    factories: dict[str, Callable] | None = None,
    holdout_days: int = 92,
    validation_days: int = 183,
) -> TrainResult:
    """Run the full training and evaluation pipeline and save the best model.

    Raises:
        ValueError: if the data does not cover training + validation + holdout.
    """
    factories = factories or MODEL_FACTORIES
    feats = make_features(demand)
    last = max(demand["date"])
    periods = split_periods(last, holdout_days, validation_days)
    if feats.empty or (periods.validation_start - feats["date"].min()).days < MIN_TRAINING_DAYS:
        needed = MAX_LOOKBACK + MIN_TRAINING_DAYS + validation_days + holdout_days
        raise ValueError(f"Not enough history: about {needed} days of data are needed")

    val_origins = forecast_origins(periods.validation_start, periods.validation_end)
    hold_origins = forecast_origins(periods.holdout_start, periods.holdout_end)

    val_scores, val_preds = compare_models(feats, factories, val_origins)
    best = str(val_scores.iloc[0]["model"])
    hold_scores, hold_preds = compare_models(feats, factories, hold_origins)
    best_preds = hold_preds[best]

    # Uncertainty is estimated on validation and checked on the holdout, never both on one.
    uncertainty = fit_uncertainty(val_preds[best])
    coverage = safety_stock_coverage(best_preds, uncertainty, service_level=0.95)

    # The deployed model uses all data. The scores reported are the backtest scores.
    final_model = factories[best]().fit(feats[FEATURE_COLUMNS], feats["units"])
    bundle = ModelBundle(
        name=best,
        model=final_model,
        trained_until=last,
        uncertainty=uncertainty,
        metrics={"holdout_backtest": metrics(best_preds["units"], best_preds["prediction"])},
    )
    bundle_path = bundle.save(Path(models_dir) / "model.joblib")

    # ---- report ----
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    by_cat = pd.DataFrame(
        [
            {"category": cat, **metrics(g["units"], g["prediction"])}
            for cat, g in best_preds.groupby("category")
        ]
    )
    _plot_comparison(hold_scores, best, reports / "model_comparison.png")
    _plot_examples(best_preds, reports / "forecast_example.png")
    # Importance is measured out of sample: a model fit before the holdout, scored on it.
    before = feats[feats["date"] < periods.holdout_start]
    pre_holdout_model = factories[best]().fit(before[FEATURE_COLUMNS], before["units"])
    holdout_feats = feats[feats["date"] >= periods.holdout_start]
    _plot_importance(pre_holdout_model, holdout_feats, reports / "feature_importance.png")
    _plot_by_category(by_cat, reports / "error_by_category.png")

    last_day = dt.timedelta(days=HORIZON - 1)
    period_info = {
        "validation_scored": f"{val_origins[0]} to {val_origins[-1] + last_day}",
        "validation_windows": len(val_origins),
        "holdout_scored": f"{hold_origins[0]} to {hold_origins[-1] + last_day}",
        "holdout_windows": len(hold_origins),
    }
    data_info = {
        "rows": len(demand),
        "series": int(demand.groupby(["hospital_id", "medicine_id"]).ngroups),
        "start": min(demand["date"]).isoformat(),
        "end": last.isoformat(),
    }
    summary = {
        "best_model": best,
        "periods": period_info,
        "data": data_info,
        "validation": _round_table(val_scores).to_dict(orient="records"),
        "holdout": _round_table(hold_scores).to_dict(orient="records"),
        "holdout_by_category": _round_table(by_cat).to_dict(orient="records"),
        "safety_stock_coverage_95": {str(k): round(v, 3) for k, v in coverage.items()},
    }
    (reports / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    naive = hold_scores.set_index("model")["wape"].get("Seasonal naive")
    best_wape = hold_scores.set_index("model").loc[best, "wape"]
    gain = (
        f"WAPE drops from **{naive:.1f}%** (seasonal naive) to **{best_wape:.1f}%** "
        f"({(naive - best_wape) / naive * 100:.0f}% fewer errors)."
        if naive is not None and best != "Seasonal naive"
        else ""
    )
    lines = [
        "# Results",
        "",
        f"Data: {data_info['rows']:,} rows, {data_info['series']} series "
        f"({data_info['start']} to {data_info['end']}).",
        "",
        f"Best model (chosen on validation): **{best}**. {gain}",
        "",
        f"## Validation ({period_info['validation_scored']}, "
        f"{len(val_origins)} forecast windows) - used to choose the model",
        "",
        _to_markdown(_round_table(val_scores)),
        "",
        f"## Holdout ({period_info['holdout_scored']}, {len(hold_origins)} forecast windows) "
        "- final check, not used to choose the model",
        "",
        _to_markdown(_round_table(hold_scores)),
        "",
        "![Model comparison](model_comparison.png)",
        "",
        "## Error by category (best model, holdout)",
        "",
        _to_markdown(_round_table(by_cat)),
        "",
        "![Error by category](error_by_category.png)",
        "",
        "## Safety stock check (target: 95% of windows without a stock-out)",
        "",
        "| lead time (days) | covered on holdout |",
        "| --- | --- |",
        *[f"| {k} | {v * 100:.1f}% |" for k, v in coverage.items()],
        "",
        "## Forecast examples",
        "",
        "![Forecast examples](forecast_example.png)",
        "",
        "## Feature importance",
        "",
        "![Feature importance](feature_importance.png)",
        "",
    ]
    (reports / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    return TrainResult(best, val_scores, hold_scores, bundle_path)
