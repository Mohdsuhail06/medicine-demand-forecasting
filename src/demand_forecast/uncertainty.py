"""How uncertain is a forecast over the supplier lead time?

The safety stock protects against the error of the *total* demand during the lead time.
If daily errors were independent, that error would grow with sqrt(lead time). In practice
errors on neighbouring days move together (a level shift, an outbreak), so they add up
faster. We therefore measure it from backtest errors:

    lead-time std(L) = factor[L] x daily RMSE of the series x sqrt(L)

`factor[L]` is 1.0 for independent errors and grows towards sqrt(L) when errors are
fully correlated. It is estimated on the validation backtest, and its coverage is then
checked on the separate holdout period.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

from demand_forecast.features import HORIZON

SERIES = ["hospital_id", "medicine_id"]


@dataclass
class UncertaintyModel:
    series_std: dict[tuple[str, str], float]  # daily RMSE per (hospital, medicine)
    default_std: float  # pooled daily RMSE, used for a series without its own value
    factors: list[float]  # one correction factor per lead time 1..HORIZON

    def lead_time_std(self, hospital_id: str, medicine_id: str, lead_time: int) -> float:
        """Standard deviation of the forecast error of total demand over `lead_time` days."""
        if not 1 <= lead_time <= len(self.factors):
            raise ValueError(f"lead_time must be between 1 and {len(self.factors)}")
        daily = self.series_std.get((hospital_id, medicine_id), self.default_std)
        return self.factors[lead_time - 1] * daily * np.sqrt(lead_time)


def lead_time_errors(preds: pd.DataFrame) -> pd.DataFrame:
    """Cumulative error, forecast and actual for lead times 1..14 within each forecast origin."""
    df = preds.sort_values([*SERIES, "origin", "date"]).copy()
    df["error"] = df["prediction"] - df["units"]
    grouped = df.groupby([*SERIES, "origin"], sort=False)
    df["lead_time"] = grouped.cumcount() + 1
    df["error_sum"] = grouped["error"].cumsum()
    df["forecast_sum"] = grouped["prediction"].cumsum()
    df["actual_sum"] = grouped["units"].cumsum()
    return df[[*SERIES, "origin", "lead_time", "error_sum", "forecast_sum", "actual_sum"]]


def fit_uncertainty(preds: pd.DataFrame) -> UncertaintyModel:
    """Estimate daily RMSE per series and the lead-time correction factors from backtest output."""
    errors = preds["prediction"] - preds["units"]
    rmse = errors.pow(2).groupby([preds[c] for c in SERIES]).mean().pow(0.5)
    series_std = {key: float(v) for key, v in rmse.items()}
    default_std = float(np.sqrt(np.mean(errors**2)))

    le = lead_time_errors(preds)
    daily = np.array(
        [series_std[k] for k in zip(le["hospital_id"], le["medicine_id"], strict=True)]
    )
    scale = np.maximum(daily, 1e-9) * np.sqrt(le["lead_time"].to_numpy())
    normalized = le["error_sum"].to_numpy() / scale
    by_lead = pd.Series(normalized**2).groupby(le["lead_time"].to_numpy()).mean().pow(0.5)
    factors = [float(by_lead.get(lead, 1.0)) for lead in range(1, HORIZON + 1)]
    return UncertaintyModel(series_std, default_std, factors)


def safety_stock_coverage(
    preds: pd.DataFrame,
    model: UncertaintyModel,
    service_level: float = 0.95,
    lead_times: tuple[int, ...] = (3, 7, 14),
) -> dict[int, float]:
    """Share of forecast windows where demand stayed below forecast + safety stock."""
    z = NormalDist().inv_cdf(service_level)
    le = lead_time_errors(preds)
    result = {}
    for lead in lead_times:
        rows = le[le["lead_time"] == lead]
        std = np.array(
            [
                model.lead_time_std(h, m, lead)
                for h, m in zip(rows["hospital_id"], rows["medicine_id"], strict=True)
            ]
        )
        covered = rows["actual_sum"].to_numpy() <= rows["forecast_sum"].to_numpy() + z * std
        result[lead] = float(covered.mean())
    return result
