"""Turn a demand forecast into a pharmacy decision: order now, how much, or wait."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from statistics import NormalDist

OVERSTOCK_DAYS = 60  # more than ~2 months of cover risks expiry


@dataclass(frozen=True)
class Recommendation:
    expected_demand: float  # units needed during the lead time
    safety_stock: float  # buffer against forecast error
    reorder_point: float  # expected demand + safety stock
    order_quantity: int  # units to order now (0 = no order)
    days_of_cover: float  # how many days current stock lasts at forecast demand
    status: str  # "Urgent order" | "Reorder" | "OK" | "Overstock"

    def to_dict(self) -> dict:
        data = asdict(self)
        data["days_of_cover"] = (
            None if math.isinf(self.days_of_cover) else round(self.days_of_cover, 1)
        )
        for key in ("expected_demand", "safety_stock", "reorder_point"):
            data[key] = round(data[key], 1)
        return data


def reorder_recommendation(
    daily_forecast: Sequence[float],
    current_stock: float,
    daily_error_std: float,
    service_level: float = 0.95,
) -> Recommendation:
    """Classic reorder-point logic with a safety stock sized from forecast error.

    Args:
        daily_forecast: forecast units for each day of the supplier lead time.
        current_stock: units on the shelf now.
        daily_error_std: standard deviation of the model's daily forecast error.
        service_level: probability of not running out during the lead time (e.g. 0.95).
    """
    if len(daily_forecast) == 0:
        raise ValueError("daily_forecast must contain at least one day")
    if current_stock < 0 or daily_error_std < 0:
        raise ValueError("current_stock and daily_error_std must be >= 0")
    if not 0 < service_level < 1:
        raise ValueError("service_level must be between 0 and 1 (exclusive)")

    lead_time = len(daily_forecast)
    expected = float(sum(daily_forecast))
    z = NormalDist().inv_cdf(service_level)
    safety = z * daily_error_std * math.sqrt(lead_time)
    reorder_point = expected + safety
    order_qty = max(0, math.ceil(reorder_point - current_stock))

    mean_daily = expected / lead_time
    cover = math.inf if mean_daily == 0 else current_stock / mean_daily

    if cover < lead_time / 2:
        status = "Urgent order"
    elif order_qty > 0:
        status = "Reorder"
    elif cover > OVERSTOCK_DAYS:
        status = "Overstock"
    else:
        status = "OK"
    return Recommendation(expected, safety, reorder_point, order_qty, cover, status)
