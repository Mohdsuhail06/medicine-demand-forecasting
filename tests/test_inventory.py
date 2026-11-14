"""Tests for turning a forecast into a reorder recommendation."""

from __future__ import annotations

import pytest

from demand_forecast.inventory import reorder_recommendation


def test_no_order_needed_when_stock_covers_demand_and_safety_stock():
    rec = reorder_recommendation([10] * 7, current_stock=500, daily_error_std=5)

    assert rec.order_quantity == 0
    assert rec.status == "OK"


def test_order_fills_the_gap_to_demand_plus_safety_stock():
    # demand over 7 days = 70, safety stock = 1.645 * 5 * sqrt(7) = 21.76 -> target 91.76
    rec = reorder_recommendation([10] * 7, current_stock=50, daily_error_std=5)

    assert rec.expected_demand == pytest.approx(70)
    assert rec.safety_stock == pytest.approx(21.76, abs=0.01)
    assert rec.order_quantity == 42  # ceil(91.76 - 50); 5 days of cover is not urgent
    assert rec.status == "Reorder"


def test_higher_service_level_means_more_safety_stock():
    low = reorder_recommendation([10] * 7, 0, 5, service_level=0.90)
    high = reorder_recommendation([10] * 7, 0, 5, service_level=0.99)

    assert high.safety_stock > low.safety_stock


def test_stock_running_out_before_half_the_lead_time_is_urgent():
    rec = reorder_recommendation([10] * 10, current_stock=20, daily_error_std=2)

    assert rec.days_of_cover == pytest.approx(2.0)
    assert rec.status == "Urgent order"


def test_far_too_much_stock_is_flagged_as_overstock():
    rec = reorder_recommendation([10] * 7, current_stock=2000, daily_error_std=2)

    assert rec.status == "Overstock"
    assert rec.days_of_cover == pytest.approx(200)


def test_zero_forecast_does_not_divide_by_zero():
    rec = reorder_recommendation([0] * 7, current_stock=10, daily_error_std=0)

    assert rec.days_of_cover == float("inf")
    assert rec.order_quantity == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"daily_forecast": [], "current_stock": 1, "daily_error_std": 1},
        {"daily_forecast": [1], "current_stock": -1, "daily_error_std": 1},
        {"daily_forecast": [1], "current_stock": 1, "daily_error_std": -1},
        {"daily_forecast": [1], "current_stock": 1, "daily_error_std": 1, "service_level": 1.0},
    ],
)
def test_invalid_inputs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        reorder_recommendation(**kwargs)
