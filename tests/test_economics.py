import numpy as np
import pytest

from powerplant.economics import (
    degradation_loss_rub,
    forecast_error_saved_mwh,
    payback_months,
    revenue_rub,
    vacuum_loss_profile,
)


def test_forecast_error_saved_scales_with_hours():
    saved = forecast_error_saved_mwh(mae_base=4.8, mae_model=3.5, hours=1000)

    assert saved == pytest.approx(1300.0)


def test_forecast_error_saved_is_never_negative():
    assert forecast_error_saved_mwh(mae_base=3.0, mae_model=3.5, hours=1000) == 0.0


def test_revenue_uses_price_and_loss_share():
    assert revenue_rub(100.0, price=2000.0, loss_share=0.5) == pytest.approx(100_000.0)


def test_degradation_loss_counts_undetected_hours():
    loss = degradation_loss_rub(plant_mw=450.0, drop_share=0.02, hours=24, price=2000.0)

    assert loss == pytest.approx(450 * 0.02 * 24 * 2000)


def test_payback_months_from_annual_effect():
    assert payback_months(cost_rub=1_200_000, annual_effect_rub=2_400_000) == pytest.approx(6.0)


def test_payback_is_infinite_without_effect():
    assert payback_months(cost_rub=1.0, annual_effect_rub=0.0) == float("inf")


def test_vacuum_loss_profile_reports_worst_hours():
    plan = np.full(100, 450.0)
    reference = np.concatenate([np.full(90, 451.0), np.full(10, 440.0)])

    profile = vacuum_loss_profile(plan, reference, worst_share=0.1)

    assert profile["share_hours_worse"] == pytest.approx(0.1)
    assert profile["worst_mean_loss_mw"] == pytest.approx(10.0)
