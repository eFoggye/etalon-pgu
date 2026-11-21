from dataclasses import astuple, replace

import numpy as np
import pytest

from powerplant.economics import (
    DowntimeInputs,
    PlantAssumptions,
    breakeven_reduction,
    cumulative_result,
    downtime_economics,
    vacuum_loss_profile,
)

# Пример ментора: 16 ч простоя, 5 000 ₽/ч, сокращение 25%, внедрение 60 000 ₽, 2 000 ₽/мес.
MENTOR = DowntimeInputs(
    downtime_hours_per_month=16,
    cost_per_hour_rub=5_000,
    reduction_share=0.25,
    implementation_rub=60_000,
    support_rub_per_month=2_000,
)


def test_downtime_economics_reproduces_mentor_template():
    result = downtime_economics(MENTOR)

    assert astuple(result) == pytest.approx((80_000, 4, 20_000, 18_000, 216_000, 156_000, 10 / 3))


def test_payback_is_infinite_when_support_eats_the_saving():
    result = downtime_economics(replace(MENTOR, reduction_share=0.02))

    assert result.payback_months == float("inf")


def test_cumulative_result_starts_at_minus_implementation():
    curve = cumulative_result(MENTOR, months=12)

    assert curve[0] == pytest.approx(-60_000)
    assert len(curve) == 13


def test_cumulative_result_ends_at_first_year_result():
    curve = cumulative_result(MENTOR, months=12)

    assert curve[12] == pytest.approx(downtime_economics(MENTOR).first_year_result_rub)


def test_payback_is_infinite_when_saving_only_covers_support():
    result = downtime_economics(replace(MENTOR, cost_per_hour_rub=500))  # 4 ч × 500 = 2 000

    assert result.payback_months == float("inf")


def test_breakeven_is_infinite_without_downtime():
    assert breakeven_reduction(replace(MENTOR, downtime_hours_per_month=0)) == float("inf")


def test_cumulative_result_crosses_zero_at_payback():
    curve = cumulative_result(MENTOR, months=12)

    assert curve[3] < 0 < curve[4]  # окупаемость 3,33 мес


def test_breakeven_reduction_pays_back_in_exactly_the_given_months():
    share = breakeven_reduction(MENTOR, months=12)

    result = downtime_economics(replace(MENTOR, reduction_share=share))
    assert result.payback_months == pytest.approx(12)


@pytest.mark.parametrize(
    ("field", "value"),
    [("reduction_share", 1.5), ("reduction_share", -0.1), ("cost_per_hour_rub", -1.0)],
)
def test_inputs_reject_impossible_values(field, value):
    with pytest.raises(ValueError):
        replace(MENTOR, **{field: value})


def test_plant_assumptions_derive_template_inputs_from_sources():
    inputs = PlantAssumptions().to_inputs()

    # 5% × половина × доля газовых турбин 313,9 / 463,9 × 730 ч; 463,9 МВт × 1 747 ₽;
    # 20 дней × 8 ч × 243 000 × 1,3 / (1 972 / 12); сервер 5 838 ₽ + 4 ч поддержки по той же ставке
    expected = (12.349, 810_433.3, 0.01, 307_570, 13_527)
    assert astuple(inputs) == pytest.approx(expected, rel=1e-3)


def test_vacuum_loss_profile_reports_worst_hours():
    plan = np.full(100, 450.0)
    reference = np.concatenate([np.full(90, 451.0), np.full(10, 440.0)])

    profile = vacuum_loss_profile(plan, reference, worst_share=0.1)

    assert profile["share_hours_worse"] == pytest.approx(0.1)
    assert profile["worst_mean_loss_mw"] == pytest.approx(10.0)
