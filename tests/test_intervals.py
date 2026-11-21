import numpy as np
import pandas as pd
import pytest

from powerplant.detector import decompose, detection_rate, flag_windows, window_threshold
from powerplant.intervals import conformal_quantile, coverage, interval, shortfall_probability
from powerplant.novelty import check_novelty, fit_novelty


def test_conformal_quantile_uses_finite_sample_rank():
    residuals = np.arange(1.0, 101.0)

    assert conformal_quantile(residuals, level=0.9) == pytest.approx(91.0)


def test_conformal_interval_covers_requested_share_on_new_data():
    rng = np.random.default_rng(0)
    calibration = np.abs(rng.normal(0, 3, 5000))
    new_errors = rng.normal(0, 3, 5000)

    q = conformal_quantile(calibration, level=0.9)
    low, high = interval(np.zeros(5000), q)

    assert coverage(new_errors, low, high) == pytest.approx(0.9, abs=0.02)


def test_shortfall_probability_grows_with_required_volume():
    residuals = np.random.default_rng(1).normal(0, 3, 2000)

    probabilities = [shortfall_probability(450.0, plan, residuals) for plan in (430, 448, 452, 470)]

    assert probabilities == sorted(probabilities)
    assert probabilities[0] == pytest.approx(0.0, abs=0.01)
    assert probabilities[-1] == pytest.approx(1.0, abs=0.01)


@pytest.fixture(scope="module")
def correlated_training() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    temperature = rng.uniform(0, 35, 3000)
    return pd.DataFrame(
        {
            "AT": temperature,
            "V": 30 + 1.3 * temperature + rng.normal(0, 3, 3000),
            "AP": rng.normal(1013, 6, 3000),
            "RH": rng.uniform(30, 100, 3000),
        }
    )


def test_novelty_is_silent_inside_the_data(correlated_training):
    model = fit_novelty(correlated_training)

    assert check_novelty(model, {"AT": 15.0, "V": 49.5, "AP": 1013.0, "RH": 70.0}) == []


def test_novelty_flags_values_outside_training_range(correlated_training):
    model = fit_novelty(correlated_training)

    warnings = check_novelty(model, {"AT": 42.0, "V": 85.0, "AP": 1013.0, "RH": 70.0})

    assert any("Температура" in w for w in warnings)


def test_novelty_flags_unseen_combination_inside_ranges(correlated_training):
    model = fit_novelty(correlated_training)

    warnings = check_novelty(model, {"AT": 33.0, "V": 35.0, "AP": 1013.0, "RH": 70.0})

    assert any("Сочетание" in w for w in warnings)


def test_decomposition_parts_add_up():
    parts = decompose(plan=460.0, reference=455.0, actual=448.0)

    assert parts.total == pytest.approx(-12.0)
    assert parts.vacuum == pytest.approx(-5.0)
    assert parts.other == pytest.approx(-7.0)


def test_daily_threshold_matches_theory_for_normal_residuals():
    residuals = np.random.default_rng(3).normal(0, 3, 20000)

    threshold = window_threshold(residuals, window=24, false_alarm_rate=0.01)

    assert threshold == pytest.approx(-2.326 * 3 / np.sqrt(24), abs=0.15)


def test_detection_rate_separates_healthy_and_degraded_plant():
    residuals = np.random.default_rng(4).normal(0, 3, 20000)
    threshold = window_threshold(residuals, window=24, false_alarm_rate=0.01)

    healthy = detection_rate(residuals, window=24, threshold=threshold, shift_mw=0.0)
    degraded = detection_rate(residuals, window=24, threshold=threshold, shift_mw=-4.5)

    assert healthy == pytest.approx(0.01, abs=0.01)
    assert degraded > 0.99


def test_flag_windows_marks_sustained_drop_only():
    deviations = np.concatenate([np.zeros(48), np.full(48, -5.0)])

    flags = flag_windows(deviations, window=24, threshold=-2.0)

    assert not flags[:48].any()
    assert flags[-1]


def test_flag_windows_gives_no_alarm_on_file_shorter_than_window():
    flags = flag_windows(np.full(10, -5.0), window=24, threshold=-2.0)

    assert not flags.any()


def test_empty_residuals_are_rejected():
    with pytest.raises(ValueError):
        shortfall_probability(450.0, 440.0, np.array([]))
    with pytest.raises(ValueError):
        window_threshold(np.array([]), window=24)


def test_novelty_survives_constant_feature():
    frame = pd.DataFrame({"AT": np.linspace(0, 30, 50), "AP": np.full(50, 1013.0)})

    model = fit_novelty(frame)

    assert check_novelty(model, {"AT": 15.0, "AP": 1013.0}) == []
