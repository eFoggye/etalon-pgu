import numpy as np
import pandas as pd
import pytest

from powerplant.evidence import (
    VacuumFromWeather,
    calibration_table,
    dietterich_t,
    novelty_error_ratio,
    paired_error_test,
)


def test_paired_test_detects_clearly_better_model():
    rng = np.random.default_rng(0)
    y = rng.normal(450, 10, 2000)
    worse, better = y + rng.normal(0, 4, 2000), y + rng.normal(0, 2, 2000)

    result = paired_error_test(y, worse, better)

    assert result["mae_a"] > result["mae_b"]
    assert result["p_value"] < 0.001


def test_paired_test_sees_no_difference_between_equal_models():
    rng = np.random.default_rng(1)
    y = rng.normal(450, 10, 2000)
    noise = rng.normal(0, 3, (2, 2000))

    result = paired_error_test(y, y + noise[0], y + noise[1])

    assert result["p_value"] > 0.01


def test_dietterich_statistic_follows_5x2cv_formula():
    diffs = np.array([[1.0, 0.5], [0.8, 0.6], [1.2, 0.4], [0.9, 0.7], [1.1, 0.5]])

    t, p = dietterich_t(diffs)

    variances = [((row - row.mean()) ** 2).sum() for row in diffs]
    assert t == pytest.approx(diffs[0, 0] / np.sqrt(np.mean(variances)))
    assert 0 < p < 1


def test_vacuum_from_weather_needs_no_vacuum_at_prediction():
    rng = np.random.default_rng(2)
    at = rng.uniform(0, 35, 600)
    frame = pd.DataFrame(
        {"AT": at, "AP": rng.normal(1013, 5, 600), "RH": rng.uniform(30, 100, 600)}
    )
    frame = frame.assign(V=30 + 1.3 * at + rng.normal(0, 2, 600))
    y = 500 - 1.5 * frame["AT"] - 0.5 * frame["V"]

    model = VacuumFromWeather().fit(frame, y)
    prediction = model.predict(frame[["AT", "AP", "RH"]])

    assert prediction.shape == (600,)
    assert np.mean(np.abs(prediction - y)) < 5


def test_novelty_error_ratio_compares_flagged_hours():
    errors = np.array([1.0, 1.0, 1.0, 4.0])
    flags = np.array([False, False, False, True])

    result = novelty_error_ratio(errors, flags)

    assert result["ratio"] == pytest.approx(4.0)
    assert result["n_flagged"] == 1


def test_calibration_matches_declared_shortfall_probability():
    rng = np.random.default_rng(3)
    residuals = rng.normal(0, 3, 5000)
    y = 450 + rng.normal(0, 3, 5000)
    prediction = np.full(5000, 450.0)

    table = calibration_table(y, prediction, residuals, levels=(0.1, 0.5))

    assert table["наблюдалось"].tolist() == pytest.approx([0.1, 0.5], abs=0.03)
