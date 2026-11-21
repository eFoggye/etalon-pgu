import numpy as np
import pandas as pd
import pytest

from powerplant.eda import (
    find_gaps,
    partial_correlation,
    quality_report,
    variance_inflation,
)

COLUMNS = ["AT", "V", "AP", "RH", "PE"]


def test_quality_report_counts_every_problem():
    df = pd.DataFrame(
        [
            (14.96, 41.76, 1024.07, 73.17, 463.26),
            (14.96, 41.76, 1024.07, 73.17, 463.26),  # точный дубль
            (14.96, 41.76, 1024.07, 73.17, 470.00),  # тот же вход, другая мощность
            (25.18, 62.96, 1020.04, 100.30, 444.37),  # влажность выше 100
            (5.11, None, 1012.16, 92.14, 488.56),  # пропуск
        ],
        columns=COLUMNS,
    )

    report = quality_report(df)

    assert report.rows == 5
    assert report.duplicates == 1
    assert report.conflicts == 1
    assert report.humidity_over_100 == 1
    assert report.missing["V"] == 1
    assert report.missing["AT"] == 0


def test_variance_inflation_flags_collinear_features():
    rng = np.random.default_rng(0)
    base = rng.normal(size=500)
    df = pd.DataFrame(
        {
            "a": base,
            "b": base + rng.normal(scale=0.05, size=500),
            "c": rng.normal(size=500),
        }
    )

    vif = variance_inflation(df, ["a", "b", "c"])

    assert vif["a"] > 10
    assert vif["b"] > 10
    assert vif["c"] == pytest.approx(1.0, abs=0.1)


def test_partial_correlation_removes_shared_driver():
    rng = np.random.default_rng(1)
    driver = rng.normal(size=2000)
    df = pd.DataFrame(
        {
            "driver": driver,
            "x": driver + rng.normal(scale=0.3, size=2000),
            "y": driver + rng.normal(scale=0.3, size=2000),
        }
    )

    raw = df["x"].corr(df["y"])
    partial = partial_correlation(df, "x", "y", ["driver"])

    assert raw > 0.8
    assert abs(partial) < 0.1


def test_find_gaps_reports_isolated_cluster():
    values = pd.Series([1.0, 1.5, 2.0, 10.0, 10.5, 11.0, 12.0])

    gaps = find_gaps(values, min_width=5.0)

    assert len(gaps) == 1
    assert gaps[0].low == pytest.approx(2.0)
    assert gaps[0].high == pytest.approx(10.0)
    assert gaps[0].count_below == 3
