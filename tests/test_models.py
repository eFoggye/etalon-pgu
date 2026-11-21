import numpy as np
import pandas as pd
import pytest

from powerplant.features import PhysicsFeatures, air_density, dew_point
from powerplant.models import (
    CvResult,
    build_ladder,
    evaluate_oof,
    physically_plausible,
    regression_metrics,
    select_model,
    with_physics,
)


def make_result(name: str, complexity: int, rmse: float) -> CvResult:
    return CvResult(
        name=name, complexity=complexity, mae=rmse, rmse=rmse, r2=0.9, mape=1.0, rmse_std=0.1,
        oof=np.zeros(3),
    )


@pytest.fixture(scope="module")
def linear_problem() -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(0)
    x = pd.DataFrame(
        {
            "AT": rng.uniform(0, 35, 400),
            "AP": rng.uniform(995, 1030, 400),
            "RH": rng.uniform(30, 100, 400),
        }
    )
    y = pd.Series(500 - 2 * x["AT"] + 0.1 * (x["AP"] - 1013) + rng.normal(0, 1, 400))
    return x, y


def test_regression_metrics_on_known_values():
    metrics = regression_metrics(np.array([100.0, 200.0]), np.array([110.0, 190.0]))

    assert metrics["MAE"] == pytest.approx(10.0)
    assert metrics["RMSE"] == pytest.approx(10.0)
    assert metrics["R2"] == pytest.approx(0.96)
    assert metrics["MAPE"] == pytest.approx(7.5)


def test_select_model_prefers_simpler_model_within_tolerance():
    results = [
        make_result("Линейная", 1, 4.5),
        make_result("Случайный лес", 3, 3.40),
        make_result("Градиентный бустинг", 4, 3.30),
    ]

    assert select_model(results, tolerance=0.05).name == "Случайный лес"


def test_select_model_takes_best_when_simpler_models_are_far_behind():
    results = [make_result("Линейная", 1, 4.5), make_result("Градиентный бустинг", 4, 3.3)]

    assert select_model(results, tolerance=0.05).name == "Градиентный бустинг"


def test_air_density_matches_standard_atmosphere():
    assert air_density(15.0, 1013.25, 0.0) == pytest.approx(1.225, abs=0.002)


def test_humid_air_is_lighter_than_dry_air():
    assert air_density(25.0, 1013.0, 90.0) < air_density(25.0, 1013.0, 10.0)


def test_dew_point_equals_temperature_at_full_humidity():
    assert dew_point(20.0, 100.0) == pytest.approx(20.0, abs=1e-6)


def test_physics_features_add_columns_without_mutating_input():
    x = pd.DataFrame({"AT": [15.0], "AP": [1013.25], "RH": [50.0]})

    out = PhysicsFeatures().fit_transform(x)

    assert list(out.columns) == ["AT", "AP", "RH", "density", "dew_point"]
    assert list(x.columns) == ["AT", "AP", "RH"]


def test_ladder_starts_with_constant_and_grows_in_complexity():
    ladder = build_ladder(seed=0)

    assert ladder[0].name == "Константа"
    assert [c.complexity for c in ladder] == sorted(c.complexity for c in ladder)


def test_evaluate_oof_is_deterministic_and_beats_constant(linear_problem):
    x, y = linear_problem
    ladder = {c.name: c for c in build_ladder(seed=0)}

    first = evaluate_oof("Линейная", ladder["Линейная"].estimator, x, y, complexity=1, seed=0)
    second = evaluate_oof("Линейная", ladder["Линейная"].estimator, x, y, complexity=1, seed=0)
    constant = evaluate_oof("Константа", ladder["Константа"].estimator, x, y, complexity=0, seed=0)

    assert len(first.oof) == len(y)
    assert np.allclose(first.oof, second.oof)
    assert first.rmse < constant.rmse / 5


def test_boosting_respects_physics_monotonicity():
    rng = np.random.default_rng(3)
    x = pd.DataFrame({"AT": rng.uniform(0, 35, 800), "AP": rng.uniform(995, 1030, 800),
                      "RH": rng.uniform(30, 100, 800)})
    wiggle = 3 * np.sin(x["AT"])  # немонотонная примесь: без ограничений бустинг выучил бы подъёмы
    y = pd.Series(500 - 2 * x["AT"] + wiggle + rng.normal(0, 0.5, 800))
    ladder = {c.name: c for c in build_ladder(seed=0, features=("AT", "AP", "RH"))}
    model = ladder["Градиентный бустинг"].estimator.fit(x, y)

    assert physically_plausible(model, x)


def test_physics_check_rejects_power_rising_with_temperature():
    x = pd.DataFrame({"AT": np.linspace(0, 35, 50), "AP": 1013.0, "RH": 70.0})

    class RisingAtEdge:
        def predict(self, frame):
            at = frame["AT"].to_numpy()
            return 500 - 2 * at + np.where(at > 30, 5 * (at - 30), 0)

    assert not physically_plausible(RisingAtEdge(), x)


def test_physics_variant_keeps_temperature_constraint():
    ladder = {c.name: c for c in build_ladder(seed=0, features=("AT", "AP", "RH"))}

    variant = with_physics(ladder["Градиентный бустинг"].estimator)

    assert variant.named_steps["model"].monotonic_cst["AT"] == -1
