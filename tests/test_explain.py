import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from powerplant.explain import shapley_contributions
from powerplant.importance import grouped_permutation_importance

FEATURES = ["AT", "V", "AP"]


@pytest.fixture(scope="module")
def background() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({col: rng.normal(size=200) for col in FEATURES})


def linear_predict(frame: pd.DataFrame) -> np.ndarray:
    at, v, ap = (frame[col].to_numpy() for col in FEATURES)
    return 500 - 2.0 * at - 1.0 * v + 0.5 * ap


def test_shapley_contributions_add_up_to_prediction(background):
    point = pd.DataFrame({"AT": [1.5], "V": [-0.7], "AP": [2.0]})

    base, contributions = shapley_contributions(linear_predict, point, background, FEATURES)

    assert base + sum(contributions.values()) == pytest.approx(linear_predict(point)[0])


def test_shapley_matches_closed_form_for_linear_model(background):
    point = pd.DataFrame({"AT": [1.5], "V": [-0.7], "AP": [2.0]})

    _, contributions = shapley_contributions(linear_predict, point, background, FEATURES)

    expected_at = -2.0 * (1.5 - background["AT"].mean())
    assert contributions["AT"] == pytest.approx(expected_at)


def test_grouped_importance_ranks_strong_feature_first():
    rng = np.random.default_rng(1)
    x = pd.DataFrame({"AT": rng.normal(size=600), "AP": rng.normal(size=600)})
    y = pd.Series(10 * x["AT"] + 5 * x["AP"] + rng.normal(scale=0.1, size=600))

    table = grouped_permutation_importance(
        LinearRegression(), x, y, groups={"AT": ["AT"], "AP": ["AP"], "AT+AP": ["AT", "AP"]}
    )

    assert table.index[0] == "AT+AP"
    assert table.loc["AT", "rmse_increase"] > table.loc["AP", "rmse_increase"] > 0
