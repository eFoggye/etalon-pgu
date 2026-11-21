"""Важность признаков тремя способами, которые не зависят от выбранной модели.

1. Сгруппированная перестановочная важность на внефолдовых данных: AT и V связаны,
   поэтому кроме каждого признака перемешиваем и пару {AT, V} разом.
2. Графики частичной зависимости (PDP) — форма и знак влияния.
3. Коэффициенты линейной модели на стандартизированных признаках.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.base import BaseEstimator, clone
from sklearn.inspection import partial_dependence
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from powerplant import config

N_REPEATS = 5
PDP_GRID = 30


def _rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def grouped_permutation_importance(
    estimator: BaseEstimator,
    x: pd.DataFrame,
    y: pd.Series,
    groups: Mapping[str, Sequence[str]],
    seed: int = config.SEED,
    repeats: int = N_REPEATS,
) -> pd.DataFrame:
    """Насколько вырастает RMSE (МВт), если перемешать строки группы признаков разом.

    Считается на валидационных частях фолдов: модель эти строки не видела.
    """
    rng = np.random.default_rng(seed)
    target = np.asarray(y, dtype=float)
    increases: dict[str, list[float]] = {name: [] for name in groups}
    folds = KFold(config.CV_FOLDS, shuffle=True, random_state=seed)
    for train_idx, valid_idx in folds.split(x):
        model = clone(estimator).fit(x.iloc[train_idx], target[train_idx])
        x_valid, y_valid = x.iloc[valid_idx], target[valid_idx]
        base = _rmse(y_valid, model.predict(x_valid))
        for name, columns in groups.items():
            for _ in range(repeats):
                order = rng.permutation(len(x_valid))
                shuffled = x_valid.assign(
                    **{col: x_valid[col].to_numpy()[order] for col in columns}
                )
                increases[name].append(_rmse(y_valid, model.predict(shuffled)) - base)
    table = pd.DataFrame(
        {
            "rmse_increase": {name: float(np.mean(vals)) for name, vals in increases.items()},
            "std": {name: float(np.std(vals)) for name, vals in increases.items()},
        }
    )
    return table.sort_values("rmse_increase", ascending=False)


def standardized_coefficients(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    """МВт на одно стандартное отклонение признака в линейной модели."""
    model = make_pipeline(StandardScaler(), LinearRegression()).fit(x, y)
    return pd.Series(model[-1].coef_, index=x.columns, name="МВт на 1 ст. откл.")


def pdp_curves(
    fitted: BaseEstimator, x: pd.DataFrame, features: Sequence[str], grid: int = PDP_GRID
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Средний прогноз при изменении одного признака и остальных как в данных."""
    curves = {}
    for col in features:
        result = partial_dependence(fitted, x, [col], grid_resolution=grid, kind="average")
        curves[col] = (np.asarray(result["grid_values"][0]), np.asarray(result["average"][0]))
    return curves


def pdp_slope(curve: tuple[np.ndarray, np.ndarray]) -> float:
    """Средний наклон кривой частичной зависимости, МВт на единицу признака."""
    grid, values = curve
    return float(np.polyfit(grid, values, 1)[0])


def plot_importance(table: pd.DataFrame, title: str) -> Figure:
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(table) + 1.2))
    ordered = table.sort_values("rmse_increase")
    ax.barh(ordered.index, ordered["rmse_increase"], xerr=ordered["std"], color="#3b6ea5")
    ax.set_xlabel("Рост ошибки RMSE при перемешивании, МВт")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_pdp(curves: Mapping[str, tuple[np.ndarray, np.ndarray]], title: str) -> Figure:
    fig, axes = plt.subplots(1, len(curves), figsize=(4 * len(curves), 3.6), sharey=True)
    for ax, (col, (grid, values)) in zip(np.atleast_1d(axes), curves.items(), strict=True):
        ax.plot(grid, values, color="#c0392b", linewidth=2.2)
        ax.set_xlabel(config.FEATURE_LABELS.get(col, col))
    np.atleast_1d(axes)[0].set_ylabel("Средний прогноз, МВт")
    fig.suptitle(title)
    fig.tight_layout()
    return fig
