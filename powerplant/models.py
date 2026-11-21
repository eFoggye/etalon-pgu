"""Лестница моделей от простой к сложной, честная проверка и правило выбора G2."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.base import BaseEstimator, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from powerplant import config
from powerplant.features import PhysicsFeatures

N_ITER = 15
# Метрики после подбора считаем на другом разбиении на фолды: это уменьшает оптимизм подбора,
# но не убирает его полностью (для этого нужна вложенная проверка). Честная цифра — отложенный тест.
FRESH_FOLD_OFFSET = 1000
STACKING_COMPLEXITY = 5


@dataclass(frozen=True)
class Candidate:
    """Ступень лестницы: модель-шаблон и пространство поиска гиперпараметров."""

    name: str
    complexity: int
    estimator: BaseEstimator
    search_space: Mapping[str, object] | None = None


@dataclass(frozen=True)
class CvResult:
    """Честные метрики модели по внефолдовым прогнозам."""

    name: str
    complexity: int
    mae: float
    rmse: float
    r2: float
    mape: float
    rmse_std: float
    oof: np.ndarray = field(repr=False)
    params: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE и RMSE в МВт, R², MAPE в процентах."""
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred) * 100),
    }


def build_ladder(
    seed: int = config.SEED, features: Sequence[str] = config.FEATURES_ALL
) -> list[Candidate]:
    """Модели от простой к сложной. Масштабирование только там, где оно нужно (линейные).

    Бустинг получает ограничения монотонности по физике для тех признаков, что есть в наборе.
    """
    monotonic = {name: config.MONOTONIC.get(name, 0) for name in features}
    polynomial = Pipeline(
        [
            ("poly", PolynomialFeatures(include_bias=False)),
            ("scale", StandardScaler()),
            ("ridge", Ridge()),
        ]
    )
    return [
        Candidate("Константа", 0, DummyRegressor()),
        Candidate("Линейная", 1, make_pipeline(StandardScaler(), LinearRegression())),
        Candidate(
            "Полином + Ridge",
            2,
            polynomial,
            {"poly__degree": [2, 3, 4], "ridge__alpha": loguniform(1e-4, 1e2)},
        ),
        Candidate(
            "Случайный лес",
            3,
            RandomForestRegressor(random_state=seed, n_jobs=-1),
            {
                "n_estimators": [200, 300],
                "min_samples_leaf": [1, 2, 3, 5],
                "max_features": [0.5, 0.75, 1.0],
            },
        ),
        Candidate(
            "Градиентный бустинг",
            4,
            HistGradientBoostingRegressor(random_state=seed, monotonic_cst=monotonic),
            {
                "learning_rate": loguniform(0.02, 0.2),
                "max_iter": [300, 500, 800],
                "max_leaf_nodes": [15, 31, 63],
                "min_samples_leaf": [10, 20, 40],
                "l2_regularization": loguniform(1e-3, 1.0),
            },
        ),
    ]


def build_stacking(forest: BaseEstimator, boosting: BaseEstimator, seed: int) -> StackingRegressor:
    """Стекинг подобранных леса и бустинга с линейной моделью сверху."""
    return StackingRegressor(
        estimators=[("forest", clone(forest)), ("boosting", clone(boosting))],
        final_estimator=RidgeCV(),
        cv=KFold(config.CV_FOLDS, shuffle=True, random_state=seed),
    )


def with_physics(estimator: BaseEstimator) -> Pipeline:
    """Та же модель, но с добавленными плотностью воздуха и точкой росы.

    Если у модели есть ограничения монотонности, к ним добавляется физика плотности,
    иначе через неограниченную плотность модель могла бы снова начать расти с жарой.
    """
    model = clone(estimator)
    constraints = getattr(model, "monotonic_cst", None)
    if isinstance(constraints, dict):
        model = model.set_params(monotonic_cst={**constraints, **config.PHYSICS_FEATURE_MONOTONIC})
    return Pipeline([("physics", PhysicsFeatures()), ("model", model)])


def tune(
    candidate: Candidate,
    x: pd.DataFrame,
    y: pd.Series,
    seed: int = config.SEED,
    n_iter: int = N_ITER,
) -> tuple[BaseEstimator, dict[str, object]]:
    """Подбор гиперпараметров на CV. Возвращает необученную модель с лучшими параметрами."""
    if not candidate.search_space:
        return clone(candidate.estimator), {}
    search = RandomizedSearchCV(
        clone(candidate.estimator),
        candidate.search_space,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=KFold(config.CV_FOLDS, shuffle=True, random_state=seed),
        random_state=seed,
    )
    search.fit(x, y)
    best = clone(candidate.estimator).set_params(**search.best_params_)
    return best, dict(search.best_params_)


def evaluate_oof(
    name: str,
    estimator: BaseEstimator,
    x: pd.DataFrame,
    y: pd.Series,
    complexity: int,
    seed: int = config.SEED,
    params: Mapping[str, object] | None = None,
) -> CvResult:
    """Честные метрики: внефолдовые прогнозы на свежем разбиении на фолды."""
    target = np.asarray(y, dtype=float)
    folds = KFold(config.CV_FOLDS, shuffle=True, random_state=seed + FRESH_FOLD_OFFSET)
    oof = cross_val_predict(clone(estimator), x, target, cv=folds)
    fold_rmse = [
        np.sqrt(mean_squared_error(target[test], oof[test])) for _, test in folds.split(x)
    ]
    metrics = regression_metrics(target, oof)
    return CvResult(
        name=name,
        complexity=complexity,
        mae=metrics["MAE"],
        rmse=metrics["RMSE"],
        r2=metrics["R2"],
        mape=metrics["MAPE"],
        rmse_std=float(np.std(fold_rmse)),
        oof=oof,
        params=MappingProxyType(dict(params or {})),
    )


def physically_plausible(
    fitted: BaseEstimator,
    x: pd.DataFrame,
    tolerance: float = config.PHYSICS_TOLERANCE_MW,
    points: int = 40,
) -> bool:
    """Физическая проверка: при росте температуры по всему диапазону данных мощность не растёт.

    Остальные признаки ставим на 10, 50 и 90-й перцентили во всех сочетаниях. Подъём больше
    `tolerance` МВт хотя бы на одном срезе — модель отбраковывается.
    """
    grid = np.linspace(x["AT"].min(), x["AT"].max(), points)
    others = [col for col in x.columns if col != "AT"]
    levels = [x[col].quantile(config.PHYSICS_SLICE_QUANTILES).to_numpy() for col in others]
    for combo in itertools.product(*levels):
        frame = pd.DataFrame({"AT": grid, **dict(zip(others, combo, strict=True))})
        prediction = np.asarray(fitted.predict(frame[list(x.columns)]), dtype=float)
        if np.max(np.diff(prediction), initial=0.0) > tolerance:
            return False
    return True


def select_model(
    results: Sequence[CvResult], tolerance: float = config.SELECTION_TOLERANCE
) -> CvResult:
    """Правило G2: самая простая модель, чей RMSE не хуже лучшего более чем на `tolerance`."""
    best_rmse = min(result.rmse for result in results)
    eligible = [result for result in results if result.rmse <= best_rmse * (1 + tolerance)]
    return min(eligible, key=lambda result: (result.complexity, result.rmse))
