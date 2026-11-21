"""Предупреждение о выходе за обучающие данные: по каждому признаку и по их сочетанию."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from powerplant import config

NEIGHBORS = 5
# Порог — 99-й перцентиль внутри обучающих данных, то есть около 1% ложных предупреждений.
THRESHOLD_PERCENTILE = 99.0
COMBINATION_WARNING = "Сочетание параметров не встречалось в обучающих данных — прогноз ненадёжен"


@dataclass(frozen=True)
class NoveltyModel:
    """Границы обучающих данных и индекс ближайших соседей в стандартизированном пространстве."""

    features: tuple[str, ...]
    low: np.ndarray
    high: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    index: NearestNeighbors
    k: int
    threshold: float


def fit_novelty(
    x: pd.DataFrame, k: int = NEIGHBORS, percentile: float = THRESHOLD_PERCENTILE
) -> NoveltyModel:
    if len(x) <= k:
        raise ValueError(f"Для проверки новизны нужно больше {k} строк")
    values = x.to_numpy(dtype=float)
    mean, spread = values.mean(axis=0), values.std(axis=0)
    scale = np.where(spread > 0, spread, 1.0)  # постоянный признак не должен давать деление на 0
    standardized = (values - mean) / scale
    index = NearestNeighbors(n_neighbors=k + 1).fit(standardized)
    distances, _ = index.kneighbors(standardized)
    typical = distances[:, 1:].mean(axis=1)  # без самой точки
    return NoveltyModel(
        features=tuple(x.columns),
        low=values.min(axis=0),
        high=values.max(axis=0),
        mean=mean,
        scale=scale,
        index=index,
        k=k,
        threshold=float(np.percentile(typical, percentile)),
    )


def novelty_distance(model: NoveltyModel, points: pd.DataFrame) -> np.ndarray:
    """Среднее расстояние до k ближайших обучающих точек."""
    standardized = (points[list(model.features)].to_numpy(dtype=float) - model.mean) / model.scale
    distances, _ = model.index.kneighbors(standardized, n_neighbors=model.k)
    return distances.mean(axis=1)


def check_novelty(model: NoveltyModel, point: Mapping[str, float]) -> list[str]:
    """Человеческие предупреждения: какой признак вне данных и не небывалое ли сочетание."""
    warnings = []
    for i, col in enumerate(model.features):
        value = float(point[col])
        if not model.low[i] <= value <= model.high[i]:
            label = config.FEATURE_LABELS.get(col, col)
            warnings.append(
                f"{label}: {value:g} — вне диапазона обучающих данных "
                f"({model.low[i]:g}–{model.high[i]:g})"
            )
    frame = pd.DataFrame([{col: float(point[col]) for col in model.features}])
    if novelty_distance(model, frame)[0] > model.threshold:
        warnings.append(COMBINATION_WARNING)
    return warnings
