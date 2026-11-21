"""Вклад каждого параметра в конкретный прогноз: точные значения Шепли без внешних библиотек.

Ценность набора признаков S — средний прогноз по фоновым строкам, где признаки из S взяты
из нашей точки, а остальные — из фона. При 3–4 признаках все 2^n наборов считаются точно
одним вызовом модели. Сумма вкладов равна разнице «прогноз − средний прогноз по фону».
Работает с любой моделью. Оговорка: между связанными признаками (AT и V) вклад делится условно.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd


def _coalition_frames(
    point: pd.DataFrame, background: pd.DataFrame, names: list[str]
) -> tuple[list[tuple[bool, ...]], pd.DataFrame]:
    masks = list(itertools.product((False, True), repeat=len(names)))
    frames = []
    for mask in masks:
        replacements = {
            name: float(point[name].iloc[0]) for name, take in zip(names, mask, strict=True) if take
        }
        frames.append(background[names].assign(**replacements))
    return masks, pd.concat(frames, ignore_index=True)


def shapley_contributions(
    predict: Callable[[pd.DataFrame], np.ndarray],
    point: pd.DataFrame,
    background: pd.DataFrame,
    features: Sequence[str],
) -> tuple[float, dict[str, float]]:
    """Возвращает средний прогноз по фону и вклад каждого признака, МВт."""
    names = list(features)
    n = len(names)
    masks, stacked = _coalition_frames(point, background, names)
    values = np.asarray(predict(stacked), dtype=float).reshape(len(masks), len(background))
    value = dict(zip(masks, values.mean(axis=1), strict=True))

    contributions = {}
    for i, name in enumerate(names):
        total = 0.0
        for mask in masks:
            if mask[i]:
                continue
            size = sum(mask)
            weight = math.factorial(size) * math.factorial(n - size - 1) / math.factorial(n)
            with_feature = tuple(True if j == i else taken for j, taken in enumerate(mask))
            total += weight * (value[with_feature] - value[mask])
        contributions[name] = float(total)
    return float(value[(False,) * n]), contributions
