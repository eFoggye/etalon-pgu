"""Конформный диапазон прогноза и вероятность не выполнить требуемый объём."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from powerplant import config


def conformal_quantile(residuals, level: float = config.INTERVAL_LEVEL) -> float:
    """Квантиль |остатков| ранга ⌈(n+1)·level⌉ — поправка на конечную выборку."""
    values = np.sort(np.abs(np.asarray(residuals, dtype=float)))
    if values.size == 0:
        raise ValueError("Для диапазона нужен хотя бы один остаток")
    rank = math.ceil((values.size + 1) * level)
    if rank > values.size:
        return float("inf")
    return float(values[rank - 1])


def interval(prediction, q: float) -> tuple[np.ndarray, np.ndarray]:
    """Диапазон «от … до …»: прогноз ± q."""
    center = np.asarray(prediction, dtype=float)
    return center - q, center + q


def coverage(y_true, low, high) -> float:
    """Доля фактических значений внутри диапазона."""
    actual = np.asarray(y_true, dtype=float)
    return float(np.mean((actual >= low) & (actual <= high)))


def shortfall_probability(prediction: float, required: float, residuals) -> float:
    """Вероятность, что факт окажется ниже требуемого объёма.

    Считается по эмпирическому распределению внефолдовых остатков «факт − прогноз»:
    факт < требуемого ⇔ остаток < требуемый − прогноз.
    """
    errors = np.asarray(residuals, dtype=float)
    if errors.size == 0:
        raise ValueError("Для оценки риска нужен хотя бы один остаток")
    return float(np.mean(errors < required - prediction))


def coverage_by_slices(
    y_true, prediction, q: float, slice_values, edges, labels
) -> pd.DataFrame:
    """Покрытие диапазона по срезам, например «холод / норма / жара»."""
    low, high = interval(prediction, q)
    actual = np.asarray(y_true, dtype=float)
    inside = (actual >= low) & (actual <= high)
    slices = pd.cut(slice_values, bins=edges, labels=labels)
    frame = pd.DataFrame({"срез": slices, "внутри": inside})
    grouped = frame.groupby("срез", observed=True)["внутри"]
    return pd.DataFrame({"покрытие": grouped.mean(), "строк": grouped.size()})
