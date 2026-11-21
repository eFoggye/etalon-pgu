"""Контроль оборудования: разложение недобора и суточное правило тревоги.

Одиночное правило «час ниже нижней границы 90%-диапазона» срабатывало бы около 5% часов,
то есть примерно раз в сутки на здоровой станции. Поэтому тревогу даём по среднему
отклонению за окно (по умолчанию сутки) с заданной долей ложных срабатываний.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from powerplant import config

DEFAULT_WINDOW = 24
DEFAULT_FALSE_ALARM = 0.01
N_SIMULATIONS = 20_000


@dataclass(frozen=True)
class Decomposition:
    """Отклонение факта от плана и его причины, МВт (минус — недобор)."""

    total: float  # факт − план по погоде
    vacuum: float  # эталон − план: вакуум хуже (минус) или лучше (плюс) обычного для такой погоды
    other: float  # факт − эталон: всё, что не объясняется погодой и вакуумом


def decompose(plan: float, reference: float, actual: float) -> Decomposition:
    return Decomposition(total=actual - plan, vacuum=reference - plan, other=actual - reference)


def _window_means(residuals, window: int, n_sim: int, seed: int) -> np.ndarray:
    values = np.asarray(residuals, dtype=float)
    if values.size == 0:
        raise ValueError("Для порога нужен хотя бы один остаток")
    rng = np.random.default_rng(seed)
    return rng.choice(values, size=(n_sim, window), replace=True).mean(axis=1)


def window_threshold(
    residuals,
    window: int = DEFAULT_WINDOW,
    false_alarm_rate: float = DEFAULT_FALSE_ALARM,
    n_sim: int = N_SIMULATIONS,
    seed: int = config.SEED,
) -> float:
    """Порог для среднего отклонения за окно: на здоровой станции ниже него — `false_alarm_rate`
    окон (доля считается на одно окно, а не на сутки: скользящие окна перекрываются).

    Окна собираются из независимых остатков. Реальные часы связаны между собой,
    поэтому на живом потоке ложных тревог может быть больше.
    """
    return float(np.quantile(_window_means(residuals, window, n_sim, seed), false_alarm_rate))


def detection_rate(
    residuals,
    window: int,
    threshold: float,
    shift_mw: float,
    n_sim: int = N_SIMULATIONS,
    seed: int = config.SEED + 1,
) -> float:
    """Доля окон, в которых правило ловит устойчивый сдвиг мощности на `shift_mw`."""
    means = _window_means(residuals, window, n_sim, seed) + shift_mw
    return float(np.mean(means < threshold))


def flag_windows(deviations, window: int, threshold: float) -> np.ndarray:
    """Скользящее среднее отклонений по подряд идущим часам; True — тревога.

    Порог посчитан для полного окна, поэтому в файле короче окна тревогу не даём:
    среднее по нескольким часам разбросано сильнее и дало бы ложные срабатывания.
    """
    values = np.asarray(deviations, dtype=float)
    if values.size < window:
        return np.zeros(values.size, dtype=bool)
    rolling = pd.Series(values).rolling(window).mean().to_numpy()
    return np.where(np.isnan(rolling), False, rolling < threshold)
