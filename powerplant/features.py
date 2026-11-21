"""Физические признаки из погоды: плотность влажного воздуха и точка росы.

Газовая турбина сжимает массу воздуха, а масса при том же объёме зависит от плотности.
Плотность считаем по закону идеального газа для смеси сухого воздуха и водяного пара,
давление насыщенного пара — по формуле Магнуса (коэффициенты Alduchov, Eskridge, 1996).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

R_DRY = 287.058  # Дж/(кг·К), газовая постоянная сухого воздуха
R_VAPOR = 461.495  # Дж/(кг·К), газовая постоянная водяного пара
KELVIN = 273.15
HPA_TO_PA = 100.0
MAGNUS_E0 = 6.1094  # гПа
MAGNUS_B = 17.625
MAGNUS_C = 243.04  # °C
MIN_RH = 1e-6  # чтобы логарифм не уходил в минус бесконечность при RH = 0


def _magnus_exponent(temp_c):
    return MAGNUS_B * temp_c / (temp_c + MAGNUS_C)


def vapor_pressure(temp_c, rh):
    """Парциальное давление водяного пара, гПа."""
    return np.asarray(rh) / 100.0 * MAGNUS_E0 * np.exp(_magnus_exponent(temp_c))


def air_density(temp_c, pressure_hpa, rh):
    """Плотность влажного воздуха, кг/м³."""
    vapor = vapor_pressure(temp_c, rh)
    temp_k = np.asarray(temp_c) + KELVIN
    dry = (np.asarray(pressure_hpa) - vapor) * HPA_TO_PA / (R_DRY * temp_k)
    wet = vapor * HPA_TO_PA / (R_VAPOR * temp_k)
    return dry + wet


def dew_point(temp_c, rh):
    """Точка росы, °C."""
    gamma = np.log(np.maximum(np.asarray(rh), MIN_RH) / 100.0) + _magnus_exponent(temp_c)
    return MAGNUS_C * gamma / (MAGNUS_B - gamma)


class PhysicsFeatures(TransformerMixin, BaseEstimator):
    """Добавляет к AT, AP, RH плотность воздуха и точку росы. Исходную таблицу не меняет."""

    def fit(self, x: pd.DataFrame, y=None) -> PhysicsFeatures:
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        return x.assign(
            density=air_density(x["AT"], x["AP"], x["RH"]),
            dew_point=dew_point(x["AT"], x["RH"]),
        )
