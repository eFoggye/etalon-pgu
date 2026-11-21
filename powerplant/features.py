"""Физические признаки из погоды: плотность влажного воздуха и точка росы.

Газовая турбина сжимает массу воздуха, а масса при том же объёме зависит от плотности.
Плотность считаем по уравнению Менделеева–Клапейрона (идеальный газ) для смеси сухого воздуха
и водяного пара, давление насыщенного пара — по формуле Магнуса (коэффициенты Alduchov,
Eskridge, 1996). Уравнение Ван-дер-Ваальса здесь только для проверки: при атмосферном давлении
оно меняет плотность меньше чем на 0,1%, и модель от него не меняется (ноутбук, раздел 8.2).
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

R_UNIVERSAL = 8.314462618  # Дж/(моль·К)
# Константы Ван-дер-Ваальса (CRC Handbook): a в Па·м⁶/моль², b в м³/моль; молярные массы, кг/моль
VDW_GASES = {
    "N2": (0.1370, 3.87e-5, 0.0280134),
    "O2": (0.1382, 3.186e-5, 0.0319988),
    "Ar": (0.1355, 3.201e-5, 0.039948),
    "H2O": (0.5536, 3.049e-5, 0.01801528),
}
DRY_AIR_MOLE_FRACTIONS = {"N2": 0.78084, "O2": 0.20946, "Ar": 0.00934}
VDW_ITERATIONS = 30  # простая итерация по молярному объёму сходится за несколько шагов


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


def air_density_van_der_waals(temp_c, pressure_hpa, rh):
    """Плотность влажного воздуха по уравнению Ван-дер-Ваальса, кг/м³. Только для проверки.

    Смесь сухого воздуха и пара сводим к одному газу правилами смешения Ван-дер-Ваальса,
    молярный объём находим простой итерацией V = b + RT / (p + a / V²).
    """
    temp_k = np.asarray(temp_c, dtype=float) + KELVIN
    pressure = np.asarray(pressure_hpa, dtype=float) * HPA_TO_PA
    vapor_share = vapor_pressure(temp_c, rh) * HPA_TO_PA / pressure
    dry_total = sum(DRY_AIR_MOLE_FRACTIONS.values())
    shares = {gas: (1 - vapor_share) * x / dry_total for gas, x in DRY_AIR_MOLE_FRACTIONS.items()}
    shares["H2O"] = vapor_share
    a_mix = sum(x * np.sqrt(VDW_GASES[gas][0]) for gas, x in shares.items()) ** 2
    b_mix = sum(x * VDW_GASES[gas][1] for gas, x in shares.items())
    molar_mass = sum(x * VDW_GASES[gas][2] for gas, x in shares.items())
    volume = R_UNIVERSAL * temp_k / pressure
    for _ in range(VDW_ITERATIONS):
        volume = b_mix + R_UNIVERSAL * temp_k / (pressure + a_mix / volume**2)
    return molar_mass / volume


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
