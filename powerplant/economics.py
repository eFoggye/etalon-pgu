"""Экономика внедрения: формулы с явными параметрами.

Числа, которые мы считаем сами (МВт·ч, МВт), идут из моделей. Цены и доли потерь — параметры:
на слайд они попадают только с источником или с подписью «допущение».
"""

from __future__ import annotations

import numpy as np

HOURS_PER_YEAR = 8760
MONTHS_PER_YEAR = 12


def forecast_error_saved_mwh(mae_base: float, mae_model: float, hours: float) -> float:
    """На сколько МВт·ч меньше ошибка прогноза за `hours` часов работы на полной нагрузке."""
    return max(0.0, (mae_base - mae_model) * hours)


def revenue_rub(mwh: float, price: float, loss_share: float) -> float:
    """Оценка выручки: МВт·ч × цена × доля потери на невыгодной цене (допущение, не прибыль)."""
    return mwh * price * loss_share


def degradation_loss_rub(plant_mw: float, drop_share: float, hours: float, price: float) -> float:
    """Выручка, потерянная за `hours` часов незамеченного падения мощности на `drop_share`."""
    return plant_mw * drop_share * hours * price


def payback_months(cost_rub: float, annual_effect_rub: float) -> float:
    """Срок окупаемости, месяцев. Без эффекта — бесконечность."""
    if annual_effect_rub <= 0:
        return float("inf")
    return cost_rub / annual_effect_rub * MONTHS_PER_YEAR


def vacuum_loss_profile(plan, reference, worst_share: float = 0.1) -> dict[str, float]:
    """Сколько МВт отнимает вакуум, когда он хуже обычного для такой погоды.

    Потеря вакуума в час = план по погоде − эталон с учётом вакуума (если положительна).
    """
    loss = np.asarray(plan, dtype=float) - np.asarray(reference, dtype=float)
    worst_count = max(1, int(round(len(loss) * worst_share)))
    worst = np.sort(loss)[-worst_count:]
    return {
        "share_hours_worse": float(np.mean(loss > 0)),
        "mean_loss_when_worse_mw": float(loss[loss > 0].mean()) if (loss > 0).any() else 0.0,
        "worst_share": worst_share,
        "worst_mean_loss_mw": float(worst.mean()),
    }
