"""Экономика внедрения по шаблону ментора: простои, их цена, сокращение, затраты, окупаемость.

Формулы повторяют шаблон один в один. Для ПГУ «простой» — эквивалентные часы простоя из-за
недобора мощности: 1% недобора за 100 часов равен одному часу полного простоя.
Источник каждой цифры или пометка «допущение» — в reports/economics.xlsx.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from powerplant import config

HOURS_PER_YEAR = 8760
MONTHS_PER_YEAR = 12
HOURS_PER_MONTH = HOURS_PER_YEAR / MONTHS_PER_YEAR
WORKDAY_HOURS = 8


@dataclass(frozen=True)
class DowntimeInputs:
    """Блок «Исходные данные» шаблона."""

    downtime_hours_per_month: float
    cost_per_hour_rub: float
    reduction_share: float
    implementation_rub: float
    support_rub_per_month: float

    def __post_init__(self) -> None:
        if not 0 <= self.reduction_share <= 1:
            raise ValueError("Сокращение простоев — доля от 0 до 1")
        amounts = (self.downtime_hours_per_month, self.cost_per_hour_rub,
                   self.implementation_rub, self.support_rub_per_month)
        if min(amounts) < 0:
            raise ValueError("Часы, цены и затраты не могут быть отрицательными")


@dataclass(frozen=True)
class DowntimeResult:
    """Блок «Расчёт экономии» шаблона, строки в том же порядке."""

    loss_rub_per_month: float
    hours_saved_per_month: float
    saving_rub_per_month: float
    net_saving_rub_per_month: float
    net_saving_rub_per_year: float
    first_year_result_rub: float
    payback_months: float


def downtime_economics(inputs: DowntimeInputs) -> DowntimeResult:
    """Расчёт экономии. Без чистой экономии окупаемость бесконечна (в шаблоне — «нет»)."""
    hours_saved = inputs.downtime_hours_per_month * inputs.reduction_share
    saving = hours_saved * inputs.cost_per_hour_rub
    net = saving - inputs.support_rub_per_month
    per_year = net * MONTHS_PER_YEAR
    return DowntimeResult(
        loss_rub_per_month=inputs.downtime_hours_per_month * inputs.cost_per_hour_rub,
        hours_saved_per_month=hours_saved,
        saving_rub_per_month=saving,
        net_saving_rub_per_month=net,
        net_saving_rub_per_year=per_year,
        first_year_result_rub=per_year - inputs.implementation_rub,
        payback_months=inputs.implementation_rub / net if net > 0 else float("inf"),
    )


def cumulative_result(inputs: DowntimeInputs, months: int) -> list[float]:
    """Накопленный результат на конец месяцев 0…months: минус внедрение плюс чистая экономия."""
    net = downtime_economics(inputs).net_saving_rub_per_month
    return [net * month - inputs.implementation_rub for month in range(months + 1)]


def breakeven_reduction(inputs: DowntimeInputs, months: float = MONTHS_PER_YEAR) -> float:
    """Минимальное сокращение простоев, при котором внедрение окупается за `months` месяцев."""
    potential = inputs.downtime_hours_per_month * inputs.cost_per_hour_rub
    if potential <= 0:
        return float("inf")
    return (inputs.implementation_rub / months + inputs.support_rub_per_month) / potential


@dataclass(frozen=True)
class PlantAssumptions:
    """Цифры для ПГУ и внедрения, из которых складываются исходные данные шаблона."""

    plant_mw: float = 463.9  # ПГУ-450 Ярославской ТЭС
    gas_turbines_mw: float = 313.9  # две ГТЭ-160: 156,2 + 157,7 МВт, компрессор есть только у них
    price_rub_per_mwh: float = config.PRICE_RUB_PER_MWH  # РСВ, 1-я ценовая зона, 2024
    fouling_loss_at_8000h: float = 0.05  # умеренное загрязнение компрессора к 8000 ч работы
    average_loss_ratio: float = 0.5  # допущение: средний недобор за цикл — половина конечного
    reduction_share: float = 0.01  # допущение: осторожная оценка
    salary_rub_per_month: float = 243_000  # медиана Python-разработчика, 1-е полугодие 2026
    insurance_share: float = 0.30  # страховые взносы работодателя
    work_hours_per_year: float = 1_972  # производственный календарь 2026, 40 часов в неделю
    implementation_person_days: float = 20  # допущение: интеграция с АСУ ТП и пилот
    server_rub_per_month: float = 5_838  # облачный сервер 4 vCPU, 8 ГБ, SSD 50 ГБ, с НДС
    support_hours_per_month: float = 4  # допущение

    def hourly_rate_rub(self) -> float:
        """Час разработчика для работодателя: оклад со взносами, делённый на часы в месяце."""
        hours_per_month = self.work_hours_per_year / MONTHS_PER_YEAR
        return self.salary_rub_per_month * (1 + self.insurance_share) / hours_per_month

    def downtime_hours_per_month(self) -> float:
        """Недобор от загрязнения в часах полного простоя блока.

        Загрязняется компрессор газовых турбин, поэтому недобор считаем только с их мощности.
        Паровая часть при этом тоже теряет, но её не учитываем: оценка выходит осторожнее.
        """
        lost_mw = self.gas_turbines_mw * self.fouling_loss_at_8000h * self.average_loss_ratio
        return lost_mw / self.plant_mw * HOURS_PER_MONTH

    def to_inputs(self) -> DowntimeInputs:
        rate = self.hourly_rate_rub()
        return DowntimeInputs(
            downtime_hours_per_month=self.downtime_hours_per_month(),
            cost_per_hour_rub=self.plant_mw * self.price_rub_per_mwh,
            reduction_share=self.reduction_share,
            implementation_rub=self.implementation_person_days * WORKDAY_HOURS * rate,
            support_rub_per_month=self.server_rub_per_month + self.support_hours_per_month * rate,
        )


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
