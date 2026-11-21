"""Схемы запросов и ответов API. Границы ввода — физически правдоподобные значения."""

from __future__ import annotations

from pydantic import BaseModel, Field

from powerplant import config

_AT, _V, _AP, _RH = (config.INPUT_RANGES[name] for name in ("AT", "V", "AP", "RH"))
MAX_POWER_MW = 1000.0
MAX_PRICE = 100_000.0


class PredictRequest(BaseModel):
    """Погода обязательна; вакуум, требуемый объём и факт — по желанию."""

    AT: float = Field(ge=_AT.low, le=_AT.high, description="Температура воздуха, °C")
    AP: float = Field(ge=_AP.low, le=_AP.high, description="Атмосферное давление, мбар")
    RH: float = Field(ge=_RH.low, le=_RH.high, description="Относительная влажность, %")
    V: float | None = Field(default=None, ge=_V.low, le=_V.high, description="Вакуум на выхлопе")
    required_mw: float | None = Field(default=None, ge=0, le=MAX_POWER_MW)
    actual_mw: float | None = Field(default=None, ge=0, le=MAX_POWER_MW)
    price_rub_per_mwh: float | None = Field(default=None, ge=0, le=MAX_PRICE)
    explain: bool = True


class Contribution(BaseModel):
    feature: str
    label: str
    mw: float


class CurvePoint(BaseModel):
    AT: float
    power: float
    low: float
    high: float
    novel: bool = False  # такого сочетания погоды в обучающих данных не было


class Forecast(BaseModel):
    model: str
    power: float
    low: float
    high: float
    base_mw: float | None = None
    contributions: list[Contribution] = []


class PlanRisk(BaseModel):
    required_mw: float
    shortfall_probability: float


class Deviation(BaseModel):
    total_mw: float
    vacuum_mw: float
    other_mw: float
    below_reference_interval: bool
    rub_per_hour: float


class PredictResponse(BaseModel):
    level: float
    plan: Forecast
    reference: Forecast | None = None
    plan_risk: PlanRisk | None = None
    deviation: Deviation | None = None
    curve: list[CurvePoint] = []
    warnings: list[str] = []
    model_version: str


class FeatureInfo(BaseModel):
    name: str
    label: str
    train_min: float
    train_max: float
    input_min: float
    input_max: float
    median: float


class ImportanceItem(BaseModel):
    group: str
    label: str
    mw: float


class MetaResponse(BaseModel):
    model_version: str
    level: float
    mean_power: float
    price_default: float
    detector_window_h: int
    detector_threshold_mw: float
    features: list[FeatureInfo]
    presets: dict[str, dict[str, float]]
    importance: dict[str, list[ImportanceItem]]
    metrics: dict[str, object]


class BatchRow(BaseModel):
    hour: int
    plan: float
    low: float
    high: float
    reference: float | None = None
    actual: float | None = None
    deviation: float | None = None
    vacuum: float | None = None
    other: float | None = None
    alarm: bool = False
    novelty: bool = False


class BatchSummary(BaseModel):
    rows: int
    has_actual: bool
    has_vacuum: bool
    alarm_hours: int
    first_alarm_hour: int | None = None
    mean_deviation_mw: float | None = None
    lost_mwh: float | None = None
    lost_rub: float | None = None
    novelty_hours: int


class BatchResponse(BaseModel):
    summary: BatchSummary
    rows: list[BatchRow]
