"""Бизнес-логика сервиса: прогноз плана, эталон, риск плана, разложение недобора, пакет часов."""

from __future__ import annotations

import io
import logging
import zipfile
from dataclasses import dataclass

import numpy as np
import pandas as pd

from powerplant import config
from powerplant.artifact import Bundle, ModelPack
from powerplant.detector import decompose, flag_windows, window_threshold
from powerplant.explain import shapley_contributions
from powerplant.intervals import interval, shortfall_probability
from powerplant.novelty import check_novelty, novelty_distance
from powerplant.schemas import (
    MAX_POWER_MW,
    BatchResponse,
    BatchRow,
    BatchSummary,
    Contribution,
    CurvePoint,
    Deviation,
    FeatureInfo,
    Forecast,
    ImportanceItem,
    MetaResponse,
    PlanRisk,
    PredictRequest,
    PredictResponse,
)

LOG = logging.getLogger(__name__)
CURVE_POINTS = 31
MAX_BATCH_ROWS = 10_000
MAX_BATCH_COLUMNS = 50
MAX_XLSX_ENTRIES = 200
MAX_XLSX_UNPACKED_BYTES = 50 * 1024 * 1024
READ_ERROR = "Не удалось прочитать файл: нужен CSV или Excel"
REQUIRED_BATCH_COLUMNS = config.FEATURES_PLAN
# Факт в пакете может быть и нулём: отключение станции — главный случай для контроля.
ACTUAL_RANGE = config.Range(0.0, MAX_POWER_MW)


class BatchError(ValueError):
    """Файл пакетного прогноза не подходит; сообщение показывается пользователю."""

    def __init__(self, message: str, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class BatchArrays:
    """Посчитанные по пакету массивы; None — колонки не было в файле."""

    plan: np.ndarray
    low: np.ndarray
    high: np.ndarray
    reference: np.ndarray | None
    actual: np.ndarray | None
    deviation: np.ndarray | None
    alarms: np.ndarray
    novelty: np.ndarray


def _round(value: float, digits: int = 2) -> float:
    return round(float(value), digits)


def _optional(values: np.ndarray | None, i: int) -> float | None:
    return None if values is None else _round(values[i])


def _guard_xlsx(content: bytes) -> None:
    """Отсекает битые архивы и zip-бомбы до того, как pandas начнёт распаковку."""
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            entries = archive.infolist()
    except zipfile.BadZipFile as exc:
        raise BatchError(READ_ERROR, status_code=400) from exc
    unpacked = sum(entry.file_size for entry in entries)
    if len(entries) > MAX_XLSX_ENTRIES or unpacked > MAX_XLSX_UNPACKED_BYTES:
        raise BatchError("Excel-файл слишком большой после распаковки", status_code=413)


def read_table(filename: str, content: bytes) -> pd.DataFrame:
    """Читает CSV (запятая или точка с запятой) или xlsx, не больше лимита строк."""
    try:
        if filename.lower().endswith(".xlsx"):
            _guard_xlsx(content)
            return pd.read_excel(io.BytesIO(content), nrows=MAX_BATCH_ROWS + 1)
        first_line = content.split(b"\n", 1)[0]
        russian = first_line.count(b";") > first_line.count(b",")
        return pd.read_csv(
            io.BytesIO(content),
            sep=";" if russian else ",",
            decimal="," if russian else ".",
            nrows=MAX_BATCH_ROWS + 1,
        )
    except BatchError:
        raise
    except Exception as exc:  # pandas и openpyxl бросают разные исключения на битых файлах
        LOG.warning("Файл %r не прочитан: %s", filename, exc)
        raise BatchError(READ_ERROR, status_code=400) from exc


def validate_batch(frame: pd.DataFrame) -> pd.DataFrame:
    """Проверяет размер, колонки и диапазоны. Возвращает только свои колонки в числах."""
    if len(frame) > MAX_BATCH_ROWS:
        raise BatchError(f"Слишком много строк: больше {MAX_BATCH_ROWS}", status_code=413)
    if frame.shape[1] > MAX_BATCH_COLUMNS:
        raise BatchError(f"Слишком много колонок: {frame.shape[1]}, максимум {MAX_BATCH_COLUMNS}")
    if frame.empty:
        raise BatchError("В файле нет строк с данными")
    missing = [col for col in REQUIRED_BATCH_COLUMNS if col not in frame.columns]
    if missing:
        raise BatchError(f"Нет обязательных колонок: {', '.join(missing)}")
    known = [col for col in (*config.FEATURES_ALL, config.TARGET) if col in frame.columns]
    numeric = frame[known].apply(pd.to_numeric, errors="coerce")
    problems = []
    for col in known:
        # Файлы станции проверяем по контракту данных: там допускается огрех датчика RH до 100.5%.
        bounds = ACTUAL_RANGE if col == config.TARGET else config.CONTRACT_RANGES[col]
        values = numeric[col]
        bad = int((values.isna() | ~values.between(bounds.low, bounds.high)).sum())
        if bad:
            problems.append(f"{col}: {bad} пустых или невозможных значений")
    if problems:
        raise BatchError("; ".join(problems))
    return numeric


class ForecastService:
    """Оборачивает файл модели и отвечает на вопросы страницы и API."""

    def __init__(self, bundle: Bundle) -> None:
        self.bundle = bundle
        self._plan_threshold = window_threshold(bundle.plan.residuals, bundle.detector_window)

    # --- одиночный прогноз -------------------------------------------------------------

    def _forecast(self, pack: ModelPack, point: pd.DataFrame, explain: bool) -> Forecast:
        power = float(pack.predict(point)[0])
        low, high = interval(power, pack.q)
        forecast = Forecast(
            model=pack.name, power=_round(power), low=_round(low), high=_round(high)
        )
        if not explain:
            return forecast
        base, parts = shapley_contributions(pack.predict, point, pack.background, pack.features)
        contributions = [
            Contribution(feature=name, label=config.FEATURE_LABELS[name], mw=_round(value))
            for name, value in parts.items()
        ]
        return forecast.model_copy(
            update={"base_mw": _round(base), "contributions": contributions}
        )

    def _curve(self, request: PredictRequest) -> list[CurvePoint]:
        """Мощность по плану при изменении температуры и остальной погоде как сейчас."""
        novelty = self.bundle.plan.novelty
        at_index = novelty.features.index("AT")
        grid = np.linspace(novelty.low[at_index], novelty.high[at_index], CURVE_POINTS)
        frame = pd.DataFrame({"AT": grid, "AP": request.AP, "RH": request.RH})
        powers = self.bundle.plan.predict(frame)
        lows, highs = interval(powers, self.bundle.plan.q)
        novel = novelty_distance(novelty, frame) > novelty.threshold
        return [
            CurvePoint(AT=_round(at, 1), power=_round(p), low=_round(lo), high=_round(hi),
                       novel=bool(flag))
            for at, p, lo, hi, flag in zip(grid, powers, lows, highs, novel, strict=True)
        ]

    def _warnings(self, request: PredictRequest) -> list[str]:
        point = request.model_dump()
        warnings = check_novelty(self.bundle.plan.novelty, point)
        if request.V is not None:
            warnings += check_novelty(self.bundle.reference.novelty, point)
        return list(dict.fromkeys(warnings))  # без повторов, порядок сохраняется

    def _deviation(
        self, request: PredictRequest, plan: Forecast, reference: Forecast
    ) -> Deviation:
        actual = request.actual_mw
        parts = decompose(plan.power, reference.power, actual)
        price = (
            self.bundle.price_default
            if request.price_rub_per_mwh is None
            else request.price_rub_per_mwh
        )
        return Deviation(
            total_mw=_round(parts.total),
            vacuum_mw=_round(parts.vacuum),
            other_mw=_round(parts.other),
            below_reference_interval=actual < reference.low,
            rub_per_hour=_round(max(0.0, -parts.total) * price, 0),
        )

    def _plan_risk(self, plan: Forecast, required: float | None) -> PlanRisk | None:
        if required is None:
            return None
        probability = shortfall_probability(plan.power, required, self.bundle.plan.residuals)
        return PlanRisk(required_mw=required, shortfall_probability=_round(probability, 3))

    def predict(self, request: PredictRequest) -> PredictResponse:
        weather = {"AT": request.AT, "AP": request.AP, "RH": request.RH}
        plan = self._forecast(self.bundle.plan, pd.DataFrame([weather]), request.explain)
        reference = None
        if request.V is not None:
            point = pd.DataFrame([{**weather, "V": request.V}])
            reference = self._forecast(self.bundle.reference, point, request.explain)
        deviation = None
        if reference is not None and request.actual_mw is not None:
            deviation = self._deviation(request, plan, reference)
        return PredictResponse(
            level=self.bundle.level,
            plan=plan,
            reference=reference,
            plan_risk=self._plan_risk(plan, request.required_mw),
            deviation=deviation,
            curve=self._curve(request),
            warnings=self._warnings(request),
            model_version=self.bundle.version,
        )

    # --- справка для страницы -----------------------------------------------------------

    def meta(self) -> MetaResponse:
        novelty = self.bundle.reference.novelty
        background = self.bundle.reference.background
        features = [
            FeatureInfo(
                name=name,
                label=config.FEATURE_LABELS[name],
                train_min=_round(novelty.low[i]),
                train_max=_round(novelty.high[i]),
                input_min=config.INPUT_RANGES[name].low,
                input_max=config.INPUT_RANGES[name].high,
                median=_round(background[name].median(), 1),
            )
            for i, name in enumerate(novelty.features)
        ]
        importance = {
            key: [ImportanceItem(**item) for item in items]
            for key, items in self.bundle.importance.items()
        }
        return MetaResponse(
            model_version=self.bundle.version,
            level=self.bundle.level,
            mean_power=_round(self.bundle.mean_power),
            price_default=self.bundle.price_default,
            detector_window_h=self.bundle.detector_window,
            detector_threshold_mw=_round(self.bundle.detector_threshold),
            features=features,
            presets=dict(self.bundle.presets),
            importance=importance,
            metrics=dict(self.bundle.metrics),
        )

    # --- пакет часов --------------------------------------------------------------------

    def example_csv(self) -> str:
        return self.bundle.example_batch.to_csv(index=False)

    def _batch_arrays(self, frame: pd.DataFrame) -> BatchArrays:
        has_vacuum, has_actual = "V" in frame.columns, config.TARGET in frame.columns
        plan = self.bundle.plan.predict(frame)
        low, high = interval(plan, self.bundle.plan.q)
        reference = self.bundle.reference.predict(frame) if has_vacuum else None
        actual = frame[config.TARGET].to_numpy(dtype=float) if has_actual else None
        expected = reference if reference is not None else plan
        deviation = actual - expected if actual is not None else None
        threshold = self.bundle.detector_threshold if has_vacuum else self._plan_threshold
        alarms = (
            flag_windows(deviation, self.bundle.detector_window, threshold)
            if deviation is not None
            else np.zeros(len(frame), dtype=bool)
        )
        pack = self.bundle.reference if has_vacuum else self.bundle.plan
        novelty = novelty_distance(pack.novelty, frame) > pack.novelty.threshold
        return BatchArrays(plan, low, high, reference, actual, deviation, alarms, novelty)

    @staticmethod
    def _batch_rows(arrays: BatchArrays) -> list[BatchRow]:
        vacuum = None if arrays.reference is None else arrays.reference - arrays.plan
        other = (
            None
            if arrays.reference is None or arrays.actual is None
            else arrays.actual - arrays.reference
        )
        return [
            BatchRow(
                hour=i + 1,
                plan=_round(arrays.plan[i]),
                low=_round(arrays.low[i]),
                high=_round(arrays.high[i]),
                reference=_optional(arrays.reference, i),
                actual=_optional(arrays.actual, i),
                deviation=_optional(arrays.deviation, i),
                vacuum=_optional(vacuum, i),
                other=_optional(other, i),
                alarm=bool(arrays.alarms[i]),
                novelty=bool(arrays.novelty[i]),
            )
            for i in range(len(arrays.plan))
        ]

    def _batch_summary(self, arrays: BatchArrays) -> BatchSummary:
        alarms = arrays.alarms
        summary = BatchSummary(
            rows=len(arrays.plan),
            has_actual=arrays.actual is not None,
            has_vacuum=arrays.reference is not None,
            alarm_hours=int(alarms.sum()),
            first_alarm_hour=int(np.argmax(alarms)) + 1 if alarms.any() else None,
            novelty_hours=int(arrays.novelty.sum()),
        )
        if arrays.deviation is None:
            return summary
        lost = float(np.clip(-arrays.deviation, 0, None).sum())
        return summary.model_copy(
            update={
                "mean_deviation_mw": _round(arrays.deviation.mean()),
                "lost_mwh": _round(lost, 1),
                "lost_rub": _round(lost * self.bundle.price_default, 0),
            }
        )

    def batch(self, filename: str, content: bytes) -> BatchResponse:
        frame = validate_batch(read_table(filename, content))
        arrays = self._batch_arrays(frame)
        return BatchResponse(summary=self._batch_summary(arrays), rows=self._batch_rows(arrays))
