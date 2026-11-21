"""Общие настройки проекта: пути, признаки, границы и параметры проверки."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
CSV_PATH = DATA_DIR / "ccpp.csv"
XLSX_PATH = DATA_DIR / "Folds5x2_pp.xlsx"
MODELS_DIR = PROJECT_ROOT / "models"
BUNDLE_PATH = MODELS_DIR / "bundle.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"

TARGET = "PE"
FEATURES_ALL: tuple[str, ...] = ("AT", "V", "AP", "RH")
# Модель плана: вакуума на завтра никто не знает, поэтому V в неё не входит.
FEATURES_PLAN: tuple[str, ...] = ("AT", "AP", "RH")

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
INTERVAL_LEVEL = 0.9
# Правило G2: берём самую простую модель, чей RMSE не хуже лучшего более чем на 5%.
SELECTION_TOLERANCE = 0.05
# Физика, которую модель обязана соблюдать (решения 15 и 17 журнала): жарче — мощность не растёт.
# Для бустинга это ограничение монотонности. V и AP не ограничиваем: шкала V физически
# неоднозначна, а частный эффект AP в данных немонотонный (опыт на CV, журнал, решение 17).
MONOTONIC = MappingProxyType({"AT": -1})
# Дополнительные ограничения для физических признаков (плотность, точка росы) не задаём:
# вариант с ними остаётся, только если проходит физическую проверку.
PHYSICS_FEATURE_MONOTONIC = MappingProxyType({})
# Допуск физической проверки: подъём мощности с ростом температуры не больше 0.5 МВт.
PHYSICS_TOLERANCE_MW = 0.5
# Цена по умолчанию для перевода МВт в рубли: индекс равновесных цен на покупку на РСВ,
# первая ценовая зона, среднее за 2024 год — 1 747,29 ₽/МВт·ч (Ассоциация «НП Совет рынка»,
# по Neftegaz.RU от 12.03.2025). На странице поле редактируемое.
PRICE_RUB_PER_MWH = 1747.0


@dataclass(frozen=True)
class Range:
    """Замкнутый диапазон допустимых значений."""

    low: float
    high: float

    def contains(self, value: float) -> bool:
        return self.low <= value <= self.high


# Контракт данных: значения за этими границами считаем браком файла.
CONTRACT_RANGES = MappingProxyType(
    {
        "AT": Range(-30.0, 50.0),
        "V": Range(20.0, 90.0),
        "AP": Range(950.0, 1060.0),
        # В данных встречается RH до 100.16 — огрех датчика, такие строки допускаем.
        "RH": Range(0.0, 100.5),
        "PE": Range(300.0, 600.0),
    }
)

# Физические границы ввода в API: за ними запрос отклоняется.
INPUT_RANGES = MappingProxyType(
    {
        "AT": Range(-30.0, 50.0),
        "V": Range(20.0, 90.0),
        "AP": Range(950.0, 1060.0),
        "RH": Range(0.0, 100.0),
    }
)

FEATURE_LABELS = MappingProxyType(
    {
        "AT": "Температура воздуха, °C",
        "V": "Вакуум на выхлопе (шкала датасета)",
        "AP": "Атмосферное давление, мбар",
        "RH": "Относительная влажность, %",
        "PE": "Мощность, МВт",
    }
)
