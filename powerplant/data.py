"""Загрузка данных станции, контракт данных и честное разбиение."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from powerplant import config

COLUMNS: tuple[str, ...] = (*config.FEATURES_ALL, config.TARGET)


class DataContractError(ValueError):
    """Данные не прошли проверку контракта."""


@dataclass(frozen=True)
class Split:
    """Обучающая и отложенная части данных."""

    train: pd.DataFrame
    test: pd.DataFrame


def load_raw(path: Path = config.CSV_PATH) -> pd.DataFrame:
    """Читает один экземпляр данных (csv с UCI, он совпадает с первым листом xlsx)."""
    df = pd.read_csv(path)
    missing = [col for col in COLUMNS if col not in df.columns]
    if missing:
        raise DataContractError(f"В файле {path.name} нет колонок: {', '.join(missing)}")
    return df.loc[:, list(COLUMNS)]


def load_sheets(path: Path = config.XLSX_PATH) -> dict[str, pd.DataFrame]:
    """Читает все листы исходного xlsx. Нужно только для демонстрации ловушки с листами."""
    sheets = pd.read_excel(path, sheet_name=None)
    return {name: frame.loc[:, list(COLUMNS)] for name, frame in sheets.items()}


def glue_sheets(sheets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Склеивает листы подряд. Так делать нельзя: функция показывает, к чему это приводит."""
    return pd.concat(list(sheets.values()), ignore_index=True)


def validate_contract(df: pd.DataFrame) -> pd.DataFrame:
    """Проверяет колонки, пропуски, типы и допустимые диапазоны. При нарушении — исключение."""
    missing_columns = [col for col in COLUMNS if col not in df.columns]
    if missing_columns:
        raise DataContractError(f"Нет колонок: {', '.join(missing_columns)}")

    problems = []
    for col in COLUMNS:
        series = df[col]
        if not pd.api.types.is_numeric_dtype(series):
            problems.append(f"{col}: не числовой тип")
            continue
        missing_count = int(series.isna().sum())
        if missing_count:
            problems.append(f"{col}: {missing_count} пропуск(ов)")
        allowed = config.CONTRACT_RANGES[col]
        outside = int((series.notna() & ~series.between(allowed.low, allowed.high)).sum())
        if outside:
            problems.append(f"{col}: {outside} значений вне [{allowed.low}, {allowed.high}]")

    if problems:
        raise DataContractError("; ".join(problems))
    return df


def drop_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Возвращает новую таблицу без точных дублей строк."""
    return df.drop_duplicates().reset_index(drop=True)


def split_holdout(
    df: pd.DataFrame,
    test_size: float = config.TEST_SIZE,
    seed: int = config.SEED,
    stratify_bins: int | None = None,
) -> Split:
    """Делит данные на обучение и отложенный тест.

    При `stratify_bins` разбиение стратифицируется по квантилям мощности, чтобы обе моды
    распределения попали в тест в той же пропорции, что и в обучение.
    """
    strata = None
    if stratify_bins:
        strata = pd.qcut(df[config.TARGET], q=stratify_bins, labels=False, duplicates="drop")
    train, test = train_test_split(df, test_size=test_size, random_state=seed, stratify=strata)
    return Split(train=train.reset_index(drop=True), test=test.reset_index(drop=True))
