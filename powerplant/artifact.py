"""Один файл модели: обе модели, диапазоны, границы данных и всё, что нужно сервису."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from powerplant import config
from powerplant.novelty import NoveltyModel


@dataclass(frozen=True)
class ModelPack:
    """Модель плана или эталона, обученная на всех данных, и её паспорт."""

    name: str
    features: tuple[str, ...]
    estimator: BaseEstimator
    q: float  # полуширина диапазона заданного уровня, МВт
    residuals: np.ndarray  # внефолдовые остатки «факт − прогноз» на всех данных
    novelty: NoveltyModel
    background: pd.DataFrame  # фоновые строки для вкладов Шепли
    physics: bool

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.estimator.predict(frame[list(self.features)])


@dataclass(frozen=True)
class Bundle:
    """Всё, что нужно странице и API, в одном файле."""

    version: str
    created_at: str
    dataset_sha256: str
    level: float
    plan: ModelPack
    reference: ModelPack
    detector_window: int
    detector_threshold: float  # МВт: порог для среднего (факт − эталон) за окно
    presets: Mapping[str, Mapping[str, float]]
    importance: Mapping[str, list[dict]]
    metrics: Mapping[str, object]
    mean_power: float
    price_default: float  # ₽/МВт·ч
    example_batch: pd.DataFrame  # демо-файл для пакетного прогноза: строки датасета с поломкой


def save_bundle(bundle: Bundle, path: Path = config.BUNDLE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path, compress=3)


def load_bundle(path: Path = config.BUNDLE_PATH) -> Bundle:
    """Грузит файл модели проекта по пути из настроек.

    joblib исполняет код из файла при загрузке, поэтому подкладывать сюда чужие файлы нельзя.
    Проверка типа ниже — защита от перепутанного файла, а не от злонамеренного.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Нет файла модели {path}. Сначала запустите: .venv/bin/python scripts/train.py"
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, Bundle):
        raise TypeError(f"{path} — не файл модели проекта (ожидался Bundle)")
    return bundle
