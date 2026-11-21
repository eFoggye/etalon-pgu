"""Разведочный анализ: качество данных, распределения, связи и покрытие пространства признаков."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression

from powerplant import config

MODE_MIN_SHARE = 0.2  # пик ниже 20% от самого высокого не считаем отдельной модой
MIN_BIN_COUNT = 30  # медиану по корзине рисуем, только если в ней достаточно точек


@dataclass(frozen=True)
class QualityReport:
    """Сводка проблем качества данных."""

    rows: int
    missing: Mapping[str, int]
    duplicates: int
    conflicts: int
    humidity_over_100: int
    out_of_contract: Mapping[str, int]


@dataclass(frozen=True)
class Gap:
    """Пустой промежуток в значениях признака."""

    low: float
    high: float
    count_below: int
    count_above: int


def quality_report(df: pd.DataFrame) -> QualityReport:
    """Считает пропуски, дубли, конфликты «тот же вход — другая мощность» и выходы за контракт."""
    features = [col for col in config.FEATURES_ALL if col in df.columns]
    unique_rows = df.drop_duplicates()
    out_of_contract = {
        col: int((df[col].notna() & ~df[col].between(rng.low, rng.high)).sum())
        for col, rng in config.CONTRACT_RANGES.items()
        if col in df.columns
    }
    return QualityReport(
        rows=len(df),
        missing=MappingProxyType({col: int(df[col].isna().sum()) for col in df.columns}),
        duplicates=int(df.duplicated().sum()),
        conflicts=int(unique_rows.duplicated(subset=features).sum()),
        humidity_over_100=int((df["RH"] > 100).sum()) if "RH" in df.columns else 0,
        out_of_contract=MappingProxyType(out_of_contract),
    )


def describe_with_shape(df: pd.DataFrame) -> pd.DataFrame:
    """Описательная статистика плюс межквартильный размах и асимметрия."""
    desc = df.describe().T
    return desc.assign(IQR=desc["75%"] - desc["25%"], асимметрия=df.skew())


def iqr_outliers(df: pd.DataFrame, factor: float = 1.5) -> pd.Series:
    """Сколько значений в каждой колонке выходит за усы boxplot (правило IQR)."""
    q1, q3 = df.quantile(0.25), df.quantile(0.75)
    spread = q3 - q1
    outside = (df < q1 - factor * spread) | (df > q3 + factor * spread)
    return outside.sum().rename("выбросов по IQR")


def correlation_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Корреляции Пирсона (линейная связь) и Спирмена (монотонная связь)."""
    return df.corr(method="pearson"), df.corr(method="spearman")


def variance_inflation(df: pd.DataFrame, features: Sequence[str]) -> pd.Series:
    """VIF: насколько признак объясняется остальными. Больше 5–10 — сильная мультиколлинеарность."""
    values = {}
    for col in features:
        others = [other for other in features if other != col]
        r2 = LinearRegression().fit(df[others], df[col]).score(df[others], df[col])
        values[col] = 1.0 / (1.0 - r2) if r2 < 1.0 else float("inf")
    return pd.Series(values, name="VIF")


def partial_correlation(df: pd.DataFrame, x: str, y: str, controls: Sequence[str]) -> float:
    """Корреляция x и y после вычитания линейного влияния контрольных признаков."""
    ctrl = df[list(controls)]
    resid_x = df[x] - LinearRegression().fit(ctrl, df[x]).predict(ctrl)
    resid_y = df[y] - LinearRegression().fit(ctrl, df[y]).predict(ctrl)
    return float(np.corrcoef(resid_x, resid_y)[0, 1])


def find_gaps(values: pd.Series, min_width: float) -> list[Gap]:
    """Ищет пустые промежутки шириной не меньше `min_width` между соседними значениями."""
    ordered = np.sort(values.dropna().to_numpy())
    wide = np.flatnonzero(np.diff(ordered) >= min_width)
    return [
        Gap(
            low=float(ordered[i]),
            high=float(ordered[i + 1]),
            count_below=int(i + 1),
            count_above=int(len(ordered) - i - 1),
        )
        for i in wide
    ]


def histogram_modes(values: pd.Series, bins: int = 30, smooth: int = 3) -> list[float]:
    """Центры заметных пиков сглаженной гистограммы: грубая проверка многомодальности."""
    counts, edges = np.histogram(values.dropna(), bins=bins)
    smoothed = np.convolve(counts, np.ones(smooth) / smooth, mode="same")
    centers = (edges[:-1] + edges[1:]) / 2
    threshold = MODE_MIN_SHARE * smoothed.max()
    return [
        float(centers[i])
        for i in range(1, len(smoothed) - 1)
        if smoothed[i] > smoothed[i - 1] and smoothed[i] >= smoothed[i + 1]
        and smoothed[i] >= threshold
    ]


def physics_sign_check(df: pd.DataFrame) -> pd.DataFrame:
    """Сравнивает знак связи признака с мощностью с ожидаемым по физике."""
    expected = {"AT": -1, "V": -1, "AP": 1}
    rows = []
    for col, sign in expected.items():
        corr = df[col].corr(df[config.TARGET])
        rows.append(
            {
                "признак": col,
                "ожидаемый знак": "+" if sign > 0 else "−",
                "корреляция с PE": round(corr, 3),
                "сходится": bool(np.sign(corr) == sign),
            }
        )
    return pd.DataFrame(rows)


def _label(col: str) -> str:
    return config.FEATURE_LABELS.get(col, col)


def plot_distributions(df: pd.DataFrame) -> Figure:
    """Гистограмма с оценкой плотности и boxplot для каждой колонки."""
    fig, axes = plt.subplots(len(df.columns), 2, figsize=(13, 3.0 * len(df.columns)))
    for row, col in enumerate(df.columns):
        sns.histplot(df[col], kde=True, bins=40, ax=axes[row, 0], color="#3b6ea5")
        axes[row, 0].set_title(f"Распределение: {_label(col)}", fontsize=11)
        axes[row, 0].set_xlabel("")
        sns.boxplot(x=df[col], ax=axes[row, 1], color="#e8a87c")
        axes[row, 1].set_title(f"Boxplot: {_label(col)}", fontsize=11)
        axes[row, 1].set_xlabel("")
    fig.tight_layout()
    return fig


def plot_correlations(pearson: pd.DataFrame, spearman: pd.DataFrame) -> Figure:
    """Две тепловые карты корреляций рядом."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, table, title in (
        (axes[0], pearson, "Пирсон (линейная связь)"),
        (axes[1], spearman, "Спирмен (монотонная связь)"),
    ):
        sns.heatmap(table, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, ax=ax)
        ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_target_relations(df: pd.DataFrame, features: Sequence[str], bins: int = 25) -> Figure:
    """Мощность против каждого признака: точки и медиана по корзинам (видна нелинейность)."""
    fig, axes = plt.subplots(1, len(features), figsize=(4.2 * len(features), 4), sharey=True)
    for ax, col in zip(np.atleast_1d(axes), features, strict=True):
        ax.scatter(df[col], df[config.TARGET], s=4, alpha=0.12, color="#2a7f62")
        groups = df.groupby(pd.cut(df[col], bins=bins), observed=True)
        medians = groups[[col, config.TARGET]].median()[groups.size() >= MIN_BIN_COUNT]
        ax.plot(medians[col], medians[config.TARGET], color="#c0392b", linewidth=2.2)
        ax.set_xlabel(_label(col))
        ax.set_title(f"r = {df[col].corr(df[config.TARGET]):.3f}")
    np.atleast_1d(axes)[0].set_ylabel(_label(config.TARGET))
    fig.tight_layout()
    return fig


def plot_coverage(df: pd.DataFrame, x: str = "AT", y: str = "V") -> Figure:
    """Плотность точек в плоскости двух признаков: где данных нет, там модель гадает."""
    fig, ax = plt.subplots(figsize=(7, 5))
    hexes = ax.hexbin(df[x], df[y], gridsize=35, mincnt=1, cmap="viridis", bins="log")
    fig.colorbar(hexes, ax=ax, label="точек в ячейке (лог. шкала)")
    ax.set_xlabel(_label(x))
    ax.set_ylabel(_label(y))
    ax.set_title("Покрытие пространства признаков: пустые места — зона без данных")
    fig.tight_layout()
    return fig
