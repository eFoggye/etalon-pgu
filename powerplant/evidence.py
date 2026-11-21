"""Обоснование решений сверх кейса: статистические проверки, на которые ссылаются слайды.

1. Вакуум: модель с V против без V (парный t-тест ошибок и 5×2cv t-тест Диттериха)
   и двухступенчатая модель «V по погоде → эталон» против модели плана.
2. Предупреждение о новизне: ошибка на часах с флагом против остальных.
3. Риск не выполнить план: калибровка заявленной вероятности на отложенных данных.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict

from powerplant import config

DIETTERICH_DF = 5  # пять повторов 2-fold проверки


def paired_error_test(y, pred_a, pred_b) -> dict[str, float]:
    """Сравнение двух моделей на одних и тех же часах: парный t-тест и Уилкоксон по |ошибкам|."""
    actual = np.asarray(y, dtype=float)
    err_a = np.abs(actual - np.asarray(pred_a, dtype=float))
    err_b = np.abs(actual - np.asarray(pred_b, dtype=float))
    t_stat, p_value = stats.ttest_rel(err_a, err_b)
    return {
        "mae_a": float(err_a.mean()),
        "mae_b": float(err_b.mean()),
        "mean_diff": float((err_a - err_b).mean()),
        "t": float(t_stat),
        "p_value": float(p_value),
        "wilcoxon_p": float(stats.wilcoxon(err_a, err_b).pvalue),
    }


def dietterich_t(diffs: np.ndarray) -> tuple[float, float]:
    """Статистика 5×2cv (Dietterich, 1998) по матрице разностей ошибок 5×2."""
    values = np.asarray(diffs, dtype=float)
    variances = ((values - values.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    t_stat = values[0, 0] / np.sqrt(variances.mean())
    return float(t_stat), float(2 * stats.t.sf(abs(t_stat), df=DIETTERICH_DF))


def dietterich_5x2cv(
    model_a: BaseEstimator,
    features_a: Sequence[str],
    model_b: BaseEstimator,
    features_b: Sequence[str],
    sheets: Sequence[pd.DataFrame],
) -> dict[str, float]:
    """5×2cv на пяти перемешиваниях авторов датасета: разность MAE модели A и модели B."""
    diffs = []
    for frame in sheets:
        half = len(frame) // 2
        row = []
        for train, test in ((frame.iloc[:half], frame.iloc[half:]),
                            (frame.iloc[half:], frame.iloc[:half])):
            maes = []
            for model, features in ((model_a, features_a), (model_b, features_b)):
                fitted = clone(model).fit(train[list(features)], train[config.TARGET])
                error = test[config.TARGET].to_numpy() - fitted.predict(test[list(features)])
                maes.append(float(np.mean(np.abs(error))))
            row.append(maes[0] - maes[1])
        diffs.append(row)
    t_stat, p_value = dietterich_t(np.array(diffs))
    return {"mean_mae_diff": float(np.mean(diffs)), "t": t_stat, "p_value": p_value}


class VacuumFromWeather(RegressorMixin, BaseEstimator):
    """Двухступенчатая модель: сначала V по погоде, затем мощность по (AT, V̂, AP, RH).

    Так выглядела бы попытка «спрогнозировать вакуум на завтра». На обучении V̂ берётся
    из кросс-фита, чтобы вторая ступень училась на таких же неточных V̂, как в работе.
    При прогнозе вакуум не нужен — только погода.
    """

    def __init__(self, vacuum_model=None, power_model=None, folds: int = config.CV_FOLDS,
                 seed: int = config.SEED) -> None:
        self.vacuum_model = vacuum_model
        self.power_model = power_model
        self.folds = folds
        self.seed = seed

    def _base(self, model):
        return clone(model) if model is not None else HistGradientBoostingRegressor(
            random_state=self.seed)

    def fit(self, x: pd.DataFrame, y) -> VacuumFromWeather:
        weather = x[list(config.FEATURES_PLAN)]
        folds = KFold(self.folds, shuffle=True, random_state=self.seed)
        vacuum_oof = cross_val_predict(self._base(self.vacuum_model), weather, x["V"], cv=folds)
        self.vacuum_model_ = self._base(self.vacuum_model).fit(weather, x["V"])
        staged = weather.assign(V=vacuum_oof)[list(config.FEATURES_ALL)]
        self.power_model_ = self._base(self.power_model).fit(staged, y)
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        weather = x[list(config.FEATURES_PLAN)]
        staged = weather.assign(V=self.vacuum_model_.predict(weather))[list(config.FEATURES_ALL)]
        return self.power_model_.predict(staged)


def novelty_error_ratio(errors, flags) -> dict[str, float]:
    """Во сколько раз ошибка на часах с флагом новизны больше, чем на остальных."""
    values = np.abs(np.asarray(errors, dtype=float))
    marked = np.asarray(flags, dtype=bool)
    flagged = float(values[marked].mean()) if marked.any() else float("nan")
    rest = float(values[~marked].mean())
    return {"mae_flagged": flagged, "mae_rest": rest, "ratio": flagged / rest,
            "n_flagged": int(marked.sum())}


def calibration_table(y, prediction, residuals, levels=(0.05, 0.1, 0.25, 0.5)) -> pd.DataFrame:
    """Заявленная вероятность недобора против наблюдаемой на отложенных данных.

    Для уровня α берём требуемый объём = прогноз + α-квантиль остатков: модель обещает,
    что факт окажется ниже него с вероятностью α.
    """
    actual = np.asarray(y, dtype=float)
    centre = np.asarray(prediction, dtype=float)
    rows = []
    for level in levels:
        required = centre + np.quantile(np.asarray(residuals, dtype=float), level)
        rows.append({"заявлено": level, "наблюдалось": float(np.mean(actual < required))})
    return pd.DataFrame(rows)
