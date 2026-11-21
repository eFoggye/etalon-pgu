"""Обучение: данные → разбиение → лестница моделей → выбор по G2 → диапазон → файл модели.

Запуск из корня проекта:
    .venv/bin/python scripts/train.py            # подобранные параметры берутся из кеша
    .venv/bin/python scripts/train.py --retune   # подобрать гиперпараметры заново
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import BaseEstimator, clone  # noqa: E402

from powerplant import config  # noqa: E402
from powerplant.artifact import Bundle, ModelPack, save_bundle  # noqa: E402
from powerplant.data import (  # noqa: E402
    Split,
    drop_exact_duplicates,
    load_raw,
    split_holdout,
    validate_contract,
)
from powerplant.detector import DEFAULT_WINDOW, detection_rate, window_threshold  # noqa: E402
from powerplant.importance import (  # noqa: E402
    grouped_permutation_importance,
    pdp_curves,
    pdp_slope,
    standardized_coefficients,
)
from powerplant.intervals import (  # noqa: E402
    conformal_quantile,
    coverage,
    coverage_by_slices,
    interval,
)
from powerplant.models import (  # noqa: E402
    STACKING_COMPLEXITY,
    CvResult,
    build_ladder,
    build_stacking,
    evaluate_oof,
    regression_metrics,
    select_model,
    tune,
    with_physics,
)
from powerplant.novelty import fit_novelty  # noqa: E402

LOG = logging.getLogger("train")
STRATIFY_BINS = 10  # решение 11 журнала: стратификация по 10 квантилям мощности
PHYSICS_MIN_GAIN = 0.01  # физические признаки оставляем, если RMSE на CV падает хотя бы на 1%
TUNED_PARAMS_PATH = config.MODELS_DIR / "tuned_params.json"
MODEL_SETS = {"plan": config.FEATURES_PLAN, "ref": config.FEATURES_ALL}
IMPORTANCE_GROUPS = {
    "plan": {"AT": ["AT"], "AP": ["AP"], "RH": ["RH"]},
    "ref": {"AT": ["AT"], "V": ["V"], "AP": ["AP"], "RH": ["RH"], "AT + V": ["AT", "V"]},
}
STACKING_NAME = "Стекинг (лес + бустинг)"
SLICE_LABELS = ["низ", "середина", "верх"]
SLICE_FEATURES = ("AT", "V", "RH")
BACKGROUND_ROWS = 300
# Демо-файл пакетного прогноза: строки, которых боевая модель не видела, с искусственной поломкой.
DEMO_ROWS = 168
DEMO_DROP = 0.02
DEGRADATION_SHARES = (0.01, 0.02, 0.03)
UNSEEN_COMBINATION = {"AT": 33.0, "V": 38.0}  # решение 13 журнала: зона без данных


@dataclass(frozen=True)
class SetOutcome:
    """Итог по одному набору признаков: отчёт, кеш параметров и выбранная модель."""

    summary: dict
    params: dict
    chosen_name: str
    estimator: BaseEstimator
    physics: bool


def _jsonable(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def load_clean() -> pd.DataFrame:
    return drop_exact_duplicates(validate_contract(load_raw()))


def tuned_ladder(
    x: pd.DataFrame, y: pd.Series, cached: dict[str, dict], retune: bool
) -> dict[str, tuple[BaseEstimator, dict, int]]:
    """Подбирает (или берёт из кеша) гиперпараметры каждой ступени лестницы."""
    ladder = {}
    for candidate in build_ladder(config.SEED):
        if candidate.name in cached and not retune:
            params = cached[candidate.name]
            estimator = clone(candidate.estimator).set_params(**params)
        else:
            estimator, params = tune(candidate, x, y, seed=config.SEED)
            params = {key: _jsonable(val) for key, val in params.items()}
        LOG.info("  %s: %s", candidate.name, params)
        ladder[candidate.name] = (estimator, params, candidate.complexity)
    forest, boosting = ladder["Случайный лес"][0], ladder["Градиентный бустинг"][0]
    ladder[STACKING_NAME] = (build_stacking(forest, boosting, config.SEED), {}, STACKING_COMPLEXITY)
    return ladder


def ladder_results(
    ladder: dict[str, tuple[BaseEstimator, dict, int]], x: pd.DataFrame, y: pd.Series
) -> list[CvResult]:
    results = []
    for name, (estimator, params, complexity) in ladder.items():
        result = evaluate_oof(name, estimator, x, y, complexity=complexity, params=params)
        LOG.info("  CV %-26s RMSE %.3f ± %.3f", name, result.rmse, result.rmse_std)
        results.append(result)
    return results


def cv_rows(results: list[CvResult]) -> list[dict[str, object]]:
    return [
        {"name": r.name, "complexity": r.complexity, "MAE": r.mae, "RMSE": r.rmse,
         "RMSE_std": r.rmse_std, "R2": r.r2, "MAPE": r.mape, "params": dict(r.params)}
        for r in results
    ]


def test_rows(
    ladder: dict[str, tuple[BaseEstimator, dict, int]], split: Split, features: list[str]
) -> list[dict[str, float]]:
    """Единственный прогон на отложенном тесте — для итоговой таблицы, без выбора по нему."""
    x_train, y_train = split.train[features], split.train[config.TARGET]
    rows = []
    for name, (estimator, _, _) in ladder.items():
        prediction = clone(estimator).fit(x_train, y_train).predict(split.test[features])
        rows.append({"name": name, **regression_metrics(split.test[config.TARGET], prediction)})
    return rows


def slice_edges(values: pd.Series) -> list[float]:
    low, high = values.quantile([1 / 3, 2 / 3])
    return [-np.inf, float(low), float(high), np.inf]


def slice_tables(
    y: pd.Series, prediction: np.ndarray, q: float, frame: pd.DataFrame, train: pd.DataFrame
) -> dict[str, list[dict]]:
    """Покрытие по третям AT, V и RH. Границы третей берутся по обучающей части."""
    tables = {}
    for col in SLICE_FEATURES:
        edges = slice_edges(train[col])
        table = coverage_by_slices(y, prediction, q, frame[col], edges, SLICE_LABELS)
        tables[col] = table.reset_index().to_dict(orient="records")
    return tables


def interval_report(
    fitted: BaseEstimator, oof: np.ndarray, split: Split, features: list[str]
) -> dict[str, object]:
    """Диапазон по внефолдовым остаткам обучения; покрытие — на тесте (G3) и по срезам (G3b).

    Покрытие относится к модели на 80% данных и её q. Боевая модель обучена на всех данных,
    её q пересчитан по внефолдовым остаткам на всех данных (поле service в отчёте).
    """
    y_train, y_test = split.train[config.TARGET], split.test[config.TARGET]
    q = conformal_quantile(y_train.to_numpy() - oof)
    prediction = fitted.predict(split.test[features])
    low, high = interval(prediction, q)
    return {
        "level": config.INTERVAL_LEVEL,
        "q_mw": q,
        "coverage_test": coverage(y_test, low, high),
        "slices_oof": slice_tables(y_train, oof, q, split.train, split.train),
        "slices_test": slice_tables(y_test, prediction, q, split.test, split.train),
        "test_metrics": regression_metrics(y_test, prediction),
    }


def explain_report(
    set_name: str, estimator: BaseEstimator, fitted: BaseEstimator, split: Split,
    features: list[str],
) -> dict[str, object]:
    """Три способа важности: группы на внефолдовых данных, PDP, линейные коэффициенты."""
    x, y = split.train[features], split.train[config.TARGET]
    table = grouped_permutation_importance(estimator, x, y, IMPORTANCE_GROUPS[set_name])
    curves = pdp_curves(fitted, x, features)
    return {
        "permutation": table.reset_index(names="group").to_dict(orient="records"),
        "coefficients": standardized_coefficients(x, y).to_dict(),
        "pdp": {col: {"grid": grid.tolist(), "value": vals.tolist()}
                for col, (grid, vals) in curves.items()},
        "pdp_slope_mw_per_unit": {col: pdp_slope(curve) for col, curve in curves.items()},
    }


def choose(
    results: list[CvResult], ladder: dict, x: pd.DataFrame, y: pd.Series
) -> tuple[CvResult, BaseEstimator, bool, dict]:
    """Выбор по G2 и опыт с физическими признаками для выбранной модели."""
    chosen = select_model(results)
    base = ladder[chosen.name][0]
    physics = evaluate_oof(chosen.name + " + физика", with_physics(base), x, y,
                           complexity=chosen.complexity)
    kept = physics.rmse < chosen.rmse * (1 - PHYSICS_MIN_GAIN)
    LOG.info("  выбрана %s; физика: RMSE %.3f → %.3f, оставляем: %s", chosen.name, chosen.rmse,
             physics.rmse, kept)
    report = {"rmse_base": chosen.rmse, "rmse_physics": physics.rmse, "kept": kept,
              "min_gain": PHYSICS_MIN_GAIN}
    if kept:
        return physics, with_physics(base), True, report
    return chosen, base, False, report


def run_set(set_name: str, split: Split, cached: dict, retune: bool) -> SetOutcome:
    features = list(MODEL_SETS[set_name])
    LOG.info("Набор «%s»: признаки %s", set_name, features)
    x, y = split.train[features], split.train[config.TARGET]
    ladder = tuned_ladder(x, y, cached.get(set_name, {}), retune)
    results = ladder_results(ladder, x, y)
    chosen, estimator, physics, physics_report = choose(results, ladder, x, y)
    fitted = clone(estimator).fit(x, y)
    summary = {
        "features": features,
        "cv": cv_rows(results),
        "chosen": chosen.name,
        "physics": physics_report,
        "test": test_rows(ladder, split, features),
        "interval": interval_report(fitted, chosen.oof, split, features),
        "importance": explain_report(set_name, estimator, fitted, split, features),
    }
    params = {name: p for name, (_, p, _) in ladder.items() if name != STACKING_NAME}
    return SetOutcome(summary, params, chosen.name, estimator, physics)


def service_pack(outcome: SetOutcome, df: pd.DataFrame) -> tuple[ModelPack, dict[str, float]]:
    """Модель для сервиса: остатки и диапазон — по внефолдовым прогнозам на всех данных."""
    features = outcome.summary["features"]
    x, y = df[features], df[config.TARGET]
    full = evaluate_oof(outcome.chosen_name, outcome.estimator, x, y, complexity=0)
    residuals = y.to_numpy() - full.oof
    pack = ModelPack(
        name=outcome.chosen_name,
        features=tuple(features),
        estimator=clone(outcome.estimator).fit(x, y),
        q=conformal_quantile(residuals),
        residuals=residuals,
        novelty=fit_novelty(x),
        background=x.sample(BACKGROUND_ROWS, random_state=config.SEED).reset_index(drop=True),
        physics=outcome.physics,
    )
    return pack, {"oof_rmse_all": full.rmse, "q_mw_all": pack.q}


def holdout_demo(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Откладывает строки для демо-файла: боевая модель их не увидит."""
    demo = df.sample(DEMO_ROWS, random_state=config.SEED)
    return demo.reset_index(drop=True), df.drop(index=demo.index).reset_index(drop=True)


def make_example_batch(demo: pd.DataFrame) -> pd.DataFrame:
    """Демо-файл: со второй половины мощность искусственно снижена на DEMO_DROP."""
    columns = [*config.FEATURES_ALL, config.TARGET]
    degraded = demo.index >= len(demo) // 2
    power = demo[config.TARGET].where(~degraded, demo[config.TARGET] * (1 - DEMO_DROP))
    return demo[columns].assign(**{config.TARGET: power.round(2)})


def make_presets(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Пресеты для показа: медианы реальных часов, а не придуманные числа."""
    def medians(rows: pd.DataFrame) -> dict[str, float]:
        return {col: round(float(rows[col].median()), 1) for col in config.FEATURES_ALL}

    at = df["AT"]
    middle = df[at.between(at.quantile(0.4), at.quantile(0.6))]
    hot = df[at > 30]
    return {
        "Типичный час": medians(df),
        "Зима": medians(df[at <= at.quantile(0.1)]),
        "Жара": medians(df[at >= at.quantile(0.9)]),
        "Плохой вакуум": {**medians(middle), "V": round(float(middle["V"].quantile(0.95)), 1)},
        "Небывалое сочетание": {**medians(hot), **UNSEEN_COMBINATION},
    }


def detector_report(reference: ModelPack, mean_power: float) -> tuple[float, dict[str, object]]:
    threshold = window_threshold(reference.residuals, window=DEFAULT_WINDOW)
    rates = {
        f"{int(share * 100)}%": detection_rate(reference.residuals, DEFAULT_WINDOW, threshold,
                                               shift_mw=-share * mean_power)
        for share in DEGRADATION_SHARES
    }
    LOG.info("Детектор: порог суточного среднего %.2f МВт, ловит %s", threshold, rates)
    return threshold, {"window_h": DEFAULT_WINDOW, "threshold_mw": threshold,
                       "false_alarm_rate": 0.01, "detection_rate_by_drop": rates}


def page_importance(summary: dict) -> list[dict[str, object]]:
    return [
        {"group": row["group"], "label": config.FEATURE_LABELS.get(row["group"], row["group"]),
         "mw": round(row["rmse_increase"], 2)}
        for row in summary["importance"]["permutation"]
    ]


def page_metrics(summary: dict, service: dict) -> dict[str, object]:
    chosen = next(row for row in summary["cv"] if row["name"] == summary["chosen"])
    test = summary["interval"]["test_metrics"]
    return {"name": summary["chosen"], "cv_rmse": chosen["RMSE"], "test_rmse": test["RMSE"],
            "test_mae": test["MAE"], "test_r2": test["R2"],
            "coverage_test": summary["interval"]["coverage_test"], "q_mw": service["q_mw_all"]}


def build_bundle(outcomes: dict[str, SetOutcome], df: pd.DataFrame, report: dict) -> Bundle:
    demo, service_df = holdout_demo(df)
    packs, services = {}, {}
    for set_name, outcome in outcomes.items():
        packs[set_name], services[set_name] = service_pack(outcome, service_df)
        report["sets"][set_name]["service"] = services[set_name]
    mean_power = float(df[config.TARGET].mean())
    threshold, detector = detector_report(packs["ref"], mean_power)
    report["detector"] = detector
    report["demo"] = {"rows": DEMO_ROWS, "drop": DEMO_DROP, "service_rows": len(service_df)}
    return Bundle(
        version=report["created_at"],
        created_at=report["created_at"],
        dataset_sha256=report["data"]["dataset_sha256"],
        level=config.INTERVAL_LEVEL,
        plan=packs["plan"],
        reference=packs["ref"],
        detector_window=DEFAULT_WINDOW,
        detector_threshold=threshold,
        presets=make_presets(df),
        importance={name: page_importance(report["sets"][name]) for name in outcomes},
        metrics={**{name: page_metrics(report["sets"][name], services[name]) for name in outcomes},
                 "detector": detector},
        mean_power=mean_power,
        price_default=config.PRICE_RUB_PER_MWH,
        example_batch=make_example_batch(demo),
    )


def cache_meta(report: dict) -> dict[str, object]:
    """С чем подбирались параметры: при другом наборе данных или разбиении кеш не годится."""
    return {"dataset_sha256": report["data"]["dataset_sha256"], "seed": config.SEED,
            "test_size": config.TEST_SIZE, "stratify_bins": STRATIFY_BINS}


def read_cache(meta: dict[str, object]) -> dict:
    if not TUNED_PARAMS_PATH.exists():
        return {}
    cached = json.loads(TUNED_PARAMS_PATH.read_text())
    if cached.get("meta") != meta:
        LOG.info("Кеш параметров от других данных или разбиения — подбираю заново")
        return {}
    return cached.get("params", {})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retune", action="store_true", help="подобрать гиперпараметры заново")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

    df = load_clean()
    split = split_holdout(df, stratify_bins=STRATIFY_BINS)
    LOG.info("Данные: %d строк, обучение %d, тест %d", len(df), len(split.train), len(split.test))
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "data": {"rows_clean": len(df), "train": len(split.train), "test": len(split.test),
                 "stratify_bins": STRATIFY_BINS,
                 "dataset_sha256": hashlib.sha256(config.CSV_PATH.read_bytes()).hexdigest()},
        "selection_tolerance": config.SELECTION_TOLERANCE,
        "sets": {},
    }
    meta = cache_meta(report)
    cached = read_cache(meta)
    outcomes = {name: run_set(name, split, cached, args.retune) for name in MODEL_SETS}
    report["sets"] = {name: outcome.summary for name, outcome in outcomes.items()}
    params = {name: outcome.params for name, outcome in outcomes.items()}
    TUNED_PARAMS_PATH.write_text(
        json.dumps({"meta": meta, "params": params}, ensure_ascii=False, indent=2)
    )

    bundle = build_bundle(outcomes, df, report)
    report["presets"] = bundle.presets
    save_bundle(bundle)
    config.METRICS_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                              default=_jsonable))
    LOG.info("Готово: %s и %s", config.METRICS_PATH, config.BUNDLE_PATH)


if __name__ == "__main__":
    main()
