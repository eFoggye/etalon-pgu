"""Смоук-тесты сборки файла модели: ловят ошибки, которые иначе вылезли бы в конце обучения."""

import importlib.util
import sys

import pytest
from sklearn.linear_model import LinearRegression

from powerplant import config
from powerplant.artifact import Bundle
from powerplant.data import drop_exact_duplicates, load_raw
from powerplant.novelty import check_novelty

_SCRIPT = config.PROJECT_ROOT / "scripts" / "train.py"
_spec = importlib.util.spec_from_file_location("train_script", _SCRIPT)
train = importlib.util.module_from_spec(_spec)
sys.modules["train_script"] = train  # dataclass ищет свой модуль в sys.modules
_spec.loader.exec_module(train)


@pytest.fixture(scope="module")
def data():
    return drop_exact_duplicates(load_raw())


def _summary(features: tuple[str, ...]) -> dict:
    return {
        "features": list(features),
        "chosen": "Линейная",
        "cv": [{"name": "Линейная", "RMSE": 4.5}],
        "interval": {"coverage_test": 0.9, "test_metrics": {"RMSE": 4.5, "MAE": 3.6, "R2": 0.93}},
        "importance": {"permutation": [{"group": "AT", "rmse_increase": 10.0}]},
    }


def test_demo_rows_are_held_out_from_service_data_and_degraded(data):
    demo, service = train.holdout_demo(data)
    example = train.make_example_batch(demo)

    assert len(demo) == train.DEMO_ROWS
    assert demo.merge(service, on=list(data.columns)).empty
    half = len(example) // 2
    assert (example["PE"].iloc[:half] == demo["PE"].iloc[:half].round(2)).all()
    assert (example["PE"].iloc[half:] < demo["PE"].iloc[half:]).all()


def test_presets_cover_demo_scenarios_and_unseen_combination(data):
    presets = train.make_presets(data)

    assert set(presets) == {"Типичный час", "Зима", "Жара", "Плохой вакуум", "Небывалое сочетание"}
    assert presets["Жара"]["AT"] > presets["Зима"]["AT"]
    assert presets["Небывалое сочетание"]["AT"] == 33.0


def test_build_bundle_assembles_every_field(data):
    outcomes = {
        name: train.SetOutcome(_summary(features), {}, "Линейная", LinearRegression(), False)
        for name, features in train.MODEL_SETS.items()
    }
    report = {
        "created_at": "test",
        "data": {"dataset_sha256": "test"},
        "sets": {name: outcome.summary for name, outcome in outcomes.items()},
    }

    bundle = train.build_bundle(outcomes, data, report)

    assert isinstance(bundle, Bundle)
    assert len(bundle.example_batch) == train.DEMO_ROWS
    assert bundle.detector_threshold < 0
    unseen = bundle.presets["Небывалое сочетание"]
    assert any("Сочетание" in w for w in check_novelty(bundle.reference.novelty, unseen))
