import joblib
import pytest

from powerplant.artifact import load_bundle


def test_load_bundle_explains_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="train.py"):
        load_bundle(tmp_path / "нет.joblib")


def test_load_bundle_rejects_foreign_object(tmp_path):
    path = tmp_path / "чужой.joblib"
    joblib.dump({"не": "бандл"}, path)

    with pytest.raises(TypeError, match="Bundle"):
        load_bundle(path)
