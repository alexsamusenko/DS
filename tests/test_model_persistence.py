"""Тесты дискового кеша L4-модели (service/routers/prediction.py) -- обучение не
должно повторяться при каждом рестарте процесса, а устаревший кеш (после
изменения конфигурации обучения) не должен молча использоваться."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from service.routers import prediction as prediction_module  # noqa: E402


def test_model_is_persisted_and_reloaded_without_retraining(tmp_path, monkeypatch):
    model_path = tmp_path / "model.joblib"
    monkeypatch.setattr(prediction_module, "_MODEL_PATH", model_path)
    monkeypatch.setattr(prediction_module, "_cache", {})

    model1, columns1 = prediction_module._get_model()
    assert model_path.exists()

    # Симуляция нового процесса: пустой in-memory кеш -- модель должна
    # подгрузиться с диска, а не обучиться заново.
    monkeypatch.setattr(prediction_module, "_cache", {})
    model2, columns2 = prediction_module._get_model()

    assert columns2 == columns1
    assert np.array_equal(model1.feature_importances_, model2.feature_importances_)


def test_stale_cache_version_is_rejected(tmp_path, monkeypatch):
    import joblib

    model_path = tmp_path / "model.joblib"
    monkeypatch.setattr(prediction_module, "_MODEL_PATH", model_path)
    monkeypatch.setattr(prediction_module, "_cache", {})

    prediction_module._get_model()
    payload = joblib.load(model_path)
    payload["cache_version"] = -1
    joblib.dump(payload, model_path)

    assert prediction_module._load_from_disk() is None


def test_stale_modalities_are_rejected(tmp_path, monkeypatch):
    import joblib

    model_path = tmp_path / "model.joblib"
    monkeypatch.setattr(prediction_module, "_MODEL_PATH", model_path)
    monkeypatch.setattr(prediction_module, "_cache", {})

    prediction_module._get_model()
    payload = joblib.load(model_path)
    payload["modalities"] = ("num",)
    joblib.dump(payload, model_path)

    assert prediction_module._load_from_disk() is None
