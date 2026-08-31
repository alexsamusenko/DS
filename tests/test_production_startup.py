"""Тесты стартовой проверки production-конфигурации (service/app.py: lifespan).

См. аудит перед развёртыванием: раньше сервис мог молча подняться в
DS_ENV=production без DS_API_KEY (открытый API) и/или без предобученной
модели L4 на диске (прогнозы обслуживались бы синтетической demo-моделью).
Теперь это фатальная ошибка старта, а не тихая деградация.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from service.app import app  # noqa: E402
from service.routers import prediction  # noqa: E402


def test_dev_mode_starts_without_api_key_or_model(monkeypatch, tmp_path):
    monkeypatch.delenv("DS_ENV", raising=False)
    monkeypatch.delenv("DS_API_KEY", raising=False)
    monkeypatch.setattr(prediction, "_MODEL_PATH", tmp_path / "missing.joblib")
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200


def test_production_mode_refuses_to_start_without_api_key(monkeypatch, tmp_path):
    monkeypatch.setenv("DS_ENV", "production")
    monkeypatch.delenv("DS_API_KEY", raising=False)
    monkeypatch.setattr(prediction, "_MODEL_PATH", tmp_path / "l4.joblib")
    (tmp_path / "l4.joblib").write_bytes(b"stub")  # модель "есть" -- проверяем именно отсутствие ключа

    with pytest.raises(RuntimeError, match="DS_API_KEY"), TestClient(app):
        pass


def test_production_mode_refuses_to_start_without_model(monkeypatch, tmp_path):
    monkeypatch.setenv("DS_ENV", "production")
    monkeypatch.setenv("DS_API_KEY", "secret123")
    monkeypatch.setattr(prediction, "_MODEL_PATH", tmp_path / "missing.joblib")

    with pytest.raises(RuntimeError, match="модель L4 не найдена"), TestClient(app):
        pass


def test_production_mode_starts_with_key_and_model(monkeypatch, tmp_path):
    monkeypatch.setenv("DS_ENV", "production")
    monkeypatch.setenv("DS_API_KEY", "secret123")
    model_path = tmp_path / "l4.joblib"
    model_path.write_bytes(b"stub")
    monkeypatch.setattr(prediction, "_MODEL_PATH", model_path)

    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
