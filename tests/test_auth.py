"""Тесты опциональной защиты API-ключом (service/auth.py) -- по умолчанию (DS_API_KEY
не задан) сервис открыт, как и раньше; остальные тесты (test_service.py и др.)
запускаются без переменной окружения и подтверждают это поведение неявно."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from service.app import app  # noqa: E402

client = TestClient(app)


def test_health_never_requires_key(monkeypatch):
    monkeypatch.setenv("DS_API_KEY", "secret123")
    resp = client.get("/health")
    assert resp.status_code == 200


def test_protected_route_rejects_missing_key(monkeypatch):
    monkeypatch.setenv("DS_API_KEY", "secret123")
    resp = client.get("/datasets")
    assert resp.status_code == 401


def test_protected_route_rejects_wrong_key(monkeypatch):
    monkeypatch.setenv("DS_API_KEY", "secret123")
    resp = client.get("/datasets", headers={"X-API-Key": "wrong"})
    assert resp.status_code == 401


def test_protected_route_accepts_correct_key(monkeypatch):
    monkeypatch.setenv("DS_API_KEY", "secret123")
    resp = client.get("/datasets", headers={"X-API-Key": "secret123"})
    assert resp.status_code == 200


def test_no_key_configured_means_open_access(monkeypatch):
    monkeypatch.delenv("DS_API_KEY", raising=False)
    resp = client.get("/datasets")
    assert resp.status_code == 200


def test_protected_route_accepts_key_via_query_param(monkeypatch):
    """EventSource (используется для GET /training/history/stream, §2.5.7)
    не умеет слать кастомные заголовки -- поэтому ключ дополнительно
    принимается через ?api_key=, см. service/auth.py."""
    monkeypatch.setenv("DS_API_KEY", "secret123")
    resp = client.get("/datasets?api_key=secret123")
    assert resp.status_code == 200


def test_protected_route_rejects_wrong_query_param_key(monkeypatch):
    monkeypatch.setenv("DS_API_KEY", "secret123")
    resp = client.get("/datasets?api_key=wrong")
    assert resp.status_code == 401
