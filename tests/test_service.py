"""Тесты FastAPI-сервиса (service/) -- каждый маршрут проверяется на реальный вызов
соответствующей функции src/ds_*, а не только на код ответа."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from service.app import app  # noqa: E402

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_openapi_schema_is_generated():
    resp = client.get("/openapi.json")
    assert resp.status_code == 200
    paths = resp.json()["paths"]
    assert "/preprocessing/fill-gaps" in paths
    assert "/prediction/predict" in paths
    assert "/optimization/optimize" in paths
    assert "/integration/export-isoxml" in paths


def test_fill_gaps_endpoint_restores_a_missing_cell():
    body = {
        "coords": [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        "times": [0.0, 1.0, 2.0, 3.0, 4.0],
        "values": [
            [1.0, 1.1, 1.2, 0.0, 1.4],
            [1.05, 1.15, 1.25, 1.35, 1.45],
            [0.95, 1.05, 1.15, 1.25, 1.35],
            [1.0, 1.1, 1.2, 1.3, 1.4],
        ],
        "mask_observed": [
            [True, True, True, False, True],
            [True, True, True, True, True],
            [True, True, True, True, True],
            [True, True, True, True, True],
        ],
    }
    resp = client.post("/preprocessing/fill-gaps", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["filled"][0][3] is not None  # пропуск восстановлен
    assert 1.0 < data["filled"][0][3] < 1.6  # правдоподобное значение


def test_fill_gaps_endpoint_rejects_invalid_input():
    body = {
        "coords": [[0.0, 0.0]],
        "times": [0.0, 1.0],
        "values": [[1.0, 1.0], [1.0, 1.0]],  # форма не согласована с coords (M=1)
        "mask_observed": [[True, True], [True, True]],
    }
    resp = client.post("/preprocessing/fill-gaps", json=body)
    assert resp.status_code == 422


def test_predict_endpoint_returns_yield_and_modality_importance():
    body = {
        "soil_moisture_mean": 30.0,
        "soil_nitrogen_mean": 15.0,
        "precip_sum": 400.0,
        "temp_sum": 2200.0,
        "ndvi_peak": 0.6,
        "ndvi_integral": 9.0,
        "disease_events_count": 1.0,
        "max_risk_stage": 1.0,
        "fertilizer_applications_count": 3.0,
        "total_dose": 100.0,
    }
    resp = client.post("/prediction/predict", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["predicted_yield"], float)
    assert set(data["modality_importance"].keys()) == {"num", "geo", "img", "text"}


def test_training_summary_endpoint():
    resp = client.get("/prediction/training-summary")
    assert resp.status_code == 200
    data = resp.json()
    assert data["rmse_all_modalities"] > 0
    assert set(data["rmse_without_modality"].keys()) == {"num", "geo", "img", "text"}


def test_optimize_endpoint_without_budget():
    body = {
        "plots": [
            {"plot_id": 0, "baseline": 40.0, "R": 8.0, "s": 60.0, "area": 1.0},
            {"plot_id": 1, "baseline": 42.0, "R": 3.0, "s": 60.0, "area": 1.0},
        ],
        "price_yield": 1300.0,
        "price_fert": 50.0,
    }
    resp = client.post("/optimization/optimize", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["doses"]) == 2
    assert data["doses"][0] > data["doses"][1]  # участок с большим R получает больше удобрения


def test_optimize_endpoint_with_budget_beats_uniform():
    body = {
        "plots": [{"plot_id": i, "baseline": 40.0, "R": 3.0 + i, "s": 60.0, "area": 1.0} for i in range(10)],
        "price_yield": 1300.0,
        "price_fert": 50.0,
        "budget": 600.0,
    }
    resp = client.post("/optimization/optimize", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_profit"] >= data["total_profit_uniform"] - 1e-6
    assert data["profit_gain_percent"] >= -1e-6


def test_optimize_endpoint_rejects_invalid_plots():
    body = {
        "plots": [{"plot_id": 0, "baseline": 40.0, "R": 8.0, "s": 0.0, "area": 1.0}],  # s=0 недопустимо
        "price_yield": 1300.0,
        "price_fert": 50.0,
    }
    resp = client.post("/optimization/optimize", json=body)
    assert resp.status_code == 422


def test_export_isoxml_endpoint_returns_valid_xml():
    import xml.etree.ElementTree as ET

    body = {
        "plots": [
            {"plot_id": 0, "baseline": 40.0, "R": 8.0, "s": 60.0, "area": 1.0},
            {"plot_id": 1, "baseline": 42.0, "R": 3.0, "s": 60.0, "area": 1.5},
        ],
        "price_yield": 1300.0,
        "price_fert": 50.0,
        "customer_name": "Тестовое хозяйство",
    }
    resp = client.post("/integration/export-isoxml", json=body)
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/xml")

    root = ET.fromstring(resp.content)
    assert root.tag == "ISO11783_TaskData"
    assert root.find("CTR").get("B") == "Тестовое хозяйство"
    assert len(root.findall("PFD")) == 2


def test_export_isoxml_endpoint_rejects_invalid_plots():
    body = {
        "plots": [{"plot_id": 0, "baseline": 40.0, "R": 8.0, "s": 0.0, "area": 1.0}],
        "price_yield": 1300.0,
        "price_fert": 50.0,
    }
    resp = client.post("/integration/export-isoxml", json=body)
    assert resp.status_code == 422
