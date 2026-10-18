"""Workbench upload smoke tests: accepted metadata must not imply validated evidence."""
import io

from fastapi.testclient import TestClient

from service.app import app


def test_workbench_upload_csv_is_registered_not_pipeline_ready(tmp_path, monkeypatch):
    from service.routers import workbench
    from service import workbench_storage

    monkeypatch.setattr(workbench, "ROOT", tmp_path)
    monkeypatch.setattr(workbench_storage, "ROOT", tmp_path)
    client = TestClient(app)
    response = client.post(
        "/workbench/files",
        files={"file": ("measurements.csv", io.BytesIO(b"field_id,ndvi\nA,0.72\n"), "text/csv")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["category"] == "table"
    assert body["preview"]["columns"] == ["field_id", "ndvi"]
    assert body["pipeline_ready"] is False
    assert body["status"] == "registered_untrusted"
    assert len(list(tmp_path.iterdir())) == 2
    listed = client.get("/workbench/files").json()["files"]
    assert len(listed) == 1
    assert listed[0]["file_id"] == body["file_id"]


def test_workbench_rejects_unsupported_extension(tmp_path, monkeypatch):
    from service.routers import workbench
    from service import workbench_storage

    monkeypatch.setattr(workbench, "ROOT", tmp_path)
    monkeypatch.setattr(workbench_storage, "ROOT", tmp_path)
    response = TestClient(app).post(
        "/workbench/files",
        files={"file": ("script.exe", b"not executable", "application/octet-stream")},
    )
    assert response.status_code == 400
    assert list(tmp_path.iterdir()) == []


def test_workbench_rejects_empty_file(tmp_path, monkeypatch):
    from service.routers import workbench
    from service import workbench_storage

    monkeypatch.setattr(workbench, "ROOT", tmp_path)
    monkeypatch.setattr(workbench_storage, "ROOT", tmp_path)
    response = TestClient(app).post(
        "/workbench/files",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )
    assert response.status_code == 400
