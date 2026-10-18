"""End-to-end HTTP contract checks for every advertised upload extension.

These checks prove upload, persistence and download, NOT full content extraction.
"""
import pytest
from fastapi.testclient import TestClient

from service import workbench_storage
from service.app import app
from service.routers import workbench


@pytest.mark.parametrize("extension", sorted(workbench.EXTENSIONS))
def test_every_advertised_extension_roundtrip(extension, tmp_path, monkeypatch):
    monkeypatch.setattr(workbench, "ROOT", tmp_path)
    monkeypatch.setattr(workbench_storage, "ROOT", tmp_path)
    client = TestClient(app)
    payload = b"test agriculture sample"
    response = client.post("/workbench/files", files={"file": ("farm" + extension, payload, "application/octet-stream")})
    assert response.status_code == 200, (extension, response.text)
    item = response.json()
    assert item["category"] == workbench.EXT_TYPES[extension]
    assert item["status"] == "registered_untrusted"
    assert item["pipeline_ready"] is False
    assert item["file_id"]
    assert client.get("/workbench/files/" + item["file_id"]).status_code == 200
    downloaded = client.get("/workbench/files/" + item["file_id"] + "/download")
    assert downloaded.status_code == 200
    assert downloaded.content == payload
    listed = client.get("/workbench/files").json()["files"]
    assert len(listed) == 1
    assert listed[0]["file_id"] == item["file_id"]


def test_extension_map_is_complete():
    assert set(workbench.EXT_TYPES) == workbench.EXTENSIONS


def test_oversize_file_does_not_persist(tmp_path, monkeypatch):
    monkeypatch.setattr(workbench, "ROOT", tmp_path)
    monkeypatch.setattr(workbench_storage, "ROOT", tmp_path)
    response = TestClient(app).post("/workbench/files", files={"file": ("huge.csv", b"x" * (workbench.MAX_FILE_BYTES + 1), "text/csv")})
    assert response.status_code == 413
    assert not tmp_path.exists()


def test_path_traversal_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(workbench, "ROOT", tmp_path)
    monkeypatch.setattr(workbench_storage, "ROOT", tmp_path)
    response = TestClient(app).post("/workbench/files", files={"file": ("../evil.csv", b"a,b", "text/csv")})
    assert response.status_code == 400
