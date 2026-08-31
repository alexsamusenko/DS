"""Тесты новых JSON-эндпоинтов для фронта (service/routers/training_stats.py,
service/routers/datasets.py) -- статистика обучения и каталог датасетов."""

import io
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient  # noqa: E402

from service import datasets_catalog  # noqa: E402
from service.app import app  # noqa: E402

client = TestClient(app)


def test_training_history_has_both_runs_keys():
    resp = client.get("/training/history")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"ner", "leaf_classifier"}
    # Каждый прогон -- либо ещё не обучен (null), либо словарь с логом эпох
    for run in body.values():
        assert run is None or "task" in run


def test_datasets_lists_plantdoc_card_metadata():
    # Карточка (docs/dataset_cards/plantdoc.md) читается независимо от того,
    # загружен ли реальный датасет на диск в этом окружении -- в отличие от
    # предыдущей версии теста, ничего не предполагает про содержимое data/
    # (в чистом чекауте/CI data/plantdoc не существует, см. .gitignore).
    resp = client.get("/datasets")
    assert resp.status_code == 200
    entries = {d["slug"]: d for d in resp.json()}
    assert "plantdoc" in entries
    assert "CC BY 4.0" in entries["plantdoc"]["license_summary"]


def test_datasets_reports_on_disk_presence_when_data_dir_has_it(monkeypatch, tmp_path):
    # on_disk/file_count -- поведение, привязанное к DATA_DIR, а не к
    # конкретно plantdoc; проверяем на изолированном временном каталоге,
    # чтобы тест не зависел от того, что реально загружено в data/.
    fake_data_dir = tmp_path / "data"
    (fake_data_dir / "plantdoc" / "classA").mkdir(parents=True)
    (fake_data_dir / "plantdoc" / "classA" / "img1.txt").write_text("x")
    monkeypatch.setattr(datasets_catalog, "DATA_DIR", fake_data_dir)

    resp = client.get("/datasets")
    assert resp.status_code == 200
    entries = {d["slug"]: d for d in resp.json()}
    assert entries["plantdoc"]["on_disk"] is True
    assert entries["plantdoc"]["file_count"] == 1


def _make_zip_bytes(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    return buf.getvalue()


def test_upload_dataset_extracts_zip_into_data_dir():
    target = Path("data/_pytest_upload_test")
    shutil.rmtree(target, ignore_errors=True)
    try:
        zip_bytes = _make_zip_bytes({"classA/img1.txt": "fake image bytes", "classB/img2.txt": "fake image bytes"})
        resp = client.post(
            "/datasets/upload",
            data={"name": "_pytest_upload_test"},
            files={"archive": ("data.zip", zip_bytes, "application/zip")},
        )
        assert resp.status_code == 200
        assert resp.json()["file_count"] == 2
        assert (target / "classA" / "img1.txt").exists()
    finally:
        shutil.rmtree(target, ignore_errors=True)


def test_upload_dataset_rejects_bad_name():
    zip_bytes = _make_zip_bytes({"a.txt": "x"})
    resp = client.post(
        "/datasets/upload",
        data={"name": "../evil"},
        files={"archive": ("data.zip", zip_bytes, "application/zip")},
    )
    assert resp.status_code == 422


def test_upload_dataset_rejects_path_traversal_inside_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../../evil.txt", "escape attempt")
    target = Path("data/_pytest_traversal_test")
    shutil.rmtree(target, ignore_errors=True)
    try:
        resp = client.post(
            "/datasets/upload",
            data={"name": "_pytest_traversal_test"},
            files={"archive": ("data.zip", buf.getvalue(), "application/zip")},
        )
        assert resp.status_code == 400
    finally:
        # Роутер проверяет все пути ДО extractall(), так что при 400 ничего
        # не распаковывается -- чистить, кроме самой (не созданной) target, нечего.
        shutil.rmtree(target, ignore_errors=True)
