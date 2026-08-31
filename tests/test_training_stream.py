"""Тесты SSE-потока статистики обучения (GET /training/history/stream) --
"в прямом эфире" означает: клиент получает новый снимок сразу после того,
как training/finetune_*.py записал очередную эпоху на диск, а не только
после завершения всего обучения. Тестируется на уровне async-генератора
напрямую (без TestClient) -- TestClient.stream() не даёт надёжно оборвать
бесконечный SSE-поток в тесте."""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import service.routers.training_stats as ts  # noqa: E402


def _run(coro, timeout=5):
    return asyncio.run(asyncio.wait_for(coro, timeout=timeout))


def test_stream_emits_initial_snapshot_immediately():
    async def scenario():
        gen = ts._history_event_stream(poll_interval=0.02)
        try:
            first = await asyncio.wait_for(gen.__anext__(), timeout=1)
            assert first.startswith("data: ")
            payload = json.loads(first[len("data: ") :])
            assert set(payload.keys()) == {"ner", "leaf_classifier"}
        finally:
            await gen.aclose()

    _run(scenario())


def test_stream_stays_silent_when_nothing_changes():
    async def scenario():
        gen = ts._history_event_stream(poll_interval=0.02)
        try:
            await asyncio.wait_for(gen.__anext__(), timeout=1)
            try:
                await asyncio.wait_for(gen.__anext__(), timeout=0.3)
                raise AssertionError("получено новое событие без изменений на диске")
            except TimeoutError:
                pass  # ожидаемо -- ничего не изменилось, событие не отправлено
        finally:
            await gen.aclose()

    _run(scenario())


def test_stream_emits_new_event_on_file_change(tmp_path, monkeypatch):
    run_path = tmp_path / "history.json"
    run_path.write_text(json.dumps({"status": "running", "epoch_log": []}), encoding="utf-8")
    monkeypatch.setattr(ts, "RUNS", {"demo": run_path})

    async def scenario():
        gen = ts._history_event_stream(poll_interval=0.02)
        try:
            await asyncio.wait_for(gen.__anext__(), timeout=1)

            run_path.write_text(json.dumps({"status": "completed", "epoch_log": [1]}), encoding="utf-8")

            second = await asyncio.wait_for(gen.__anext__(), timeout=1)
            payload = json.loads(second[len("data: ") :])
            assert payload["demo"]["status"] == "completed"
        finally:
            await gen.aclose()

    _run(scenario())
