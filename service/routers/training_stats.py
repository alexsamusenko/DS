"""Статистика обучения для фронта -- читает history.json, который training/finetune_*.py
пишет рядом с чекпоинтом (build/<run>/history.json, см. §2.5.6 model_training.md).
Оба скрипта дообучения пишут history.json после КАЖДОЙ эпохи (не только в
конце), со статусом "running"/"completed" -- это то, что делает возможным
GET /training/history/stream: SSE-поток, который опрашивает файлы и отдаёт
клиенту новый снимок, как только он реально изменился на диске.

Сами тренировки запускаются вне сервиса (training/finetune_*.py, обычно в
контейнере train -- docker-compose.yml); этот роутер только читает уже
сохранённый результат, ничего не запускает и не блокирует запрос.
"""

import asyncio
import json
import logging
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

logger = logging.getLogger("ds.training_stats")

router = APIRouter(prefix="/training", tags=["Статистика обучения"])

RUNS = {
    "ner": Path("build/ner_model/history.json"),
    "leaf_classifier": Path("build/leaf_classifier/history.json"),
}

POLL_INTERVAL_SECONDS = 1.0


def _load_run(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # training/finetune_*.py дописывает history.json после каждой эпохи --
        # опрос (в т.ч. SSE-стрим ниже) может застать файл в момент записи
        # (усечённый JSON). Это не ошибка клиента и не повод падать 500-кой:
        # следующий опрос через POLL_INTERVAL_SECONDS увидит уже дописанный файл.
        logger.debug("history.json временно нечитаем (вероятно, идёт запись): %s", path, exc_info=True)
        return None


def _snapshot() -> dict:
    return {name: _load_run(path) for name, path in RUNS.items()}


@router.get("/history", summary="История эпох по всем прогонам обучения, для которых есть history.json")
def training_history() -> dict:
    return _snapshot()


async def _history_event_stream(poll_interval: float = POLL_INTERVAL_SECONDS):
    last = None
    while True:
        current = _snapshot()
        if current != last:
            yield f"data: {json.dumps(current, ensure_ascii=False)}\n\n"
            last = current
        await asyncio.sleep(poll_interval)


@router.get(
    "/history/stream",
    summary="То же, что /training/history, но SSE-поток -- обновляется в прямом эфире по мере обучения (опрос файла раз в секунду)",
)
def training_history_stream() -> StreamingResponse:
    return StreamingResponse(_history_event_stream(), media_type="text/event-stream")
