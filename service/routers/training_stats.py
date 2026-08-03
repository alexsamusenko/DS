"""Статистика обучения для фронта -- читает history.json, который training/finetune_*.py
пишет рядом с чекпоинтом (build/<run>/history.json, см. §2.5.6 model_training.md).

Сами тренировки запускаются вне сервиса (training/finetune_*.py, обычно в
контейнере train -- docker-compose.yml); этот роутер только читает уже
сохранённый результат, ничего не запускает и не блокирует запрос.
"""

import json
from pathlib import Path

from fastapi import APIRouter

router = APIRouter(prefix="/training", tags=["Статистика обучения"])

RUNS = {
    "ner": Path("build/ner_model/history.json"),
    "leaf_classifier": Path("build/leaf_classifier/history.json"),
}


def _load_run(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@router.get("/history", summary="История эпох по всем прогонам обучения, для которых есть history.json")
def training_history() -> dict:
    return {name: _load_run(path) for name, path in RUNS.items()}
