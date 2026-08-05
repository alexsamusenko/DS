"""Опциональная защита API-ключом для внешних вызовов (L7, §2.6 -- когда сервис
открыт внешней ИС агрохолдинга, а не только собственному frontend/ в локальном
запуске). Не ломает поведение по умолчанию: если DS_API_KEY не задан в
окружении, проверка отключена -- ровно то же поведение, что было раньше
(локальная разработка, pytest, docker compose run --rm test/validate не
требуют никакой настройки).
"""

import os
import secrets

from fastapi import Header, HTTPException, Query


def require_api_key(
    x_api_key: str | None = Header(default=None),
    api_key: str | None = Query(default=None, description="Резерв для EventSource -- браузер не умеет слать кастомные заголовки в SSE-подключениях"),
) -> None:
    expected = os.environ.get("DS_API_KEY")
    if expected is None:
        return  # ключ не настроен -- сервис открыт, как и раньше (локальный прототип)
    provided = x_api_key or api_key
    if provided is None or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Неверный или отсутствующий X-API-Key")
