"""Опциональная (в dev) / обязательная (в production) защита API-ключом --
см. `service/config.py` (DS_ENV, DS_API_KEY) и README, раздел «Развёртывание».

В dev (DS_ENV не задан или не "production"): если DS_API_KEY не задан в
окружении, проверка отключена -- ровно то же поведение, что было раньше
(локальная разработка, pytest, docker compose run --rm test/validate не
требуют никакой настройки).

В production (DS_ENV=production): отсутствие DS_API_KEY -- фатальная ошибка
конфигурации, проверяется при старте приложения (service/app.py, lifespan),
а не только здесь -- так сервис вообще не поднимется в небезопасной
конфигурации, вместо того чтобы молча обслуживать открытый API.
"""

import secrets

from fastapi import Header, HTTPException, Query

from .config import get_api_key


def require_api_key(
    x_api_key: str | None = Header(default=None),
    api_key: str | None = Query(
        default=None, description="Резерв для EventSource -- браузер не умеет слать кастомные заголовки в SSE-подключениях"
    ),
) -> None:
    expected = get_api_key()
    if expected is None:
        return  # ключ не настроен -- сервис открыт (dev/локальный прототип); в production это отловлено на старте
    provided = x_api_key or api_key
    if provided is None or not secrets.compare_digest(provided, expected):
        raise HTTPException(status_code=401, detail="Неверный или отсутствующий X-API-Key")
