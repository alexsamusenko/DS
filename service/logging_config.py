"""Структурное логирование сервиса -- до этого модуля в проекте не было ни
одного вызова `logging`: единственным операционным сигналом был access-log
uvicorn по умолчанию. Этого недостаточно, чтобы понять по логам, например,
что прогноз L4 обслуживается синтетической demo-моделью, а не обученной на
диске (см. service/routers/prediction.py) -- события такого рода теперь
логируются явно на уровне WARNING/ERROR.
"""

import logging
import time
from collections.abc import Awaitable, Callable

from fastapi import Request, Response

from .config import LOG_LEVEL

_CONFIGURED = False


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    _CONFIGURED = True


logger = logging.getLogger("ds")


async def log_requests(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """Минимальный request-log middleware: метод, путь, статус, время ответа.

    Не заменяет полноценные метрики/трейсинг (см. аудит перед развёртыванием --
    Prometheus/OpenTelemetry явно оставлены как отдельная задача), но даёт
    оператору хоть какую-то видимость поверх голого access-log uvicorn --
    в частности, единый формат с остальными логами приложения.
    """
    start = time.monotonic()
    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.monotonic() - start) * 1000
        logger.exception("%s %s -- необработанное исключение (%.1f мс)", request.method, request.url.path, duration_ms)
        raise
    duration_ms = (time.monotonic() - start) * 1000
    level = logging.WARNING if response.status_code >= 500 else logging.INFO
    logger.log(level, "%s %s -> %d (%.1f мс)", request.method, request.url.path, response.status_code, duration_ms)
    return response
