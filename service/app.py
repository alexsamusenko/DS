"""Единый FastAPI-сервис поверх L3-L5 (компонент L6 «API», артефакт «Архитектура DS»).

Каждый маршрут -- тонкая обёртка над уже формализованными и протестированными
функциями `src/ds_*`: сервис не содержит собственной бизнес-логики, только
валидацию HTTP-входа (Pydantic, service/schemas.py) и вызов существующего кода.

Запуск:
    uvicorn service.app:app --reload --port 8000
Документация (автогенерируется из схем):
    http://localhost:8000/docs

Конфигурация -- переменные окружения, см. .env.example и service/config.py.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import config
from .auth import require_api_key
from .config import CORS_ORIGINS, PREDICTION_MODEL_PATH
from .logging_config import configure_logging, log_requests, logger
from .routers import datasets, integration, optimization, prediction, preprocessing, training_stats


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    configure_logging()

    # В production сервис не должен молча подниматься в конфигурации, которая
    # либо небезопасна (API открыт), либо вводит в заблуждение (прогнозы
    # обслуживаются синтетической demo-моделью вместо обученной на реальных
    # данных хозяйства) -- см. аудит перед развёртыванием. В dev (по
    # умолчанию) обе ситуации допустимы и ожидаемы.
    if config.is_production():
        problems = []
        if config.get_api_key() is None:
            problems.append("DS_API_KEY не задан -- API будет полностью открыт")
        if not prediction.model_available():
            problems.append(
                f"обученная модель L4 не найдена по пути {PREDICTION_MODEL_PATH} -- "
                "смонтируйте том build/ с результатом обучения на реальных данных "
                "(см. README, «Развёртывание»), иначе сервис будет обслуживать "
                "прогнозы синтетической demo-моделью"
            )
        if problems:
            message = "Отказ от старта в DS_ENV=production:\n  - " + "\n  - ".join(problems)
            logger.error(message)
            raise RuntimeError(message)
        logger.info("Стартовые проверки production-конфигурации пройдены")
    else:
        logger.info("DS_ENV=%s (не production) -- стартовые проверки production-конфигурации пропущены", "dev")

    yield


app = FastAPI(
    title="DS API",
    description=(
        "Единая точка доступа к предобработке (L3), мультимодальному прогнозу (L4), "
        "оптимизации внесения удобрений (L5) и интеграции с внешними ИС (L7, ISO 11783-10) "
        "-- практическая часть диссертации."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.middleware("http")(log_requests)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Defence-in-depth: отдельные роутеры (prediction.py и т.п.) уже
    # переводят ожидаемые ошибки в осмысленные HTTP-коды; этот обработчик --
    # последний рубеж для всего, что не было предусмотрено явно, чтобы клиент
    # получал единообразный JSON, а не студенистый traceback FastAPI по
    # умолчанию (который к тому же может утечь детали реализации).
    logger.exception("Необработанное исключение на %s %s", request.method, request.url.path)
    return JSONResponse(status_code=500, content={"detail": "Внутренняя ошибка сервиса"})


# Разрешает фронту (npm run dev, порт 5173 по умолчанию) обращаться к API,
# который в разработке слушает отдельный порт (8000). В production фронт
# собирается статикой и отдаётся этим же приложением (см. ниже) -- там CORS
# не участвует, т.к. запросы идут с того же origin. Настраивается через
# DS_CORS_ORIGINS (service/config.py) для нестандартных деплоев.
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# /health и /ready намеренно без require_api_key -- нужны незащищёнными для
# проб (Docker healthcheck, балансировщик) независимо от того, настроен ли ключ.
_protected = [Depends(require_api_key)]
app.include_router(preprocessing.router, dependencies=_protected)
app.include_router(prediction.router, dependencies=_protected)
app.include_router(optimization.router, dependencies=_protected)
app.include_router(training_stats.router, dependencies=_protected)
app.include_router(datasets.router, dependencies=_protected)
app.include_router(integration.router, dependencies=_protected)


@app.get("/health", tags=["служебное"], summary="Liveness: процесс жив и отвечает на запросы")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready", tags=["служебное"], summary="Readiness: сервис готов обслуживать реальный трафик")
def ready() -> JSONResponse:
    # В отличие от /health (просто "процесс жив"), эта проверка отвечает на
    # вопрос "можно ли направлять сюда реальный трафик" -- сейчас единственный
    # содержательный критерий: доступна ли модель L4 (на диске или уже в
    # памяти процесса). По мере появления других состояний готовности
    # (граф знаний, кэш преобразований) сюда стоит добавлять новые проверки.
    checks = {"prediction_model": prediction.model_available() or prediction.model_loaded_in_memory()}
    ok = all(checks.values())
    return JSONResponse(status_code=200 if ok else 503, content={"status": "ok" if ok else "not_ready", "checks": checks})


# Собранный фронт (frontend/dist, см. frontend/README.md) -- монтируется, только
# если он реально собран (`npm run build`); без этого сервис остаётся чистым API
# (как раньше), что не ломает тесты и локальный API-only запуск.
_frontend_dist = Path("frontend/dist")
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="frontend")
