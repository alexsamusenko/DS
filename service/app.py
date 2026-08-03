"""Единый FastAPI-сервис поверх L3-L5 (компонент L6 «API», артефакт «Архитектура DS»).

Каждый маршрут -- тонкая обёртка над уже формализованными и протестированными
функциями `src/ds_*`: сервис не содержит собственной бизнес-логики, только
валидацию HTTP-входа (Pydantic, service/schemas.py) и вызов существующего кода.

Запуск:
    uvicorn service.app:app --reload --port 8000
Документация (автогенерируется из схем):
    http://localhost:8000/docs
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .routers import datasets, optimization, prediction, preprocessing, training_stats

app = FastAPI(
    title="DS API",
    description=(
        "Единая точка доступа к предобработке (L3), мультимодальному прогнозу (L4) "
        "и оптимизации внесения удобрений (L5) -- практическая часть диссертации."
    ),
    version="0.1.0",
)

# Разрешает фронту (npm run dev, порт 5173 по умолчанию) обращаться к API,
# который в разработке слушает отдельный порт (8000). В production фронт
# собирается статикой и отдаётся этим же приложением (см. ниже) -- там CORS
# не участвует, т.к. запросы идут с того же origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(preprocessing.router)
app.include_router(prediction.router)
app.include_router(optimization.router)
app.include_router(training_stats.router)
app.include_router(datasets.router)


@app.get("/health", tags=["служебное"], summary="Проверка работоспособности сервиса")
def health() -> dict:
    return {"status": "ok"}


# Собранный фронт (frontend/dist, см. frontend/README.md) -- монтируется, только
# если он реально собран (`npm run build`); без этого сервис остаётся чистым API
# (как раньше), что не ломает тесты и локальный API-only запуск.
_frontend_dist = Path("frontend/dist")
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="frontend")
