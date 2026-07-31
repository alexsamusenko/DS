"""Единый FastAPI-сервис поверх L3-L5 (компонент L6 «API», артефакт «Архитектура DS»).

Каждый маршрут -- тонкая обёртка над уже формализованными и протестированными
функциями `src/ds_*`: сервис не содержит собственной бизнес-логики, только
валидацию HTTP-входа (Pydantic, service/schemas.py) и вызов существующего кода.

Запуск:
    uvicorn service.app:app --reload --port 8000
Документация (автогенерируется из схем):
    http://localhost:8000/docs
"""

from fastapi import FastAPI

from .routers import optimization, prediction, preprocessing

app = FastAPI(
    title="DS API",
    description=(
        "Единая точка доступа к предобработке (L3), мультимодальному прогнозу (L4) "
        "и оптимизации внесения удобрений (L5) -- практическая часть диссертации."
    ),
    version="0.1.0",
)

app.include_router(preprocessing.router)
app.include_router(prediction.router)
app.include_router(optimization.router)


@app.get("/health", tags=["служебное"], summary="Проверка работоспособности сервиса")
def health() -> dict:
    return {"status": "ok"}
