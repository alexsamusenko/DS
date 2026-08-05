"""L4 -- мультимодальный прогноз + SHAP, обёртка над ds_prediction.

Модель обучается лениво (при первом запросе) на демонстрационных
контролируемых данных (ds_prediction.synthetic.generate_dataset, фиксированный
seed -- детерминированный датасет) и кешируется в памяти процесса, а также на
диск (build/models/), чтобы не переобучаться заново при каждом рестарте
процесса -- обучение GradientBoostingRegressor занимает секунды, но при
частых рестартах (reload в dev, много воркеров) это накопительно заметно.
Постоянное хранилище моделей/переобучение на реальных данных хозяйства --
отдельная задача главы 4.
"""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import APIRouter

from ds_prediction.explain import modality_importance
from ds_prediction.features import select_modalities
from ds_prediction.model import DEFAULT_MODALITIES, evaluate_grouped_cv, train_model
from ds_prediction.synthetic import generate_dataset

from ..schemas import PredictionFeatures, PredictionResponse, TrainingSummary

router = APIRouter(prefix="/prediction", tags=["L4 -- мультимодальный прогноз"])

_cache: dict = {}
_MODEL_PATH = Path("build/models/l4_gradient_boosting.joblib")
# Бампится при изменении состава модальностей/обучающих данных -- иначе
# устаревший кеш на диске молча продолжал бы обслуживать старую модель.
_CACHE_VERSION = 1


def _get_model():
    if "model" not in _cache:
        cached = _load_from_disk()
        if cached is not None:
            _cache["model"], _cache["columns"] = cached
        else:
            df = generate_dataset()
            model = train_model(df, modalities=DEFAULT_MODALITIES)
            columns = select_modalities(df, DEFAULT_MODALITIES).columns.tolist()
            _save_to_disk(model, columns)
            _cache["model"], _cache["columns"] = model, columns
    return _cache["model"], _cache["columns"]


def _load_from_disk():
    if not _MODEL_PATH.exists():
        return None
    payload = joblib.load(_MODEL_PATH)
    if payload.get("cache_version") != _CACHE_VERSION or payload.get("modalities") != DEFAULT_MODALITIES:
        return None  # конфигурация обучения изменилась -- переобучить, а не отдать устаревшую модель
    return payload["model"], payload["columns"]


def _save_to_disk(model, columns):
    _MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"cache_version": _CACHE_VERSION, "modalities": DEFAULT_MODALITIES, "model": model, "columns": columns}, _MODEL_PATH)


@router.post("/predict", response_model=PredictionResponse, summary="Прогноз урожайности + вклад модальностей по SHAP (§2.3.3-2.3.4)")
def predict(features: PredictionFeatures) -> PredictionResponse:
    model, columns = _get_model()
    row = pd.DataFrame([features.model_dump()])[columns]

    prediction = float(model.predict(row)[0])
    importance = modality_importance(model, row)

    return PredictionResponse(predicted_yield=prediction, modality_importance=importance)


@router.get("/training-summary", response_model=TrainingSummary, summary="RMSE по модальностям на демо-данных, GroupKFold по полю (§2.3.5-2.3.6)")
def training_summary() -> TrainingSummary:
    df = generate_dataset()

    rmse_full = evaluate_grouped_cv(df, modalities=DEFAULT_MODALITIES)
    rmse_without = {
        m: evaluate_grouped_cv(df, modalities=tuple(x for x in DEFAULT_MODALITIES if x != m))
        for m in DEFAULT_MODALITIES
    }

    model, columns = _get_model()
    importance = modality_importance(model, df[columns])

    return TrainingSummary(rmse_all_modalities=rmse_full, rmse_without_modality=rmse_without, modality_importance=importance)
