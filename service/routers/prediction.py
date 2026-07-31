"""L4 -- мультимодальный прогноз + SHAP, обёртка над ds_prediction.

Модель обучается один раз лениво (при первом запросе) на демонстрационных
контролируемых данных (ds_prediction.synthetic.generate_dataset) и кешируется
в памяти процесса -- для прототипа этого достаточно; постоянное хранилище
моделей/переобучение на реальных данных хозяйства -- отдельная задача главы 4.
"""

import pandas as pd
from fastapi import APIRouter

from ds_prediction.explain import modality_importance
from ds_prediction.features import select_modalities
from ds_prediction.model import DEFAULT_MODALITIES, evaluate_grouped_cv, train_model
from ds_prediction.synthetic import generate_dataset

from ..schemas import PredictionFeatures, PredictionResponse, TrainingSummary

router = APIRouter(prefix="/prediction", tags=["L4 -- мультимодальный прогноз"])

_cache: dict = {}


def _get_model():
    if "model" not in _cache:
        df = generate_dataset()
        _cache["model"] = train_model(df, modalities=DEFAULT_MODALITIES)
        _cache["columns"] = select_modalities(df, DEFAULT_MODALITIES).columns.tolist()
    return _cache["model"], _cache["columns"]


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
