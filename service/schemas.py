"""Pydantic-схемы запросов/ответов сервиса -- тонкая обёртка над src/ds_*."""

from pydantic import BaseModel, Field


# ---- L3: предобработка (docs/chapter2/preprocessing_model.md, §2.2) ----

class FillGapsRequest(BaseModel):
    coords: list[list[float]] = Field(..., description="Координаты точек s_1..s_M, форма (M, 2)")
    times: list[float] = Field(..., description="Числовая временная шкала, форма (T,)")
    values: list[list[float]] = Field(..., description="Матрица значений (M, T); произвольны там, где mask_observed=false")
    mask_observed: list[list[bool]] = Field(..., description="Маска наблюдаемости (M, T)")
    drop_anomalies: bool = Field(True, description="Исключать значения, признанные аномалиями (§2.2.2), из наблюдаемых")


class FillGapsResponse(BaseModel):
    filled: list[list[float | None]] = Field(..., description="Исходные + восстановленные значения; null там, где не хватило данных")
    unrestored: list[list[bool]] = Field(..., description="Ячейки, для которых не нашлось ни пространственной, ни временной оценки")
    anomalies: list[list[bool]] = Field(..., description="Обнаруженные аномалии среди исходно наблюдаемых значений")


# ---- L4: прогноз (docs/chapter2/prediction_model.md, §2.3) ----

class PredictionFeatures(BaseModel):
    """Один вектор x_{m,y} -- признаки по всем четырём модальностям (§2.3.2)."""

    soil_moisture_mean: float = Field(..., description="num: среднесезонная влажность почвы, %")
    soil_nitrogen_mean: float = Field(..., description="num: среднесезонное содержание азота")
    precip_sum: float = Field(..., description="num: сумма осадков за сезон, мм")
    temp_sum: float = Field(..., description="num: сумма эффективных температур")
    ndvi_peak: float = Field(..., description="geo: пиковый NDVI за сезон")
    ndvi_integral: float = Field(..., description="geo: интеграл NDVI за сезон")
    disease_events_count: float = Field(..., description="img: число зафиксированных по фото случаев поражения")
    max_risk_stage: float = Field(..., description="img: максимальная стадия_riska за сезон")
    fertilizer_applications_count: float = Field(..., description="text: число агроприёмов внесения удобрений за сезон")
    total_dose: float = Field(..., description="text: суммарная доза внесённых удобрений, кг")


class PredictionResponse(BaseModel):
    predicted_yield: float = Field(..., description="Ŷ_{m,y}, ц/га")
    modality_importance: dict[str, float] = Field(..., description="Phi_k -- агрегированный |SHAP| по модальности (§2.3.4)")


class TrainingSummary(BaseModel):
    rmse_all_modalities: float = Field(..., description="RMSE по GroupKFold на всех модальностях (§2.3.5)")
    rmse_without_modality: dict[str, float] = Field(..., description="RMSE при исключении каждой модальности по очереди (§2.3.6)")
    modality_importance: dict[str, float]


# ---- L5: оптимизация внесения (docs/chapter2/optimization_model.md, §2.4) ----

class Plot(BaseModel):
    plot_id: int
    baseline: float = Field(..., description="Базовый потенциал урожайности без доп. внесения, ц/га")
    R: float = Field(..., ge=0, description="Предельная прибавка урожая от удобрения, ц/га (§2.4.2)")
    s: float = Field(..., gt=0, description="Масштаб насыщения кривой отклика, кг/га")
    area: float = Field(..., gt=0, description="Площадь участка, га")


class OptimizeRequest(BaseModel):
    plots: list[Plot]
    price_yield: float = Field(..., gt=0, description="Цена продукции, руб./ц")
    price_fert: float = Field(..., gt=0, description="Цена удобрения, руб./кг")
    dose_min: float = Field(0.0, ge=0)
    dose_max: float = Field(150.0, description="Верхняя агрономическая граница дозы, кг/га")
    budget: float | None = Field(
        None, description="Суммарный бюджет удобрений, кг. Если не задан -- оптимум без ограничения (§2.4.3), иначе -- метод Лагранжа (§2.4.4)"
    )


class OptimizeResponse(BaseModel):
    doses: list[float] = Field(..., description="Оптимальные дозы по участкам, кг/га")
    total_profit: float = Field(..., description="Суммарная прибыль дифференцированного сценария")
    total_profit_uniform: float = Field(..., description="Прибыль при равномерном внесении того же суммарного количества (§2.4.5)")
    profit_gain_percent: float = Field(..., description="Прирост прибыли дифференцированного сценария относительно равномерного, %")


# ---- L7: интеграция с внешними ИС (docs/chapter2/integration_model.md, §2.6) ----

class IsoxmlExportRequest(OptimizeRequest):
    customer_name: str = Field("Демо-хозяйство", description="Значение CTR/B в TASKDATA.XML")
    farm_name: str = Field("Демо-поле", description="Значение FRM/B в TASKDATA.XML")
    task_designator: str = Field("Карта-задание", description="Значение TSK/B в TASKDATA.XML")
