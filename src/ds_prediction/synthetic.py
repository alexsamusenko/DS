"""Контролируемый многолетний многополевой набор с известной функцией
формирования урожайности -- нужен эталон для проверки ошибки (см. обоснование
в docs/chapter2/prediction_model.md §2.3.7, по аналогии с §2.2.7).

Урожайность зависит от всех четырёх модальностей (см. формулу ниже) и от
скрытого, ненаблюдаемого моделью эффекта плодородия поля -- это создаёт
реалистичную корреляцию "поле-эффект" между наблюдениями одного поля в разные
годы, из-за которой случайное (не групповое) разбиение на train/test завышало
бы оценку качества (§2.3.5).
"""

import numpy as np
import pandas as pd


def generate_dataset(n_fields=30, n_years=5, seed=11):
    rng = np.random.default_rng(seed)

    field_ids = np.repeat(np.arange(n_fields), n_years)
    year_ids = np.tile(np.arange(n_years), n_fields)
    n = n_fields * n_years

    fertility = rng.normal(0, 1, size=n_fields)[field_ids]

    soil_moisture_mean = rng.normal(30, 5, size=n) + 2.0 * fertility
    soil_nitrogen_mean = rng.normal(15, 3, size=n) + 1.5 * fertility
    precip_sum = rng.normal(400, 80, size=n)
    temp_sum = rng.normal(2200, 150, size=n)

    ndvi_peak = 0.55 + 0.05 * fertility + 0.002 * soil_moisture_mean + rng.normal(0, 0.03, size=n)
    ndvi_integral = 8.0 + 1.2 * fertility + 0.02 * soil_moisture_mean + rng.normal(0, 0.4, size=n)

    disease_rate = np.clip(2.0 - 0.6 * fertility - 0.02 * soil_moisture_mean, 0.2, None)
    disease_events_count = rng.poisson(disease_rate).astype(float)
    max_risk_stage = np.clip(disease_events_count + rng.normal(0, 0.5, size=n), 0, None)

    fertilizer_applications_count = rng.poisson(3, size=n).astype(float)
    total_dose = fertilizer_applications_count * rng.uniform(25, 45, size=n)

    noise = rng.normal(0, 1.5, size=n)

    yield_true = (
        40.0
        + 3.0 * fertility  # скрытый эффект, моделью напрямую не наблюдается
        + 6.0 * ndvi_integral  # geo -- главный фактор
        + 0.5 * soil_nitrogen_mean  # num
        - 1.8 * disease_events_count  # img -- отрицательный вклад
        + 2.0 * np.log1p(fertilizer_applications_count)  # text, насыщающийся эффект
        + noise
    )

    return pd.DataFrame(
        {
            "field_id": field_ids,
            "year": year_ids,
            "soil_moisture_mean": soil_moisture_mean,
            "soil_nitrogen_mean": soil_nitrogen_mean,
            "precip_sum": precip_sum,
            "temp_sum": temp_sum,
            "ndvi_peak": ndvi_peak,
            "ndvi_integral": ndvi_integral,
            "disease_events_count": disease_events_count,
            "max_risk_stage": max_risk_stage,
            "fertilizer_applications_count": fertilizer_applications_count,
            "total_dose": total_dose,
            "yield": yield_true,
        }
    )
