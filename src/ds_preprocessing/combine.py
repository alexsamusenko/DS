"""Комбинирование пространственной и временной оценок по обратной дисперсии, §2.2.5."""

import numpy as np

from .anomaly import detect_anomalies
from .spatial import spatial_estimate
from .temporal import temporal_estimate
from .validation import validate_inputs


def fill_gaps(coords, times, X, mask_observed, drop_anomalies=True):
    """Восстановить пропуски в X комбинированным алгоритмом (§2.2).

    coords : np.ndarray (M, 2) -- координаты точек.
    times : np.ndarray (T,) -- числовая временная шкала.
    X : np.ndarray (M, T) -- наблюдаемые значения (произвольные там, где
        mask_observed=False -- они игнорируются).
    mask_observed : np.ndarray (M, T), bool.
    drop_anomalies : если True, значения, признанные аномалиями (§2.2.2),
        исключаются из наблюдаемых перед восстановлением.

    Поднимает ds_preprocessing.validation.DataValidationError, если формы,
    типы или содержимое входных массивов некорректны (см. validation.py).

    Возвращает dict с полями:
        filled       -- np.ndarray (M, T), исходные значения + восстановленные
        unrestored   -- bool-маска (M, T): True там, где не хватило данных
                         ни для пространственной, ни для временной оценки
        spatial_only, temporal_only -- отдельные оценки (для сравнения качества,
                         см. tests/test_preprocessing.py)
        anomalies    -- bool-маска обнаруженных аномалий
    """
    validate_inputs(coords, times, X, mask_observed)

    mask = mask_observed.copy()
    anomalies = detect_anomalies(X, mask)
    if drop_anomalies:
        mask = mask & ~anomalies

    spatial_est, spatial_var = spatial_estimate(coords, X, mask)
    temporal_est, temporal_var = temporal_estimate(times, X, mask)

    filled = np.where(mask, X, np.nan)
    unrestored = np.zeros_like(mask)

    M, T = X.shape
    for m in range(M):
        for t in range(T):
            if mask[m, t]:
                continue

            has_spatial = not np.isnan(spatial_est[m, t])
            has_temporal = not np.isnan(temporal_est[m, t])

            if has_spatial and has_temporal:
                w_s = 1.0 / spatial_var[m, t]
                w_t = 1.0 / temporal_var[m, t]
                filled[m, t] = (w_s * spatial_est[m, t] + w_t * temporal_est[m, t]) / (w_s + w_t)
            elif has_spatial:
                filled[m, t] = spatial_est[m, t]
            elif has_temporal:
                filled[m, t] = temporal_est[m, t]
            else:
                unrestored[m, t] = True

    return {
        "filled": filled,
        "unrestored": unrestored,
        "spatial_only": spatial_est,
        "temporal_only": temporal_est,
        "anomalies": anomalies,
    }
