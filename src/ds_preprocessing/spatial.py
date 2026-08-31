"""Пространственная оценка обыкновенным кригингом, §2.2.3."""

import numpy as np
from pykrige.ok import OrdinaryKriging

MIN_POINTS_FOR_KRIGING = 3


def spatial_estimate(coords, X, mask_observed):
    """Оценить пропуски по срезам одного момента времени (кригинг по m).

    coords : np.ndarray формы (M, 2) -- координаты (x, y) точек s_1..s_M.
    X : np.ndarray формы (M, T).
    mask_observed : np.ndarray формы (M, T), bool.

    Возвращает (estimate, variance) -- оба формы (M, T), заполнены np.nan
    там, где оценка недоступна (< MIN_POINTS_FOR_KRIGING наблюдений в срезе).
    """
    M, T = X.shape
    estimate = np.full((M, T), np.nan)
    variance = np.full((M, T), np.nan)

    for t in range(T):
        observed_idx = np.where(mask_observed[:, t])[0]
        missing_idx = np.where(~mask_observed[:, t])[0]
        if missing_idx.size == 0 or observed_idx.size < MIN_POINTS_FOR_KRIGING:
            continue

        ok = OrdinaryKriging(
            coords[observed_idx, 0],
            coords[observed_idx, 1],
            X[observed_idx, t],
            variogram_model="spherical",
            verbose=False,
            enable_plotting=False,
        )
        pred, var = ok.execute("points", coords[missing_idx, 0], coords[missing_idx, 1])
        estimate[missing_idx, t] = np.asarray(pred)
        variance[missing_idx, t] = np.asarray(var)

    return estimate, variance
