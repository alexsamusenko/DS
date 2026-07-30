"""Временная оценка локальной полиномиальной аппроксимацией, §2.2.4."""

import numpy as np

MIN_POINTS_FOR_TREND = 2
MAX_POLY_DEGREE = 2


def temporal_estimate(times, X, mask_observed):
    """Оценить пропуски по ряду каждой точки s_m независимо.

    times : np.ndarray формы (T,) -- числовая шкала времени (например, номер дня).
    X : np.ndarray формы (M, T).
    mask_observed : np.ndarray формы (M, T), bool.

    Возвращает (estimate, variance) -- оба формы (M, T), np.nan там, где
    оценка недоступна (< MIN_POINTS_FOR_TREND наблюдений в ряде).
    """
    M, T = X.shape
    estimate = np.full((M, T), np.nan)
    variance = np.full((M, T), np.nan)

    for m in range(M):
        observed_idx = np.where(mask_observed[m])[0]
        missing_idx = np.where(~mask_observed[m])[0]
        n_obs = observed_idx.size
        if missing_idx.size == 0 or n_obs < MIN_POINTS_FOR_TREND:
            continue

        t_obs = times[observed_idx]
        x_obs = X[m, observed_idx]
        degree = min(MAX_POLY_DEGREE, max(0, n_obs - 2))

        # Дисперсия по leave-one-out, а не по остаткам на обучающих точках:
        # подгонка по немногим точкам почти всегда переобучается и занижает
        # собственную ошибку, что при взвешивании по обратной дисперсии
        # ошибочно придаёт временной оценке завышенный вес (см. build_demo.py).
        loo_residuals = []
        for i in range(n_obs):
            keep = np.arange(n_obs) != i
            if keep.sum() < degree + 1:
                continue
            coeffs_i = np.polyfit(t_obs[keep], x_obs[keep], degree)
            pred_i = np.poly1d(coeffs_i)(t_obs[i])
            loo_residuals.append(x_obs[i] - pred_i)

        resid_var = float(np.mean(np.square(loo_residuals))) if loo_residuals else float(np.var(x_obs))
        resid_var = max(resid_var, 1e-4)

        coeffs = np.polyfit(t_obs, x_obs, degree)
        poly = np.poly1d(coeffs)

        estimate[m, missing_idx] = poly(times[missing_idx])
        variance[m, missing_idx] = resid_var

    return estimate, variance
