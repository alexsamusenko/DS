"""Детекция аномалий по модифицированной z-оценке (MAD), §2.2.2."""

import numpy as np

DEFAULT_THRESHOLD = 3.5


def detect_anomalies(X, mask_observed, threshold=DEFAULT_THRESHOLD):
    """Найти выбросы во временном ряде каждой пространственной точки.

    X : np.ndarray формы (M, T) -- значения показателя.
    mask_observed : np.ndarray формы (M, T), bool -- True там, где значение
        реально наблюдалось (а не уже пропущено).
    threshold : порог модифицированной z-оценки (по умолчанию 3.5).

    Ряды агрономических показателей, как показано в §1.2.2, как правило
    нестационарны (несут временной тренд). Поэтому MAD считается не по сырым
    значениям, а по остаткам после снятия локального тренда (полином низкой
    степени) -- иначе точки на пиках самого тренда ложно помечаются как
    аномалии.

    Возвращает bool-маску формы (M, T): True в ячейках, признанных аномалиями
    среди наблюдаемых (пропущенные ячейки аномалиями не считаются).
    """
    M, T = X.shape
    is_anomaly = np.zeros((M, T), dtype=bool)
    times = np.arange(T, dtype=float)

    for m in range(M):
        observed_idx = np.where(mask_observed[m])[0]
        if observed_idx.size < 4:
            continue

        t_obs = times[observed_idx]
        values = X[m, observed_idx]

        degree = min(2, observed_idx.size - 2)
        trend = np.poly1d(np.polyfit(t_obs, values, degree))
        residuals = values - trend(t_obs)

        med = np.median(residuals)
        mad = np.median(np.abs(residuals - med))
        if mad == 0:
            continue

        mz = 0.6745 * (residuals - med) / mad
        is_anomaly[m, observed_idx[np.abs(mz) > threshold]] = True

    return is_anomaly
