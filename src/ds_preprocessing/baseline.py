"""Наивная базовая линия восстановления пропусков -- простая линейная
интерполяция по времени, без кригинга, без LOOCV-взвешивания и без учёта
пространственной структуры (§10.4 спецификации: "простая базовая линия",
обязательная точка сравнения наряду с кригингом/трендом и комбинированным
методом). Это ровно тот метод, который в §1.2.2 диссертации назван
"наиболее распространённым на практике" -- используется здесь как реальная,
а не декларативная точка сравнения.
"""

import numpy as np


def naive_interpolation_baseline(times, X, mask_observed):
    """Восстановить пропуски линейной интерполяцией по времени, независимо по каждой точке.

    times : np.ndarray формы (T,).
    X : np.ndarray формы (M, T).
    mask_observed : np.ndarray формы (M, T), bool.

    Возвращает estimate формы (M, T), np.nan там, где в ряде точки m нет ни
    одного наблюдения (интерполяция невозможна в принципе, не только этим
    методом). За пределами диапазона наблюдённых времён значение держится
    на уровне ближайшего наблюдения (реализация np.interp по умолчанию) --
    простейшее допущение, типичное для базовой линии.
    """
    M, T = X.shape
    estimate = np.full((M, T), np.nan)

    for m in range(M):
        observed_idx = np.where(mask_observed[m])[0]
        missing_idx = np.where(~mask_observed[m])[0]
        if observed_idx.size == 0 or missing_idx.size == 0:
            continue
        estimate[m, missing_idx] = np.interp(
            times[missing_idx], times[observed_idx], X[m, observed_idx]
        )

    return estimate
