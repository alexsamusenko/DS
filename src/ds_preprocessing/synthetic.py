"""Контролируемые синтетические пространственно-временные данные для валидации.

Используются вместо реальных открытых источников (docs/datasets.md) именно
здесь, потому что оценка ошибки восстановления требует известного эталона,
недоступного в реальных пропущенных данных, -- см. §2.2.6, §2.2.7
docs/chapter2_preprocessing_model.md.
"""

import numpy as np


def generate_field(grid_size=6, n_times=20, noise_std=0.15, seed=42):
    """Сгенерировать поле с независимой пространственной и временной структурой.

    Возвращает (coords, times, X_true): coords (M,2), times (T,), X_true (M,T),
    где M = grid_size**2.
    """
    rng = np.random.default_rng(seed)

    xs, ys = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    coords = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
    times = np.arange(n_times, dtype=float)

    spatial_field = np.sin(coords[:, 0] / 2.5) + np.cos(coords[:, 1] / 3.0) * 1.5
    temporal_trend = 0.08 * times + 1.2 * np.sin(2 * np.pi * times / 40.0)

    X_true = spatial_field[:, None] + temporal_trend[None, :]
    X_true += rng.normal(scale=noise_std, size=X_true.shape)

    return coords, times, X_true


def punch_holes(X_true, missing_fraction=0.2, n_outliers=3, seed=7):
    """Скрыть часть значений (для теста восстановления) и внести аномалии.

    Возвращает (X_observed, mask_observed_true, mask_test):
        X_observed        -- значения с внесёнными аномалиями (без самих пропусков -- пропуски содержат X_true, но помечены как ненаблюдаемые)
        mask_observed_true -- bool-маска (M,T): что реально считается наблюдаемым на входе алгоритма
        mask_test          -- bool-маска ячеек, скрытых намеренно для проверки качества (подмножество ~mask_observed_true, без аномалий)
    """
    rng = np.random.default_rng(seed)
    M, T = X_true.shape

    mask_test = rng.random((M, T)) < missing_fraction
    mask_observed_true = ~mask_test.copy()

    X_observed = X_true.copy()

    outlier_candidates = np.argwhere(mask_observed_true)
    outlier_idx = rng.choice(len(outlier_candidates), size=min(n_outliers, len(outlier_candidates)), replace=False)
    for i in outlier_idx:
        m, t = outlier_candidates[i]
        X_observed[m, t] += rng.choice([-1, 1]) * rng.uniform(6, 9)

    return X_observed, mask_observed_true, mask_test
