"""Валидация входных данных на границе публичного API (fill_gaps), §2.2.

Проверки сосредоточены здесь, а не продублированы в spatial.py/temporal.py --
внутренние функции конвейера могут полагаться на то, что форма и типы данных
уже корректны.
"""

import numpy as np


class DataValidationError(ValueError):
    """Некорректные входные данные для fill_gaps."""


def validate_inputs(coords, times, X, mask_observed):
    """Проверить согласованность форм, типов и содержимого входных массивов.

    Возвращает (M, T) при успешной проверке; иначе поднимает DataValidationError
    с понятным описанием, что именно не так.
    """
    coords = np.asarray(coords)
    times = np.asarray(times)
    X = np.asarray(X)
    mask_observed = np.asarray(mask_observed)

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise DataValidationError(f"coords должен иметь форму (M, 2), получено {coords.shape}")
    if not np.issubdtype(coords.dtype, np.number):
        raise DataValidationError("coords должен содержать числовые координаты")

    if times.ndim != 1:
        raise DataValidationError(f"times должен иметь форму (T,), получено {times.shape}")
    if not np.issubdtype(times.dtype, np.number):
        raise DataValidationError("times должен содержать числовую временную шкалу")

    M, T = coords.shape[0], times.shape[0]
    if M == 0 or T == 0:
        raise DataValidationError("coords и times не должны быть пустыми")

    if X.shape != (M, T):
        raise DataValidationError(f"X должен иметь форму (M, T) = ({M}, {T}), получено {X.shape}")
    if not np.issubdtype(X.dtype, np.number):
        raise DataValidationError("X должен содержать числовые значения показателя")

    if mask_observed.shape != (M, T):
        raise DataValidationError(
            f"mask_observed должен иметь форму (M, T) = ({M}, {T}), получено {mask_observed.shape}"
        )
    if mask_observed.dtype != np.bool_:
        raise DataValidationError("mask_observed должен быть булевой маской (dtype=bool)")

    if not mask_observed.any():
        raise DataValidationError("mask_observed не содержит ни одного наблюдаемого значения -- восстанавливать нечего")

    if np.isnan(X[mask_observed]).any():
        raise DataValidationError("X содержит NaN среди значений, помеченных как наблюдаемые (mask_observed=True)")

    unique_coords = np.unique(coords, axis=0)
    if unique_coords.shape[0] != coords.shape[0]:
        raise DataValidationError("coords содержит повторяющиеся координаты точек -- кригинг требует уникальных позиций")

    return M, T
