"""Валидация входных параметров оптимизации, §2.4.

Проверяется на границе публичного API (optimize_unconstrained,
optimize_with_budget), по аналогии с src/ds_preprocessing/validation.py.
"""

import numpy as np


class OptimizationInputError(ValueError):
    """Некорректные входные параметры для оптимизации внесения удобрений."""


REQUIRED_COLUMNS = ["baseline", "R", "s", "area"]


def validate_plots(plots, dose_min, dose_max, price_yield, price_fert):
    missing = [c for c in REQUIRED_COLUMNS if c not in plots.columns]
    if missing:
        raise OptimizationInputError(f"В таблице участков отсутствуют столбцы: {missing}")

    if len(plots) == 0:
        raise OptimizationInputError("Таблица участков пуста")

    if plots[REQUIRED_COLUMNS].isna().any().any():
        raise OptimizationInputError("Таблица участков содержит NaN в baseline/R/s/area")

    if (plots["s"] <= 0).any():
        raise OptimizationInputError("Масштаб насыщения s должен быть строго положительным для всех участков")

    if (plots["R"] < 0).any():
        raise OptimizationInputError("Предельная прибавка урожая R не может быть отрицательной")

    if (plots["area"] <= 0).any():
        raise OptimizationInputError("Площадь участка area должна быть строго положительной")

    if dose_min < 0 or dose_max <= dose_min:
        raise OptimizationInputError(f"Некорректный диапазон доз: [{dose_min}, {dose_max}]")

    if price_yield <= 0 or price_fert <= 0:
        raise OptimizationInputError("Цены price_yield и price_fert должны быть строго положительными")


def validate_budget(plots, budget, dose_min):
    total_min_area = float(np.sum(plots["area"])) * dose_min
    if budget < total_min_area:
        raise OptimizationInputError(
            f"Бюджет {budget} меньше стоимости минимально допустимой дозы на всех участках ({total_min_area})"
        )


def validate_plots_multi(plots, nutrient_names, dose_min, dose_max, price_yield, price_fert):
    """Валидация для многокомпонентного внесения (§2.4.8) -- аналог validate_plots,
    но s задаётся отдельной колонкой s_<nutrient> на каждый вид удобрения."""
    s_columns = [f"s_{n}" for n in nutrient_names]
    required = ["baseline", "R", "area"] + s_columns
    missing = [c for c in required if c not in plots.columns]
    if missing:
        raise OptimizationInputError(f"В таблице участков отсутствуют столбцы: {missing}")

    if len(plots) == 0:
        raise OptimizationInputError("Таблица участков пуста")

    if plots[required].isna().any().any():
        raise OptimizationInputError("Таблица участков содержит NaN в baseline/R/area/s_*")

    if (plots[s_columns] <= 0).any().any():
        raise OptimizationInputError("Масштаб насыщения s должен быть строго положительным для всех участков и видов удобрений")

    if (plots["R"] < 0).any():
        raise OptimizationInputError("Предельная прибавка урожая R не может быть отрицательной")

    if (plots["area"] <= 0).any():
        raise OptimizationInputError("Площадь участка area должна быть строго положительной")

    if dose_min < 0 or dose_max <= dose_min:
        raise OptimizationInputError(f"Некорректный диапазон доз: [{dose_min}, {dose_max}]")

    price_fert = np.asarray(price_fert, dtype=float)
    if price_fert.shape != (len(nutrient_names),):
        raise OptimizationInputError(
            f"price_fert должен содержать {len(nutrient_names)} значений (по числу видов удобрений), получено {price_fert.shape}"
        )
    if price_yield <= 0 or (price_fert <= 0).any():
        raise OptimizationInputError("Цены price_yield и все price_fert должны быть строго положительными")
