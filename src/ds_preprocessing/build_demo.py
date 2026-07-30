"""Демонстрация комбинированного алгоритма на контролируемых данных.

Сравнивает RMSE трёх вариантов -- только пространственный, только временной,
комбинированный -- воспроизводя качественный результат Ли и др. (2021,
§1.2.2 docs/chapter1... / §2.2.6): комбинированная оценка не хуже, а как
правило лучше, каждой из оценок по отдельности.

Запуск: PYTHONPATH=src python3 -m ds_preprocessing.build_demo
"""

import numpy as np

from ds_preprocessing.combine import fill_gaps
from ds_preprocessing.metrics import mae, rmse
from ds_preprocessing.synthetic import generate_field, punch_holes


def run_demo(n_trials=10):
    """Усреднить сравнение методов по нескольким случайным маскам пропусков.

    Единичный запуск (одна случайная маска) статистически неустойчив --
    взвешивание по обратной дисперсии оптимально в среднем, а не гарантированно
    на каждой конкретной выборке (§2.2.6). Поэтому сравнение, как и в Ли и др.
    (2021), проводится по агрегированной ошибке на серии испытаний.
    """
    coords, times, X_true = generate_field()

    rmses = {"spatial": [], "temporal": [], "combined": []}
    unrestored_total, anomalies_total = 0, 0

    for seed in range(n_trials):
        X_observed, mask_observed, mask_test = punch_holes(X_true, seed=seed)
        result = fill_gaps(coords, times, X_observed, mask_observed)

        rmses["spatial"].append(rmse(result["spatial_only"][mask_test], X_true[mask_test]))
        rmses["temporal"].append(rmse(result["temporal_only"][mask_test], X_true[mask_test]))
        rmses["combined"].append(rmse(result["filled"][mask_test], X_true[mask_test]))
        unrestored_total += int(result["unrestored"].sum())
        anomalies_total += int(result["anomalies"].sum())

    mean_rmse = {k: float(np.mean(v)) for k, v in rmses.items()}

    print(f"Восстановление пропусков -- сравнение методов (§2.2.6), усреднено по {n_trials} испытаниям")
    print(f"{'Метод':<25}{'Средний RMSE':>14}")
    print(f"{'только пространственный':<25}{mean_rmse['spatial']:>14.4f}")
    print(f"{'только временной':<25}{mean_rmse['temporal']:>14.4f}")
    print(f"{'комбинированный':<25}{mean_rmse['combined']:>14.4f}")
    print()
    print(f"Невосстановимых ячеек (суммарно): {unrestored_total}")
    print(f"Обнаружено аномалий (суммарно): {anomalies_total}")

    assert mean_rmse["combined"] <= min(mean_rmse["spatial"], mean_rmse["temporal"]) + 1e-9, (
        "Комбинированный алгоритм не должен уступать в среднем оценке по одному источнику (§2.2.6)"
    )

    return mean_rmse


if __name__ == "__main__":
    run_demo()
