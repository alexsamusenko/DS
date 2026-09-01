"""Демонстрация комбинированного алгоритма на контролируемых данных.

Сравнивает RMSE четырёх вариантов -- наивная базовая линия (линейная
интерполяция по времени, §10.4), только пространственный (кригинг), только
временной (локальный полиномиальный тренд), комбинированный -- воспроизводя
качественный результат Ли и др. (2021, §1.2.2 docs/chapter1... / §2.2.6):
комбинированная оценка не хуже, а как правило лучше, каждой из оценок по
отдельности, и все три содержательных метода не хуже наивной базовой линии.

Запуск: PYTHONPATH=src python3 -m ds_preprocessing.build_demo
"""

import numpy as np

from ds_preprocessing.baseline import naive_interpolation_baseline
from ds_preprocessing.combine import fill_gaps
from ds_preprocessing.metrics import rmse
from ds_preprocessing.synthetic import generate_field, punch_holes


def run_demo(n_trials=10):
    """Усреднить сравнение методов по нескольким случайным маскам пропусков.

    Единичный запуск (одна случайная маска) статистически неустойчив --
    взвешивание по обратной дисперсии оптимально в среднем, а не гарантированно
    на каждой конкретной выборке (§2.2.6). Поэтому сравнение, как и в Ли и др.
    (2021), проводится по агрегированной ошибке на серии испытаний, а для
    статистической корректности (§10.6) наряду со средним отдельно
    сохраняется и разброс (стандартное отклонение) по этим испытаниям, а не
    только точечная оценка.
    """
    coords, times, X_true = generate_field()

    rmses: dict[str, list[float]] = {"naive": [], "spatial": [], "temporal": [], "combined": []}
    unrestored_total, anomalies_total = 0, 0

    for seed in range(n_trials):
        X_observed, mask_observed, mask_test = punch_holes(X_true, seed=seed)
        result = fill_gaps(coords, times, X_observed, mask_observed)
        naive_est = naive_interpolation_baseline(times, X_observed, mask_observed)

        rmses["naive"].append(rmse(naive_est[mask_test], X_true[mask_test]))
        rmses["spatial"].append(rmse(result["spatial_only"][mask_test], X_true[mask_test]))
        rmses["temporal"].append(rmse(result["temporal_only"][mask_test], X_true[mask_test]))
        rmses["combined"].append(rmse(result["filled"][mask_test], X_true[mask_test]))
        unrestored_total += int(result["unrestored"].sum())
        anomalies_total += int(result["anomalies"].sum())

    mean_rmse = {k: float(np.mean(v)) for k, v in rmses.items()}
    std_rmse = {k: float(np.std(v, ddof=1)) for k, v in rmses.items()}

    print(f"Восстановление пропусков -- сравнение методов (§2.2.6), усреднено по {n_trials} испытаниям")
    print(f"{'Метод':<25}{'Средний RMSE':>14}{'Станд. откл.':>16}")
    print(f"{'наивная база (интерп.)':<25}{mean_rmse['naive']:>14.4f}{std_rmse['naive']:>16.4f}")
    print(f"{'только пространственный':<25}{mean_rmse['spatial']:>14.4f}{std_rmse['spatial']:>16.4f}")
    print(f"{'только временной':<25}{mean_rmse['temporal']:>14.4f}{std_rmse['temporal']:>16.4f}")
    print(f"{'комбинированный':<25}{mean_rmse['combined']:>14.4f}{std_rmse['combined']:>16.4f}")
    print()
    print(f"Невосстановимых ячеек (суммарно): {unrestored_total}")
    print(f"Обнаружено аномалий (суммарно): {anomalies_total}")

    assert mean_rmse["combined"] <= min(mean_rmse["spatial"], mean_rmse["temporal"]) + 1e-9, (
        "Комбинированный алгоритм не должен уступать в среднем оценке по одному источнику (§2.2.6)"
    )

    return {"mean": mean_rmse, "std": std_rmse, "per_trial": rmses}


def run_robustness_missing_fraction(fractions=(0.1, 0.2, 0.3, 0.4, 0.5), n_trials=10):
    """Устойчивость комбинированного метода к доле пропусков (§10.5): RMSE
    комбинированного метода и наивной базы при доле пропусков от 10 до 50%
    того же поля, что и в run_demo() -- проверка того, что преимущество
    метода не является особенностью конкретного (20%) уровня пропусков.
    """
    coords, times, X_true = generate_field()
    rows = []
    for frac in fractions:
        combined_vals, naive_vals, unrestored_total = [], [], 0
        for seed in range(n_trials):
            X_observed, mask_observed, mask_test = punch_holes(X_true, missing_fraction=frac, seed=seed)
            result = fill_gaps(coords, times, X_observed, mask_observed)
            naive_est = naive_interpolation_baseline(times, X_observed, mask_observed)
            combined_vals.append(rmse(result["filled"][mask_test], X_true[mask_test]))
            naive_vals.append(rmse(naive_est[mask_test], X_true[mask_test]))
            unrestored_total += int(result["unrestored"].sum())
        rows.append({
            "fraction": frac,
            "combined_mean": float(np.mean(combined_vals)), "combined_std": float(np.std(combined_vals, ddof=1)),
            "naive_mean": float(np.mean(naive_vals)), "naive_std": float(np.std(naive_vals, ddof=1)),
            "unrestored": unrestored_total,
        })

    print(f"\nУстойчивость к доле пропусков (§10.5), {n_trials} испытаний на каждый уровень")
    print(f"{'Доля пропусков':>16}{'Комбинированный':>20}{'Наивная база':>20}{'Невосст.':>10}")
    for r in rows:
        print(f"{r['fraction']:>16.1f}{r['combined_mean']:>13.4f}±{r['combined_std']:<5.4f}"
              f"{r['naive_mean']:>13.4f}±{r['naive_std']:<5.4f}{r['unrestored']:>10}")

    return rows


if __name__ == "__main__":
    run_demo()
    run_robustness_missing_fraction()
