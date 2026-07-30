"""Демонстрация: дифференцированное внесение против равномерного при одинаковом бюджете.

Запуск: PYTHONPATH=src python3 -m ds_optimization.build_demo
"""

import numpy as np

from ds_optimization.economics import profit
from ds_optimization.optimize import optimize_with_budget
from ds_optimization.synthetic import generate_plots

PRICE_YIELD = 1300.0  # руб./ц продукции (баллистика: зерновые, порядок величины)
PRICE_FERT = 50.0  # руб./кг д.в. удобрения
DOSE_MIN, DOSE_MAX = 0.0, 150.0  # кг/га


def total_profit(plots, doses, price_yield, price_fert):
    per_area_profit = profit(
        plots["baseline"].to_numpy(), plots["R"].to_numpy(), plots["s"].to_numpy(), doses, price_yield, price_fert
    )
    return float(np.sum(per_area_profit * plots["area"].to_numpy()))


def run_demo():
    plots = generate_plots()
    total_area = float(plots["area"].sum())
    budget = 70.0 * total_area  # средняя доза 70 кг/га по всему полю

    doses_diff = optimize_with_budget(plots, budget, DOSE_MIN, DOSE_MAX, PRICE_YIELD, PRICE_FERT)
    dose_uniform = budget / total_area
    doses_uniform = np.full(len(plots), dose_uniform)

    profit_diff = total_profit(plots, doses_diff, PRICE_YIELD, PRICE_FERT)
    profit_uniform = total_profit(plots, doses_uniform, PRICE_YIELD, PRICE_FERT)

    print("Дифференцированное внесение против равномерного (§2.4.5)")
    print(f"Суммарный бюджет удобрений: {budget:.1f} кг (в среднем {dose_uniform:.1f} кг/га)")
    print(f"{'Сценарий':<25}{'Суммарная прибыль':>20}")
    print(f"{'равномерный':<25}{profit_uniform:>20.1f}")
    print(f"{'дифференцированный':<25}{profit_diff:>20.1f}")
    print(f"Прирост прибыли: {profit_diff - profit_uniform:.1f} ({(profit_diff / profit_uniform - 1) * 100:.1f}%)")

    print()
    print("Дозы по участкам (первые 5 из {}):".format(len(plots)))
    print(f"{'участок':<10}{'R (отклик)':>12}{'равномерно':>14}{'дифференц.':>14}")
    for i in range(min(5, len(plots))):
        print(f"{i:<10}{plots['R'].iloc[i]:>12.2f}{dose_uniform:>14.1f}{doses_diff[i]:>14.1f}")

    assert profit_diff >= profit_uniform - 1e-6, (
        "Дифференцированное внесение не должно уступать равномерному при том же бюджете (§2.4.5)"
    )

    return profit_diff, profit_uniform


if __name__ == "__main__":
    run_demo()
