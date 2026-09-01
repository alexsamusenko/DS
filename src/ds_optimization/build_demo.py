"""Демонстрация: дифференцированное внесение против равномерного при одинаковом бюджете.

Запуск: PYTHONPATH=src python3 -m ds_optimization.build_demo
"""

import time

import numpy as np

from ds_optimization.economics import profit
from ds_optimization.optimize import optimize_with_budget
from ds_optimization.synthetic import generate_plots

PRICE_YIELD = 1300.0  # руб./ц продукции (баллистика: зерновые, порядок величины)
PRICE_FERT = 50.0  # руб./кг д.в. удобрения
DOSE_MIN, DOSE_MAX = 0.0, 150.0  # кг/га


def total_profit(plots, doses, price_yield, price_fert):
    per_area_profit = profit(plots["baseline"].to_numpy(), plots["R"].to_numpy(), plots["s"].to_numpy(), doses, price_yield, price_fert)
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
    print(f"Дозы по участкам (первые 5 из {len(plots)}):")
    print(f"{'участок':<10}{'R (отклик)':>12}{'равномерно':>14}{'дифференц.':>14}")
    for i in range(min(5, len(plots))):
        print(f"{i:<10}{plots['R'].iloc[i]:>12.2f}{dose_uniform:>14.1f}{doses_diff[i]:>14.1f}")

    assert profit_diff >= profit_uniform - 1e-6, "Дифференцированное внесение не должно уступать равномерному при том же бюджете (§2.4.5)"

    return profit_diff, profit_uniform


def run_variability(seeds=tuple(range(1, 11))):
    """Оценить разброс относительного выигрыша дифференцированного внесения
    над равномерным по нескольким независимым перегенерациям набора участков
    (§10.6) -- единственный расчётный пример (seed=5, используемый в run_demo)
    статистически не отличим от случайно удачной конфигурации без такой
    проверки.
    """
    from ds_optimization.synthetic import generate_plots

    gains = []
    for seed in seeds:
        plots = generate_plots(seed=seed)
        total_area = float(plots["area"].sum())
        budget = 70.0 * total_area
        doses_diff = optimize_with_budget(plots, budget, DOSE_MIN, DOSE_MAX, PRICE_YIELD, PRICE_FERT)
        dose_uniform = budget / total_area
        doses_uniform = np.full(len(plots), dose_uniform)
        profit_diff = total_profit(plots, doses_diff, PRICE_YIELD, PRICE_FERT)
        profit_uniform = total_profit(plots, doses_uniform, PRICE_YIELD, PRICE_FERT)
        gains.append((profit_diff / profit_uniform - 1) * 100)

    gains = np.array(gains)
    print(f"\nРазброс относительного выигрыша по {len(seeds)} независимым конфигурациям участков (seeds={seeds})")
    print(f"Среднее: {gains.mean():.2f}%  Станд. откл.: {gains.std(ddof=1):.2f}%  Мин/Макс: {gains.min():.2f}% / {gains.max():.2f}%")

    assert (gains >= -1e-6).all(), "Дифференцированное внесение не должно уступать равномерному ни на одной из конфигураций (§2.4.5)"

    return gains


def run_cost(n_calls=50):
    """Измерить вычислительную стоимость оптимизации уровня L5 (§10.7, §4.3):
    время формирования карты-задания для одного поля на процессоре, без
    оптимизации инференса -- нужно для оценки практического эффекта
    (сокращение времени цикла принятия решения, §4.3.1).
    """
    plots = generate_plots()
    total_area = float(plots["area"].sum())
    budget = 70.0 * total_area

    for _ in range(3):
        optimize_with_budget(plots, budget, DOSE_MIN, DOSE_MAX, PRICE_YIELD, PRICE_FERT)

    times = []
    for _ in range(n_calls):
        t0 = time.perf_counter()
        optimize_with_budget(plots, budget, DOSE_MIN, DOSE_MAX, PRICE_YIELD, PRICE_FERT)
        times.append(time.perf_counter() - t0)

    times_ms = np.array(times) * 1000
    print(f"\nВычислительная стоимость оптимизации L5 ({len(plots)} участков), {n_calls} вызовов, CPU без GPU")
    print(f"Формирование карты-задания: {times_ms.mean():.2f} ± {times_ms.std(ddof=1):.2f} мс (p50={np.percentile(times_ms, 50):.2f}, p95={np.percentile(times_ms, 95):.2f})")

    return times_ms


def run_sensitivity_dose_level(avg_doses=(30, 50, 70, 90, 110, 130)):
    """Чувствительность относительного выигрыша дифференцированного внесения
    к среднему уровню бюджета (§10.5) -- run_demo() и run_variability()
    фиксируют средний бюджет на уровне 70 кг/га; здесь проверяется,
    сохраняется ли неотрицательность выигрыша и как меняется его величина
    при отклонении от этого уровня (недо- и перерасход относительно
    локально оптимальной дозы конкретного участка).
    """
    plots = generate_plots()
    total_area = float(plots["area"].sum())
    rows = []
    for avg_dose in avg_doses:
        budget = avg_dose * total_area
        doses_diff = optimize_with_budget(plots, budget, DOSE_MIN, DOSE_MAX, PRICE_YIELD, PRICE_FERT)
        doses_uniform = np.full(len(plots), budget / total_area)
        profit_diff = total_profit(plots, doses_diff, PRICE_YIELD, PRICE_FERT)
        profit_uniform = total_profit(plots, doses_uniform, PRICE_YIELD, PRICE_FERT)
        gain_pct = (profit_diff / profit_uniform - 1) * 100
        rows.append({"avg_dose": avg_dose, "profit_diff": profit_diff, "profit_uniform": profit_uniform, "gain_pct": gain_pct})

    print(f"\nЧувствительность выигрыша к среднему уровню бюджета (§10.5)")
    print(f"{'Средняя доза, кг/га':>20}{'Прибыль дифф.':>16}{'Прибыль равн.':>16}{'Выигрыш, %':>12}")
    for r in rows:
        print(f"{r['avg_dose']:>20}{r['profit_diff']:>16.1f}{r['profit_uniform']:>16.1f}{r['gain_pct']:>12.3f}")

    assert all(r["profit_diff"] >= r["profit_uniform"] - 1e-6 for r in rows), (
        "Дифференцированное внесение не должно уступать равномерному ни при одном уровне бюджета (§2.4.5)"
    )

    return rows


if __name__ == "__main__":
    run_demo()
    run_variability()
    run_cost()
    run_sensitivity_dose_level()
