"""Тесты оптимизации дифференцированного внесения удобрений (docs/chapter2/optimization_model.md, §2.4)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ds_optimization.economics import profit  # noqa: E402
from ds_optimization.optimize import optimize_unconstrained, optimize_with_budget  # noqa: E402
from ds_optimization.response import marginal_response, yield_response  # noqa: E402
from ds_optimization.synthetic import generate_plots  # noqa: E402
from ds_optimization.validation import OptimizationInputError, validate_budget, validate_plots  # noqa: E402

PRICE_YIELD = 1300.0
PRICE_FERT = 50.0
DOSE_MIN, DOSE_MAX = 0.0, 150.0


def _total_profit(plots, doses):
    per_area = profit(plots["baseline"].to_numpy(), plots["R"].to_numpy(), plots["s"].to_numpy(), doses, PRICE_YIELD, PRICE_FERT)
    return float(np.sum(per_area * plots["area"].to_numpy()))


def test_yield_response_is_increasing_and_concave():
    d = np.linspace(0, 150, 50)
    y = yield_response(baseline=40.0, R=8.0, s=60.0, d=d)
    assert np.all(np.diff(y) >= 0)  # монотонно неубывающая

    m = marginal_response(R=8.0, s=60.0, d=d)
    assert np.all(np.diff(m) <= 0)  # убывающая предельная отдача


def test_unconstrained_optimum_satisfies_first_order_condition():
    """В точке оптимума (не на границе) предельная выручка = предельные затраты (§2.4.3)."""
    plots = generate_plots(n_plots=10, seed=1)
    doses = optimize_unconstrained(plots, DOSE_MIN, DOSE_MAX, PRICE_YIELD, PRICE_FERT)

    for i, row in enumerate(plots.itertuples()):
        d = doses[i]
        if DOSE_MIN < d < DOSE_MAX:
            marginal_revenue = PRICE_YIELD * marginal_response(row.R, row.s, d)
            assert marginal_revenue == pytest.approx(PRICE_FERT, rel=1e-3)


def test_unconstrained_returns_dose_min_when_unprofitable():
    """Если предельная выручка при d=0 уже ниже цены удобрения, оптимум -- dose_min."""
    import pandas as pd

    plots = pd.DataFrame({"plot_id": [0], "baseline": [40.0], "R": [1.0], "s": [80.0], "area": [1.0]})
    doses = optimize_unconstrained(plots, DOSE_MIN, DOSE_MAX, price_yield=10.0, price_fert=50.0)
    assert doses[0] == pytest.approx(DOSE_MIN)


def test_budget_constraint_is_respected():
    plots = generate_plots(n_plots=15, seed=2)
    total_area = float(plots["area"].sum())
    budget = 30.0 * total_area  # заведомо связывающий бюджет (ниже свободного оптимума)

    doses = optimize_with_budget(plots, budget, DOSE_MIN, DOSE_MAX, PRICE_YIELD, PRICE_FERT)
    spent = float(np.sum(doses * plots["area"].to_numpy()))
    assert spent <= budget + 1e-4


def test_budget_not_binding_matches_unconstrained_optimum():
    plots = generate_plots(n_plots=12, seed=3)
    doses_unconstrained = optimize_unconstrained(plots, DOSE_MIN, DOSE_MAX, PRICE_YIELD, PRICE_FERT)
    huge_budget = float(np.sum(plots["area"])) * DOSE_MAX * 10  # заведомо не связывающий

    doses_with_budget = optimize_with_budget(plots, huge_budget, DOSE_MIN, DOSE_MAX, PRICE_YIELD, PRICE_FERT)
    np.testing.assert_allclose(doses_with_budget, doses_unconstrained, atol=1e-6)


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("dose_uniform", [30.0, 60.0, 90.0])
def test_differentiated_beats_uniform_at_same_budget(seed, dose_uniform):
    """Ключевое свойство §2.4.5: при одинаковом суммарном расходе дифференцированное
    внесение не должно уступать равномерному -- это формальное следствие оптимальности
    решения задачи с бюджетным ограничением, а не статистическая закономерность."""
    plots = generate_plots(n_plots=20, seed=seed)
    total_area = float(plots["area"].sum())
    budget = dose_uniform * total_area

    doses_diff = optimize_with_budget(plots, budget, DOSE_MIN, DOSE_MAX, PRICE_YIELD, PRICE_FERT)
    doses_uniform = np.full(len(plots), dose_uniform)

    profit_diff = _total_profit(plots, doses_diff)
    profit_uniform = _total_profit(plots, doses_uniform)

    assert profit_diff >= profit_uniform - 1e-6


def test_validate_plots_rejects_missing_columns():
    import pandas as pd

    plots = pd.DataFrame({"baseline": [40.0], "R": [5.0], "area": [1.0]})  # нет "s"
    with pytest.raises(OptimizationInputError, match="отсутствуют столбцы"):
        validate_plots(plots, DOSE_MIN, DOSE_MAX, PRICE_YIELD, PRICE_FERT)


def test_validate_plots_rejects_nonpositive_s():
    import pandas as pd

    plots = pd.DataFrame({"baseline": [40.0], "R": [5.0], "s": [0.0], "area": [1.0]})
    with pytest.raises(OptimizationInputError, match="Масштаб насыщения"):
        validate_plots(plots, DOSE_MIN, DOSE_MAX, PRICE_YIELD, PRICE_FERT)


def test_validate_plots_rejects_bad_dose_range():
    plots = generate_plots(n_plots=3)
    with pytest.raises(OptimizationInputError, match="диапазон доз"):
        validate_plots(plots, dose_min=100.0, dose_max=50.0, price_yield=PRICE_YIELD, price_fert=PRICE_FERT)


def test_validate_plots_rejects_nonpositive_prices():
    plots = generate_plots(n_plots=3)
    with pytest.raises(OptimizationInputError, match="Цены"):
        validate_plots(plots, DOSE_MIN, DOSE_MAX, price_yield=0.0, price_fert=PRICE_FERT)


def test_validate_budget_rejects_infeasible_budget():
    plots = generate_plots(n_plots=5)
    with pytest.raises(OptimizationInputError, match="меньше стоимости"):
        validate_budget(plots, budget=-1.0, dose_min=10.0)
