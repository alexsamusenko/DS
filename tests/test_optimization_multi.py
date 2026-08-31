"""Тесты многокомпонентного внесения удобрений (docs/chapter2/optimization_model.md, §2.4.8)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ds_optimization.economics import profit_multi  # noqa: E402
from ds_optimization.optimize import optimize_unconstrained, optimize_unconstrained_multi  # noqa: E402
from ds_optimization.response import yield_response_multi  # noqa: E402
from ds_optimization.synthetic import generate_plots  # noqa: E402
from ds_optimization.validation import OptimizationInputError  # noqa: E402

PRICE_YIELD = 1300.0
DOSE_MIN, DOSE_MAX = 0.0, 150.0
NUTRIENTS = ("N", "P", "K")


def test_yield_response_multi_increases_with_each_dose():
    d = np.array([10.0, 10.0, 10.0])
    y0 = yield_response_multi(baseline=40.0, R=15.0, s=[60.0, 70.0, 80.0], d=d)
    d_more_n = np.array([50.0, 10.0, 10.0])
    y1 = yield_response_multi(baseline=40.0, R=15.0, s=[60.0, 70.0, 80.0], d=d_more_n)
    assert y1 > y0


def test_yield_response_multi_obeys_liebig_minimum_law():
    """Если один вид удобрения не внесён вовсе (d_j=0), урожай равен baseline
    независимо от того, сколько внесено остальных видов -- избыток одного
    компонента не компенсирует полное отсутствие другого (§2.4.8)."""
    baseline = 40.0
    y = yield_response_multi(baseline=baseline, R=20.0, s=[60.0, 70.0, 80.0], d=[0.0, 1000.0, 1000.0])
    assert y == pytest.approx(baseline, abs=1e-6)


def test_profit_multi_matches_manual_calculation():
    baseline, R = 40.0, 15.0
    s = np.array([60.0, 70.0, 80.0])
    d = np.array([20.0, 30.0, 10.0])
    price_fert = np.array([50.0, 60.0, 40.0])

    expected_yield = baseline + R * np.prod(1 - np.exp(-d / s))
    expected_cost = np.sum(d * price_fert)
    expected_profit = PRICE_YIELD * expected_yield - expected_cost

    result = profit_multi(baseline, R, s, d, PRICE_YIELD, price_fert)
    assert result == pytest.approx(expected_profit)


def _multi_plots(n_plots=6, seed=3):
    base = generate_plots(n_plots=n_plots, seed=seed)
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "plot_id": base["plot_id"],
            "baseline": base["baseline"],
            "R": base["R"],
            "area": base["area"],
            "s_N": base["s"],
            "s_P": base["s"] * rng.uniform(0.8, 1.3, size=n_plots),
            "s_K": base["s"] * rng.uniform(0.8, 1.3, size=n_plots),
        }
    )


def test_optimize_multi_returns_correct_shape():
    plots = _multi_plots()
    doses = optimize_unconstrained_multi(plots, NUTRIENTS, DOSE_MIN, DOSE_MAX, PRICE_YIELD, price_fert=[50.0, 60.0, 40.0])
    assert doses.shape == (len(plots), len(NUTRIENTS))
    assert np.all(doses >= DOSE_MIN) and np.all(doses <= DOSE_MAX)


def test_optimize_multi_with_single_nutrient_matches_analytical_solution():
    """При J=1 численный оптимизатор должен сойтись к тому же решению, что
    аналитический optimize_unconstrained (§2.4.3) -- перекрёстная проверка
    корректности численного метода против уже проверенного аналитического."""
    plots_single = generate_plots(n_plots=8, seed=4)
    analytical = optimize_unconstrained(plots_single, DOSE_MIN, DOSE_MAX, PRICE_YIELD, price_fert=50.0)

    plots_multi = pd.DataFrame(
        {
            "plot_id": plots_single["plot_id"],
            "baseline": plots_single["baseline"],
            "R": plots_single["R"],
            "area": plots_single["area"],
            "s_N": plots_single["s"],
        }
    )
    numerical = optimize_unconstrained_multi(plots_multi, ("N",), DOSE_MIN, DOSE_MAX, PRICE_YIELD, price_fert=[50.0])

    np.testing.assert_allclose(numerical[:, 0], analytical, atol=0.5)


def test_optimize_multi_rejects_wrong_price_fert_length():
    plots = _multi_plots()
    with pytest.raises(OptimizationInputError):
        optimize_unconstrained_multi(
            plots, NUTRIENTS, DOSE_MIN, DOSE_MAX, PRICE_YIELD, price_fert=[50.0, 60.0]
        )  # только 2 цены на 3 удобрения


def test_optimize_multi_rejects_missing_s_column():
    plots = _multi_plots().drop(columns=["s_K"])
    with pytest.raises(OptimizationInputError):
        optimize_unconstrained_multi(plots, NUTRIENTS, DOSE_MIN, DOSE_MAX, PRICE_YIELD, price_fert=[50.0, 60.0, 40.0])
