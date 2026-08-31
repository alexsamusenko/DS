"""Оптимизация доз по участкам: без бюджета (§2.4.3) и с бюджетом (§2.4.4)."""

import numpy as np
from scipy.optimize import brentq, minimize

from .economics import profit_multi
from .validation import validate_budget, validate_plots, validate_plots_multi

EPS_R = 1e-9


def _dose_for_shadow_price(R, s, shadow_price, price_yield, dose_min, dose_max):
    """d_k(lambda) = clip(-s * ln(shadow_price * s / (price_yield * R)), dmin, dmax).

    shadow_price = price_fert (без бюджета) или price_fert + lambda*area_k (с бюджетом).
    """
    if R <= EPS_R:
        return dose_min

    ratio = shadow_price * s / (price_yield * R)
    if ratio <= 0:
        return dose_max
    raw = -s * np.log(ratio)
    return float(np.clip(raw, dose_min, dose_max))


def optimize_unconstrained(plots, dose_min, dose_max, price_yield, price_fert):
    """Оптимум по каждому участку независимо (§2.4.3). Возвращает np.ndarray доз."""
    validate_plots(plots, dose_min, dose_max, price_yield, price_fert)

    doses = np.array([_dose_for_shadow_price(row.R, row.s, price_fert, price_yield, dose_min, dose_max) for row in plots.itertuples()])
    return doses


def _total_dose_at_lambda(plots, lam, price_yield, price_fert, dose_min, dose_max):
    doses = np.array(
        [_dose_for_shadow_price(row.R, row.s, price_fert + lam * row.area, price_yield, dose_min, dose_max) for row in plots.itertuples()]
    )
    return doses, float(np.sum(doses * plots["area"].to_numpy()))


def optimize_with_budget(plots, budget, dose_min, dose_max, price_yield, price_fert):
    """Оптимум с ограничением суммарного расхода (§2.4.4, метод Лагранжа).

    Возвращает np.ndarray доз, удовлетворяющих sum(area_k * d_k) <= budget.
    """
    validate_plots(plots, dose_min, dose_max, price_yield, price_fert)
    validate_budget(plots, budget, dose_min)

    doses_unconstrained, total_unconstrained = _total_dose_at_lambda(plots, 0.0, price_yield, price_fert, dose_min, dose_max)
    if total_unconstrained <= budget:
        return doses_unconstrained  # ограничение не связывающее (lambda = 0)

    def excess(lam):
        _, total = _total_dose_at_lambda(plots, lam, price_yield, price_fert, dose_min, dose_max)
        return total - budget

    lam_hi = 1.0
    while excess(lam_hi) > 0:
        lam_hi *= 2.0
        if lam_hi > 1e12:
            raise RuntimeError("Не удалось найти верхнюю границу lambda для бюджетного ограничения")

    lam_star = brentq(excess, 0.0, lam_hi, xtol=1e-8)
    doses, _ = _total_dose_at_lambda(plots, lam_star, price_yield, price_fert, dose_min, dose_max)
    return doses


def optimize_unconstrained_multi(plots, nutrient_names, dose_min, dose_max, price_yield, price_fert):
    """Оптимум по каждому участку независимо для J>1 видов удобрений (§2.4.8,
    закон Митчерлиха-Бауле). В отличие от optimize_unconstrained, решается
    численно (L-BFGS-B) на каждом участке -- при J>1 произведение вогнутых по
    каждой dose_j функций не обязано быть совместно вогнутым, аналитического
    решения через условие первого порядка, как при J=1, в общем виде нет.

    plots : DataFrame со столбцами baseline, R, area, s_<nutrient> для каждого nutrient_names.
    price_fert : (J,) -- цена по видам удобрений, тот же порядок, что nutrient_names.
    Возвращает np.ndarray формы (K, J) -- доза по каждому участку и виду удобрения.
    """
    validate_plots_multi(plots, nutrient_names, dose_min, dose_max, price_yield, price_fert)

    s_columns = [f"s_{n}" for n in nutrient_names]
    J = len(nutrient_names)
    price_fert = np.asarray(price_fert, dtype=float)
    bounds = [(dose_min, dose_max)] * J
    x0 = np.full(J, (dose_min + dose_max) / 2)

    doses = np.empty((len(plots), J))
    for i, row in enumerate(plots.itertuples()):
        s = np.array([getattr(row, col) for col in s_columns])
        baseline, R = row.baseline, row.R

        def neg_profit(d, baseline=baseline, R=R, s=s):
            return -profit_multi(baseline, R, s, d, price_yield, price_fert)

        result = minimize(neg_profit, x0, bounds=bounds, method="L-BFGS-B")
        doses[i] = np.clip(result.x, dose_min, dose_max)

    return doses
