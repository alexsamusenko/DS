"""Функция прибыли участка, §2.4.1."""

import numpy as np

from .response import yield_response, yield_response_multi


def profit(baseline, R, s, d, price_yield, price_fert):
    """Profit(d) = p_Y * Y(d) - p_D * d, на единицу площади."""
    return price_yield * yield_response(baseline, R, s, d) - price_fert * d


def profit_multi(baseline, R, s, d, price_yield, price_fert):
    """Многокомпонентная прибыль, §2.4.8: Profit(d) = p_Y*Y(d) - sum_j p_j*d_j.

    d, s : (J,) для одного участка или (K, J); price_fert : (J,) -- цена по видам удобрений.
    """
    d = np.asarray(d, dtype=float)
    price_fert = np.asarray(price_fert, dtype=float)
    fert_cost = np.sum(d * price_fert, axis=-1)
    return price_yield * yield_response_multi(baseline, R, s, d) - fert_cost
