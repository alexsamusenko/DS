"""Функция прибыли участка, §2.4.1."""

from .response import yield_response


def profit(baseline, R, s, d, price_yield, price_fert):
    """Profit(d) = p_Y * Y(d) - p_D * d, на единицу площади."""
    return price_yield * yield_response(baseline, R, s, d) - price_fert * d
