"""Оптимизация дифференцированного внесения удобрений, §2.4.

docs/chapter2/optimization_model.md: кривая отклика на дозу (продолжение
модели L4 законом Митчерлиха), оптимизация по участкам без бюджета и с
бюджетным ограничением (метод Лагранжа), сравнение с равномерным внесением.
"""

from .economics import profit, profit_multi
from .optimize import optimize_unconstrained, optimize_unconstrained_multi, optimize_with_budget
from .response import yield_response, yield_response_multi

__all__ = [
    "yield_response",
    "yield_response_multi",
    "profit",
    "profit_multi",
    "optimize_unconstrained",
    "optimize_unconstrained_multi",
    "optimize_with_budget",
]
