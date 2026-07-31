"""Оптимизация дифференцированного внесения удобрений, §2.4.

docs/chapter2/optimization_model.md: кривая отклика на дозу (продолжение
модели L4 законом Митчерлиха), оптимизация по участкам без бюджета и с
бюджетным ограничением (метод Лагранжа), сравнение с равномерным внесением.
"""

from .economics import profit
from .optimize import optimize_unconstrained, optimize_with_budget
from .response import yield_response

__all__ = ["yield_response", "profit", "optimize_unconstrained", "optimize_with_budget"]
