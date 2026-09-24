"""DS ontology package.

Реализует формальную модель O_t = <C, R_O, R_D, Ax, I, tau>
динамической предметной онтологии агрономических данных.
"""

from .schema import build_schema

__all__ = ["build_schema"]
