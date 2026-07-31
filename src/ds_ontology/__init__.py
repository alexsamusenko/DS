"""DS ontology package.

Реализует формальную модель O_t = <C, R_O, R_D, Ax, I, tau>,
описанную в docs/chapter2/ontology_model.md (§2.1).
"""

from .schema import build_schema

__all__ = ["build_schema"]
