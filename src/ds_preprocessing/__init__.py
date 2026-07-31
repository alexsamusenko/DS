"""Комбинированный алгоритм предобработки пространственно-временных данных.

Реализует §2.2 docs/chapter2/preprocessing_model.md: детекцию аномалий,
пространственную (кригинг) и временную (тренд) оценки, их комбинирование
по обратной дисперсии.
"""

from .combine import fill_gaps
from .metrics import mae, rmse
from .validation import DataValidationError

__all__ = ["fill_gaps", "rmse", "mae", "DataValidationError"]
