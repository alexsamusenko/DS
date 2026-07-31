"""Мультимодальная модель прогнозирования урожайности, §2.3.

docs/chapter2/prediction_model.md: конструирование признаков по модальностям
(включая фото и голосовые записи, сведённые к тексту через ASR), ансамблевая
модель, SHAP-интерпретация по модальностям, GroupKFold-валидация по полю.
"""

from .explain import explain_predictions, modality_importance
from .features import MODALITY_COLUMNS, select_modalities
from .model import evaluate_grouped_cv, train_model

__all__ = [
    "MODALITY_COLUMNS",
    "select_modalities",
    "train_model",
    "evaluate_grouped_cv",
    "explain_predictions",
    "modality_importance",
]
