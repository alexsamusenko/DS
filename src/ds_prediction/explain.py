"""SHAP-интерпретация и агрегация вклада по модальностям, §2.3.4."""

import numpy as np
import shap

from .features import modality_of


def explain_predictions(model, X):
    """Значения Шепли для каждого признака каждого наблюдения."""
    explainer = shap.TreeExplainer(model)
    return explainer(X)


def modality_importance(model, X):
    """Средний |SHAP| по признакам, агрегированный по модальности (Phi_k, §2.3.4)."""
    shap_values = explain_predictions(model, X)
    mean_abs = np.abs(shap_values.values).mean(axis=0)

    importance = {}
    for column, value in zip(X.columns, mean_abs):
        modality = modality_of(column)
        importance[modality] = importance.get(modality, 0.0) + float(value)

    return importance
