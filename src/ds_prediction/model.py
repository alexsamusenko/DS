"""Ансамблевая модель и групповая кросс-валидация по полю, §2.3.3, §2.3.5."""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from .features import select_modalities

DEFAULT_MODALITIES = ("num", "geo", "img", "text")


def make_model(random_state=0):
    return GradientBoostingRegressor(random_state=random_state, n_estimators=150, max_depth=3, learning_rate=0.05)


def train_model(df, modalities=DEFAULT_MODALITIES, random_state=0):
    """Обучить модель F(x) = Y_hat на всех переданных данных (§2.3.3)."""
    X = select_modalities(df, modalities)
    y = df["yield"].to_numpy()
    model = make_model(random_state=random_state)
    model.fit(X, y)
    return model


def evaluate_grouped_cv(df, modalities=DEFAULT_MODALITIES, n_splits=5, random_state=0):
    """RMSE по GroupKFold с группой field_id (§2.3.5).

    Именно группировка по полю, а не случайное разбиение, проверяет
    обобщение модели на НЕвиденные поля -- см. обоснование в §2.3.5.
    """
    X = select_modalities(df, modalities)
    y = df["yield"].to_numpy()
    groups = df["field_id"].to_numpy()

    gkf = GroupKFold(n_splits=n_splits)
    errors = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        model = make_model(random_state=random_state)
        model.fit(X.iloc[train_idx], y[train_idx])
        pred = model.predict(X.iloc[test_idx])
        errors.append(np.sqrt(np.mean((pred - y[test_idx]) ** 2)))

    return float(np.mean(errors))
