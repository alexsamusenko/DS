"""Ансамблевая модель и групповая кросс-валидация по полю, §2.3.3, §2.3.5."""

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from .features import select_modalities

DEFAULT_MODALITIES = ("num", "geo", "img", "text")


def make_model(random_state=0):
    return GradientBoostingRegressor(random_state=random_state, n_estimators=150, max_depth=3, learning_rate=0.05)


def make_baseline_model(random_state=0):
    """Простая базовая линия (§10.4 спецификации): обычная линейная регрессия
    по тем же признакам, без учёта взаимодействий и нелинейностей -- точка
    сравнения для содержательной оценки прироста от ансамблевой модели, а не
    только внутреннего сравнения вариантов одной и той же модели (ablation).
    random_state не используется (OLS детерминирован), аргумент сохранён для
    единообразия сигнатуры с make_model."""
    return LinearRegression()


def train_model(df, modalities=DEFAULT_MODALITIES, random_state=0, model_factory=make_model):
    """Обучить модель F(x) = Y_hat на всех переданных данных (§2.3.3)."""
    X = select_modalities(df, modalities)
    y = df["yield"].to_numpy()
    model = model_factory(random_state=random_state)
    model.fit(X, y)
    return model


def evaluate_grouped_cv(df, modalities=DEFAULT_MODALITIES, n_splits=5, random_state=0, model_factory=make_model):
    """RMSE по GroupKFold с группой field_id (§2.3.5).

    Именно группировка по полю, а не случайное разбиение, проверяет
    обобщение модели на НЕвиденные поля -- см. обоснование в §2.3.5.
    model_factory позволяет прогнать ту же схему валидации для другой модели
    (см. make_baseline_model) -- используется для честного сравнения с
    простой базовой линией на идентичных разбиениях (§10.4).
    """
    X = select_modalities(df, modalities)
    y = df["yield"].to_numpy()
    groups = df["field_id"].to_numpy()

    gkf = GroupKFold(n_splits=n_splits)
    errors = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        model = model_factory(random_state=random_state)
        model.fit(X.iloc[train_idx], y[train_idx])
        pred = model.predict(X.iloc[test_idx])
        errors.append(np.sqrt(np.mean((pred - y[test_idx]) ** 2)))

    return float(np.mean(errors))
