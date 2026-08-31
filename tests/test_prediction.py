"""Тесты мультимодальной прогнозной модели (docs/chapter2/prediction_model.md, §2.3)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ds_prediction.explain import modality_importance  # noqa: E402
from ds_prediction.features import MODALITY_COLUMNS, modality_of, select_modalities  # noqa: E402
from ds_prediction.model import DEFAULT_MODALITIES, evaluate_grouped_cv, train_model  # noqa: E402
from ds_prediction.synthetic import generate_dataset  # noqa: E402


def test_modality_of_resolves_known_columns():
    assert modality_of("ndvi_integral") == "geo"
    assert modality_of("disease_events_count") == "img"
    assert modality_of("fertilizer_applications_count") == "text"
    assert modality_of("soil_nitrogen_mean") == "num"


def test_modality_of_rejects_unknown_column():
    with pytest.raises(KeyError):
        modality_of("not_a_real_column")


def test_select_modalities_returns_only_requested_columns():
    df = generate_dataset(n_fields=5, n_years=2)
    subset = select_modalities(df, ["geo", "text"])
    expected = set(MODALITY_COLUMNS["geo"]) | set(MODALITY_COLUMNS["text"])
    assert set(subset.columns) == expected


def test_generated_dataset_has_expected_shape_and_no_nans():
    df = generate_dataset(n_fields=10, n_years=4)
    assert len(df) == 40
    assert df.isna().sum().sum() == 0
    assert set(df["field_id"].unique()) == set(range(10))


def test_multimodal_beats_single_modality_on_grouped_cv():
    """Ключевой критерий §2.3.6: урожайность в синтетических данных зависит
    от всех четырёх модальностей -- удаление любой должно ухудшать RMSE."""
    df = generate_dataset()

    rmse_full = evaluate_grouped_cv(df, modalities=DEFAULT_MODALITIES)

    for modality in DEFAULT_MODALITIES:
        remaining = tuple(m for m in DEFAULT_MODALITIES if m != modality)
        rmse_without = evaluate_grouped_cv(df, modalities=remaining)
        assert rmse_full <= rmse_without + 1e-9, f"Полная модель не должна уступать модели без модальности {modality!r}"


def test_grouped_cv_prevents_leakage_across_years_of_same_field():
    """GroupKFold по field_id не должен ни разу поместить один и тот же
    field_id одновременно в train и test ни на одном разбиении (§2.3.5)."""
    from sklearn.model_selection import GroupKFold

    df = generate_dataset(n_fields=12, n_years=3)
    X = select_modalities(df, DEFAULT_MODALITIES)
    groups = df["field_id"].to_numpy()

    gkf = GroupKFold(n_splits=4)
    for train_idx, test_idx in gkf.split(X, df["yield"], groups):
        train_fields = set(groups[train_idx])
        test_fields = set(groups[test_idx])
        assert train_fields.isdisjoint(test_fields)


def test_shap_importance_ranks_dominant_modality_highest():
    """SHAP-агрегация по модальностям должна выявлять geo (самый сильный
    истинный коэффициент в synthetic.py) как главный фактор, а img (самый
    слабый) -- как наименее значимый (§2.3.4)."""
    df = generate_dataset()
    model = train_model(df, modalities=DEFAULT_MODALITIES)
    X = select_modalities(df, DEFAULT_MODALITIES)

    importance = modality_importance(model, X)

    assert importance["geo"] > importance["img"]
    assert importance["geo"] == max(importance.values())
