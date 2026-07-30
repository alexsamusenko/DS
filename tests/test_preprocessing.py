"""Тесты комбинированного алгоритма предобработки (docs/chapter2_preprocessing_model.md, §2.2)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ds_preprocessing.anomaly import detect_anomalies  # noqa: E402
from ds_preprocessing.combine import fill_gaps  # noqa: E402
from ds_preprocessing.metrics import mae, rmse  # noqa: E402
from ds_preprocessing.spatial import spatial_estimate  # noqa: E402
from ds_preprocessing.synthetic import generate_field, punch_holes  # noqa: E402
from ds_preprocessing.temporal import temporal_estimate  # noqa: E402
from ds_preprocessing.validation import DataValidationError, validate_inputs  # noqa: E402


def test_metrics_rmse_mae():
    estimate = np.array([1.0, 2.0, 3.0, np.nan])
    truth = np.array([1.0, 2.0, 5.0, 10.0])
    # только первые три ячейки учитываются (последняя -- NaN в estimate)
    assert rmse(estimate, truth) == pytest.approx(np.sqrt((0 + 0 + 4) / 3))
    assert mae(estimate, truth) == pytest.approx((0 + 0 + 2) / 3)


def test_anomaly_detection_flags_injected_outlier():
    coords, times, X_true = generate_field(grid_size=5, n_times=15)
    X_observed, mask_observed, _ = punch_holes(X_true, missing_fraction=0.1, n_outliers=5, seed=1)

    anomalies = detect_anomalies(X_observed, mask_observed)
    # выбросы вносятся только в наблюдаемые ячейки -- аномалии должны быть подмножеством наблюдаемых
    assert not anomalies[~mask_observed].any()
    assert anomalies.sum() >= 1


def test_spatial_estimate_recovers_smooth_field():
    coords, times, X_true = generate_field(grid_size=6, n_times=10)
    X_observed, mask_observed, mask_test = punch_holes(X_true, missing_fraction=0.15, n_outliers=0, seed=2)

    estimate, variance = spatial_estimate(coords, X_observed, mask_observed)
    error = rmse(estimate[mask_test], X_true[mask_test])

    assert error < 0.6
    assert np.all(variance[mask_test] >= 0)


def test_temporal_estimate_recovers_smooth_series():
    coords, times, X_true = generate_field(grid_size=6, n_times=20)
    X_observed, mask_observed, mask_test = punch_holes(X_true, missing_fraction=0.15, n_outliers=0, seed=3)

    estimate, variance = temporal_estimate(times, X_observed, mask_observed)
    error = rmse(estimate[mask_test], X_true[mask_test])

    assert error < 0.6
    assert np.all(variance[mask_test] >= 1e-4)


def test_combined_beats_either_alone_on_average():
    """Ключевой критерий §2.2.6: в среднем по серии испытаний комбинированная
    оценка не должна уступать оценке по одному источнику -- как в Ли и др. (2021)."""
    coords, times, X_true = generate_field()

    rmses = {"spatial": [], "temporal": [], "combined": []}
    for seed in range(8):
        X_observed, mask_observed, mask_test = punch_holes(X_true, seed=seed)
        result = fill_gaps(coords, times, X_observed, mask_observed)

        rmses["spatial"].append(rmse(result["spatial_only"][mask_test], X_true[mask_test]))
        rmses["temporal"].append(rmse(result["temporal_only"][mask_test], X_true[mask_test]))
        rmses["combined"].append(rmse(result["filled"][mask_test], X_true[mask_test]))

    mean_rmse = {k: float(np.mean(v)) for k, v in rmses.items()}
    assert mean_rmse["combined"] <= min(mean_rmse["spatial"], mean_rmse["temporal"]) + 1e-9


def test_degenerate_cases_fall_back_to_single_source():
    """Одна точка (нет пространственных соседей) -> только временная оценка;
    один момент времени (нет истории) -> только пространственная (§2.2.5)."""
    coords, times, X_true = generate_field(grid_size=1, n_times=10)
    X_observed, mask_observed, mask_test = punch_holes(X_true, missing_fraction=0.3, n_outliers=0, seed=4)
    result = fill_gaps(coords, times, X_observed, mask_observed)
    assert np.all(np.isnan(result["spatial_only"][mask_test]))
    assert not np.any(np.isnan(result["filled"][mask_test]))  # временная оценка их закрыла


def test_validate_inputs_rejects_shape_mismatch():
    coords = np.zeros((4, 2))
    times = np.zeros(5)
    X = np.zeros((4, 6))  # неверная форма: должно быть (4, 5)
    mask = np.ones((4, 6), dtype=bool)
    with pytest.raises(DataValidationError, match="X должен иметь форму"):
        validate_inputs(coords, times, X, mask)


def test_validate_inputs_rejects_nan_in_observed_cells():
    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    times = np.array([0.0, 1.0, 2.0])
    X = np.array([[1.0, np.nan, 3.0], [1.0, 2.0, 3.0]])
    mask = np.ones((2, 3), dtype=bool)
    with pytest.raises(DataValidationError, match="NaN"):
        validate_inputs(coords, times, X, mask)


def test_validate_inputs_rejects_duplicate_coordinates():
    coords = np.array([[0.0, 0.0], [0.0, 0.0]])
    times = np.array([0.0, 1.0])
    X = np.zeros((2, 2))
    mask = np.ones((2, 2), dtype=bool)
    with pytest.raises(DataValidationError, match="повторяющиеся координаты"):
        validate_inputs(coords, times, X, mask)


def test_validate_inputs_rejects_non_boolean_mask():
    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    times = np.array([0.0, 1.0])
    X = np.zeros((2, 2))
    mask = np.ones((2, 2), dtype=int)  # не bool
    with pytest.raises(DataValidationError, match="булевой маской"):
        validate_inputs(coords, times, X, mask)


def test_validate_inputs_rejects_all_missing():
    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    times = np.array([0.0, 1.0])
    X = np.zeros((2, 2))
    mask = np.zeros((2, 2), dtype=bool)  # нет ни одного наблюдения
    with pytest.raises(DataValidationError, match="не содержит ни одного наблюдаемого"):
        validate_inputs(coords, times, X, mask)


def test_fill_gaps_raises_on_bad_input():
    coords = np.array([[0.0, 0.0]])
    times = np.array([0.0, 1.0])
    X = np.zeros((2, 2))  # M=1 по coords, но X подразумевает M=2 -- несогласованно
    mask = np.ones((2, 2), dtype=bool)
    with pytest.raises(DataValidationError):
        fill_gaps(coords, times, X, mask)
