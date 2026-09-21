"""Тесты комбинированного алгоритма предобработки (docs/chapter2/preprocessing_model.md, §2.2)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import ds_preprocessing.combine as combine_module  # noqa: E402
from ds_preprocessing.anomaly import detect_anomalies  # noqa: E402
from ds_preprocessing.baseline import naive_interpolation_baseline  # noqa: E402
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


def test_anomaly_detection_skips_series_with_too_few_observations():
    """При < 4 наблюдениях в ряде локальный тренд ненадёжен -- детектор должен
    молча пропустить точку (ни одной аномалии, без деления на ноль/ошибок),
    а не пытаться что-то определить по единственному-двум наблюдениям."""
    X = np.array([[1.0, 2.0, 100.0]])  # выброс есть, но наблюдений всего 3
    mask_observed = np.array([[True, True, True]])

    anomalies = detect_anomalies(X, mask_observed)

    assert not anomalies.any()


def test_spatial_estimate_handles_degenerate_constant_slice():
    """Вырожденная пространственная конфигурация: все наблюдения одного среза
    времени точно совпадают -- pykrige не может подобрать вариограмму и
    поднимает ValueError при оптимизации. spatial_estimate обязана не упасть,
    а оставить оценку этого среза как NaN (оценка попросту недоступна),
    делегируя решение вызывающему коду (fill_gaps -> запасной источник или
    'невосстановлено')."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])
    X = np.full((5, 3), 5.0)  # совершенно постоянное поле в каждом срезе
    mask_observed = np.ones((5, 3), dtype=bool)
    mask_observed[0, 1] = False  # один пропуск, который нужно оценить кригингом

    estimate, variance = spatial_estimate(coords, X, mask_observed)

    assert np.isnan(estimate[0, 1])
    assert np.isnan(variance[0, 1])


def test_fill_gaps_survives_degenerate_spatial_configuration():
    """Тот же вырожденный случай (константный срез) через полный конвейер
    fill_gaps: раньше необработанное исключение pykrige приводило к падению
    всего восстановления. Теперь пропуск для этой ячейки закрывается
    временной оценкой (она не зависит от пространственной конфигурации),
    либо явно помечается 'невосстановлено' -- но fill_gaps не падает."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 0.0]])
    times = np.arange(6, dtype=float)
    X = np.full((5, 6), 5.0)
    mask_observed = np.ones((5, 6), dtype=bool)
    mask_observed[0, 2] = False

    result = fill_gaps(coords, times, X, mask_observed)

    assert not result["unrestored"][0, 2]
    assert result["filled"][0, 2] == pytest.approx(5.0)


def test_fill_gaps_falls_back_to_spatial_only_when_temporal_unavailable():
    """Единственный момент времени -> временной тренд в принципе не строится
    (MIN_POINTS_FOR_TREND=2), но пространственных соседей достаточно --
    комбинированная оценка обязана выродиться в чисто пространственную,
    без NaN и без деления на нулевой вес (§2.2.5)."""
    coords, times, X_true = generate_field(grid_size=4, n_times=1)
    mask_observed = np.ones(X_true.shape, dtype=bool)
    mask_observed[0, 0] = False

    result = fill_gaps(coords, times, X_true, mask_observed)

    assert np.isnan(result["temporal_only"][0, 0])
    assert not np.isnan(result["spatial_only"][0, 0])
    assert not result["unrestored"][0, 0]
    assert result["filled"][0, 0] == pytest.approx(result["spatial_only"][0, 0])


def test_fill_gaps_marks_unrestored_when_neither_source_available():
    """Единственная точка (нет пространственных соседей) и единственное
    наблюдение в её ряде (нет временного тренда) -- ни одна из оценок
    недоступна, ячейка обязана быть явно помечена как невосстановленная, а
    не получить произвольное/нулевое значение (принцип §2.2.5 и реферата)."""
    coords = np.array([[0.0, 0.0]])
    times = np.array([0.0, 1.0])
    X = np.array([[1.0, 1.0]])
    mask_observed = np.array([[True, False]])

    result = fill_gaps(coords, times, X, mask_observed)

    assert result["unrestored"][0, 1]
    assert np.isnan(result["filled"][0, 1])


def test_fill_gaps_weight_guard_against_nonpositive_variance(monkeypatch):
    """Регрессионный тест на исправленный баг взвешивания по обратной
    дисперсии: если один из источников (гипотетически, из-за численной
    неустойчивости) дал дисперсию <= 0, деление на неё же раньше давало
    inf/inf = NaN в комбинированной оценке -- тихо испорченный результат
    вместо числа или явного 'невосстановлено'. Проверяем через monkeypatch
    (естественным путём pykrige/LOOCV нулевую дисперсию не выдают, см.
    test_spatial_estimate_recovers_smooth_field)."""
    coords = np.array([[0.0, 0.0], [1.0, 0.0]])
    times = np.array([0.0, 1.0])
    X = np.array([[1.0, 2.0], [3.0, 4.0]])
    mask_observed = np.array([[True, False], [True, True]])

    def fake_spatial(coords, X, mask):
        est = np.full(X.shape, np.nan)
        var = np.full(X.shape, np.nan)
        est[0, 1] = 10.0
        var[0, 1] = 0.0  # вырожденная нулевая дисперсия
        return est, var

    def fake_temporal(times, X, mask):
        est = np.full(X.shape, np.nan)
        var = np.full(X.shape, np.nan)
        est[0, 1] = 20.0
        var[0, 1] = 4.0
        return est, var

    monkeypatch.setattr(combine_module, "spatial_estimate", fake_spatial)
    monkeypatch.setattr(combine_module, "temporal_estimate", fake_temporal)

    result = fill_gaps(coords, times, X, mask_observed)

    # при var_s -> 0 вес пространственной оценки доминирует -- итог должен
    # быть конечным числом, близким к пространственной оценке, а не NaN
    assert np.isfinite(result["filled"][0, 1])
    assert result["filled"][0, 1] == pytest.approx(10.0, abs=1e-6)


def test_fill_gaps_drop_anomalies_false_keeps_outlier_in_fit():
    """drop_anomalies=False -- явно документированная альтернативная ветка
    (используется, например, для построения графика 'до устранения
    аномалий'): аномальные точки должны остаться среди наблюдаемых и
    участвовать в оценках, а не быть исключёнными."""
    coords, times, X_true = generate_field(grid_size=5, n_times=15)
    X_observed, mask_observed, _ = punch_holes(X_true, missing_fraction=0.1, n_outliers=5, seed=1)

    result_dropped = fill_gaps(coords, times, X_observed, mask_observed, drop_anomalies=True)
    result_kept = fill_gaps(coords, times, X_observed, mask_observed, drop_anomalies=False)

    assert result_dropped["anomalies"].sum() >= 1
    # аномалии всё равно детектируются одинаково (детекция не зависит от drop_anomalies)
    assert np.array_equal(result_dropped["anomalies"], result_kept["anomalies"])
    # но при drop_anomalies=False аномальные наблюдения остаются в filled как есть
    anomaly_cells = result_kept["anomalies"]
    assert np.array_equal(result_kept["filled"][anomaly_cells], X_observed[anomaly_cells])


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


def test_naive_baseline_recovers_linear_series():
    """На точно линейном ряду линейная интерполяция обязана быть почти точной --
    это минимальная проверка корректности реализации (baseline.py), отдельная
    от статистических свойств, которые проверяются на зашумлённых полях."""
    times = np.arange(10, dtype=float)
    true_row = 2.0 * times + 5.0  # y = 2t + 5
    X = np.tile(true_row, (2, 1))

    mask_observed = np.ones((2, 10), dtype=bool)
    # прячем несколько внутренних точек (не крайние -- иначе это уже экстраполяция)
    hidden = [2, 3, 6]
    mask_observed[0, hidden] = False

    estimate = naive_interpolation_baseline(times, X, mask_observed)

    assert np.allclose(estimate[0, hidden], true_row[hidden], atol=1e-9)
    # вторая строка полностью наблюдаема -- baseline не должен её трогать (NaN, не 0)
    assert np.all(np.isnan(estimate[1]))


def test_naive_baseline_row_without_observations_is_all_nan():
    """Если для точки m нет ни одного наблюдения, интерполяция невозможна в
    принципе -- вся строка должна остаться NaN, а не какое-то произвольное
    значение (0, среднее и т.п.)."""
    times = np.arange(5, dtype=float)
    X = np.zeros((2, 5))
    mask_observed = np.zeros((2, 5), dtype=bool)
    mask_observed[1] = True  # у второй точки наблюдения есть, у первой -- нет

    estimate = naive_interpolation_baseline(times, X, mask_observed)

    assert np.all(np.isnan(estimate[0]))
    # у второй строки все точки наблюдаемы -- восстанавливать нечего, тоже NaN
    assert np.all(np.isnan(estimate[1]))


def test_naive_baseline_holds_constant_with_single_observation():
    """С единственным наблюдением интерполяция вырождается в константу
    (поведение np.interp по умолчанию за пределами диапазона наблюдений) --
    документированное, но нетривиальное упрощение простой базовой линии."""
    times = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    X = np.zeros((1, 5))
    X[0, 2] = 7.0
    mask_observed = np.zeros((1, 5), dtype=bool)
    mask_observed[0, 2] = True  # единственное наблюдение -- в середине ряда

    estimate = naive_interpolation_baseline(times, X, mask_observed)

    missing = [0, 1, 3, 4]
    assert np.allclose(estimate[0, missing], 7.0)


def test_combined_beats_naive_baseline_per_trial():
    """Реферат заявляет, что комбинированный метод не уступает в точности
    использованию только пространственной или только временной информации.
    Отдельно проверяем и более сильное практическое сравнение -- с простой
    базовой линией (baseline.py), которая в §1.2.2 названа наиболее
    распространённой на практике: на контролируемых данных комбинированный
    метод обязан быть не хуже неё в каждом отдельном испытании, а не только
    в среднем по серии."""
    coords, times, X_true = generate_field()

    for seed in range(8):
        X_observed, mask_observed, mask_test = punch_holes(X_true, seed=seed)
        result = fill_gaps(coords, times, X_observed, mask_observed)
        naive_est = naive_interpolation_baseline(times, X_observed, mask_observed)

        combined_error = rmse(result["filled"][mask_test], X_true[mask_test])
        naive_error = rmse(naive_est[mask_test], X_true[mask_test])

        assert combined_error <= naive_error, (
            f"seed={seed}: комбинированный метод (RMSE={combined_error:.4f}) "
            f"хуже наивной базы (RMSE={naive_error:.4f})"
        )
