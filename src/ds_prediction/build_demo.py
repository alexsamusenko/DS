"""Демонстрация: мультимодальный прогноз против одномодальных + простой
базовой линии (§10.4) + SHAP по модальностям + оценка разброса по нескольким
независимым перегенерациям данных (§10.6) + вычислительная стоимость (§10.7).

Запуск: PYTHONPATH=src python3 -m ds_prediction.build_demo
"""

import time

import numpy as np

from ds_prediction.explain import modality_importance
from ds_prediction.features import select_modalities
from ds_prediction.model import (
    DEFAULT_MODALITIES,
    evaluate_grouped_cv,
    make_baseline_model,
    make_model,
    train_model,
)
from ds_prediction.synthetic import generate_dataset


def run_demo():
    df = generate_dataset()

    print("Прогноз урожайности -- сравнение по модальностям (§2.3.6)")
    print(f"{'Набор модальностей':<30}{'RMSE (GroupKFold по полю)':>28}")

    rmse_full = evaluate_grouped_cv(df, modalities=DEFAULT_MODALITIES, model_factory=make_model)
    print(f"{'все модальности (GBM)':<30}{rmse_full:>28.3f}")

    rmse_baseline = evaluate_grouped_cv(df, modalities=DEFAULT_MODALITIES, model_factory=make_baseline_model)
    print(f"{'простая база (лин. регрессия)':<30}{rmse_baseline:>28.3f}")

    rmse_without = {}
    for m in DEFAULT_MODALITIES:
        remaining = tuple(x for x in DEFAULT_MODALITIES if x != m)
        rmse_without[m] = evaluate_grouped_cv(df, modalities=remaining, model_factory=make_model)
        print(f"{'без ' + m:<30}{rmse_without[m]:>28.3f}")

    print()
    model = train_model(df, modalities=DEFAULT_MODALITIES)
    X = select_modalities(df, DEFAULT_MODALITIES)
    importance = modality_importance(model, X)

    print("Вклад модальностей в прогноз (агрегированный |SHAP|, §2.3.4)")
    for modality, value in sorted(importance.items(), key=lambda kv: -kv[1]):
        print(f"  {modality:<10}{value:>10.3f}")

    for m in DEFAULT_MODALITIES:
        assert rmse_full <= rmse_without[m] + 1e-9, f"Мультимодальная модель не должна уступать модели без модальности {m!r} (§2.3.6)"
    # Внимание: превосходство ансамбля над линейной базовой линией НЕ проверяется
    # утверждением (assert) -- в отличие от свойства ablation выше, оно не следует
    # из построения алгоритма и является эмпирическим фактом, который на данном
    # синтетическом наборе (почти линейный по построению generate_dataset,
    # §2.3.7) НЕ выполняется: rmse_baseline < rmse_full. Это честно
    # зафиксировано и обсуждается в главе 3 (§3.3), а не скрывается.

    return rmse_full, rmse_baseline, rmse_without, importance


def run_variability(n_repeats=5, seeds=(11, 12, 13, 14, 15)):
    """Оценить разброс RMSE по нескольким независимым перегенерациям набора
    данных (разные seed generate_dataset), а не по одной фиксированной
    выборке -- статистическая корректность §10.6: точечная оценка без
    указания разброса недостаточна для содержательного вывода о превосходстве
    метода.
    """
    assert len(seeds) == n_repeats
    full_scores, baseline_scores = [], []

    for seed in seeds:
        df = generate_dataset(seed=seed)
        full_scores.append(evaluate_grouped_cv(df, modalities=DEFAULT_MODALITIES, model_factory=make_model))
        baseline_scores.append(evaluate_grouped_cv(df, modalities=DEFAULT_MODALITIES, model_factory=make_baseline_model))

    full_scores = np.array(full_scores)
    baseline_scores = np.array(baseline_scores)

    print(f"\nРазброс RMSE по {n_repeats} независимым перегенерациям данных (seeds={seeds})")
    print(f"{'Модель':<30}{'Среднее RMSE':>14}{'Станд. откл.':>16}")
    print(f"{'ансамбль (GBM)':<30}{full_scores.mean():>14.3f}{full_scores.std(ddof=1):>16.3f}")
    print(f"{'база (лин. регрессия)':<30}{baseline_scores.mean():>14.3f}{baseline_scores.std(ddof=1):>16.3f}")

    return {"gbm": full_scores, "baseline": baseline_scores}


def run_cost(n_repeats_train=5, n_predict_calls=200):
    """Измерить вычислительную стоимость модели L4 (§10.7): время обучения
    на полном наборе и латентность единичного прогноза -- на процессоре,
    без GPU, без специальной оптимизации инференса (ONNX и т.п.), поскольку
    именно так модель обслуживает запросы уровня L6 в текущей реализации.
    """
    df = generate_dataset()
    X = select_modalities(df, DEFAULT_MODALITIES)
    y = df["yield"].to_numpy()

    train_times = []
    for _ in range(n_repeats_train):
        model = make_model()
        t0 = time.perf_counter()
        model.fit(X, y)
        train_times.append(time.perf_counter() - t0)

    model = make_model()
    model.fit(X, y)
    single_row = X.iloc[[0]]

    # прогрев (JIT/кеш) перед измерением, не входит в статистику
    for _ in range(10):
        model.predict(single_row)

    predict_times = []
    for _ in range(n_predict_calls):
        t0 = time.perf_counter()
        model.predict(single_row)
        predict_times.append(time.perf_counter() - t0)

    predict_times_ms = np.array(predict_times) * 1000
    train_times = np.array(train_times)

    print(f"\nВычислительная стоимость модели L4 (§10.7), {n_repeats_train} обучений, {n_predict_calls} прогнозов, CPU без GPU")
    print(f"Обучение на {len(X)} наблюдениях: {train_times.mean() * 1000:.1f} ± {train_times.std(ddof=1) * 1000:.1f} мс")
    print(f"Единичный прогноз: {np.mean(predict_times_ms):.2f} мс (p50={np.percentile(predict_times_ms, 50):.2f}, p95={np.percentile(predict_times_ms, 95):.2f})")
    print(f"Пропускная способность (последовательные единичные вызовы): {1000 / np.mean(predict_times_ms):.0f} прогнозов/с")

    return {"train_ms": train_times * 1000, "predict_ms": predict_times_ms}


def run_sensitivity_sample_size(sizes=(15, 30, 60, 120, 240), n_years=5, seed=11):
    """Анализ чувствительности (§10.5): как соотношение RMSE ансамбля и
    простой базовой линии (§3.3.2) меняется с ростом объёма обучающих
    данных -- проверка гипотезы о том, что отставание ансамбля на 150
    наблюдениях объясняется малым размером выборки, а не структурным
    недостатком модели.
    """
    rows = []
    for n_fields in sizes:
        df = generate_dataset(n_fields=n_fields, n_years=n_years, seed=seed)
        n_splits = min(5, n_fields)
        gbm = evaluate_grouped_cv(df, modalities=DEFAULT_MODALITIES, model_factory=make_model, n_splits=n_splits)
        base = evaluate_grouped_cv(df, modalities=DEFAULT_MODALITIES, model_factory=make_baseline_model, n_splits=n_splits)
        rows.append({"n_fields": n_fields, "n_obs": n_fields * n_years, "gbm": gbm, "baseline": base, "ratio": gbm / base})

    print(f"\nЧувствительность соотношения RMSE(GBM)/RMSE(база) к объёму данных (§10.5)")
    print(f"{'n_fields':>8}{'n_obs':>8}{'GBM':>10}{'база':>10}{'GBM/база':>12}")
    for r in rows:
        print(f"{r['n_fields']:>8}{r['n_obs']:>8}{r['gbm']:>10.3f}{r['baseline']:>10.3f}{r['ratio']:>12.3f}")

    return rows


if __name__ == "__main__":
    run_demo()
    run_variability()
    run_cost()
    run_sensitivity_sample_size()
