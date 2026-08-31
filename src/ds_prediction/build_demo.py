"""Демонстрация: мультимодальный прогноз против одномодальных + SHAP по модальностям.

Запуск: PYTHONPATH=src python3 -m ds_prediction.build_demo
"""

from ds_prediction.explain import modality_importance
from ds_prediction.features import select_modalities
from ds_prediction.model import DEFAULT_MODALITIES, evaluate_grouped_cv, train_model
from ds_prediction.synthetic import generate_dataset


def run_demo():
    df = generate_dataset()

    print("Прогноз урожайности -- сравнение по модальностям (§2.3.6)")
    print(f"{'Набор модальностей':<30}{'RMSE (GroupKFold по полю)':>28}")

    rmse_full = evaluate_grouped_cv(df, modalities=DEFAULT_MODALITIES)
    print(f"{'все модальности':<30}{rmse_full:>28.3f}")

    rmse_without = {}
    for m in DEFAULT_MODALITIES:
        remaining = tuple(x for x in DEFAULT_MODALITIES if x != m)
        rmse_without[m] = evaluate_grouped_cv(df, modalities=remaining)
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

    return rmse_full, rmse_without, importance


if __name__ == "__main__":
    run_demo()
