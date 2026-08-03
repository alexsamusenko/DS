"""Демонстрация: экспорт результата L5 в ISO 11783-10 Task Data (§2.6.3).

Запуск: PYTHONPATH=src python3 -m ds_integration.build_demo
"""

from pathlib import Path

from ds_integration.isoxml_export import export_task_data
from ds_optimization.optimize import optimize_with_budget
from ds_optimization.synthetic import generate_plots

PRICE_YIELD = 1300.0
PRICE_FERT = 50.0
DOSE_MIN, DOSE_MAX = 0.0, 150.0


def run_demo():
    plots = generate_plots(n_plots=6)
    total_area = float(plots["area"].sum())
    budget = 70.0 * total_area

    doses = optimize_with_budget(plots, budget, DOSE_MIN, DOSE_MAX, PRICE_YIELD, PRICE_FERT)
    xml_text = export_task_data(plots, doses)

    output_dir = Path("build/integration")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "TASKDATA.xml"
    output_path.write_text(xml_text, encoding="utf-8")

    print("Экспорт карты-задания в ISO 11783-10 Task Data (§2.6.3)")
    print(f"Участков: {len(plots)}, суммарный бюджет: {budget:.1f} кг")
    print(f"Сохранено: {output_path} ({len(xml_text)} байт)")
    print()
    print(xml_text[:800] + ("..." if len(xml_text) > 800 else ""))

    return xml_text


if __name__ == "__main__":
    run_demo()
