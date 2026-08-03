"""Тесты L7 -- экспорт результата L5 в ISO 11783-10 Task Data (docs/chapter2/integration_model.md, §2.6)."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ds_integration.geometry import plot_square_polygon  # noqa: E402
from ds_integration.isoxml_export import export_task_data  # noqa: E402
from ds_optimization.synthetic import generate_plots  # noqa: E402


def test_export_produces_well_formed_xml():
    plots = generate_plots(n_plots=4)
    doses = np.array([10.0, 20.0, 30.0, 40.0])

    xml_text = export_task_data(plots, doses)

    root = ET.fromstring(xml_text)
    assert root.tag == "ISO11783_TaskData"


def test_export_has_one_zone_and_field_per_plot():
    plots = generate_plots(n_plots=5)
    doses = np.arange(5, dtype=float) * 10.0

    root = ET.fromstring(export_task_data(plots, doses))

    assert len(root.findall("PFD")) == 5
    assert len(root.findall("TSK/TZN")) == 5


def test_export_dose_values_match_input():
    plots = generate_plots(n_plots=3)
    doses = np.array([12.5, 47.0, 99.9])

    root = ET.fromstring(export_task_data(plots, doses))

    exported_doses = [float(pdv.get("B")) for pdv in root.findall("TSK/TZN/PDV")]
    assert exported_doses == [12.5, 47.0, 99.9]


def test_export_rejects_mismatched_lengths():
    plots = generate_plots(n_plots=3)
    doses = np.array([1.0, 2.0])

    try:
        export_task_data(plots, doses)
        assert False, "ожидалась ValueError при несовпадении числа участков и доз"
    except ValueError:
        pass


def test_export_customer_farm_task_names_are_applied():
    plots = generate_plots(n_plots=1)
    doses = np.array([5.0])

    root = ET.fromstring(export_task_data(plots, doses, customer_name="ООО Тест", farm_name="Поле 1", task_designator="Задание А"))

    assert root.find("CTR").get("B") == "ООО Тест"
    assert root.find("FRM").get("B") == "Поле 1"
    assert root.find("TSK").get("B") == "Задание А"


def test_polygon_is_closed_and_has_four_corners():
    polygon = plot_square_polygon(index=0, area_ha=1.0)

    assert len(polygon) == 5  # 4 угла + повтор первой точки, чтобы полигон был замкнут
    assert polygon[0] == polygon[-1]


def test_polygon_area_scales_with_input():
    small = plot_square_polygon(index=0, area_ha=0.5)
    large = plot_square_polygon(index=0, area_ha=4.0)

    def side_length(polygon):
        lat0, lon0 = polygon[0]
        lat1, lon1 = polygon[1]
        return abs(lat1 - lat0) + abs(lon1 - lon0)

    assert side_length(large) > side_length(small)
