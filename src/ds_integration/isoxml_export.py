"""psi_ISOXML: результат L5 (участок -> доза) -> ISO 11783-10 Task Data.

Структура документа (элементы TASKDATA/CTR/FRM/PFD/TSK/TZN/PDV, атрибуты
A/B/C/D..., идентификаторы вида "PFD1") соответствует открыто
документированной схеме стандарта ISO 11783-10 (Task Data). Числовой код DDI
(Data Dictionary Identifier, ISO 11783-11) для показателя "доза внесения"
ниже -- "0006", по общепринятому в литературе по ISOBUS примеру для setpoint
application rate -- требует финальной сверки с официальным реестром AEF
ISOBUS DDI перед использованием с реальной техникой конкретного
производителя (docs/governance/licenses.md, раздел "ISO 11783 / DDI").
"""

import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .geometry import plot_square_polygon

DDI_APPLICATION_RATE = "0006"  # см. предупреждение в докстринге модуля


def _polygon_element(parent: ET.Element, polygon: list[tuple[float, float]], polygon_type: str = "1") -> None:
    pln = ET.SubElement(parent, "PLN", A=polygon_type)
    lsg = ET.SubElement(pln, "LSG", A="1")
    for lat, lon in polygon:
        ET.SubElement(lsg, "PNT", A="2", C=f"{lat:.7f}", D=f"{lon:.7f}")


def export_task_data(
    plots_df: pd.DataFrame,
    doses: np.ndarray,
    *,
    customer_name: str = "Демо-хозяйство",
    farm_name: str = "Демо-поле",
    task_designator: str = "Карта-задание (демо)",
) -> str:
    """Построить TASKDATA.XML для участков `plots_df` (колонки plot_id, area) с
    дозами `doses` (тот же порядок и длина, что и строки `plots_df`)."""
    if len(plots_df) != len(doses):
        raise ValueError(f"Число участков ({len(plots_df)}) не совпадает с числом доз ({len(doses)})")

    duplicated_ids = plots_df["plot_id"][plots_df["plot_id"].duplicated()].unique().tolist()
    if duplicated_ids:
        # PFD/TZN получают идентификаторы вида "PFD{plot_id}"/"TZN{plot_id}" --
        # при повторяющихся plot_id документ получил бы неуникальные ID,
        # что нарушает требование ISO 11783-10 к уникальности идентификаторов.
        raise ValueError(f"Повторяющиеся plot_id во входных данных: {duplicated_ids}")

    root = ET.Element(
        "ISO11783_TaskData",
        VersionMajor="4",
        VersionMinor="3",
        ManagementSoftwareManufacturer="DS",
        ManagementSoftwareVersion="0.1.0",
        DataTransferOrigin="1",
    )

    ET.SubElement(root, "CTR", A="CTR1", B=customer_name)
    ET.SubElement(root, "FRM", A="FRM1", B=farm_name, I="CTR1")
    ET.SubElement(root, "PDT", A="PDT1", B="Удобрение (общее)")

    tsk = ET.SubElement(root, "TSK", A="TSK1", B=task_designator, C="1", F="FRM1")

    for i, (_, plot) in enumerate(plots_df.reset_index(drop=True).iterrows()):
        plot_id = int(plot["plot_id"])
        area_ha = float(plot["area"])
        dose = float(doses[i])
        polygon = plot_square_polygon(plot_id, area_ha)

        pfd = ET.SubElement(root, "PFD", A=f"PFD{plot_id}", B=f"Участок {plot_id}", C=f"{area_ha:.4f}", I="CTR1")
        _polygon_element(pfd, polygon)

        tzn = ET.SubElement(tsk, "TZN", A=f"TZN{plot_id}", B=f"Зона {plot_id}", D=f"PFD{plot_id}")
        # C -- ProductIdRef (не D, который в ISO 11783-10 -- DeviceElementIdRef
        # и не должен указывать на продукт).
        ET.SubElement(tzn, "PDV", A=DDI_APPLICATION_RATE, B=f"{dose:.3f}", C="PDT1")
        _polygon_element(tzn, polygon, polygon_type="4")  # 4 = Treatment Zone, ISO 11783-10

    ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + ET.tostring(root, encoding="unicode")
