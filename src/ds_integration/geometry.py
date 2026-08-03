"""Синтетическая геометрия участков для демонстрации экспорта в ISO-XML (§2.6.3).

В системе нет источника реальных границ полей (кадастр/ГИС хозяйства не
подключены -- задача 6, docs/datasets.md), поэтому полигон каждого участка
строится как квадрат регулярной сетки со стороной, производной от площади --
это явное упрощение для демонстрации структуры документа, не попытка выдать
синтетику за реальные границы. Перед использованием с реальной техникой
геометрия должна браться из фактических границ полей хозяйства.
"""

import math

# Точка отсчёта сетки -- ориентировочный центр аграрного региона (Краснодарский
# край), выбрана только для правдоподобных координат в демонстрационном выводе.
BASE_LAT = 45.0354
BASE_LON = 39.0214
KM_PER_DEGREE_LAT = 111.0
CELL_SPACING_KM = 1.0  # шаг сетки между центрами участков, км


def _meters_per_degree_lon(lat_deg: float) -> float:
    return KM_PER_DEGREE_LAT * math.cos(math.radians(lat_deg))


def plot_square_polygon(index: int, area_ha: float, grid_cols: int = 5) -> list[tuple[float, float]]:
    """Вернуть замкнутый полигон (список (lat, lon), первая точка = последняя)
    для участка `index` площадью `area_ha`, размещённого на регулярной сетке."""
    row, col = divmod(index, grid_cols)
    center_lat = BASE_LAT + row * (CELL_SPACING_KM / KM_PER_DEGREE_LAT)
    center_lon = BASE_LON + col * (CELL_SPACING_KM / _meters_per_degree_lon(BASE_LAT))

    side_km = math.sqrt(area_ha * 0.01)  # 1 га = 0.01 км^2
    half_lat = (side_km / 2) / KM_PER_DEGREE_LAT
    half_lon = (side_km / 2) / _meters_per_degree_lon(center_lat)

    corners = [
        (center_lat - half_lat, center_lon - half_lon),
        (center_lat - half_lat, center_lon + half_lon),
        (center_lat + half_lat, center_lon + half_lon),
        (center_lat + half_lat, center_lon - half_lon),
    ]
    return corners + [corners[0]]
