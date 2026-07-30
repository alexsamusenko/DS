"""Таксономия модальностей и признаков, §2.3.2.

Каждая модальность -- это группа столбцов итогового вектора признаков
x_{m,y} = [phi_num ; phi_geo ; phi_img ; phi_text]. Голосовые записи не
образуют отдельную модальность (§2.1.4): после ASR они становятся текстом
и учтены внутри "text" через признаки, извлечённые NER-конвейером.
"""

MODALITY_COLUMNS = {
    "num": ["soil_moisture_mean", "soil_nitrogen_mean", "precip_sum", "temp_sum"],
    "geo": ["ndvi_peak", "ndvi_integral"],
    "img": ["disease_events_count", "max_risk_stage"],
    "text": ["fertilizer_applications_count", "total_dose"],
}

ALL_COLUMNS = [c for cols in MODALITY_COLUMNS.values() for c in cols]


def modality_of(column):
    """Определить, к какой модальности относится столбец признаков."""
    for modality, cols in MODALITY_COLUMNS.items():
        if column in cols:
            return modality
    raise KeyError(f"Столбец {column!r} не отнесён ни к одной модальности")


def select_modalities(df, modalities):
    """Выбрать из таблицы признаков только столбцы указанных модальностей."""
    columns = [c for m in modalities for c in MODALITY_COLUMNS[m]]
    return df[columns]
