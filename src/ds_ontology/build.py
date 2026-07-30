"""Демонстрационная сборка онтологии: TBox (build_schema) + пример ABox.

Показывает работу mu = phi (union) lambda (integration.py) на игрушечном
наборе разнородных фактов об одном поле -- сенсор, метео, агроприём,
текстовое упоминание вредителя -- и сохраняет результат в build/agro_demo.owl.

Запуск: PYTHONPATH=src python3 -m ds_ontology.build
"""

from pathlib import Path

from ds_ontology.integration import (
    assert_agro_priem,
    assert_soil_reading,
    assert_weather_event,
    get_or_create_istochnik,
    get_or_create_pole,
    resolve_entity_mention,
)
from ds_ontology.schema import build_schema

OUTPUT_PATH = Path(__file__).resolve().parents[2] / "build" / "agro_demo.owl"


def run_demo():
    onto = build_schema()

    pole = get_or_create_pole(onto, pole_id="7", geometriya="POLYGON((0 0,0 1,1 1,1 0,0 0))", ploschad=54.3, region="Ростовская область")

    kultura = onto.Kultura()
    kultura.nazvanie_kultury = "Пшеница озимая"
    kultura.sort = "Ермак"
    kultura.fenofaza = "кущение"
    kultura.regionalnoe_nazvanie.append("озимка")
    pole.vyraschivaetsya.append(kultura)

    sensor_source = get_or_create_istochnik(onto, tip="IoT-датчик почвы", format_istochnika="MQTT/JSON", dostovernost=0.95)
    assert_soil_reading(onto, pole, tip_pokazatelya="влажность почвы", znachenie=27.4, data_izmereniya="2026-05-14", edinitsa_izmereniya="%", istochnik=sensor_source)

    lab_source = get_or_create_istochnik(onto, tip="Лабораторный анализ", format_istochnika="XLSX", dostovernost=0.99)
    assert_soil_reading(onto, pole, tip_pokazatelya="содержание азота", znachenie=18.2, data_izmereniya="2026-04-02", edinitsa_izmereniya="мг/кг", istochnik=lab_source)

    assert_weather_event(onto, pole, tip_sobytiya="заморозок", intensivnost=-3.5, period="2026-04-10")

    journal_source = get_or_create_istochnik(onto, tip="Журнал полевых работ", format_istochnika="текст", dostovernost=0.8)
    assert_agro_priem(onto, pole, tip_operatsii="внесение азотных удобрений", doza=120.0, data_priema="2026-04-15", istochnik=journal_source)

    # lambda: текстовое упоминание "озимка" должно разрешиться в уже созданный
    # экземпляр Kultura через regionalnoe_nazvanie (§2.1.5), а не породить дубликат
    matched = resolve_entity_mention(onto, "озимка", "Kultura")
    print("resolve_entity_mention('озимка') ->", matched, "(ожидается тот же экземпляр, что и Пшеница озимая)")
    assert matched is kultura, "lambda не нашла существующий экземпляр по синониму"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    onto.save(file=str(OUTPUT_PATH), format="rdfxml")
    print(f"Онтология сохранена: {OUTPUT_PATH}")
    print("Классы (C):", sorted(c.name for c in onto.classes()))
    print("Экземпляры (I):", sorted(i.name for i in onto.individuals()))

    return onto


if __name__ == "__main__":
    run_demo()
