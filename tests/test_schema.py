"""Smoke-тесты для схемы онтологии (docs/chapter2/ontology_model.md, §2.1)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ds_ontology.integration import (  # noqa: E402
    assert_agro_priem,
    assert_soil_reading,
    assert_weather_event,
    get_or_create_istochnik,
    get_or_create_pole,
    resolve_entity_mention,
)
from ds_ontology.schema import build_schema  # noqa: E402

EXPECTED_CLASSES = {
    "Pole",
    "Kultura",
    "PokazatelPochvy",
    "PogodnoeSobytie",
    "AgroPriem",
    "VreditelBolezn",
    "Istochnik",
}

EXPECTED_OBJECT_PROPERTIES = {
    "vyraschivaetsya",
    "harakterizuetsya",
    "podverzheno",
    "obrabatyvaetsya",
    "podverzhena",
    "zafiksirovan_v",
    "opisan_v",
}


def test_all_classes_present():
    onto = build_schema()
    assert {c.name for c in onto.classes()} == EXPECTED_CLASSES


def test_all_object_properties_present():
    onto = build_schema()
    assert {p.name for p in onto.object_properties()} == EXPECTED_OBJECT_PROPERTIES


def test_classes_are_pairwise_disjoint():
    onto = build_schema()
    # AllDisjoint(C) должна быть зарегистрирована в TBox как аксиома Ax
    # (фактическую непротиворечивость для конкретных индивидов проверяет ризонер -- см. test_reasoner_finds_no_inconsistency)
    assert any(onto.Pole in d.entities and onto.Kultura in d.entities for d in onto.disjoint_classes())


def test_functional_property_keeps_single_value():
    onto = build_schema()
    pole = get_or_create_pole(onto, pole_id="1")
    pole.pole_id = "1"
    pole.pole_id = "2"
    # FunctionalProperty в owlready2 хранит одно значение, а не список
    assert pole.pole_id == "2"


def test_mu_integration_creates_anchored_triples():
    """phi: показание сенсора должно присоединиться к экземпляру Pole (anchor entity)."""
    onto = build_schema()
    pole = get_or_create_pole(onto, pole_id="42")
    istochnik = get_or_create_istochnik(onto, tip="IoT-датчик почвы")
    reading = assert_soil_reading(
        onto,
        pole,
        tip_pokazatelya="влажность почвы",
        znachenie=30.1,
        data_izmereniya="2026-05-01",
        edinitsa_izmereniya="%",
        istochnik=istochnik,
    )
    assert reading in pole.harakterizuetsya
    assert reading.zafiksirovan_v is istochnik


def test_assert_weather_event_creates_anchored_triple():
    """phi_i: метеособытие должно присоединиться к Pole через podverzheno
    (Поле -> ПогодноеСобытие, объектное отношение R_O из §2.1.3) с заданными
    атрибутами. Метеособытие намеренно не привязывается к Источнику (см.
    integration.py) -- это проверяется отдельно ниже."""
    onto = build_schema()
    pole = get_or_create_pole(onto, pole_id="100")

    sobytie = assert_weather_event(onto, pole, tip_sobytiya="заморозок", intensivnost=-3.5, period="2026-04-10")

    assert sobytie.tip_sobytiya == "заморозок"
    assert sobytie.intensivnost == -3.5
    assert sobytie.period == "2026-04-10"
    assert sobytie in pole.podverzheno


def test_assert_agro_priem_creates_anchored_triple_with_source():
    """phi_i: агроприём должен присоединиться к Pole через obrabatyvaetsya
    (Поле -> АгроПриём) и хранить функциональную ссылку opisan_v на
    Источник (в отличие от метеособытия, агроприём привязывается к
    источнику происхождения для последующей оценки достоверности)."""
    onto = build_schema()
    pole = get_or_create_pole(onto, pole_id="101")
    istochnik = get_or_create_istochnik(onto, tip="Журнал полевых работ", format_istochnika="текст", dostovernost=0.8)

    priem = assert_agro_priem(
        onto,
        pole,
        tip_operatsii="внесение азотных удобрений",
        doza=120.0,
        data_priema="2026-04-15",
        istochnik=istochnik,
    )

    assert priem.tip_operatsii == "внесение азотных удобрений"
    assert priem.doza == 120.0
    assert priem.data_priema == "2026-04-15"
    assert priem.opisan_v is istochnik
    assert priem in pole.obrabatyvaetsya


def test_get_or_create_pole_is_idempotent():
    """Повторный вызов с тем же pole_id должен вернуть тот же экземпляр,
    а не создать дубликат (важно при повторной обработке данных одного
    и того же поля из разных источников в разное время)."""
    onto = build_schema()
    first = get_or_create_pole(onto, pole_id="200")
    second = get_or_create_pole(onto, pole_id="200")
    other = get_or_create_pole(onto, pole_id="201")

    assert second is first
    assert other is not first
    assert len(list(onto.Pole.instances())) == 2


def test_get_or_create_istochnik_is_idempotent():
    """Повторный вызов с теми же идентифицирующими данными должен вернуть
    тот же экземпляр Источника, а не создать дубликат."""
    onto = build_schema()
    first = get_or_create_istochnik(onto, tip="IoT-датчик почвы", format_istochnika="JSON")
    second = get_or_create_istochnik(onto, tip="IoT-датчик почвы", format_istochnika="JSON")
    other = get_or_create_istochnik(onto, tip="Лабораторный анализ", format_istochnika="PDF")

    assert second is first
    assert other is not first
    assert len(list(onto.Istochnik.instances())) == 2


def test_lambda_resolves_regional_synonym():
    """lambda/resolve: региональный синоним должен указывать на существующий экземпляр (§2.1.5)."""
    onto = build_schema()
    kultura = onto.Kultura()
    kultura.nazvanie_kultury = "Пшеница озимая"
    kultura.regionalnoe_nazvanie.append("озимка")

    match = resolve_entity_mention(onto, "озимка", "Kultura")
    assert match is kultura

    no_match = resolve_entity_mention(onto, "картофель", "Kultura")
    assert no_match is None


def test_reasoner_finds_no_inconsistency():
    """Проверка Ax на непротиворечивость (OWL 2 DL, HermiT). Пропускается, если Java недоступна."""
    import subprocess

    if subprocess.run(["which", "java"], capture_output=True).returncode != 0:
        import pytest

        pytest.skip("Java недоступна в окружении -- HermiT не может быть запущен")

    from owlready2 import sync_reasoner

    onto = build_schema()
    pole = get_or_create_pole(onto, pole_id="1")
    kultura = onto.Kultura()
    pole.vyraschivaetsya.append(kultura)

    with onto:
        sync_reasoner(infer_property_values=True)

    assert list(onto.inconsistent_classes()) == []
