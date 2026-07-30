"""Smoke-тесты для схемы онтологии (docs/chapter2_ontology_model.md, §2.1)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ds_ontology.integration import (  # noqa: E402
    assert_soil_reading,
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
        onto, pole,
        tip_pokazatelya="влажность почвы", znachenie=30.1,
        data_izmereniya="2026-05-01", edinitsa_izmereniya="%",
        istochnik=istochnik,
    )
    assert reading in pole.harakterizuetsya
    assert reading.zafiksirovan_v is istochnik


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
