"""OWL 2 DL schema for the DS ontology.

Реализует формальную модель из docs/chapter2_ontology_model.md (§2.1.3-2.1.4):
классы C, объектные отношения R_O, атрибуты (data properties) R_D и аксиомы Ax.
"""

from owlready2 import (
    AllDisjoint,
    AnnotationProperty,
    FunctionalProperty,
    Thing,
    World,
)

ONTOLOGY_IRI = "http://ds-thesis.local/onto/agro.owl"


def build_schema(world=None):
    """Собрать TBox онтологии (классы C, отношения R_O/R_D, аксиомы Ax).

    Каждый вызов строит схему в независимом owlready2.World, чтобы модули
    и тесты могли работать с изолированными экземплярами онтологии.
    """
    world = world or World()
    onto = world.get_ontology(ONTOLOGY_IRI)

    with onto:

        class Pole(Thing):
            """Поле — якорная сущность (anchor entity), §2.1.3."""

        class Kultura(Thing):
            """Культура — расширяет верхнеуровневые концепты AGROVOC / Crop Ontology."""

        class PokazatelPochvy(Thing):
            """ПоказательПочвы."""

        class PogodnoeSobytie(Thing):
            """ПогодноеСобытие."""

        class AgroPriem(Thing):
            """АгроПриём."""

        class VreditelBolezn(Thing):
            """ВредительБолезнь — расширяет BioAGgressor Ontology (BAGO)."""

        class Istochnik(Thing):
            """Источник данных: тип/формат/достоверность."""

        AllDisjoint([Pole, Kultura, PokazatelPochvy, PogodnoeSobytie, AgroPriem, VreditelBolezn, Istochnik])

        # ---- R_O: объектные отношения ----

        class vyraschivaetsya(Pole >> Kultura):
            """Поле -> Культура."""

        class harakterizuetsya(Pole >> PokazatelPochvy):
            """Поле -> ПоказательПочвы."""

        class podverzheno(Pole >> PogodnoeSobytie):
            """Поле -> ПогодноеСобытие."""

        class obrabatyvaetsya(Pole >> AgroPriem):
            """Поле -> АгроПриём."""

        class podverzhena(Kultura >> VreditelBolezn):
            """Культура -> ВредительБолезнь."""

        class zafiksirovan_v(PokazatelPochvy >> Istochnik, FunctionalProperty):
            """ПоказательПочвы -> Источник (функционально: один источник на показатель)."""

        class opisan_v(AgroPriem >> Istochnik, FunctionalProperty):
            """АгроПриём -> Источник."""

        # ---- R_D: атрибуты (data properties), таблица §2.1.3 ----

        class pole_id(Pole >> str, FunctionalProperty):
            pass

        class geometriya(Pole >> str, FunctionalProperty):
            """WKT-представление геометрии поля."""

        class ploschad(Pole >> float, FunctionalProperty):
            pass

        class region(Pole >> str, FunctionalProperty):
            pass

        class nazvanie_kultury(Kultura >> str, FunctionalProperty):
            pass

        class sort(Kultura >> str, FunctionalProperty):
            pass

        class fenofaza(Kultura >> str, FunctionalProperty):
            pass

        class tip_pokazatelya(PokazatelPochvy >> str, FunctionalProperty):
            pass

        class znachenie(PokazatelPochvy >> float, FunctionalProperty):
            pass

        class data_izmereniya(PokazatelPochvy >> str, FunctionalProperty):
            """ISO-8601 дата/время измерения."""

        class edinitsa_izmereniya(PokazatelPochvy >> str, FunctionalProperty):
            pass

        class tip_sobytiya(PogodnoeSobytie >> str, FunctionalProperty):
            pass

        class intensivnost(PogodnoeSobytie >> float, FunctionalProperty):
            pass

        class period(PogodnoeSobytie >> str, FunctionalProperty):
            pass

        class tip_operatsii(AgroPriem >> str, FunctionalProperty):
            pass

        class doza(AgroPriem >> float, FunctionalProperty):
            pass

        class data_priema(AgroPriem >> str, FunctionalProperty):
            pass

        class nazvanie_ugrozy(VreditelBolezn >> str, FunctionalProperty):
            pass

        class stadiya_riska(VreditelBolezn >> str, FunctionalProperty):
            pass

        class tip_istochnika(Istochnik >> str, FunctionalProperty):
            pass

        class format_istochnika(Istochnik >> str, FunctionalProperty):
            pass

        class dostovernost(Istochnik >> float, FunctionalProperty):
            pass

        # ---- Syn(c): региональные/устаревшие названия, §2.1.5 ----

        class regionalnoe_nazvanie(AnnotationProperty):
            """Аннотация-синоним: региональный или устаревший вариант термина (ru)."""

    return onto
