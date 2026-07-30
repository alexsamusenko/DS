"""Функция семантической интеграции mu = phi (union) lambda, §2.1.4.

phi_i: структурные источники (сенсоры, лабораторные анализы, агроприёмы) --
данные попадают в граф напрямую по известной схеме источника.

lambda: неструктурный источник (текстовое упоминание) -- сущность
связывается с существующим экземпляром или порождает новый (§2.1.4-2.1.5).

Каждая функция принимает онтологию, собранную build_schema(), и возвращает
или обновляет экземпляр (I, ABox), не трогая TBox (C, R_O, R_D, Ax).
"""

from difflib import SequenceMatcher


def get_or_create_pole(onto, pole_id, geometriya=None, ploschad=None, region=None):
    """Найти или создать экземпляр Поле — якорную сущность (§2.1.3)."""
    Pole = onto.Pole
    for individual in Pole.instances():
        if individual.pole_id == pole_id:
            return individual

    pole = Pole(f"pole_{pole_id}")
    pole.pole_id = pole_id
    if geometriya is not None:
        pole.geometriya = geometriya
    if ploschad is not None:
        pole.ploschad = ploschad
    if region is not None:
        pole.region = region
    return pole


def get_or_create_istochnik(onto, tip, format_istochnika=None, dostovernost=None):
    """phi для описания источника (Источник), на который ссылаются другие факты."""
    Istochnik = onto.Istochnik
    for individual in Istochnik.instances():
        if individual.tip_istochnika == tip and individual.format_istochnika == format_istochnika:
            return individual

    istochnik = Istochnik()
    istochnik.tip_istochnika = tip
    if format_istochnika is not None:
        istochnik.format_istochnika = format_istochnika
    if dostovernost is not None:
        istochnik.dostovernost = dostovernost
    return istochnik


def assert_soil_reading(onto, pole, tip_pokazatelya, znachenie, data_izmereniya, edinitsa_izmereniya, istochnik):
    """phi_i для показаний почвенных сенсоров/лабораторных анализов.

    Реализует триплет (Pole_i, harakterizuetsya, PokazatelPochvy) из §2.1.4.
    """
    pokazatel = onto.PokazatelPochvy()
    pokazatel.tip_pokazatelya = tip_pokazatelya
    pokazatel.znachenie = float(znachenie)
    pokazatel.data_izmereniya = data_izmereniya
    pokazatel.edinitsa_izmereniya = edinitsa_izmereniya
    pokazatel.zafiksirovan_v = istochnik
    pole.harakterizuetsya.append(pokazatel)
    return pokazatel


def assert_weather_event(onto, pole, tip_sobytiya, intensivnost, period):
    """phi_i для метеоданных (Open-Meteo/NASA POWER, см. docs/datasets.md)."""
    sobytie = onto.PogodnoeSobytie()
    sobytie.tip_sobytiya = tip_sobytiya
    sobytie.intensivnost = float(intensivnost)
    sobytie.period = period
    pole.podverzheno.append(sobytie)
    return sobytie


def assert_agro_priem(onto, pole, tip_operatsii, doza, data_priema, istochnik):
    """phi_i для записей журнала полевых работ (внесение удобрений/СЗР и т.п.)."""
    priem = onto.AgroPriem()
    priem.tip_operatsii = tip_operatsii
    priem.doza = float(doza)
    priem.data_priema = data_priema
    priem.opisan_v = istochnik
    pole.obrabatyvaetsya.append(priem)
    return priem


def _similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def resolve_entity_mention(onto, mention_text, target_class_name, threshold=0.82):
    """lambda: связать текстовое упоминание сущности с существующим экземпляром.

    Базовая (baseline) реализация resolve()/Syn(c) из §2.1.5 -- сравнение по
    строковому сходству названия и по аннотациям regionalnoe_nazvanie.
    Возвращает существующий экземпляр либо None, если совпадений не найдено
    (в таком случае вызывающий код создаёт новый экземпляр -- см. build.py).
    Реализация -- отправная точка; в проде заменяется эмбеддинговым сходством
    и ограничениями графового соседства, как оговорено в §2.1.4.
    """
    target_class = getattr(onto, target_class_name)
    name_attr = {
        "Kultura": "nazvanie_kultury",
        "VreditelBolezn": "nazvanie_ugrozy",
    }.get(target_class_name)
    if name_attr is None:
        raise ValueError(f"Нет настроенного атрибута названия для класса {target_class_name}")

    best_match, best_score = None, 0.0
    for individual in target_class.instances():
        candidates = [getattr(individual, name_attr) or ""]
        candidates += [str(s) for s in individual.regionalnoe_nazvanie]
        score = max(_similarity(mention_text, c) for c in candidates if c)
        if score > best_score:
            best_match, best_score = individual, score

    return best_match if best_score >= threshold else None
