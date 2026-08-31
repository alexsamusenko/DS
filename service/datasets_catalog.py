"""Каталог датасетов для фронта: сопоставляет карточки датасетов (docs/dataset_cards/)
с тем, что реально лежит на диске в data/ -- см. docs/governance/licenses.md про то,
почему статус лицензии здесь важен не меньше, чем факт наличия файлов."""

import re
from pathlib import Path

DATASET_CARDS_DIR = Path("docs/dataset_cards")
DATA_DIR = Path("data")


def _extract_section(text: str, heading: str) -> str | None:
    match = re.search(rf"^##\s*{re.escape(heading)}\s*\n(.+?)(?=\n##|\Z)", text, re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else None


def _parse_card(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")

    title_match = re.search(r"^#\s*Dataset Card:\s*(.+)$", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else path.stem

    # Разделы соответствуют docs/dataset_cards/TEMPLATE.md (по образцу Datasheets
    # for Datasets, Gebru et al. 2018) -- показываем на фронте все, не только лицензию,
    # чтобы карточка была информативна сама по себе, без перехода в docs/.
    return {
        "slug": path.stem,
        "title": title,
        "license_summary": _extract_section(text, "Лицензия") or "не указана",
        "source": _extract_section(text, "Источник и мотивация"),
        "composition": _extract_section(text, "Состав"),
        "limitations": _extract_section(text, "Известные ограничения"),
        "usage_in_project": _extract_section(text, "Использование в проекте"),
    }


def _dir_stats(path: Path) -> dict:
    files = [f for f in path.rglob("*") if f.is_file()]
    total_size = sum(f.stat().st_size for f in files)
    return {"file_count": len(files), "total_size_bytes": total_size}


def list_datasets() -> list[dict]:
    entries = []

    cards = {}
    if DATASET_CARDS_DIR.exists():
        for card_path in sorted(DATASET_CARDS_DIR.glob("*.md")):
            if card_path.stem == "TEMPLATE":
                continue
            cards[card_path.stem] = _parse_card(card_path)

    on_disk = {}
    if DATA_DIR.exists():
        on_disk = {d.name: d for d in DATA_DIR.iterdir() if d.is_dir()}

    for slug, card in cards.items():
        entry = {**card, "on_disk": slug in on_disk}
        if slug in on_disk:
            entry.update(_dir_stats(on_disk[slug]))
        entries.append(entry)

    # Папки в data/, для которых нет карточки -- показываем тоже, но явно
    # помечаем как недокументированные (см. правило "лицензия/карточка на
    # каждый добавляемый актив", docs/governance/best_practices.md).
    for name, path in on_disk.items():
        if name not in cards:
            entries.append(
                {
                    "slug": name,
                    "title": name,
                    "license_summary": "⚠️ карточка датасета не найдена",
                    "source": None,
                    "composition": None,
                    "limitations": None,
                    "usage_in_project": None,
                    "on_disk": True,
                    **_dir_stats(path),
                }
            )

    return entries
