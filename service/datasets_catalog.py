"""Каталог датасетов для фронта: сопоставляет карточки датасетов (docs/dataset_cards/)
с тем, что реально лежит на диске в data/ -- см. docs/governance/licenses.md про то,
почему статус лицензии здесь важен не меньше, чем факт наличия файлов."""

import re
from pathlib import Path

DATASET_CARDS_DIR = Path("docs/dataset_cards")
DATA_DIR = Path("data")


def _parse_card(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")

    title_match = re.search(r"^#\s*Dataset Card:\s*(.+)$", text, re.MULTILINE)
    title = title_match.group(1).strip() if title_match else path.stem

    license_match = re.search(r"^##\s*Лицензия\s*\n(.+?)(?=\n##|\Z)", text, re.MULTILINE | re.DOTALL)
    license_text = license_match.group(1).strip() if license_match else "не указана"

    return {"slug": path.stem, "title": title, "license_summary": license_text}


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
            entries.append({
                "slug": name, "title": name, "license_summary": "⚠️ карточка датасета не найдена",
                "on_disk": True, **_dir_stats(path),
            })

    return entries
