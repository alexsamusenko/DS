"""Persistent local catalog and best-effort extraction for untrusted farm documents."""
import csv
import io
import json
import os
import sqlite3
import xml.etree.ElementTree as ET
import zipfile
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(os.environ.get("DS_WORKBENCH_DIR", "build/workbench_uploads"))


def connection():
    ROOT.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(ROOT / "catalog.sqlite3")
    db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE IF NOT EXISTS files (id TEXT PRIMARY KEY, name TEXT, extension TEXT, category TEXT, size INTEGER, digest TEXT, preview TEXT, created_at TEXT)")
    return db


def save(file_id, filename, extension, category, size, digest, preview):
    with connection() as db:
        db.execute("INSERT INTO files VALUES (?,?,?,?,?,?,?,?)", (file_id, filename, extension, category, size, digest, json.dumps(preview, ensure_ascii=False), datetime.now(UTC).isoformat()))
        db.commit()


def as_dict(row):
    return {
        "file_id": row["id"], "filename": row["name"], "extension": row["extension"],
        "category": row["category"], "size_bytes": row["size"], "sha256": row["digest"],
        "preview": json.loads(row["preview"]), "created_at": row["created_at"],
        "status": "registered_untrusted", "pipeline_ready": False,
    }


def list_files():
    with connection() as db:
        return [as_dict(row) for row in db.execute("SELECT * FROM files ORDER BY created_at DESC LIMIT 500").fetchall()]


def get_file(file_id):
    with connection() as db:
        row = db.execute("SELECT * FROM files WHERE id=?", (file_id,)).fetchone()
    return as_dict(row) if row else None


def extract(extension, data):
    try:
        if extension == ".docx":
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                root = ET.fromstring(archive.read("word/document.xml"))
            paragraphs = ["".join(t.text or "" for t in p.iter() if t.tag.endswith("}t")) for p in root.iter() if p.tag.endswith("}p")]
            return {"kind": "document_text", "text": "\n".join(paragraphs)[:10000], "paragraph_count": len(paragraphs)}
        if extension in {".xlsx", ".xlsm"}:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                shared = []
                if "xl/sharedStrings.xml" in archive.namelist():
                    root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
                    shared = ["".join(t.text or "" for t in item.iter() if t.tag.endswith("}t")) for item in root if item.tag.endswith("}si")]
                sheets = []
                for path in sorted(p for p in archive.namelist() if p.startswith("xl/worksheets/sheet") and p.endswith(".xml"))[:20]:
                    root = ET.fromstring(archive.read(path))
                    rows = []
                    for row in (node for node in root.iter() if node.tag.endswith("}row")):
                        cells = []
                        for cell in (node for node in row if node.tag.endswith("}c")):
                            v = next((node.text for node in cell if node.tag.endswith("}v")), "")
                            if cell.get("t") == "s" and v:
                                v = shared[int(v)]
                            elif cell.get("t") == "inlineStr":
                                v = "".join(t.text or "" for t in cell.iter() if t.tag.endswith("}t"))
                            cells.append({"address": cell.get("r"), "value": v})
                        rows.append(cells[:40])
                        if len(rows) >= 12:
                            break
                    sheets.append({"sheet": path.split("/")[-1], "rows": rows})
            return {"kind": "spreadsheet_preview", "sheets": sheets, "warning": "Formula values and units require verification"}
        if extension == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            return {"kind": "pdf_text", "pages": len(reader.pages), "text": "\n".join((p.extract_text() or "") for p in reader.pages[:10])[:10000], "warning": "Scans and handwriting require OCR"}
        if extension in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}:
            import pytesseract
            from PIL import Image
            with Image.open(io.BytesIO(data)) as img:
                img.thumbnail((2200, 2200))
                text = pytesseract.image_to_string(img, lang="rus+eng", timeout=20)
            return {"kind": "ocr_text", "text": text[:10000], "warning": "Handwriting recognition is experimental and unverified"}
        if extension in {".txt", ".log", ".xml", ".kml", ".gpx", ".gml"}:
            return {"kind": "text_preview", "text": data[:10000].decode("utf-8-sig")[:5000]}
    except Exception as exc:
        return {"kind": "unreadable", "reason": "PARSER_FAILED", "error_type": type(exc).__name__}
    return None
