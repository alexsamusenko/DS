"""Local prototype intake for heterogeneous agricultural source files.

Uploading is not extraction or validation. Files are quarantined as untrusted inputs.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter(prefix="/workbench", tags=["АПК: загрузка исходных данных"])
ROOT = Path(os.environ.get("DS_WORKBENCH_DIR", "build/workbench_uploads"))
MAX_FILE_BYTES = 20 * 1024 * 1024
EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".txt", ".docx", ".doc", ".odt", ".ods", ".xlsm", ".heic", ".webp", ".geojson", ".kml", ".gpx", ".gml", ".xml", ".tif", ".tiff", ".zip", ".shp", ".dbf", ".shx", ".prj", ".gpkg", ".las", ".log"}
EXT_TYPES = {
    ".csv": "table", ".tsv": "table", ".xlsx": "table", ".xls": "table", ".xlsm": "table", ".ods": "table",
    ".json": "structured", ".xml": "structured", ".geojson": "geospatial", ".kml": "geospatial", ".gpx": "geospatial", ".gml": "geospatial", ".shp": "geospatial", ".shx": "geospatial", ".dbf": "geospatial", ".prj": "geospatial", ".gpkg": "geospatial", ".las": "geospatial", ".pdf": "document", ".docx": "document", ".doc": "document", ".odt": "document",
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".tif": "image", ".tiff": "image", ".heic": "image", ".webp": "image", ".zip": "archive", ".log": "telemetry",
    ".txt": "document",
}


def _preview(ext: str, content: bytes) -> dict:
    if ext == ".json":
        try:
            obj = json.loads(content.decode("utf-8-sig"))
            return {"kind": "json", "top_level_type": type(obj).__name__, "keys": list(obj)[:30] if isinstance(obj, dict) else []}
        except (UnicodeError, ValueError):
            return {"kind": "unreadable", "reason": "JSON_DECODE_FAILED"}
    if ext in {".csv", ".tsv"}:
        try:
            sample = content[:128 * 1024].decode("utf-8-sig")
            reader = csv.reader(io.StringIO(sample), delimiter="\t" if ext == ".tsv" else ",")
            rows = []
            for index, row in enumerate(reader):
                if index >= 6:
                    break
                rows.append([cell[:160] for cell in row[:25]])
            return {"kind": "table_preview", "columns": rows[0] if rows else [], "sample_rows": rows[1:]}
        except (UnicodeError, csv.Error):
            return {"kind": "unreadable", "reason": "CSV_DECODE_FAILED"}
    if ext in {".xlsx", ".xls", ".xlsm", ".ods"}:
        return {"kind": "table", "reason": "SPREADSHEET_PARSING_NOT_IMPLEMENTED"}
    return {"kind": EXT_TYPES[ext], "reason": "CONTENT_EXTRACTION_NOT_IMPLEMENTED"}


@router.post("/files")
async def upload_source_file(file: UploadFile = File(...)) -> dict:
    original = (file.filename or "").strip()
    ext = Path(original).suffix.lower()
    if ext not in EXTENSIONS or not re.fullmatch(r"[^/\\\x00]{1,180}", original):
        raise HTTPException(status_code=400, detail="Unsupported extension or unsafe filename")
    chunks: list[bytes] = []
    total = 0
    try:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_FILE_BYTES:
                raise HTTPException(status_code=413, detail="File exceeds 20 MiB limit")
            chunks.append(chunk)
    finally:
        await file.close()
    if total == 0:
        raise HTTPException(status_code=400, detail="Empty file")
    data = b"".join(chunks)
    digest = hashlib.sha256(data).hexdigest()
    file_id = uuid.uuid4().hex
    ROOT.mkdir(parents=True, exist_ok=True)
    target = ROOT / f"{file_id}{ext}"
    with target.open("xb") as out:
        out.write(data)
    # No path is exposed to the browser; uploaded content is never executed.
    return {
        "file_id": file_id,
        "filename": original,
        "category": EXT_TYPES[ext],
        "size_bytes": total,
        "sha256": digest,
        "status": "registered_untrusted",
        "preview": _preview(ext, data),
        "pipeline_ready": False,
        "warnings": ["File registered, not semantically validated; extracted values must be mapped and reviewed before pipeline use."],
    }
