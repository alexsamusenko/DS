"""Storage parser and restart persistence tests with synthetic documents."""
import io
import zipfile

from service import workbench_storage


def test_catalog_survives_new_connection(tmp_path, monkeypatch):
    monkeypatch.setattr(workbench_storage, "ROOT", tmp_path)
    workbench_storage.save("id-1", "test.csv", ".csv", "table", 3, "abc", {"kind": "table_preview"})
    assert workbench_storage.get_file("id-1")["filename"] == "test.csv"
    assert workbench_storage.list_files()[0]["pipeline_ready"] is False


def test_docx_extracts_text():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("word/document.xml", '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>Пшеница</w:t></w:r></w:p></w:body></w:document>')
    result = workbench_storage.extract(".docx", buffer.getvalue())
    assert result["kind"] == "document_text"
    assert "Пшеница" in result["text"]


def test_xlsx_extracts_preview():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>Поле</t></is></c><c r="B1"><v>10</v></c></row></sheetData></worksheet>')
    result = workbench_storage.extract(".xlsx", buffer.getvalue())
    assert result["kind"] == "spreadsheet_preview"
    assert result["sheets"][0]["rows"][0][0]["value"] == "Поле"
    assert result["sheets"][0]["rows"][0][1]["value"] == "10"


def test_pdf_text_layer_parser():
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=300, height=300)
    buffer = io.BytesIO()
    writer.write(buffer)
    result = workbench_storage.extract(".pdf", buffer.getvalue())
    assert result["kind"] == "pdf_text"
    assert result["pages"] == 1


def test_invalid_image_fails_closed():
    result = workbench_storage.extract(".png", b"not-a-real-image")
    assert result["kind"] == "unreadable"
    assert result["reason"] == "PARSER_FAILED"
