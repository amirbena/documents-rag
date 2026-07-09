"""Tests for DocumentTextExtractor: .txt/.md/.pdf extraction, errors, and Hebrew/Unicode text."""

import io
import uuid
from pathlib import Path

import pytest
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject
from pypdf.generic import StreamObject as _StreamObject

from app.models.document import Document
from app.services.document_text_extractor import DocumentTextExtractionError, DocumentTextExtractor


def _make_document(path: Path, content_type: str = "text/plain") -> Document:
    return Document(
        id=str(uuid.uuid4()),
        original_filename=path.name,
        stored_filename=path.name,
        content_type=content_type,
        file_size=path.stat().st_size if path.exists() else 0,
        stored_path=str(path),
    )


def _build_pdf_bytes(page_texts: list[str]) -> bytes:
    """Build a minimal multi-page PDF with real extractable text, using pypdf only.

    pypdf has no public "set page content" API, so this uses `_add_object` (private) to attach
    a raw content stream to a blank page — a known idiom for building pypdf test fixtures
    without a separate PDF-writing dependency. Test-fixture generation only, not production code.
    """
    writer = PdfWriter()
    for text in page_texts:
        page = writer.add_blank_page(width=200, height=200)

        stream = _StreamObject()
        stream.set_data(f"BT /F1 12 Tf 20 100 Td ({text}) Tj ET".encode("latin-1"))
        stream_ref = writer._add_object(stream)
        page[NameObject("/Contents")] = stream_ref

        font = DictionaryObject()
        font[NameObject("/Type")] = NameObject("/Font")
        font[NameObject("/Subtype")] = NameObject("/Type1")
        font[NameObject("/BaseFont")] = NameObject("/Helvetica")
        font_ref = writer._add_object(font)

        fonts = DictionaryObject()
        fonts[NameObject("/F1")] = font_ref
        resources = DictionaryObject()
        resources[NameObject("/Font")] = fonts
        page[NameObject("/Resources")] = resources

    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


async def test_txt_extraction(tmp_path: Path) -> None:
    """A .txt file should extract as a single unnumbered page."""
    path = tmp_path / "notes.txt"
    path.write_text("hello world", encoding="utf-8")
    document = _make_document(path)

    result = await DocumentTextExtractor().extract(document)

    assert result.document_id == document.id
    assert len(result.pages) == 1
    assert result.pages[0].text == "hello world"
    assert result.pages[0].page_number is None


async def test_markdown_extraction(tmp_path: Path) -> None:
    """A .md file should extract as a single unnumbered page, raw text (no markdown parsing)."""
    path = tmp_path / "readme.md"
    path.write_text("# Title\n\nSome **bold** text.", encoding="utf-8")
    document = _make_document(path, content_type="text/markdown")

    result = await DocumentTextExtractor().extract(document)

    assert len(result.pages) == 1
    assert result.pages[0].text == "# Title\n\nSome **bold** text."
    assert result.pages[0].page_number is None


async def test_pdf_extraction_with_page_numbers(tmp_path: Path) -> None:
    """A PDF should extract text page by page, with 1-indexed page numbers preserved."""
    path = tmp_path / "handbook.pdf"
    path.write_bytes(_build_pdf_bytes(["Page one text", "Page two text"]))
    document = _make_document(path, content_type="application/pdf")

    result = await DocumentTextExtractor().extract(document)

    assert len(result.pages) == 2
    assert result.pages[0].page_number == 1
    assert "Page one text" in result.pages[0].text
    assert result.pages[1].page_number == 2
    assert "Page two text" in result.pages[1].text
    assert "Page one text" in result.full_text
    assert "Page two text" in result.full_text


async def test_unsupported_file_type_fails(tmp_path: Path) -> None:
    """An unsupported extension should raise DocumentTextExtractionError."""
    path = tmp_path / "archive.zip"
    path.write_bytes(b"not a real zip, just bytes")
    document = _make_document(path, content_type="application/zip")

    with pytest.raises(DocumentTextExtractionError):
        await DocumentTextExtractor().extract(document)


async def test_missing_file_fails(tmp_path: Path) -> None:
    """A stored_path that doesn't exist on disk should raise DocumentTextExtractionError."""
    path = tmp_path / "does_not_exist.txt"
    document = _make_document(path)

    with pytest.raises(DocumentTextExtractionError):
        await DocumentTextExtractor().extract(document)


async def test_empty_extracted_text_fails(tmp_path: Path) -> None:
    """A file with no meaningful text content should raise DocumentTextExtractionError."""
    path = tmp_path / "blank.txt"
    path.write_text("   \n\n  ", encoding="utf-8")
    document = _make_document(path)

    with pytest.raises(DocumentTextExtractionError):
        await DocumentTextExtractor().extract(document)


async def test_hebrew_text_extraction(tmp_path: Path) -> None:
    """A UTF-8 .txt file with Hebrew content should extract correctly, preserved exactly."""
    path = tmp_path / "חוזה.txt"
    hebrew_text = "שלום עולם, זהו מסמך לדוגמה."
    path.write_text(hebrew_text, encoding="utf-8")
    document = _make_document(path)

    result = await DocumentTextExtractor().extract(document)

    assert result.pages[0].text == hebrew_text
