"""Extracts text from a stored document file: .txt, .md, .pdf, .docx, and .xlsx only.

No chunking, embedding, or Qdrant upsert here — this is purely the "load the file and get raw
text out of it" step. Routing is by file extension only, no content sniffing. PDF text is
extracted page by page via pypdf, preserving page numbers; XLSX is extracted sheet by sheet via
openpyxl, preserving sheet names; DOCX is extracted as plain paragraph text via python-docx;
plain text/Markdown files are treated as a single unnumbered page.
"""

import asyncio
from dataclasses import dataclass
from pathlib import Path

import docx
from openpyxl import load_workbook
from pypdf import PdfReader

from app.models.document import Document

_PLAIN_TEXT_SUFFIXES = {".txt", ".md"}


class DocumentTextExtractionError(Exception):
    """Raised when a document's stored file is missing, unsupported, or has no extractable text."""


@dataclass
class ExtractedPage:
    """One page's extracted text.

    `page_number` is set for PDFs, `sheet_name` for XLSX sheets — both None for plain
    text/Markdown/DOCX, which have no natural pagination.
    """

    text: str
    page_number: int | None = None
    sheet_name: str | None = None


@dataclass
class ExtractedDocument:
    """All extracted text for one Document, as an ordered list of pages."""

    document_id: str
    pages: list[ExtractedPage]

    @property
    def full_text(self) -> str:
        """Return all pages' text concatenated in order, separated by newlines."""
        return "\n".join(page.text for page in self.pages)


class DocumentTextExtractor:
    """Extracts text from a Document's stored file into an ExtractedDocument."""

    async def extract(self, document: Document) -> ExtractedDocument:
        """Load the document's stored file and extract its text (blocking work off the loop)."""
        return await asyncio.to_thread(self._extract_sync, document)

    def _extract_sync(self, document: Document) -> ExtractedDocument:
        path = Path(document.stored_path)
        if not path.exists():
            raise DocumentTextExtractionError(f"Stored file not found: {document.stored_path}")

        suffix = path.suffix.lower()
        if suffix in _PLAIN_TEXT_SUFFIXES:
            pages = [self._extract_plain_text(path)]
        elif suffix == ".pdf":
            pages = self._extract_pdf(path)
        elif suffix == ".docx":
            pages = [self._extract_docx(path)]
        elif suffix == ".xlsx":
            pages = self._extract_xlsx(path)
        else:
            raise DocumentTextExtractionError(f"Unsupported file type: {suffix or '(no extension)'}")

        if not any(page.text.strip() for page in pages):
            raise DocumentTextExtractionError("No extractable text found in document.")

        return ExtractedDocument(document_id=document.id, pages=pages)

    @staticmethod
    def _extract_plain_text(path: Path) -> ExtractedPage:
        """Read a .txt/.md file as UTF-8 text (Hebrew and other Unicode content supported)."""
        text = path.read_text(encoding="utf-8")
        return ExtractedPage(text=text, page_number=None)

    @staticmethod
    def _extract_pdf(path: Path) -> list[ExtractedPage]:
        """Extract text page by page from a PDF, preserving 1-indexed page numbers."""
        reader = PdfReader(str(path))
        return [
            ExtractedPage(text=page.extract_text() or "", page_number=index + 1)
            for index, page in enumerate(reader.pages)
        ]

    @staticmethod
    def _extract_docx(path: Path) -> ExtractedPage:
        """Extract plain paragraph text from a DOCX (no tables, headers/footers, or pagination)."""
        document = docx.Document(str(path))
        text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        return ExtractedPage(text=text)

    @staticmethod
    def _extract_xlsx(path: Path) -> list[ExtractedPage]:
        """Extract text sheet by sheet from an XLSX, preserving each sheet's name."""
        workbook = load_workbook(str(path), read_only=True, data_only=True)
        pages = []
        for sheet in workbook.worksheets:
            rows_text = [
                "\t".join(str(cell) for cell in row if cell is not None)
                for row in sheet.iter_rows(values_only=True)
                if any(cell is not None for cell in row)
            ]
            pages.append(ExtractedPage(text="\n".join(rows_text), sheet_name=sheet.title))
        return pages
