"""Tests for DocumentChunker: sizing, overlap, metadata preservation, and determinism."""

import pytest

from app.services.documents.chunker import DocumentChunk, DocumentChunker
from app.services.documents.text_extractor import ExtractedDocument, ExtractedPage


def test_single_chunk_for_short_text() -> None:
    """Text shorter than chunk_size should produce exactly one chunk."""
    document = ExtractedDocument(
        document_id="doc-1", pages=[ExtractedPage(text="hello world", page_number=None)]
    )
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=200)

    chunks = chunker.chunk(document)

    assert len(chunks) == 1
    assert isinstance(chunks[0], DocumentChunk)
    assert chunks[0].text == "hello world"
    assert chunks[0].document_id == "doc-1"
    assert chunks[0].chunk_index == 0
    assert chunks[0].chunk_id == "doc-1-0"


def test_multiple_chunks_for_long_text() -> None:
    """Text longer than chunk_size should be split into multiple chunks."""
    text = " ".join(f"word{i}" for i in range(50))
    document = ExtractedDocument(document_id="doc-1", pages=[ExtractedPage(text=text, page_number=1)])
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)

    chunks = chunker.chunk(document)

    assert len(chunks) > 1
    for chunk_text in chunks:
        assert len(chunk_text.text) <= 50
        # No chunk should start or end mid-word (each chunk is whole, space-joined words).
        assert " " not in chunk_text.text or all(part for part in chunk_text.text.split(" "))


def test_no_chunk_splits_inside_a_word() -> None:
    """Every chunk should be made of whole words — never a fragment of a word."""
    text = " ".join(f"word{i}" for i in range(50))
    document = ExtractedDocument(document_id="doc-1", pages=[ExtractedPage(text=text)])
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)

    chunks = chunker.chunk(document)

    all_words = set(text.split())
    for chunk_text in chunks:
        for word in chunk_text.text.split():
            assert word in all_words


def test_overlap_correctness() -> None:
    """Consecutive chunks should share trailing/leading words matching chunk_overlap."""
    text = " ".join(f"word{i}" for i in range(50))
    document = ExtractedDocument(document_id="doc-1", pages=[ExtractedPage(text=text)])
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)

    chunks = chunker.chunk(document)

    for i in range(len(chunks) - 1):
        current_last_word = chunks[i].text.split()[-1]
        next_first_word = chunks[i + 1].text.split()[0]
        assert current_last_word == next_first_word


def test_chunk_index_and_ids_are_sequential() -> None:
    """chunk_index should be sequential starting at 0, and chunk_id should reflect it."""
    text = " ".join(f"word{i}" for i in range(50))
    document = ExtractedDocument(document_id="doc-1", pages=[ExtractedPage(text=text)])
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)

    chunks = chunker.chunk(document)

    for index, chunk_text in enumerate(chunks):
        assert chunk_text.chunk_index == index
        assert chunk_text.chunk_id == f"doc-1-{index}"


def test_page_metadata_preserved() -> None:
    """Chunks from a PDF page should preserve that page's page_number."""
    document = ExtractedDocument(
        document_id="doc-1",
        pages=[
            ExtractedPage(text="first page content here", page_number=1),
            ExtractedPage(text="second page content here", page_number=2),
        ],
    )
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=100)

    chunks = chunker.chunk(document)

    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[0].sheet_name is None
    assert chunks[1].page_number == 2
    assert chunks[1].sheet_name is None


def test_sheet_metadata_preserved() -> None:
    """Chunks from an XLSX sheet should preserve that sheet's sheet_name."""
    document = ExtractedDocument(
        document_id="doc-1",
        pages=[
            ExtractedPage(text="revenue data here", sheet_name="Q1"),
            ExtractedPage(text="expense data here", sheet_name="Q2"),
        ],
    )
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=100)

    chunks = chunker.chunk(document)

    assert len(chunks) == 2
    assert chunks[0].sheet_name == "Q1"
    assert chunks[0].page_number is None
    assert chunks[1].sheet_name == "Q2"
    assert chunks[1].page_number is None


def test_txt_and_docx_pages_have_no_page_or_sheet_metadata() -> None:
    """Chunks from a plain text/DOCX page should have both page_number and sheet_name as None."""
    document = ExtractedDocument(
        document_id="doc-1", pages=[ExtractedPage(text="plain text content")]
    )
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=100)

    chunks = chunker.chunk(document)

    assert chunks[0].page_number is None
    assert chunks[0].sheet_name is None


def test_hebrew_text_chunking() -> None:
    """Hebrew text should be split on whitespace correctly, preserved exactly."""
    hebrew_text = "שלום עולם זהו מסמך לדוגמה בעברית עם מספר מילים נוספות כדי לבדוק חלוקה לחלקים"
    document = ExtractedDocument(document_id="doc-1", pages=[ExtractedPage(text=hebrew_text)])
    chunker = DocumentChunker(chunk_size=30, chunk_overlap=5)

    chunks = chunker.chunk(document)

    assert len(chunks) > 1
    all_words = set(hebrew_text.split())
    for chunk_text in chunks:
        for word in chunk_text.text.split():
            assert word in all_words


def test_empty_chunks_are_ignored() -> None:
    """A page with only whitespace text should produce zero chunks, not an empty-text chunk."""
    document = ExtractedDocument(document_id="doc-1", pages=[ExtractedPage(text="   \n\n  ")])
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=100)

    chunks = chunker.chunk(document)

    assert chunks == []


def test_empty_document_produces_no_chunks() -> None:
    """A document with no pages at all should produce zero chunks."""
    document = ExtractedDocument(document_id="doc-1", pages=[])
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=100)

    chunks = chunker.chunk(document)

    assert chunks == []


def test_deterministic_output() -> None:
    """Chunking the same document twice should produce identical chunks, in the same order."""
    text = " ".join(f"word{i}" for i in range(50))
    document = ExtractedDocument(document_id="doc-1", pages=[ExtractedPage(text=text, page_number=1)])
    chunker = DocumentChunker(chunk_size=50, chunk_overlap=10)

    first_run = chunker.chunk(document)
    second_run = chunker.chunk(document)

    assert first_run == second_run


def test_multi_page_document_chunk_index_continues_across_pages() -> None:
    """chunk_index should increment continuously across pages, not reset per page."""
    document = ExtractedDocument(
        document_id="doc-1",
        pages=[
            ExtractedPage(text="first page short text", page_number=1),
            ExtractedPage(text="second page short text", page_number=2),
        ],
    )
    chunker = DocumentChunker(chunk_size=1000, chunk_overlap=100)

    chunks = chunker.chunk(document)

    assert [chunk_text.chunk_index for chunk_text in chunks] == [0, 1]
    assert [chunk_text.chunk_id for chunk_text in chunks] == ["doc-1-0", "doc-1-1"]


def test_invalid_chunk_overlap_raises() -> None:
    """Constructing a chunker with overlap >= size should raise ValueError."""
    with pytest.raises(ValueError, match="chunk_overlap"):
        DocumentChunker(chunk_size=100, chunk_overlap=100)
