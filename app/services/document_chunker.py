"""Splits ExtractedDocument text into fixed-size, overlapping chunks for later embedding.

No embedding generation, no Qdrant upsert, no retrieval — purely text splitting, preserving
page_number (PDF) / sheet_name (XLSX) metadata per chunk. Chunking is word-boundary-aware
(never splits inside a word) and deterministic: the same ExtractedDocument always produces the
same chunks, in the same order, with the same chunk_ids.
"""

from dataclasses import dataclass

from app.services.document_text_extractor import ExtractedDocument


@dataclass
class DocumentChunk:
    """One chunk of extracted text, ready for embedding, with its source metadata preserved."""

    document_id: str
    chunk_id: str
    text: str
    chunk_index: int
    page_number: int | None = None
    sheet_name: str | None = None


class DocumentChunker:
    """Splits an ExtractedDocument's pages into fixed-size, word-boundary-aware chunks."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def chunk(self, document: ExtractedDocument) -> list[DocumentChunk]:
        """Split every page's text into chunks, preserving page/sheet metadata, in order."""
        chunks: list[DocumentChunk] = []
        chunk_index = 0
        for page in document.pages:
            for text in self._split_text(page.text):
                chunks.append(
                    DocumentChunk(
                        document_id=document.document_id,
                        chunk_id=f"{document.document_id}-{chunk_index}",
                        text=text,
                        chunk_index=chunk_index,
                        page_number=page.page_number,
                        sheet_name=page.sheet_name,
                    )
                )
                chunk_index += 1
        return chunks

    def _split_text(self, text: str) -> list[str]:
        """Split text into word-boundary-aware chunks with the configured size/overlap."""
        words = text.split()
        if not words:
            return []

        chunks: list[str] = []
        current_words: list[str] = []
        current_len = 0

        for word in words:
            added_len = len(word) + (1 if current_words else 0)
            if current_words and current_len + added_len > self._chunk_size:
                chunks.append(" ".join(current_words))
                current_words = self._take_overlap(current_words)
                current_len = len(" ".join(current_words)) if current_words else 0
                added_len = len(word) + (1 if current_words else 0)
            current_words.append(word)
            current_len += added_len

        if current_words:
            chunks.append(" ".join(current_words))

        return [chunk_text for chunk_text in chunks if chunk_text.strip()]

    def _take_overlap(self, words: list[str]) -> list[str]:
        """Return trailing words from the end totalling roughly chunk_overlap characters."""
        overlap: list[str] = []
        total = 0
        for word in reversed(words):
            additional = len(word) + (1 if overlap else 0)
            if total + additional > self._chunk_overlap:
                break
            overlap.insert(0, word)
            total += additional
        return overlap
