"""Shared EmbeddingIndexConfig/Document builders for the indexing package's unit tests.

Used by tests/unit/services/indexing/test_collection_registry.py,
test_vector_deletion_service.py, and test_cleanup_job_service.py.
"""

import uuid

from app.models.document import Document
from app.rag.embedding_config import EmbeddingIndexConfig


def build_embedding_config(**overrides: object) -> EmbeddingIndexConfig:
    """Build an EmbeddingIndexConfig with sensible defaults."""
    fields: dict[str, object] = {
        "collection_prefix": "documents",
        "provider": "ollama",
        "model": "nomic-embed-text",
        "dimension": 768,
        "embedding_version": "v1",
        "chunking_version": "v1",
    }
    fields.update(overrides)
    return EmbeddingIndexConfig(**fields)  # type: ignore[arg-type]


def build_document(**overrides: object) -> Document:
    """Build a Document with sensible defaults for indexing tests."""
    fields: dict[str, object] = {
        "id": str(uuid.uuid4()),
        "original_filename": "notes.txt",
        "stored_filename": f"{uuid.uuid4().hex}.txt",
        "content_type": "text/plain",
        "file_size": 100,
        "stored_path": "unset",
    }
    fields.update(overrides)
    return Document(**fields)  # type: ignore[arg-type]
