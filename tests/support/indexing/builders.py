"""Shared EmbeddingIndexConfig/Document/ReindexJob builders for the indexing package's unit tests.

Used by tests/unit/services/indexing/test_collection_registry.py, test_vector_deletion_service.py,
test_cleanup_job_service.py, and test_reindex_scheduling_service.py.
"""

import uuid
from datetime import UTC, datetime, timedelta

from app.models.document import Document
from app.models.reindex_job import ReindexJob, ReindexJobStatus
from app.rag.embedding_config import EmbeddingIndexConfig

_BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


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


def build_reindex_job(
    document_id: str,
    status: ReindexJobStatus,
    *,
    target_collection_name: str = "documents__ollama__target-model__ev1__cv1__d768",
    created_at: datetime | None = None,
    **overrides: object,
) -> ReindexJob:
    """Build a ReindexJob for `document_id` at the given status, for full-deletion resolution tests."""
    fields: dict[str, object] = dict(
        id=str(uuid.uuid4()),
        document_id=document_id,
        target_collection_name=target_collection_name,
        target_chunk_size=500,
        target_chunk_overlap=50,
        status=status,
        created_at=created_at or (_BASE_TIME - timedelta(minutes=30)),
    )
    fields.update(overrides)
    return ReindexJob(**fields)  # type: ignore[arg-type]
