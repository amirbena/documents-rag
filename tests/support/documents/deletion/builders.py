"""Shared Document/IngestionJob/DocumentDeletionJob builders for deletion unit tests.

Used by both tests/unit/services/documents/test_deletion_service.py and test_deletion_worker.py —
extracted here rather than duplicated, since both modules need the exact same fixture shapes.
"""

import uuid
from datetime import UTC, datetime, timedelta

from app.models.document import Document
from app.models.document_deletion_job import DocumentDeletionJob, DocumentDeletionStatus
from app.models.ingestion_job import IngestionJob, IngestionStatus
from app.models.reindex_job import ReindexJob, ReindexJobStatus

NOW = datetime(2026, 7, 13, 12, 0, 0, tzinfo=UTC)


def build_document(document_id: str | None = None, **overrides: object) -> Document:
    """Build a Document with sane defaults for deletion unit tests, overriding as needed."""
    fields: dict[str, object] = {
        "id": document_id or str(uuid.uuid4()),
        "original_filename": "a.pdf",
        "stored_filename": "a.pdf",
        "content_type": "application/pdf",
        "file_size": 10,
        "stored_path": "a.pdf",
        "storage_provider": "local",
        "storage_key": "documents/a/a.pdf",
        "created_at": NOW - timedelta(days=1),
    }
    fields.update(overrides)
    return Document(**fields)  # type: ignore[arg-type]


def build_ingestion_job(
    document_id: str, status: IngestionStatus, *, created_at: datetime | None = None
) -> IngestionJob:
    """Build an IngestionJob for `document_id` at the given status."""
    created_at = created_at or (NOW - timedelta(hours=1))
    return IngestionJob(
        id=str(uuid.uuid4()),
        document_id=document_id,
        status=status,
        created_at=created_at,
        updated_at=created_at,
    )


def build_deletion_job(
    document_id: str, status: DocumentDeletionStatus, *, created_at: datetime | None = None
) -> DocumentDeletionJob:
    """Build a DocumentDeletionJob for `document_id` at the given status."""
    created_at = created_at or (NOW - timedelta(minutes=30))
    return DocumentDeletionJob(
        id=str(uuid.uuid4()),
        document_id=document_id,
        status=status,
        vector_cleanup_completed=False,
        storage_cleanup_completed=False,
        created_at=created_at,
        updated_at=created_at,
    )


def build_reindex_job(
    document_id: str,
    status: ReindexJobStatus,
    *,
    target_collection_name: str = "documents__ollama__target-model__ev1__cv1__d3",
    created_at: datetime | None = None,
) -> ReindexJob:
    """Build a ReindexJob for `document_id` at the given status."""
    created_at = created_at or (NOW - timedelta(minutes=30))
    return ReindexJob(
        id=str(uuid.uuid4()),
        document_id=document_id,
        target_collection_name=target_collection_name,
        target_chunk_size=500,
        target_chunk_overlap=50,
        status=status,
        created_at=created_at,
        updated_at=created_at,
    )
