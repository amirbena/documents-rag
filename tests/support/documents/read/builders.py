"""Shared Document/IngestionJob/DocumentDeletionJob builders for the document read/download unit tests.

Used by both tests/unit/services/documents/test_query_service.py and test_download_service.py.
"""

import uuid
from datetime import UTC, datetime, timedelta

from app.models.document import Document
from app.models.document_deletion_job import DocumentDeletionJob, DocumentDeletionStatus
from app.models.ingestion_job import IngestionJob, IngestionStatus

BASE_TIME = datetime(2026, 1, 1, tzinfo=UTC)


def build_document(index: int = 0, **overrides: object) -> Document:
    """Build a Document with sensible defaults, indexed by `index` for uniqueness/ordering."""
    defaults: dict[str, object] = dict(
        id=str(uuid.uuid4()),
        original_filename=f"file-{index}.pdf",
        stored_filename=f"stored-{index}.pdf",
        content_type="application/pdf",
        file_size=100 + index,
        stored_path=f"documents/doc-{index}/stored-{index}.pdf",
        created_at=BASE_TIME + timedelta(minutes=index),
        storage_provider="local",
        storage_bucket=None,
        storage_key=f"documents/doc-{index}/stored-{index}.pdf",
        storage_etag="etag-value",
        collection_name=None,
        embedding_provider=None,
        embedding_model=None,
        embedding_dimension=None,
        embedding_version=None,
        chunking_version=None,
        indexed_at=None,
    )
    defaults.update(overrides)
    return Document(**defaults)  # type: ignore[arg-type]


def build_ingestion_job(
    document_id: str, status: IngestionStatus, *, minutes: int = 0, **overrides: object
) -> IngestionJob:
    """Build an IngestionJob for `document_id`, `minutes` after BASE_TIME."""
    defaults: dict[str, object] = dict(
        id=str(uuid.uuid4()),
        document_id=document_id,
        status=status,
        error_message=None,
        created_at=BASE_TIME + timedelta(minutes=minutes),
        updated_at=BASE_TIME + timedelta(minutes=minutes),
    )
    defaults.update(overrides)
    return IngestionJob(**defaults)  # type: ignore[arg-type]


def build_deletion_job(
    document_id: str, status: DocumentDeletionStatus, *, minutes: int = 0
) -> DocumentDeletionJob:
    """Build a DocumentDeletionJob for `document_id`, `minutes` after BASE_TIME."""
    return DocumentDeletionJob(
        id=str(uuid.uuid4()),
        document_id=document_id,
        status=status,
        vector_cleanup_completed=False,
        storage_cleanup_completed=False,
        created_at=BASE_TIME + timedelta(minutes=minutes),
        updated_at=BASE_TIME + timedelta(minutes=minutes),
    )
