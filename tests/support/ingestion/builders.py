"""Shared Document/IngestionJob builders for the ingestion retry/stale-recovery unit tests.

Used by both tests/unit/services/ingestion/test_retry_service.py and test_stale_recovery_service.py.
"""

import uuid
from datetime import UTC, datetime, timedelta

from app.models.document import Document
from app.models.ingestion_job import IngestionJob, IngestionStatus

NOW = datetime(2026, 7, 13, 12, 0, 0, tzinfo=UTC)


def build_document(document_id: str | None = None) -> Document:
    """Build a Document with sensible defaults for retry/stale-recovery tests."""
    return Document(
        id=document_id or str(uuid.uuid4()),
        original_filename="a.pdf",
        stored_filename="a.pdf",
        content_type="application/pdf",
        file_size=10,
        stored_path="a.pdf",
        storage_provider="local",
        storage_key="a.pdf",
        created_at=NOW - timedelta(days=1),
    )


def build_ingestion_job(
    document_id: str,
    status: IngestionStatus,
    *,
    created_at: datetime | None = None,
    updated_at: datetime | None = None,
) -> IngestionJob:
    """Build an IngestionJob for `document_id`, defaulting created_at/updated_at relative to NOW."""
    created_at = created_at or (NOW - timedelta(minutes=30))
    return IngestionJob(
        id=str(uuid.uuid4()),
        document_id=document_id,
        status=status,
        created_at=created_at,
        updated_at=updated_at or created_at,
    )
