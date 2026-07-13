"""Response schemas for document upload/ingestion/read endpoints.

Every schema field traces to a real column on `Document`/`IngestionJob` or a genuinely-derivable
value (see `app/services/document_query_service.py`). No document response ever includes
`storage_key`/`storage_bucket`/`storage_etag` or any other internal storage-provider detail —
only `storage_provider` (e.g. "local"/"minio") is exposed, per the storage-abstraction governance
rule in CLAUDE.md.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel

from app.models.ingestion_job import IngestionStatus


class DocumentLifecycleStatus(StrEnum):
    """A document's derived lifecycle state — see `document_query_service.derive_lifecycle_status`.

    Defined here (not in the service module) so `app/schemas/documents.py` and
    `app/services/document_query_service.py` don't import each other in a cycle: the service
    module builds these response schemas, so the schema module must not depend back on it.
    """

    UPLOADED = "uploaded"
    PENDING = "pending"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"


class DocumentUploadResponse(BaseModel):
    """Shape returned by POST /api/v1/documents."""

    document_id: str
    job_id: str
    status: IngestionStatus


class DocumentSummaryResponse(BaseModel):
    """One document row in the GET /api/v1/documents list — no storage/embedding internals."""

    id: str
    original_filename: str
    content_type: str
    size_bytes: int
    status: DocumentLifecycleStatus
    created_at: datetime
    latest_ingestion_activity_at: datetime | None


class DocumentListResponse(BaseModel):
    """Shape returned by GET /api/v1/documents: one page of documents plus paging metadata."""

    items: list[DocumentSummaryResponse]
    total: int
    limit: int
    offset: int


class DocumentDetailResponse(BaseModel):
    """Shape returned by GET /api/v1/documents/{document_id}.

    `storage_provider` is exposed; `storage_bucket`/`storage_key`/`storage_etag` deliberately are
    not (see module docstring).
    """

    id: str
    original_filename: str
    content_type: str
    size_bytes: int
    storage_provider: str | None
    status: DocumentLifecycleStatus
    collection_name: str | None
    embedding_version: str | None
    chunking_version: str | None
    indexed_at: datetime | None
    latest_ingestion_job_id: str | None
    latest_ingestion_status: IngestionStatus | None
    created_at: datetime


class IngestionStatusResponse(BaseModel):
    """Shape returned by GET /api/v1/documents/{document_id}/ingestion.

    `IngestionJob` has no dedicated `started_at`/`failed_at`/`attempt_count` columns — `created_at`
    is used as a "job first created" surrogate and `updated_at` as a "last status transition"
    surrogate (it has `onupdate=func.now()`). All fields are null when the document has no
    ingestion job yet (see DocumentLifecycleStatus.UPLOADED in document_query_service.py).
    """

    document_id: str
    job_id: str | None
    status: IngestionStatus | None
    created_at: datetime | None
    updated_at: datetime | None


class IngestionFailureResponse(BaseModel):
    """Shape returned by GET /api/v1/documents/{document_id}/failure.

    `safe_message` is a fixed, generic message — never the raw `IngestionJob.error_message` (see
    `sanitize_ingestion_error` in document_query_service.py for why). No `retryable` field is
    included: there is no retry endpoint or attempt-count tracking in this codebase yet, so a
    boolean here would be fabricated rather than genuinely derived.
    """

    document_id: str
    job_id: str
    status: IngestionStatus
    safe_message: str
    failed_at: datetime
