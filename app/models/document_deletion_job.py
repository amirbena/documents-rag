"""ORM model for a full-document-deletion attempt (Phase 2.8.4).

One row per deletion attempt for a document, mirroring `IngestionJob`'s append-only lifecycle
style (`app/models/ingestion_job.py`): a `PARTIALLY_FAILED` row is never reset back to `PENDING`
and never deleted — retrying always creates a brand-new row for the same `document_id`. At most
one `PENDING`/`PROCESSING` ("active") row may exist per document at a time, enforced by the
partial unique index `ix_document_deletion_jobs_one_active_per_document`
(migration `c8f3a2b6d1e7`), not application logic alone — see
`app/services/documents/deletion_service.py`.

`vector_cleanup_completed`/`storage_cleanup_completed` record this attempt's progress through the
two-step cleanup order (vectors before storage — see the service module) so a partial failure is
observable; `error_code` is a fixed, machine-identifiable marker (never a raw provider exception),
`error_message` is the raw internal detail retained for operator/log inspection only, never
returned by a public API verbatim.
"""

from datetime import datetime
from enum import StrEnum

from sqlalchemy import Boolean, DateTime, Enum, ForeignKey, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class DocumentDeletionStatus(StrEnum):
    """Lifecycle status of one full-document-deletion attempt."""

    PENDING = "pending"
    PROCESSING = "processing"
    PARTIALLY_FAILED = "partially_failed"
    COMPLETED = "completed"


class DocumentDeletionJob(Base):
    """Tracks one attempt to fully delete a document's vectors and stored object."""

    __tablename__ = "document_deletion_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id"), nullable=False, index=True
    )
    status: Mapped[DocumentDeletionStatus] = mapped_column(
        Enum(
            DocumentDeletionStatus,
            native_enum=False,
            length=20,
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        default=DocumentDeletionStatus.PENDING,
        nullable=False,
        index=True,
    )
    vector_cleanup_completed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    storage_cleanup_completed: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
