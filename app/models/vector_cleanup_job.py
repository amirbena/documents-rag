"""ORM model tracking a pending/failed/completed legacy-vector cleanup after a re-index.

One row per (document, previous collection) pair whose old vectors still need deleting after a
document was successfully re-indexed into a new collection (see app/services/reindex_service.py).
Exists because a cleanup failure (the Qdrant delete-by-filter call failing after the new
collection and Document indexing metadata already committed) must be durably tracked and
retryable — it is never conflated with re-index failure, and never silently lost. Multiple rows
for the same document (one per still-uncleaned historical collection) are supported; cleanup is
idempotent, so retrying an already-completed row's collection is harmless.
"""

from datetime import datetime
from enum import StrEnum

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class VectorCleanupStatus(StrEnum):
    """Lifecycle status of one legacy-vector cleanup attempt."""

    PENDING = "pending"
    FAILED = "failed"
    COMPLETED = "completed"


class VectorCleanupJob(Base):
    """One outstanding (or resolved) legacy-collection vector cleanup for a re-indexed document."""

    __tablename__ = "vector_cleanup_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(String(36), ForeignKey("documents.id"), nullable=False)
    collection_name: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[VectorCleanupStatus] = mapped_column(
        Enum(
            VectorCleanupStatus,
            native_enum=False,
            length=20,
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        default=VectorCleanupStatus.PENDING,
        nullable=False,
    )
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
