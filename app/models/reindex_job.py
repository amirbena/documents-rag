"""ORM model for one re-index build attempt for a document (Phase 2.8.6, subtask 2).

One row per build attempt, mirroring `IngestionJob`/`DocumentDeletionJob`'s append-only lifecycle
style: a `FAILED` row is never reset or reused — retrying always creates a brand-new row for the
same `document_id`. At most one `PENDING`/`PROCESSING` ("active") row may exist per document at a
time, enforced by the partial unique index `ix_reindex_jobs_one_active_per_document`
(migration `a8685da857f3`), not application logic alone — see
`app/services/indexing/reindex_scheduling_service.py`.

`target_collection_name`/`target_chunk_size`/`target_chunk_overlap` are a fully pinned,
reproducible build snapshot captured once at scheduling time — never re-derived from whatever the
live process's `Settings` say when a worker later claims this job (see
`app.services.indexing.reindex_service.build_settings_for_target`). `target_collection_name` is a
foreign key into `IndexCollection`, which already durably holds the target's full embedding
provider/model/dimension/embedding_version/chunking_version identity — so this row never needs to
duplicate those fields itself.

`error_message` is an internal value only, retained for operator/log inspection — never yet
exposed through any public API (no such API exists as of this subtask).
"""

from datetime import datetime
from enum import StrEnum

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class ReindexJobStatus(StrEnum):
    """Lifecycle status of one re-index build attempt."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ReindexJob(Base):
    """Tracks one attempt to build a document's vectors under a pinned target configuration."""

    __tablename__ = "reindex_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id"), nullable=False, index=True
    )
    target_collection_name: Mapped[str] = mapped_column(
        String(255), ForeignKey("index_collections.collection_name"), nullable=False
    )
    target_chunk_size: Mapped[int] = mapped_column(Integer, nullable=False)
    target_chunk_overlap: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[ReindexJobStatus] = mapped_column(
        Enum(
            ReindexJobStatus,
            native_enum=False,
            length=20,
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        default=ReindexJobStatus.PENDING,
        nullable=False,
        index=True,
    )
    error_message: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
