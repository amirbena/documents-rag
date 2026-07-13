"""ORM model for an uploaded document's storage/original filename metadata, plus which indexing
configuration (if any) it was last successfully indexed with.

`storage_provider`/`storage_bucket`/`storage_key` are the provider-neutral storage identity
introduced in Phase 2.6/2.7 — `storage_key` is what `FileStorage.read()`/`.delete()` are called
with, never `stored_path`. `stored_filename`/`stored_path` are retained for backward
compatibility with rows written before this migration: a row with `storage_key IS NULL` is a
pre-migration local document whose `stored_path` value is treated as its local storage key (see
`app.storage.keys.resolve_document_storage_key` and the migration in
`alembic/versions/`) — this is documented, not silently unreadable.
"""

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class Document(Base):
    """An uploaded document: original + stored filename, content type, size, storage path.

    The `embedding_*`/`chunking_version`/`collection_name`/`indexed_at` columns are populated
    only after a *successful* indexing (or re-index) run — see app/services/indexing/collection_registry.py.
    They stay NULL until then, and a failed re-index never updates them, so a document's stored
    indexing configuration always reflects the last version it was genuinely, successfully
    indexed with. Comparing these columns against the platform's current active
    EmbeddingIndexConfig is how staleness is detected (see EmbeddingIndexConfig.collection_name).
    """

    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    original_filename: Mapped[str] = mapped_column(String(1024), nullable=False)
    stored_filename: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    content_type: Mapped[str] = mapped_column(String(255), nullable=False)
    file_size: Mapped[int] = mapped_column(Integer, nullable=False)
    stored_path: Mapped[str] = mapped_column(String(2048), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    # Provider-neutral storage identity (Phase 2.6/2.7) — see module docstring.
    storage_provider: Mapped[str | None] = mapped_column(String(32), nullable=True)
    storage_bucket: Mapped[str | None] = mapped_column(String(255), nullable=True)
    storage_key: Mapped[str | None] = mapped_column(String(2048), nullable=True)
    storage_etag: Mapped[str | None] = mapped_column(String(255), nullable=True)

    embedding_provider: Mapped[str | None] = mapped_column(String(64), nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(255), nullable=True)
    embedding_dimension: Mapped[int | None] = mapped_column(Integer, nullable=True)
    embedding_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    chunking_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    collection_name: Mapped[str | None] = mapped_column(
        String(255), ForeignKey("index_collections.collection_name"), nullable=True
    )
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
