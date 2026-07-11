"""ORM model tracking every Qdrant collection this platform has ever indexed into.

One row per distinct EmbeddingIndexConfig.collection_name (see app/rag/embedding_config.py) —
lets a collection's configuration be inspected/validated without querying Qdrant directly, and
lets `active` vs `retired` collections be tracked across an embedding/chunking version migration.
"""

from datetime import datetime
from enum import StrEnum

from sqlalchemy import DateTime, Enum, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class IndexCollectionStatus(StrEnum):
    """Lifecycle status of one tracked Qdrant collection."""

    ACTIVE = "active"
    RETIRED = "retired"


class IndexCollection(Base):
    """One row per distinct, versioned Qdrant collection this platform has created."""

    __tablename__ = "index_collections"

    collection_name: Mapped[str] = mapped_column(String(255), primary_key=True)
    embedding_provider: Mapped[str] = mapped_column(String(64), nullable=False)
    embedding_model: Mapped[str] = mapped_column(String(255), nullable=False)
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False)
    embedding_version: Mapped[str] = mapped_column(String(64), nullable=False)
    chunking_version: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[IndexCollectionStatus] = mapped_column(
        Enum(
            IndexCollectionStatus,
            native_enum=False,
            length=20,
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        default=IndexCollectionStatus.ACTIVE,
        nullable=False,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
