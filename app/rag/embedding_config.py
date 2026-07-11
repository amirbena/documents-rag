"""Framework-neutral embedding/index configuration: the versioned identity of "how this
platform is currently indexing documents."

Both ingestion (write side, via IngestionWorker/app/services/index_registry.py) and retrieval
(read side, via RetrievalService) must resolve the same active EmbeddingIndexConfig, so query
embeddings and document embeddings are always compatible — no engine constructs or selects an
embedding model itself; both CustomRagEngine and LangChainRagEngine reach Qdrant only through
RetrievalService/IngestionWorker, which read this module's active configuration.
"""

import re
from dataclasses import dataclass

from app.core.config import Settings, get_settings


class InvalidEmbeddingIndexConfigError(ValueError):
    """Raised when an EmbeddingIndexConfig field is missing, empty, or otherwise invalid."""


_SANITIZE_PATTERN = re.compile(r"[^a-z0-9_-]+")


def _sanitize(value: str) -> str:
    """Lowercase and replace every non alnum/-/_ run with a single '-', for a safe collection name."""
    return _SANITIZE_PATTERN.sub("-", value.strip().lower()).strip("-")


@dataclass(frozen=True)
class EmbeddingIndexConfig:
    """The active indexing configuration: which embedding model/version/chunking produced (or
    must produce) a given set of vectors, and which collection they belong to.

    Two configurations that differ in any field are incompatible — never share a collection.
    """

    collection_prefix: str
    provider: str
    model: str
    dimension: int
    embedding_version: str
    chunking_version: str

    def __post_init__(self) -> None:
        for field_name in (
            "collection_prefix",
            "provider",
            "model",
            "embedding_version",
            "chunking_version",
        ):
            if not getattr(self, field_name).strip():
                raise InvalidEmbeddingIndexConfigError(f"{field_name} must not be empty")
        if self.dimension <= 0:
            raise InvalidEmbeddingIndexConfigError("dimension must be a positive integer")

    @property
    def collection_name(self) -> str:
        """A deterministic, sanitized Qdrant collection name derived from every field above.

        Two configs that differ in any field (provider, model, dimension, embedding_version, or
        chunking_version) always produce a different collection name — they can never
        accidentally share a collection.
        """
        parts = [
            self.collection_prefix,
            self.provider,
            self.model,
            f"e{self.embedding_version}",
            f"c{self.chunking_version}",
            f"d{self.dimension}",
        ]
        return "__".join(_sanitize(part) for part in parts)


def get_active_embedding_config(settings: Settings | None = None) -> EmbeddingIndexConfig:
    """Return the platform's single active embedding/index configuration.

    This is the only place that should read EMBEDDING_PROVIDER/EMBEDDING_MODEL/VECTOR_SIZE/
    EMBEDDING_VERSION/CHUNKING_VERSION for indexing purposes — both ingestion and retrieval call
    this function rather than reading those settings directly, so they can never drift apart.
    """
    settings = settings or get_settings()
    return EmbeddingIndexConfig(
        collection_prefix=settings.qdrant_collection_name,
        provider=settings.embedding_provider,
        model=settings.resolved_embedding_model,
        dimension=settings.vector_size,
        embedding_version=settings.embedding_version,
        chunking_version=settings.chunking_version,
    )
