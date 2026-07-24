"""Backend re-index primitives: build a replacement index for a document under an explicit
target configuration, and — as a separate operation — activate it (Phase 2.8.6, subtask 1).

Two deliberately distinct operations, never combined into one function:

`build_reindex_target()` re-derives a Document's vectors from its already-persisted stored file —
no new upload required — and writes them into the *target* collection identified by an explicit,
caller-supplied `EmbeddingIndexConfig`. It never touches the document's current serving state:
`Document.collection_name`/`embedding_*`/`chunking_version`/`indexed_at` are left exactly as they
are, and nothing about the document's *previous* (currently-serving) collection is read, deleted,
or scheduled for deletion. A successful build proves only "target B was built successfully" — it
does not mean "target B is active." This is what makes build-ahead migration safe: the running
process keeps serving whatever `Document.collection_name` already says, untouched, for as long as
the operator wants, regardless of how many documents have already been built toward a new target.

`activate_reindexed_document()` is the separate, later operation that actually switches a
document's serving identity to a target configuration — updating `Document.collection_name`/
`embedding_*`/`chunking_version`/`indexed_at` and, in the same commit, persisting a
`VectorCleanupJob` for whatever collection the document is leaving behind. It never deletes Qdrant
vectors itself; actual old-vector removal is left entirely to the existing, independently-retryable
`cleanup_job_service.retry_cleanup_job()` path. Nothing in this module wires either primitive into
a worker, script, or API yet — that is deliberately out of scope for this subtask.

Both primitives resolve their embedding provider/model/chunking config from an explicit,
caller-pinned target — never from whatever the live process's `Settings` currently say — via
`build_settings_for_target()`, so a build can never silently generate embeddings under one model
while writing them into a collection whose identity claims another.
"""

import uuid
from dataclasses import dataclass
from enum import StrEnum

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.models.document import Document
from app.models.vector_cleanup_job import VectorCleanupJob, VectorCleanupStatus
from app.rag.embedding_config import EmbeddingIndexConfig, get_active_embedding_config
from app.rag.embedding_validation import validate_embeddings
from app.rag.providers.provider_factory import get_embedding_provider, get_vector_store
from app.services.documents.chunker import DocumentChunker
from app.services.documents.text_extractor import DocumentTextExtractor
from app.services.indexing.collection_registry import ensure_active_collection, mark_document_indexed
from app.services.ingestion.worker import to_vector_point
from app.storage.contract import FileStorage

# Category (Phase 2.10, see app/core/errors.py): ConfigurationError.


class TargetConfigurationMismatchError(ValueError):
    """Raised when settings derived for a build do not reproduce the exact requested target.

    A build must never proceed if the settings actually used to resolve the embedding provider/
    model/vector store would produce an `EmbeddingIndexConfig` different from the one the caller
    explicitly pinned — that would mean generating embeddings under one identity while writing
    them into a collection whose name claims another.
    """

    def __init__(self, expected: EmbeddingIndexConfig, actual: EmbeddingIndexConfig) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"Settings derived for this build resolve to {actual!r}, but the requested target "
            f"configuration was {expected!r} — refusing to build under a mismatched identity."
        )


def build_settings_for_target(
    base_settings: Settings,
    target_config: EmbeddingIndexConfig,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> Settings:
    """Return a new Settings instance scoped to `target_config`, never mutating `base_settings`.

    Every field `get_active_embedding_config()`/the embedding provider/the chunker actually read is
    explicitly overridden: `embedding_provider`, both `embedding_model` and `ollama_embedding_model`
    (kept identical — `OllamaEmbeddingProvider` reads `ollama_embedding_model` directly, while
    `resolved_embedding_model` prefers `embedding_model`; a build must never let these diverge),
    `vector_size`, `embedding_version`, `chunking_version`, `qdrant_collection_name` (the collection
    *prefix* `target_config.collection_prefix` was itself derived from), and the explicit numeric
    `chunk_size`/`chunk_overlap` — never re-derived from the chunking-version label, live settings,
    or environment defaults, since `EmbeddingIndexConfig` does not carry them.
    """
    return base_settings.model_copy(
        update={
            "embedding_provider": target_config.provider,
            "embedding_model": target_config.model,
            "ollama_embedding_model": target_config.model,
            "vector_size": target_config.dimension,
            "embedding_version": target_config.embedding_version,
            "chunking_version": target_config.chunking_version,
            "qdrant_collection_name": target_config.collection_prefix,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
        }
    )


class ReindexBuildOutcome(StrEnum):
    """The distinct outcomes `build_reindex_target()` can report for a successful build."""

    BUILT = "built"
    BUILT_EMPTY = "built_empty"


@dataclass(frozen=True)
class ReindexBuildResult:
    """The outcome of one `build_reindex_target()` call.

    Proves only that the target collection was built successfully — never that it is active. Not
    a durable readiness record itself; a future `ReindexJob(COMPLETED)` row is what later subtasks
    will treat as the durable proof of a successful build, so this type deliberately carries only
    what an in-process caller needs immediately after the call returns.
    """

    outcome: ReindexBuildOutcome
    target_collection_name: str
    chunk_count: int
    vector_count: int


def _resolved_config_matches_target(
    settings: Settings, target_config: EmbeddingIndexConfig
) -> EmbeddingIndexConfig | None:
    """Return the config `settings` actually resolves to, or None if it matches `target_config`."""
    resolved = get_active_embedding_config(settings)
    return None if resolved == target_config else resolved


async def build_reindex_target(
    document: Document,
    session: AsyncSession,
    settings: Settings,
    file_storage: FileStorage,
    target_config: EmbeddingIndexConfig,
    *,
    target_chunk_size: int,
    target_chunk_overlap: int,
) -> ReindexBuildResult:
    """Re-derive `document`'s vectors under `target_config` and write them into its collection.

    Never touches the document's current serving state: `Document.collection_name`/`embedding_*`/
    `chunking_version`/`indexed_at` are left exactly as they are, and the document's *previous*
    (currently-serving) collection is never read, deleted, or scheduled for deletion here — see the
    module docstring. Raises `TargetConfigurationMismatchError` before any storage read, extraction,
    or embedding call if the settings derived for this build do not reproduce `target_config`
    exactly. Raises on any extraction/chunking/embedding/validation/vector-store-write failure;
    since no Document metadata is ever mutated by this function, such a failure leaves the document
    exactly as it was before the call.
    """
    build_settings = build_settings_for_target(
        settings, target_config, chunk_size=target_chunk_size, chunk_overlap=target_chunk_overlap
    )
    mismatch = _resolved_config_matches_target(build_settings, target_config)
    if mismatch is not None:
        raise TargetConfigurationMismatchError(expected=target_config, actual=mismatch)

    extracted = await DocumentTextExtractor(storage=file_storage).extract(document)
    chunker = DocumentChunker(
        chunk_size=build_settings.chunk_size, chunk_overlap=build_settings.chunk_overlap
    )
    chunks = chunker.chunk(extracted)

    vector_store = get_vector_store(build_settings)

    if chunks:
        embedding_provider = get_embedding_provider(build_settings)
        vectors = await embedding_provider.embed([chunk.text for chunk in chunks])
        validate_embeddings(vectors, expected_count=len(chunks), expected_dimension=target_config.dimension)
        points = [
            to_vector_point(chunk, vector, document.original_filename)
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        await ensure_active_collection(vector_store, session, target_config)
        await vector_store.upsert_vectors(target_config.collection_name, points)

    return ReindexBuildResult(
        outcome=ReindexBuildOutcome.BUILT if chunks else ReindexBuildOutcome.BUILT_EMPTY,
        target_collection_name=target_config.collection_name,
        chunk_count=len(chunks),
        vector_count=len(chunks) if chunks else 0,
    )


@dataclass(frozen=True)
class ReindexActivationResult:
    """The outcome of one `activate_reindexed_document()` call.

    `activated=False` means the document already carried `target_config`'s identity — a no-op,
    idempotent re-activation; `cleanup_job` is always `None` in that case, since there is nothing
    to vacate. `activated=True` means the document's serving identity was switched this call;
    `cleanup_job` is populated only if there was a previous collection to vacate (never for a
    document that had never been indexed before).
    """

    document: Document
    activated: bool
    cleanup_job: VectorCleanupJob | None


async def activate_reindexed_document(
    document: Document, session: AsyncSession, target_config: EmbeddingIndexConfig
) -> ReindexActivationResult:
    """Atomically switch `document`'s serving identity to `target_config` and defer old cleanup.

    Never deletes Qdrant vectors itself — actual removal of the vacated collection's vectors is
    left entirely to `cleanup_job_service.retry_cleanup_job()`, run later and independently. The
    document-metadata switch and the new `VectorCleanupJob` (when one is needed) are persisted in
    exactly one commit — a crash between them is impossible by construction, never "metadata
    committed, cleanup job to follow." Idempotent: calling this again once the document already
    carries `target_config`'s identity does nothing and creates no cleanup job.
    """
    if document.collection_name == target_config.collection_name:
        return ReindexActivationResult(document=document, activated=False, cleanup_job=None)

    previous_collection_name = document.collection_name
    mark_document_indexed(document, target_config)

    cleanup_job: VectorCleanupJob | None = None
    if previous_collection_name is not None:
        cleanup_job = VectorCleanupJob(
            id=str(uuid.uuid4()),
            document_id=document.id,
            collection_name=previous_collection_name,
            status=VectorCleanupStatus.PENDING,
            attempts=0,
            last_error=None,
        )
        session.add(cleanup_job)

    try:
        await session.commit()
    except Exception:
        await session.rollback()
        session.expire(document)
        raise

    return ReindexActivationResult(document=document, activated=True, cleanup_job=cleanup_job)


__all__ = [
    "ReindexActivationResult",
    "ReindexBuildOutcome",
    "ReindexBuildResult",
    "TargetConfigurationMismatchError",
    "activate_reindexed_document",
    "build_reindex_target",
    "build_settings_for_target",
]
