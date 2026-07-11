"""Async worker that processes pending ingestion jobs one at a time.

Internal service only — no public API. Claims a job with a row-level lock, transitions
pending -> processing -> completed/failed with clear transaction boundaries, and never
re-processes a job that's already completed or failed. The default processing step runs
Document -> extraction -> chunking -> embedding -> versioned-collection upsert, resolving the
job to completed on success or failed (with the error stored) if any step raises. On success,
the document's indexing metadata (embedding provider/model/dimension/version, chunking version,
collection name, indexed_at) is persisted via app/services/index_registry.py — a failure never
marks a document as indexed.
"""

import uuid
from collections.abc import Awaitable, Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.document import Document
from app.models.ingestion_job import IngestionJob, IngestionStatus
from app.rag.embedding_config import get_active_embedding_config
from app.rag.providers.provider_factory import get_embedding_provider, get_vector_store
from app.rag.providers.vector_store import VectorPoint
from app.services.document_chunker import DocumentChunk, DocumentChunker
from app.services.document_text_extractor import DocumentTextExtractor
from app.services.index_registry import ensure_active_collection, mark_document_indexed

ProcessDocumentFn = Callable[[Document | None, IngestionJob, AsyncSession], Awaitable[None]]


def to_vector_point(chunk: DocumentChunk, vector: list[float], source: str) -> VectorPoint:
    """Build a VectorPoint from a DocumentChunk and its embedding, preserving all metadata.

    Public (not `_`-prefixed) because app/services/reindex_service.py reuses it verbatim — the
    point ID must be derived identically on both the initial-ingest and re-index paths, or the
    same chunk would upsert under two different point IDs and silently duplicate.
    """
    return VectorPoint(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
        vector=vector,
        document_id=chunk.document_id,
        chunk_id=chunk.chunk_id,
        text=chunk.text,
        source=source,
        page_number=chunk.page_number,
        sheet_name=chunk.sheet_name,
    )


async def _default_process_document(
    document: Document | None, job: IngestionJob, session: AsyncSession
) -> None:
    """Extract text, chunk it, embed each chunk, and upsert the vectors into the active collection."""
    if document is None:
        raise ValueError(f"Document not found for job {job.id}")

    extracted = await DocumentTextExtractor().extract(document)

    settings = get_settings()
    config = get_active_embedding_config(settings)
    chunker = DocumentChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    chunks = chunker.chunk(extracted)
    if not chunks:
        return

    embedding_provider = get_embedding_provider(settings)
    vectors = await embedding_provider.embed([chunk.text for chunk in chunks])

    points = [
        to_vector_point(chunk, vector, document.original_filename)
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]

    vector_store = get_vector_store(settings)
    await ensure_active_collection(vector_store, session, config)
    await vector_store.upsert_vectors(config.collection_name, points)

    mark_document_indexed(document, config)


class IngestionWorker:
    """Claims and processes one pending IngestionJob at a time."""

    def __init__(self, process_document: ProcessDocumentFn | None = None) -> None:
        self._process_document = process_document or _default_process_document

    async def process_next_job(self, session: AsyncSession) -> IngestionJob | None:
        """Claim one pending job, run the processing step, and resolve its final status.

        Returns None if there is no pending job to claim. Idempotent: a job already
        `completed` or `failed` is never selected again, so repeated calls never re-process it.
        """
        job = await self._claim_next_pending_job(session)
        if job is None:
            return None

        job.status = IngestionStatus.PROCESSING
        await session.commit()

        document = await session.get(Document, job.document_id)

        try:
            await self._process_document(document, job, session)
        except Exception as exc:
            job.status = IngestionStatus.FAILED
            job.error_message = str(exc)
            await session.commit()
            return job

        job.status = IngestionStatus.COMPLETED
        await session.commit()
        return job

    async def _claim_next_pending_job(self, session: AsyncSession) -> IngestionJob | None:
        """Select-for-update the oldest pending job, skipping rows locked by another worker."""
        stmt = (
            select(IngestionJob)
            .where(IngestionJob.status == IngestionStatus.PENDING)
            .order_by(IngestionJob.created_at)
            .limit(1)
            .with_for_update(skip_locked=True)
        )
        result = await session.execute(stmt)
        return result.scalar_one_or_none()
