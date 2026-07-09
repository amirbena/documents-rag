"""Async worker that processes pending ingestion jobs one at a time.

Internal service only — no public API. Claims a job with a row-level lock, transitions
pending -> processing -> completed/failed with clear transaction boundaries, and never
re-processes a job that's already completed or failed. The default processing step runs
Document -> extraction -> chunking, and stops there — embedding generation and Qdrant upsert
are still placeholders for a later milestone. This worker never calls EmbeddingProvider,
LLMProvider, or VectorStore itself.
"""

from collections.abc import Awaitable, Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.document import Document
from app.models.ingestion_job import IngestionJob, IngestionStatus
from app.services.document_chunker import DocumentChunker
from app.services.document_text_extractor import DocumentTextExtractor

ProcessDocumentFn = Callable[[Document | None, IngestionJob], Awaitable[None]]


async def _default_process_document(document: Document | None, job: IngestionJob) -> None:
    """Extract text, then split it into chunks. Stops there — no embedding or Qdrant upsert yet."""
    if document is None:
        raise ValueError(f"Document not found for job {job.id}")

    extracted = await DocumentTextExtractor().extract(document)

    settings = get_settings()
    chunker = DocumentChunker(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
    chunker.chunk(extracted)


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
            await self._process_document(document, job)
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
