"""Async worker that processes pending ingestion jobs one at a time.

Internal service only — no public API. Claims a job with a row-level lock, transitions
pending -> processing -> completed/failed with clear transaction boundaries, and never
re-processes a job that's already completed or failed. The processing step itself is a
placeholder for now: it will later become PDF extraction, chunking, embedding generation, and
Qdrant upsert. This worker never calls EmbeddingProvider, LLMProvider, or VectorStore itself.
"""

from collections.abc import Awaitable, Callable

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.ingestion_job import IngestionJob, IngestionStatus

ProcessDocumentFn = Callable[[Document | None, IngestionJob], Awaitable[None]]


async def _default_process_document(document: Document | None, job: IngestionJob) -> None:
    """Placeholder for PDF extraction, chunking, embedding generation, and Qdrant upsert."""


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
