"""Tests for IngestionWorker against a fake session double — no real database.

The worker is designed around Postgres-specific locking semantics
(`SELECT ... FOR UPDATE SKIP LOCKED`), which SQLite does not represent correctly even when it
accepts the same SQLAlchemy call — so these tests use a fake session that faithfully simulates
the WHERE status='pending' ... LIMIT 1 filter and Document lookup, without any real database.
"""

import uuid
from pathlib import Path
from typing import Any

from app.models.document import Document
from app.models.ingestion_job import IngestionJob, IngestionStatus
from app.services.ingestion_worker import IngestionWorker


class _FakeScalarResult:
    """Stand-in for the object returned by AsyncSession.execute()."""

    def __init__(self, value: IngestionJob | None) -> None:
        self._value = value

    def scalar_one_or_none(self) -> IngestionJob | None:
        return self._value


class _FakeAsyncSession:
    """Fake AsyncSession simulating the pending-job query and Document lookup, no real DB."""

    def __init__(self) -> None:
        self._documents: dict[str, Document] = {}
        self._jobs: dict[str, IngestionJob] = {}
        self.commit_count = 0

    def add(self, instance: Document | IngestionJob) -> None:
        if isinstance(instance, Document):
            self._documents[instance.id] = instance
        elif isinstance(instance, IngestionJob):
            self._jobs[instance.id] = instance

    async def execute(self, stmt: Any) -> _FakeScalarResult:
        """Simulate: SELECT ... WHERE status='pending' ORDER BY created_at LIMIT 1 FOR UPDATE."""
        pending = [job for job in self._jobs.values() if job.status == IngestionStatus.PENDING]
        job = pending[0] if pending else None
        return _FakeScalarResult(job)

    async def get(self, model: type, instance_id: str) -> Document | None:
        if model is Document:
            return self._documents.get(instance_id)
        return None

    async def commit(self) -> None:
        self.commit_count += 1


def _add_document_and_job(
    session: _FakeAsyncSession,
    status: IngestionStatus = IngestionStatus.PENDING,
    stored_path: str = "storage/documents/x.pdf",
) -> IngestionJob:
    document = Document(
        id=str(uuid.uuid4()),
        original_filename="handbook.pdf",
        stored_filename=f"{uuid.uuid4().hex}.pdf",
        content_type="application/pdf",
        file_size=123,
        stored_path=stored_path,
    )
    session.add(document)
    job = IngestionJob(id=str(uuid.uuid4()), document_id=document.id, status=status)
    session.add(job)
    return job


async def test_pending_job_transitions_to_completed() -> None:
    """A pending job should be processed and end up completed."""

    async def _noop(document: Document | None, job: IngestionJob) -> None:
        return None

    session = _FakeAsyncSession()
    job = _add_document_and_job(session)
    worker = IngestionWorker(process_document=_noop)

    result = await worker.process_next_job(session)

    assert result is not None
    assert result.id == job.id
    assert result.status == IngestionStatus.COMPLETED
    assert result.error_message is None


async def test_processing_exception_marks_job_failed() -> None:
    """A processing step that raises should mark the job failed with the error message stored."""
    session = _FakeAsyncSession()
    _add_document_and_job(session)

    async def _boom(document: Document | None, job: IngestionJob) -> None:
        raise RuntimeError("boom: extraction not implemented")

    worker = IngestionWorker(process_document=_boom)

    result = await worker.process_next_job(session)

    assert result is not None
    assert result.status == IngestionStatus.FAILED
    assert result.error_message == "boom: extraction not implemented"


async def test_no_pending_jobs_returns_none() -> None:
    """With no pending jobs at all, process_next_job should return None."""
    session = _FakeAsyncSession()
    worker = IngestionWorker()

    result = await worker.process_next_job(session)

    assert result is None
    assert session.commit_count == 0


async def test_completed_job_is_ignored() -> None:
    """A job already completed must never be selected again."""
    session = _FakeAsyncSession()
    _add_document_and_job(session, status=IngestionStatus.COMPLETED)
    worker = IngestionWorker()

    result = await worker.process_next_job(session)

    assert result is None


async def test_failed_job_is_ignored() -> None:
    """A job already failed must never be selected again."""
    session = _FakeAsyncSession()
    _add_document_and_job(session, status=IngestionStatus.FAILED)
    worker = IngestionWorker()

    result = await worker.process_next_job(session)

    assert result is None


async def test_repeated_calls_do_not_reprocess_completed_job() -> None:
    """Running process_next_job() repeatedly must not re-process a job already completed."""
    call_count = 0

    async def _counting_process(document: Document | None, job: IngestionJob) -> None:
        nonlocal call_count
        call_count += 1

    session = _FakeAsyncSession()
    _add_document_and_job(session)
    worker = IngestionWorker(process_document=_counting_process)

    first = await worker.process_next_job(session)
    second = await worker.process_next_job(session)

    assert first is not None
    assert first.status == IngestionStatus.COMPLETED
    assert second is None
    assert call_count == 1


async def test_placeholder_processing_called_exactly_once_with_document_and_job() -> None:
    """The processing step should be invoked exactly once, with the claimed document and job."""
    calls: list[tuple[Document | None, IngestionJob]] = []

    async def _recording_process(document: Document | None, job: IngestionJob) -> None:
        calls.append((document, job))

    session = _FakeAsyncSession()
    job = _add_document_and_job(session)
    worker = IngestionWorker(process_document=_recording_process)

    await worker.process_next_job(session)

    assert len(calls) == 1
    document, passed_job = calls[0]
    assert passed_job.id == job.id
    assert document is not None
    assert document.id == job.document_id


async def test_worker_never_imports_embedding_llm_or_vector_store_providers() -> None:
    """The worker module must not import EmbeddingProvider, LLMProvider, or VectorStore."""
    import app.services.ingestion_worker as worker_module

    module_names = vars(worker_module)
    for forbidden in ("EmbeddingProvider", "LLMProvider", "VectorStore", "provider_factory"):
        assert forbidden not in module_names


async def test_worker_marks_completed_when_extraction_succeeds(tmp_path: Path) -> None:
    """The real default processing step (text extraction) should complete the job on success."""
    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world", encoding="utf-8")

    session = _FakeAsyncSession()
    _add_document_and_job(session, stored_path=str(file_path))
    worker = IngestionWorker()

    result = await worker.process_next_job(session)

    assert result is not None
    assert result.status == IngestionStatus.COMPLETED
    assert result.error_message is None


async def test_worker_marks_failed_when_extraction_fails(tmp_path: Path) -> None:
    """The real default processing step should fail the job when the stored file is missing."""
    missing_path = tmp_path / "does_not_exist.txt"

    session = _FakeAsyncSession()
    _add_document_and_job(session, stored_path=str(missing_path))
    worker = IngestionWorker()

    result = await worker.process_next_job(session)

    assert result is not None
    assert result.status == IngestionStatus.FAILED
    assert result.error_message is not None
    assert "not found" in result.error_message.lower()
