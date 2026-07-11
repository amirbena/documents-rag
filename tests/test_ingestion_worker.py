"""Tests for IngestionWorker against a fake session double — no real database.

The worker is designed around Postgres-specific locking semantics
(`SELECT ... FOR UPDATE SKIP LOCKED`), which SQLite does not represent correctly even when it
accepts the same SQLAlchemy call — so these tests use a fake session that faithfully simulates
the WHERE status='pending' ... LIMIT 1 filter and Document lookup, without any real database.
"""

import uuid
from pathlib import Path
from typing import Any

import pytest

import app.services.ingestion_worker as ingestion_worker_module
from app.core.config import get_settings
from app.models.document import Document
from app.models.ingestion_job import IngestionJob, IngestionStatus
from app.rag.embedding_config import get_active_embedding_config
from app.rag.providers.vector_store import VectorPoint
from app.services.document_chunker import DocumentChunker
from app.services.ingestion_worker import IngestionWorker


@pytest.fixture(autouse=True)
def _fake_embedding_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    """Match the active config's dimension to _FakeEmbeddingProvider's 3-dim output.

    Required since app.rag.embedding_validation.validate_embeddings now rejects any embedding
    batch whose vector length doesn't match the active EmbeddingIndexConfig.dimension.
    """
    monkeypatch.setattr(get_settings(), "vector_size", 3)


class _FakeEmbeddingProvider:
    """Records the texts it's asked to embed and returns one fixed-length vector per text."""

    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or [0.1, 0.2, 0.3]
        self.embed_calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(texts)
        return [self.vector for _ in texts]


class _FailingEmbeddingProvider:
    """Always raises, simulating an embedding provider failure."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding provider unavailable")


class _FakeVectorStore:
    """Records collection creation and upserted points instead of calling real Qdrant."""

    def __init__(self) -> None:
        self.created_collections: list[tuple[str, int]] = []
        self.upserted_points: list[VectorPoint] = []

    async def create_collection_if_not_exists(self, collection_name: str, vector_size: int) -> None:
        self.created_collections.append((collection_name, vector_size))

    async def upsert_vectors(self, collection_name: str, points: list[VectorPoint]) -> None:
        self.upserted_points.extend(points)

    async def get_collection_vector_size(self, collection_name: str) -> int | None:
        return None

    async def delete_by_document_id(self, collection_name: str, document_id: str) -> None:
        return None


class _FailingVectorStore:
    """create_collection_if_not_exists succeeds; upsert_vectors always raises."""

    async def create_collection_if_not_exists(self, collection_name: str, vector_size: int) -> None:
        return None

    async def upsert_vectors(self, collection_name: str, points: list[VectorPoint]) -> None:
        raise RuntimeError("vector store unavailable")

    async def get_collection_vector_size(self, collection_name: str) -> int | None:
        return None

    async def delete_by_document_id(self, collection_name: str, document_id: str) -> None:
        return None


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

    async def _noop(document: Document | None, job: IngestionJob, session: object) -> None:
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

    async def _boom(document: Document | None, job: IngestionJob, session: object) -> None:
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

    async def _counting_process(document: Document | None, job: IngestionJob, session: object) -> None:
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

    async def _recording_process(document: Document | None, job: IngestionJob, session: object) -> None:
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


async def test_worker_never_imports_llm_provider() -> None:
    """The worker module must not import LLMProvider — no chat/generation calls from ingestion."""
    module_names = vars(ingestion_worker_module)
    assert "LLMProvider" not in module_names
    assert "get_llm_provider" not in module_names


async def test_worker_marks_completed_when_extraction_succeeds(
    tmp_path: Path, monkeypatch
) -> None:
    """The real default processing step should complete the job on success."""
    monkeypatch.setattr(
        ingestion_worker_module, "get_embedding_provider", lambda settings: _FakeEmbeddingProvider()
    )
    monkeypatch.setattr(
        ingestion_worker_module, "get_vector_store", lambda settings: _FakeVectorStore()
    )

    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world", encoding="utf-8")

    session = _FakeAsyncSession()
    _add_document_and_job(session, stored_path=str(file_path))
    worker = IngestionWorker()

    result = await worker.process_next_job(session)

    assert result is not None
    assert result.status == IngestionStatus.COMPLETED
    assert result.error_message is None


async def test_worker_marks_zero_chunk_document_indexed_with_no_vectors(
    tmp_path: Path, monkeypatch
) -> None:
    """A document producing zero chunks is still marked indexed, with no vectors written."""
    vector_store = _FakeVectorStore()
    monkeypatch.setattr(
        ingestion_worker_module, "get_embedding_provider", lambda settings: _FakeEmbeddingProvider()
    )
    monkeypatch.setattr(ingestion_worker_module, "get_vector_store", lambda settings: vector_store)
    monkeypatch.setattr(DocumentChunker, "chunk", lambda self, extracted: [])

    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world", encoding="utf-8")

    session = _FakeAsyncSession()
    job = _add_document_and_job(session, stored_path=str(file_path))
    document = session._documents[job.document_id]
    worker = IngestionWorker()

    result = await worker.process_next_job(session)

    assert result is not None
    assert result.status == IngestionStatus.COMPLETED
    assert document.indexed_at is not None
    assert document.collection_name == get_active_embedding_config().collection_name
    assert vector_store.upserted_points == []


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


async def test_worker_default_pipeline_extracts_then_chunks(tmp_path: Path, monkeypatch) -> None:
    """The real default processing step should extract text, then hand it to DocumentChunker."""
    monkeypatch.setattr(
        ingestion_worker_module, "get_embedding_provider", lambda settings: _FakeEmbeddingProvider()
    )
    monkeypatch.setattr(
        ingestion_worker_module, "get_vector_store", lambda settings: _FakeVectorStore()
    )

    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world " * 100, encoding="utf-8")

    chunk_calls = []
    original_chunk = DocumentChunker.chunk

    def _spying_chunk(self, extracted):
        chunk_calls.append(extracted)
        return original_chunk(self, extracted)

    monkeypatch.setattr(DocumentChunker, "chunk", _spying_chunk)

    session = _FakeAsyncSession()
    _add_document_and_job(session, stored_path=str(file_path))
    worker = IngestionWorker()

    result = await worker.process_next_job(session)

    assert result is not None
    assert result.status == IngestionStatus.COMPLETED
    assert len(chunk_calls) == 1
    assert chunk_calls[0].full_text.strip() != ""


async def test_worker_embeds_each_chunk(tmp_path: Path, monkeypatch) -> None:
    """The default pipeline should call the embedding provider with every chunk's text."""
    embedding_provider = _FakeEmbeddingProvider()
    monkeypatch.setattr(
        ingestion_worker_module, "get_embedding_provider", lambda settings: embedding_provider
    )
    monkeypatch.setattr(
        ingestion_worker_module, "get_vector_store", lambda settings: _FakeVectorStore()
    )

    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world " * 300, encoding="utf-8")

    session = _FakeAsyncSession()
    _add_document_and_job(session, stored_path=str(file_path))
    worker = IngestionWorker()

    result = await worker.process_next_job(session)

    assert result is not None
    assert result.status == IngestionStatus.COMPLETED
    assert len(embedding_provider.embed_calls) == 1
    embedded_texts = embedding_provider.embed_calls[0]
    assert len(embedded_texts) > 1
    assert all(text.strip() for text in embedded_texts)


async def test_worker_upserts_vectors_preserving_metadata(tmp_path: Path, monkeypatch) -> None:
    """Upserted VectorPoints should preserve document_id, chunk_id, text, source, page_number."""
    monkeypatch.setattr(
        ingestion_worker_module,
        "get_embedding_provider",
        lambda settings: _FakeEmbeddingProvider(vector=[1.0, 2.0, 3.0]),
    )
    vector_store = _FakeVectorStore()
    monkeypatch.setattr(ingestion_worker_module, "get_vector_store", lambda settings: vector_store)

    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world", encoding="utf-8")

    session = _FakeAsyncSession()
    job = _add_document_and_job(session, stored_path=str(file_path))
    document = session._documents[job.document_id]
    worker = IngestionWorker()

    result = await worker.process_next_job(session)

    assert result is not None
    assert result.status == IngestionStatus.COMPLETED
    active_config = get_active_embedding_config()
    assert vector_store.created_collections == [(active_config.collection_name, active_config.dimension)]
    assert document.collection_name == active_config.collection_name
    assert document.embedding_version == active_config.embedding_version
    assert document.indexed_at is not None
    assert len(vector_store.upserted_points) == 1
    point = vector_store.upserted_points[0]
    assert point.vector == [1.0, 2.0, 3.0]
    assert point.document_id == document.id
    assert point.chunk_id == f"{document.id}-0"
    assert point.text == "hello world"
    assert point.source == document.original_filename
    assert point.page_number is None


async def test_worker_marks_failed_when_embedding_fails(tmp_path: Path, monkeypatch) -> None:
    """A failure inside the embedding provider should mark the job failed with the error stored."""
    monkeypatch.setattr(
        ingestion_worker_module, "get_embedding_provider", lambda settings: _FailingEmbeddingProvider()
    )
    monkeypatch.setattr(
        ingestion_worker_module, "get_vector_store", lambda settings: _FakeVectorStore()
    )

    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world", encoding="utf-8")

    session = _FakeAsyncSession()
    _add_document_and_job(session, stored_path=str(file_path))
    worker = IngestionWorker()

    result = await worker.process_next_job(session)

    assert result is not None
    assert result.status == IngestionStatus.FAILED
    assert result.error_message == "embedding provider unavailable"


async def test_worker_marks_failed_when_vector_store_fails(tmp_path: Path, monkeypatch) -> None:
    """A failure inside the vector store upsert should mark the job failed with the error stored."""
    monkeypatch.setattr(
        ingestion_worker_module, "get_embedding_provider", lambda settings: _FakeEmbeddingProvider()
    )
    monkeypatch.setattr(
        ingestion_worker_module, "get_vector_store", lambda settings: _FailingVectorStore()
    )

    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world", encoding="utf-8")

    session = _FakeAsyncSession()
    _add_document_and_job(session, stored_path=str(file_path))
    worker = IngestionWorker()

    result = await worker.process_next_job(session)

    assert result is not None
    assert result.status == IngestionStatus.FAILED
    assert result.error_message == "vector store unavailable"


async def test_worker_never_calls_llm_provider_during_ingestion(tmp_path: Path, monkeypatch) -> None:
    """The default pipeline must never invoke an LLM — ingestion only embeds and indexes."""

    def _fail_if_called(settings=None):
        raise AssertionError("get_llm_provider must never be called during ingestion")

    monkeypatch.setattr(
        ingestion_worker_module, "get_embedding_provider", lambda settings: _FakeEmbeddingProvider()
    )
    monkeypatch.setattr(
        ingestion_worker_module, "get_vector_store", lambda settings: _FakeVectorStore()
    )
    monkeypatch.setattr(
        "app.rag.providers.provider_factory.get_llm_provider", _fail_if_called
    )

    file_path = tmp_path / "notes.txt"
    file_path.write_text("hello world", encoding="utf-8")

    session = _FakeAsyncSession()
    _add_document_and_job(session, stored_path=str(file_path))
    worker = IngestionWorker()

    result = await worker.process_next_job(session)

    assert result is not None
    assert result.status == IngestionStatus.COMPLETED
