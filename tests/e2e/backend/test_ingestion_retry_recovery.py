"""Backend E2E: ingestion retry and stale-job recovery through the real public HTTP boundary.

Same real-HTTP, real-Postgres/Qdrant, fake-AI-provider setup as
test_upload_to_streaming_chat.py/test_backend_failure_paths.py (see conftest.py). Covers what
tests/integration/test_ingestion_retry_postgres.py cannot: the actual
POST /api/v1/documents/{id}/ingestion/retry contract, and that retry/recovery preserve visible
history through GET .../failure and GET .../ingestion.
"""

from collections.abc import AsyncIterator

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import app.services.ingestion_worker as ingestion_worker_module
from app.core.config import get_settings
from app.models.ingestion_job import IngestionJob, IngestionStatus
from app.services.ingestion_retry_service import (
    STALE_RECOVERY_ERROR_PREFIX,
    recover_stale_ingestion_jobs,
)
from tests.e2e.backend.fakes import FakeEmbeddingProvider

pytestmark = pytest.mark.e2e

_VALID_CONTENT = b"Plain text content for the ingestion retry E2E test.\n"


class _FailOnceThenDelegateEmbeddingProvider:
    """Raises on its first call, then delegates to a real FakeEmbeddingProvider on every later call.

    Simulates a transient provider failure (e.g. a momentary Ollama/Qdrant blip) that a retry
    should recover from — never a real network call, still fully deterministic.
    """

    def __init__(self, vector_size: int) -> None:
        self._delegate = FakeEmbeddingProvider(vector_size=vector_size)
        self._calls = 0

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self._calls += 1
        if self._calls == 1:
            raise RuntimeError("simulated transient embedding provider failure")
        return await self._delegate.embed(texts)


@pytest.fixture
async def app_client_no_provider_override(
    e2e_session_factory: async_sessionmaker[AsyncSession],
    tmp_path,
    isolated_test_state: None,
) -> AsyncIterator[httpx.AsyncClient]:
    """An app_client whose embedding provider is not yet fixed — the test installs its own fake.

    Mirrors conftest.py's app_client fixture exactly, minus e2e_provider_overrides, so each test
    in this module can install a scenario-specific (e.g. fail-once-then-succeed) embedding fake.
    """
    from app.api.v1.routes.documents import get_file_storage
    from app.db.session import get_db_session
    from app.main import app
    from app.storage.local_storage import LocalFileStorage

    def _db_override():
        async def _override():
            async with e2e_session_factory() as session:
                yield session

        return _override

    app.dependency_overrides[get_db_session] = _db_override()
    app.dependency_overrides[get_file_storage] = lambda: LocalFileStorage(root=tmp_path)

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://e2e-testserver") as client:
        try:
            yield client
        finally:
            app.dependency_overrides.pop(get_db_session, None)
            app.dependency_overrides.pop(get_file_storage, None)


async def test_retry_after_failure_creates_new_job_and_preserves_old_failure(
    app_client: httpx.AsyncClient, process_pending_job
) -> None:
    """A FAILED document's retry returns 202 with a new job; the original failure stays visible."""
    upload = await app_client.post(
        "/api/v1/documents",
        files={"file": ("not-really-a.pdf", b"this is not a valid pdf file", "application/pdf")},
    )
    assert upload.status_code == 202
    document_id = upload.json()["document_id"]

    first_result = await process_pending_job()
    assert first_result is not None
    assert first_result.status == IngestionStatus.FAILED

    failure_before = await app_client.get(f"/api/v1/documents/{document_id}/failure")
    assert failure_before.status_code == 200
    old_job_id = failure_before.json()["job_id"]
    assert old_job_id == first_result.id

    retry = await app_client.post(f"/api/v1/documents/{document_id}/ingestion/retry")
    assert retry.status_code == 202
    body = retry.json()
    assert body["created"] is True
    assert body["status"] == "pending"
    new_job_id = body["job_id"]
    assert new_job_id != old_job_id

    # History is preserved: the original failure is still visible via the read API, unmodified,
    # even though a new active job now exists for the same document.
    failure_after = await app_client.get(f"/api/v1/documents/{document_id}/failure")
    assert failure_after.status_code == 200
    assert failure_after.json()["job_id"] == old_job_id

    ingestion = await app_client.get(f"/api/v1/documents/{document_id}/ingestion")
    assert ingestion.status_code == 200
    assert ingestion.json()["job_id"] == new_job_id
    assert ingestion.json()["status"] == "pending"


async def test_retry_after_transient_failure_reaches_completed_and_becomes_searchable(
    app_client_no_provider_override: httpx.AsyncClient,
    process_pending_job,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transient embedding-provider failure fails the first attempt; retry completes and indexes."""
    client = app_client_no_provider_override
    settings = get_settings()
    fake_provider = _FailOnceThenDelegateEmbeddingProvider(vector_size=settings.vector_size)
    monkeypatch.setattr(
        ingestion_worker_module, "get_embedding_provider", lambda settings=None: fake_provider
    )

    upload = await client.post(
        "/api/v1/documents", files={"file": ("notes.txt", _VALID_CONTENT, "text/plain")}
    )
    assert upload.status_code == 202
    document_id = upload.json()["document_id"]

    first_result = await process_pending_job()
    assert first_result is not None
    assert first_result.status == IngestionStatus.FAILED

    retry = await client.post(f"/api/v1/documents/{document_id}/ingestion/retry")
    assert retry.status_code == 202
    assert retry.json()["created"] is True

    second_result = await process_pending_job()
    assert second_result is not None
    assert second_result.status == IngestionStatus.COMPLETED

    detail = await client.get(f"/api/v1/documents/{document_id}")
    assert detail.status_code == 200
    assert detail.json()["status"] == "indexed"
    assert detail.json()["indexed_at"] is not None


async def test_stale_processing_job_is_recovered_and_read_apis_reflect_it(
    app_client: httpx.AsyncClient,
    process_pending_job,
    e2e_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A manufactured stale PROCESSING row is recovered; read APIs show the new job and old history."""
    upload = await app_client.post(
        "/api/v1/documents", files={"file": ("notes.txt", _VALID_CONTENT, "text/plain")}
    )
    assert upload.status_code == 202
    document_id = upload.json()["document_id"]

    # Manufacture a stale PROCESSING row directly: claim the real pending job as PROCESSING, then
    # force its updated_at into the past so it looks abandoned by a dead worker.
    async with e2e_session_factory() as session:
        pending_job_id = upload.json()["job_id"]
        job = await session.get(IngestionJob, pending_job_id)
        assert job is not None
        job.status = IngestionStatus.PROCESSING
        await session.commit()
        await session.execute(
            text("UPDATE ingestion_jobs SET updated_at = now() - interval '1 hour' WHERE id = :id"),
            {"id": pending_job_id},
        )
        await session.commit()

    async with e2e_session_factory() as session:
        result = await recover_stale_ingestion_jobs(session, batch_size=50, stale_after_seconds=60)
    assert result.count == 1
    assert result.recovered[0].stale_job_id == pending_job_id

    failure = await app_client.get(f"/api/v1/documents/{document_id}/failure")
    assert failure.status_code == 200
    assert failure.json()["job_id"] == pending_job_id

    async with e2e_session_factory() as session:
        recovered_job = await session.get(IngestionJob, pending_job_id)
        assert recovered_job is not None
        assert recovered_job.status == IngestionStatus.FAILED
        assert recovered_job.error_message is not None
        assert recovered_job.error_message.startswith(STALE_RECOVERY_ERROR_PREFIX)

    ingestion = await app_client.get(f"/api/v1/documents/{document_id}/ingestion")
    assert ingestion.status_code == 200
    replacement_job_id = ingestion.json()["job_id"]
    assert replacement_job_id != pending_job_id
    assert ingestion.json()["status"] == "pending"

    # The replacement job completes normally through the standard worker path.
    completed = await process_pending_job()
    assert completed is not None
    assert completed.id == replacement_job_id
    assert completed.status == IngestionStatus.COMPLETED
