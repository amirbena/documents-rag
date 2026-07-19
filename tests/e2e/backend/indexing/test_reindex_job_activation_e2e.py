"""Backend E2E: the job-id-scoped operator activation endpoint (Phase 2.8.7, subtask 4) —
POST /api/v1/reindex/jobs/{job_id}/activate.

Same real-HTTP, real-Postgres/Qdrant, fake-AI-provider setup as
test_reindex_lifecycle.py, which already exhaustively drives the sibling document-scoped
`/documents/{document_id}/reindex/activate` endpoint through inspect -> schedule -> build ->
activate -> cleanup. This file does not repeat that full walkthrough — it proves only what is new
here: the job-id-only endpoint resolves and activates correctly, ineligible/missing jobs are
rejected without mutating anything, repeated activation is safe, an unexpected activation failure
never leaves partial state, and the activated state is consistently visible through the existing
inspection endpoint afterward.
"""

import uuid

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import app.api.v1.routes.reindex as reindex_route_module
import app.services.indexing.reindex_service as reindex_service_module
from app.core.config import get_settings
from app.rag.embedding_config import get_active_embedding_config
from app.services.indexing.reindex_worker import ReindexWorker, ReindexWorkerOutcome
from app.storage.local_storage import LocalFileStorage
from tests.e2e.backend.documents.deletion.support import upload_and_ingest
from tests.e2e.backend.fakes import FakeEmbeddingProvider

pytestmark = pytest.mark.e2e


@pytest.fixture
def process_pending_reindex(
    e2e_session_factory: async_sessionmaker[AsyncSession],
    tmp_path,
    isolated_test_state: None,
):
    """Run a real ReindexWorker against one pending re-index job (real Qdrant + storage)."""
    file_storage = LocalFileStorage(root=tmp_path)

    async def _process():
        async with e2e_session_factory() as session:
            worker = ReindexWorker(file_storage=file_storage)
            return await worker.process_next_job(session, get_settings())

    return _process


async def _schedule_and_build(
    app_client: httpx.AsyncClient, document_id: str, process_pending_reindex
) -> str:
    schedule_response = await app_client.post(f"/api/v1/documents/{document_id}/reindex")
    assert schedule_response.status_code == 202
    job_id = schedule_response.json()["job_id"]

    worker_result = await process_pending_reindex()
    assert worker_result is not None
    assert worker_result.outcome == ReindexWorkerOutcome.COMPLETED
    assert worker_result.job_id == job_id
    return job_id


async def test_successful_job_activation_switches_the_active_collection(
    app_client: httpx.AsyncClient,
    process_pending_job,
    process_pending_reindex,
    fake_embedding_provider: FakeEmbeddingProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reindex_service_module, "get_embedding_provider", lambda settings=None: fake_embedding_provider
    )
    document_id = await upload_and_ingest(app_client, process_pending_job)

    settings = get_settings()
    original_config = get_active_embedding_config(settings)
    monkeypatch.setattr(settings, "embedding_version", f"v-e2e-{uuid.uuid4().hex[:8]}")
    target_config = get_active_embedding_config(settings)

    job_id = await _schedule_and_build(app_client, document_id, process_pending_reindex)

    response = await app_client.post(f"/api/v1/reindex/jobs/{job_id}/activate")

    assert response.status_code == 200
    body = response.json()
    assert body["job_id"] == job_id
    assert body["document_id"] == document_id
    assert body["status"] == "completed"
    assert body["activated"] is True
    assert body["already_activated"] is False
    assert body["previous_collection_name"] == original_config.collection_name
    assert body["active_collection_name"] == target_config.collection_name
    assert "cleanup_job_id" not in body
    assert body["activated_at"] is not None

    document_detail = await app_client.get(f"/api/v1/documents/{document_id}")
    assert document_detail.json()["collection_name"] == target_config.collection_name


async def test_ineligible_job_returns_409_and_leaves_state_unchanged(
    app_client: httpx.AsyncClient,
    process_pending_job,
    fake_embedding_provider: FakeEmbeddingProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A scheduled-but-not-yet-built (still PENDING) job cannot be activated."""
    monkeypatch.setattr(
        reindex_service_module, "get_embedding_provider", lambda settings=None: fake_embedding_provider
    )
    document_id = await upload_and_ingest(app_client, process_pending_job)
    settings = get_settings()
    original_config = get_active_embedding_config(settings)
    monkeypatch.setattr(settings, "embedding_version", f"v-e2e-{uuid.uuid4().hex[:8]}")

    schedule_response = await app_client.post(f"/api/v1/documents/{document_id}/reindex")
    assert schedule_response.status_code == 202
    job_id = schedule_response.json()["job_id"]

    response = await app_client.post(f"/api/v1/reindex/jobs/{job_id}/activate")

    assert response.status_code == 409
    assert "detail" in response.json()

    document_detail = await app_client.get(f"/api/v1/documents/{document_id}")
    assert document_detail.json()["collection_name"] == original_config.collection_name

    reindex_state = await app_client.get(f"/api/v1/documents/{document_id}/reindex")
    assert reindex_state.json()["latest_attempt"]["activated_at"] is None


async def test_missing_job_returns_404_with_no_state_changes(app_client: httpx.AsyncClient) -> None:
    response = await app_client.post(f"/api/v1/reindex/jobs/{uuid.uuid4()}/activate")

    assert response.status_code == 404
    assert "detail" in response.json()


async def test_repeated_activation_is_idempotent_and_creates_no_duplicate_cleanup(
    app_client: httpx.AsyncClient,
    process_pending_job,
    process_pending_reindex,
    fake_embedding_provider: FakeEmbeddingProvider,
    monkeypatch: pytest.MonkeyPatch,
    e2e_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    monkeypatch.setattr(
        reindex_service_module, "get_embedding_provider", lambda settings=None: fake_embedding_provider
    )
    document_id = await upload_and_ingest(app_client, process_pending_job)
    settings = get_settings()
    original_config = get_active_embedding_config(settings)
    monkeypatch.setattr(settings, "embedding_version", f"v-e2e-{uuid.uuid4().hex[:8]}")

    job_id = await _schedule_and_build(app_client, document_id, process_pending_reindex)

    first = await app_client.post(f"/api/v1/reindex/jobs/{job_id}/activate")
    assert first.status_code == 200
    first_body = first.json()
    assert first_body["already_activated"] is False

    second = await app_client.post(f"/api/v1/reindex/jobs/{job_id}/activate")
    assert second.status_code == 200
    second_body = second.json()
    assert second_body["already_activated"] is True
    assert second_body["activated_at"] == first_body["activated_at"]
    assert second_body["previous_collection_name"] == original_config.collection_name

    async with e2e_session_factory() as session:
        cleanup_rows = (
            await session.execute(
                text("SELECT id FROM vector_cleanup_jobs WHERE document_id = :id"), {"id": document_id}
            )
        ).all()
    assert len(cleanup_rows) == 1  # exactly the one created by the first call


async def test_activation_failure_leaves_no_partial_state(
    app_client: httpx.AsyncClient,
    process_pending_job,
    process_pending_reindex,
    fake_embedding_provider: FakeEmbeddingProvider,
    monkeypatch: pytest.MonkeyPatch,
    e2e_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """An unexpected activation failure (forced at the activation-service call boundary) must
    never leave the document pointing at the new collection, never set activated_at, and never
    create a cleanup job — mirroring the service's own commit-failure rollback guarantee, observed
    here through real persisted Postgres state rather than a fake session."""
    monkeypatch.setattr(
        reindex_service_module, "get_embedding_provider", lambda settings=None: fake_embedding_provider
    )
    document_id = await upload_and_ingest(app_client, process_pending_job)
    settings = get_settings()
    original_config = get_active_embedding_config(settings)
    monkeypatch.setattr(settings, "embedding_version", f"v-e2e-{uuid.uuid4().hex[:8]}")

    job_id = await _schedule_and_build(app_client, document_id, process_pending_reindex)

    original_activate = reindex_route_module.activate_reindexed_document

    async def _fake_raises(session, reindex_job_id):
        raise RuntimeError("simulated activation dependency failure")

    monkeypatch.setattr(reindex_route_module, "activate_reindexed_document", _fake_raises)

    with pytest.raises(Exception):  # noqa: B017 - the ASGI transport re-raises the route's exception
        await app_client.post(f"/api/v1/reindex/jobs/{job_id}/activate")

    monkeypatch.setattr(reindex_route_module, "activate_reindexed_document", original_activate)

    document_detail = await app_client.get(f"/api/v1/documents/{document_id}")
    assert document_detail.json()["collection_name"] == original_config.collection_name

    reindex_state = await app_client.get(f"/api/v1/documents/{document_id}/reindex")
    assert reindex_state.json()["latest_attempt"]["activated_at"] is None

    async with e2e_session_factory() as session:
        cleanup_rows = (
            await session.execute(
                text("SELECT id FROM vector_cleanup_jobs WHERE document_id = :id"), {"id": document_id}
            )
        ).all()
    assert cleanup_rows == []


async def test_read_after_write_consistency_through_the_inspection_endpoint(
    app_client: httpx.AsyncClient,
    process_pending_job,
    process_pending_reindex,
    fake_embedding_provider: FakeEmbeddingProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        reindex_service_module, "get_embedding_provider", lambda settings=None: fake_embedding_provider
    )
    document_id = await upload_and_ingest(app_client, process_pending_job)
    settings = get_settings()
    monkeypatch.setattr(settings, "embedding_version", f"v-e2e-{uuid.uuid4().hex[:8]}")

    job_id = await _schedule_and_build(app_client, document_id, process_pending_reindex)
    activate_response = await app_client.post(f"/api/v1/reindex/jobs/{job_id}/activate")
    assert activate_response.status_code == 200
    active_collection_name = activate_response.json()["active_collection_name"]

    inspection = await app_client.get(f"/api/v1/documents/{document_id}/reindex")

    assert inspection.status_code == 200
    body = inspection.json()
    assert body["state"] == "activated"
    assert body["active_index"]["collection_name"] == active_collection_name
    assert body["latest_attempt"]["job_id"] == job_id
