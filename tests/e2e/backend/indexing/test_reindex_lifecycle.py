"""Backend E2E: the manual single-document re-index lifecycle, through the real HTTP boundary
(Phase 2.8.6, subtask 6).

Same real-HTTP, real-Postgres/Qdrant, fake-AI-provider setup as
test_successful_deletion.py/test_ingestion_retry_recovery.py (see tests/e2e/backend/conftest.py).
Drives inspect -> schedule -> real ReindexWorker build -> inspect -> activate -> inspect entirely
through `app_client`, never calling a service function directly except to run the existing
out-of-band `ReindexWorker` (mirroring how `process_pending_job`/`process_pending_deletion` drive
their respective workers in sibling E2E modules) — the cleanup worker is deliberately never run
here, since cleanup execution is out of scope for this subtask.
"""

import uuid

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import app.services.indexing.reindex_service as reindex_service_module
from app.core.config import get_settings
from app.models.vector_cleanup_job import VectorCleanupStatus
from app.rag.embedding_config import get_active_embedding_config
from app.rag.providers.qdrant_vector_store import QdrantVectorStore
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


async def test_reindex_lifecycle_inspect_schedule_build_activate(
    app_client: httpx.AsyncClient,
    process_pending_job,
    process_pending_reindex,
    fake_embedding_provider: FakeEmbeddingProvider,
    e2e_session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Document indexed in A; desired config bumped to B; inspect -> schedule -> build -> activate."""
    monkeypatch.setattr(
        reindex_service_module, "get_embedding_provider", lambda settings=None: fake_embedding_provider
    )
    document_id = await upload_and_ingest(app_client, process_pending_job)

    settings = get_settings()
    original_config = get_active_embedding_config(settings)

    # Bump the desired configuration -> B, distinct from A.
    monkeypatch.setattr(settings, "embedding_version", f"v-e2e-{uuid.uuid4().hex[:8]}")
    target_config = get_active_embedding_config(settings)
    assert target_config.collection_name != original_config.collection_name

    # 1. GET reindex state -> reports stale.
    initial_state = await app_client.get(f"/api/v1/documents/{document_id}/reindex")
    assert initial_state.status_code == 200
    initial_body = initial_state.json()
    assert initial_body["is_stale"] is True
    assert initial_body["state"] == "stale"
    assert initial_body["can_schedule"] is True
    assert initial_body["latest_attempt"] is None

    # 2. POST reindex -> returns accepted PENDING attempt.
    schedule_response = await app_client.post(f"/api/v1/documents/{document_id}/reindex")
    assert schedule_response.status_code == 202
    schedule_body = schedule_response.json()
    assert schedule_body["created"] is True
    assert schedule_body["status"] == "pending"
    job_id = schedule_body["job_id"]

    # 3. Run existing ReindexWorker -> attempt becomes COMPLETED.
    worker_result = await process_pending_reindex()
    assert worker_result is not None
    assert worker_result.outcome == ReindexWorkerOutcome.COMPLETED
    assert worker_result.job_id == job_id

    # 4. GET reindex state -> reports target built, not activated; document still serves A.
    built_state = await app_client.get(f"/api/v1/documents/{document_id}/reindex")
    assert built_state.status_code == 200
    built_body = built_state.json()
    assert built_body["state"] == "target_built"
    assert built_body["can_activate"] is True
    assert built_body["latest_attempt"]["status"] == "completed"
    assert built_body["latest_attempt"]["activated_at"] is None
    assert built_body["active_index"]["collection_name"] == original_config.collection_name

    document_detail_before_activation = await app_client.get(f"/api/v1/documents/{document_id}")
    assert document_detail_before_activation.json()["collection_name"] == original_config.collection_name

    # Vectors already exist in B after the worker ran, even though the document still serves A.
    vector_store = QdrantVectorStore(settings=settings)
    query_vector = (await fake_embedding_provider.embed(["notes"]))[0]
    results_in_target_before_activation = await vector_store.search_similar(
        target_config.collection_name, query_vector, limit=10
    )
    assert any(r.document_id == document_id for r in results_in_target_before_activation)

    # 5. POST activate -> activation succeeds.
    activate_response = await app_client.post(f"/api/v1/documents/{document_id}/reindex/activate")
    assert activate_response.status_code == 200
    activate_body = activate_response.json()
    assert activate_body["already_activated"] is False
    assert activate_body["job_id"] == job_id

    # 6. GET reindex state -> reports activated; document now serves B; cleanup job exists for A.
    activated_state = await app_client.get(f"/api/v1/documents/{document_id}/reindex")
    assert activated_state.status_code == 200
    activated_body = activated_state.json()
    assert activated_body["state"] == "activated"
    assert activated_body["is_stale"] is False
    assert activated_body["can_activate"] is False
    assert activated_body["latest_attempt"]["activated_at"] is not None
    assert activated_body["active_index"]["collection_name"] == target_config.collection_name

    # Document metadata points to B only after activation — never before.
    document_detail_after_activation = await app_client.get(f"/api/v1/documents/{document_id}")
    assert document_detail_after_activation.json()["collection_name"] == target_config.collection_name

    # Vectors remain in A after activation — no cleanup execution occurs through this API.
    results_in_original_after_activation = await vector_store.search_similar(
        original_config.collection_name, query_vector, limit=10
    )
    assert any(r.document_id == document_id for r in results_in_original_after_activation)

    # Vectors still exist in B too — activation never deletes anything.
    results_in_target_after_activation = await vector_store.search_similar(
        target_config.collection_name, query_vector, limit=10
    )
    assert any(r.document_id == document_id for r in results_in_target_after_activation)

    # A VectorCleanupJob for A now exists (created by activation), but no cleanup worker ran here.
    async with e2e_session_factory() as session:
        cleanup_rows = (
            await session.execute(
                text("SELECT collection_name, status FROM vector_cleanup_jobs WHERE document_id = :id"),
                {"id": document_id},
            )
        ).all()
    assert len(cleanup_rows) == 1
    assert cleanup_rows[0].collection_name == original_config.collection_name
    assert cleanup_rows[0].status == VectorCleanupStatus.PENDING
