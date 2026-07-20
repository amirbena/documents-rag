"""Backend E2E: the complete manual re-index lifecycle, through the real HTTP boundary and the real
historical-cleanup worker (Phase 2.8.6, subtasks 6 and 7 — the canonical complete lifecycle test).

Same real-HTTP, real-Postgres/Qdrant, fake-AI-provider setup as
test_successful_deletion.py/test_ingestion_retry_recovery.py (see tests/e2e/backend/conftest.py).
Drives inspect -> schedule -> real ReindexWorker build -> inspect -> activate -> inspect -> real
historical cleanup entirely through `app_client` plus the existing out-of-band
`ReindexWorker`/`process_next_vector_cleanup_job()` (mirroring how `process_pending_job`/
`process_pending_deletion` drive their respective workers in sibling E2E modules) — cleanup
execution is deliberately never triggered through the API itself, only through the same
out-of-band worker function a real deployment's own scheduled runner would call.
"""

import uuid

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import app.services.indexing.reindex_service as reindex_service_module
from app.core.config import get_settings
from app.models.ingestion_job import IngestionStatus
from app.models.vector_cleanup_job import VectorCleanupStatus
from app.rag.embedding_config import get_active_embedding_config
from app.rag.providers.qdrant_vector_store import QdrantVectorStore
from app.services.indexing.cleanup_job_service import (
    VectorCleanupWorkerOutcome,
    process_next_vector_cleanup_job,
)
from app.services.indexing.reindex_worker import ReindexWorker, ReindexWorkerOutcome
from app.storage.local_storage import LocalFileStorage
from tests.e2e.backend.documents.deletion.support import upload_and_ingest
from tests.e2e.backend.fakes import FakeEmbeddingProvider

pytestmark = pytest.mark.e2e


async def _upload_and_ingest_unrelated(app_client: httpx.AsyncClient, process_pending_job) -> str:
    """Upload a second, content-distinct document and drive it to COMPLETED — distinct content
    avoids Phase 2.8.5 upload deduplication reusing the first document's row/vectors."""
    upload = await app_client.post(
        "/api/v1/documents",
        files={"file": ("unrelated.txt", b"An unrelated document's content.\n", "text/plain")},
    )
    assert upload.status_code == 202
    document_id = upload.json()["document_id"]

    result = await process_pending_job()
    assert result is not None
    assert result.status == IngestionStatus.COMPLETED
    return document_id


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


@pytest.fixture
def process_pending_cleanup(
    e2e_session_factory: async_sessionmaker[AsyncSession],
    isolated_test_state: None,
):
    """Run process_next_vector_cleanup_job() against one pending cleanup job (real Qdrant)."""
    vector_store = QdrantVectorStore(settings=get_settings())

    async def _process():
        async with e2e_session_factory() as session:
            return await process_next_vector_cleanup_job(session, vector_store)

    return _process


async def test_delete_returns_409_while_reindex_job_is_active(
    app_client: httpx.AsyncClient,
    process_pending_job,
    fake_embedding_provider: FakeEmbeddingProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DELETE against a document with an active (PENDING) ReindexJob must return 409, not 500 —
    regression test for the previously-uncaught `assert result.job is not None` path."""
    monkeypatch.setattr(
        reindex_service_module, "get_embedding_provider", lambda settings=None: fake_embedding_provider
    )
    document_id = await upload_and_ingest(app_client, process_pending_job)

    settings = get_settings()
    monkeypatch.setattr(settings, "embedding_version", f"v-e2e-{uuid.uuid4().hex[:8]}")

    schedule_response = await app_client.post(f"/api/v1/documents/{document_id}/reindex")
    assert schedule_response.status_code == 202
    assert schedule_response.json()["status"] == "pending"

    delete_response = await app_client.delete(f"/api/v1/documents/{document_id}")

    assert delete_response.status_code == 409
    assert "AssertionError" not in delete_response.text

    # No deletion job was created — the document's lifecycle status is unaffected.
    detail = await app_client.get(f"/api/v1/documents/{document_id}")
    assert detail.json()["status"] != "deleted"


async def test_reindex_lifecycle_inspect_schedule_build_activate_cleanup(
    app_client: httpx.AsyncClient,
    process_pending_job,
    process_pending_reindex,
    process_pending_cleanup,
    fake_embedding_provider: FakeEmbeddingProvider,
    e2e_session_factory: async_sessionmaker[AsyncSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Document indexed in A; desired config bumped to B; inspect -> schedule -> build -> activate
    -> historical cleanup. An unrelated document stays in A throughout, proving cleanup is scoped
    to exactly one document's vectors."""
    monkeypatch.setattr(
        reindex_service_module, "get_embedding_provider", lambda settings=None: fake_embedding_provider
    )
    document_id = await upload_and_ingest(app_client, process_pending_job)
    unrelated_document_id = await _upload_and_ingest_unrelated(app_client, process_pending_job)

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

    # 7. Run the existing historical-cleanup worker -> the cleanup job succeeds.
    cleanup_result = await process_pending_cleanup()
    assert cleanup_result.outcome == VectorCleanupWorkerOutcome.COMPLETED
    assert cleanup_result.collection_name == original_config.collection_name

    async with e2e_session_factory() as session:
        completed_cleanup_rows = (
            await session.execute(
                text("SELECT status FROM vector_cleanup_jobs WHERE document_id = :id"), {"id": document_id}
            )
        ).all()
    assert len(completed_cleanup_rows) == 1
    assert completed_cleanup_rows[0].status == VectorCleanupStatus.COMPLETED

    # ReindexJob build/activation fields are untouched by cleanup.
    reindex_state_after_cleanup = await app_client.get(f"/api/v1/documents/{document_id}/reindex")
    assert reindex_state_after_cleanup.json()["latest_attempt"]["status"] == "completed"
    assert reindex_state_after_cleanup.json()["latest_attempt"]["activated_at"] is not None

    # Document vectors are now absent from A — only this document's vectors were removed.
    results_in_original_after_cleanup = await vector_store.search_similar(
        original_config.collection_name, query_vector, limit=10
    )
    assert all(r.document_id != document_id for r in results_in_original_after_cleanup)

    # The unrelated document's vectors in A remain untouched by this document's cleanup.
    assert any(r.document_id == unrelated_document_id for r in results_in_original_after_cleanup)

    # Document vectors remain in B — cleanup never touches the active/target collection.
    results_in_target_after_cleanup = await vector_store.search_similar(
        target_config.collection_name, query_vector, limit=10
    )
    assert any(r.document_id == document_id for r in results_in_target_after_cleanup)

    # Document still serves B after cleanup — cleanup never mutates serving metadata.
    document_detail_after_cleanup = await app_client.get(f"/api/v1/documents/{document_id}")
    assert document_detail_after_cleanup.json()["collection_name"] == target_config.collection_name

    # Neither collection A nor B was deleted — only individual document vectors were removed.
    assert await vector_store.get_collection_vector_size(original_config.collection_name) is not None
    assert await vector_store.get_collection_vector_size(target_config.collection_name) is not None

    # Object Storage remains unchanged — the original downloadable content is still intact.
    download_after_cleanup = await app_client.get(f"/api/v1/documents/{document_id}/download")
    assert download_after_cleanup.status_code == 200
    unrelated_download_after_cleanup = await app_client.get(
        f"/api/v1/documents/{unrelated_document_id}/download"
    )
    assert unrelated_download_after_cleanup.status_code == 200
