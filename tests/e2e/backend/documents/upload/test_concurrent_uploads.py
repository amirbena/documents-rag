"""Backend E2E: identical-content uploads racing an in-flight or genuinely concurrent lifecycle.

Same real-HTTP, real-Postgres/Qdrant, fake-AI-provider setup as
test_upload_to_streaming_chat.py/documents/deletion/test_concurrent_requests.py (see
tests/e2e/backend/conftest.py). The "sequential while active" scenario never calls
`process_pending_job` between the two uploads, so the first job deterministically stays PENDING —
no sleeps, no timing luck. The "genuinely concurrent" scenario fires two real HTTP requests via
`asyncio.gather`, each with its own DB session (see conftest.py's `_db_session_override`), proving
the same `uq_documents_content_hash` race-recovery guarantee
tests/integration/documents/upload/test_concurrency.py proves at the service layer, through the
real public API this time.
"""

import asyncio
from collections.abc import Awaitable, Callable

import httpx
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import get_settings
from app.models.ingestion_job import IngestionJob, IngestionStatus
from app.rag.embedding_config import get_active_embedding_config
from app.rag.providers.qdrant_vector_store import QdrantVectorStore
from tests.e2e.backend.fakes import FakeEmbeddingProvider

pytestmark = pytest.mark.e2e

_SEQUENTIAL_ACTIVE_CONTENT = b"Topic: sequential upload while ingestion is still pending.\n"


async def _table_count(session_factory: async_sessionmaker[AsyncSession], table: str) -> int:
    async with session_factory() as session:
        result = await session.execute(text(f"SELECT count(*) FROM {table}"))
        return result.scalar_one()


async def test_second_upload_while_first_ingestion_still_pending_returns_reused_active(
    app_client: httpx.AsyncClient,
    process_pending_job: Callable[[], Awaitable[IngestionJob | None]],
    e2e_session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """A duplicate upload while the first job is still PENDING must reuse it, never race it."""
    first = await app_client.post(
        "/api/v1/documents",
        files={"file": ("a.txt", _SEQUENTIAL_ACTIVE_CONTENT, "text/plain")},
    )
    assert first.status_code == 202
    first_body = first.json()
    assert first_body["status"] == "pending"

    # Deliberately never call process_pending_job() here — the job stays PENDING deterministically.
    second = await app_client.post(
        "/api/v1/documents",
        files={"file": ("b.txt", _SEQUENTIAL_ACTIVE_CONTENT, "text/plain")},
    )

    assert second.status_code == 200
    second_body = second.json()
    assert second_body["outcome"] == "REUSED_ACTIVE"
    assert second_body["document_id"] == first_body["document_id"]
    assert second_body["job_id"] == first_body["job_id"]

    assert await _table_count(e2e_session_factory, "documents") == 1
    assert await _table_count(e2e_session_factory, "ingestion_jobs") == 1


@pytest.mark.parametrize("run", range(3))
async def test_concurrent_identical_http_uploads_converge_on_one_document(
    app_client: httpx.AsyncClient,
    process_pending_job: Callable[[], Awaitable[IngestionJob | None]],
    fake_embedding_provider: FakeEmbeddingProvider,
    e2e_session_factory: async_sessionmaker[AsyncSession],
    run: int,
) -> None:
    """Two genuinely concurrent identical uploads must converge on one document/job, no 500s."""
    content = f"Topic: concurrent race content for run {run}.\n".encode()

    first, second = await asyncio.gather(
        app_client.post("/api/v1/documents", files={"file": ("r1.txt", content, "text/plain")}),
        app_client.post("/api/v1/documents", files={"file": ("r2.txt", content, "text/plain")}),
    )

    assert {first.status_code, second.status_code} == {200, 202}
    winner, loser = (first, second) if first.status_code == 202 else (second, first)
    winner_body, loser_body = winner.json(), loser.json()

    assert winner_body["outcome"] == "CREATED"
    assert loser_body["outcome"] in ("REUSED_ACTIVE", "REUSED_INDEXED", "REUSED_FAILED")
    assert winner_body["document_id"] == loser_body["document_id"]
    assert winner_body["job_id"] == loser_body["job_id"]

    document_id = winner_body["document_id"]
    assert await _table_count(e2e_session_factory, "documents") == 1
    assert await _table_count(e2e_session_factory, "ingestion_jobs") == 1

    result = await process_pending_job()
    assert result is not None
    assert result.status == IngestionStatus.COMPLETED
    # No second job silently exists to be processed.
    assert await process_pending_job() is None

    settings = get_settings()
    active_config = get_active_embedding_config(settings)
    vector_store = QdrantVectorStore(settings=settings)
    query_vector = (await fake_embedding_provider.embed(["concurrent"]))[0]
    search_results = await vector_store.search_similar(active_config.collection_name, query_vector, limit=50)
    matching_chunk_ids = {
        result.chunk_id for result in search_results if result.document_id == document_id
    }
    assert matching_chunk_ids, "the converged document must remain searchable after ingestion"
