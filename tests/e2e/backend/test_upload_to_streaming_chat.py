"""Main backend E2E happy path: document upload -> ingestion -> retrieval -> streaming chat.

Drives the real FastAPI app over a real ASGI HTTP client, against real ephemeral Postgres and
Qdrant containers, with only the embedding/LLM providers faked (see conftest.py). SSE responses
are consumed incrementally, not as one fully-buffered string, so event order/timing is genuinely
exercised.
"""

import httpx
import pytest

from app.core.config import get_settings
from app.models.ingestion_job import IngestionStatus
from app.rag.embedding_config import get_active_embedding_config
from app.rag.providers.qdrant_vector_store import QdrantVectorStore
from tests.e2e.backend.fakes import FakeEmbeddingProvider, FakeStreamingLLMProvider
from tests.e2e.backend.sse import iter_sse_events

pytestmark = pytest.mark.e2e

_VACATION_DOCUMENT = (
    "מדיניות חופשה שנתית\n"
    "עובדי החברה זכאים ל-20 ימי חופשה בשנה, בהתאם לוותק.\n"
    "Topic: vacation policy and vacation days entitlement.\n"
).encode()

_DECOY_DOCUMENT = (
    b"Pizza dough recipe: flour, water, yeast, salt, and olive oil, kneaded for ten minutes.\n"
)

_QUESTION = "What does the uploaded document say about vacation days?"


async def _upload(app_client: httpx.AsyncClient, content: bytes, filename: str) -> dict:
    """Upload one document and return its 202 JSON body."""
    response = await app_client.post(
        "/api/v1/documents",
        files={"file": (filename, content, "text/plain")},
    )
    assert response.status_code == 202
    return response.json()


async def _process_all_pending_jobs(process_pending_job) -> list:
    """Drain every pending ingestion job through the real IngestionWorker."""
    processed = []
    while True:
        result = await process_pending_job()
        if result is None:
            break
        processed.append(result)
    return processed


async def test_full_backend_flow_upload_to_streaming_chat(
    app_client: httpx.AsyncClient,
    process_pending_job,
    fake_embedding_provider: FakeEmbeddingProvider,
    fake_llm_provider: FakeStreamingLLMProvider,
) -> None:
    """Upload a document, ingest it, retrieve it, and stream a document-grounded answer."""
    # A. Platform health under the E2E dependency overrides.
    live = await app_client.get("/health/live")
    assert live.status_code == 200

    ready = await app_client.get("/health/ready")
    ready_body = ready.json()
    postgres_check = next(check for check in ready_body["checks"] if check["name"] == "postgres")
    qdrant_check = next(check for check in ready_body["checks"] if check["name"] == "qdrant")
    assert postgres_check["status"] == "ok"
    assert qdrant_check["status"] == "ok"
    # Real Ollama is intentionally never run in this suite (see CLAUDE.md's E2E rules), so the
    # ollama checks are expected to fail readiness even though Postgres/Qdrant are genuinely healthy.

    # B. Upload a UTF-8/Hebrew document, plus an irrelevant decoy document.
    vacation_upload = await _upload(app_client, _VACATION_DOCUMENT, "vacation-policy.txt")
    await _upload(app_client, _DECOY_DOCUMENT, "pizza.txt")

    document_id = vacation_upload["document_id"]
    assert vacation_upload["status"] == IngestionStatus.PENDING

    # C. Process both pending ingestion jobs through the real IngestionWorker (real extraction,
    # real chunking, fake embeddings, real Qdrant upsert).
    processed_jobs = await _process_all_pending_jobs(process_pending_job)
    assert len(processed_jobs) == 2
    assert all(job.status == IngestionStatus.COMPLETED for job in processed_jobs)

    # D. Verify persistence: vectors exist in the configured Qdrant collection, metadata preserved.
    settings = get_settings()
    active_config = get_active_embedding_config(settings)
    vector_store = QdrantVectorStore(settings=settings)
    query_vector = (await fake_embedding_provider.embed(["vacation days"]))[0]
    search_results = await vector_store.search_similar(
        active_config.collection_name, query_vector, limit=10
    )
    assert search_results
    vacation_results = [result for result in search_results if result.document_id == document_id]
    assert vacation_results
    assert any("vacation" in result.text.lower() for result in vacation_results)
    assert all(result.source in {"vacation-policy.txt", "pizza.txt"} for result in search_results)

    # E. Ask a document-related question and consume the real SSE stream incrementally.
    async with app_client.stream("POST", "/api/v1/chat", json={"question": _QUESTION}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = [event async for event in iter_sse_events(response)]

    event_names = [name for name, _ in events]
    assert event_names[0] == "metadata"
    assert event_names[-1] == "done"
    assert event_names.count("done") == 1
    assert "error" not in event_names
    metadata_index = event_names.index("metadata")
    token_indices = [index for index, name in enumerate(event_names) if name == "token"]
    assert all(index > metadata_index for index in token_indices), "tokens must follow metadata"

    # F. Verify retrieval metadata: decision, retrieval_used, and source attribution.
    metadata_event = events[0][1]
    assert metadata_event["decision"] == "needs_retrieval"
    assert metadata_event["retrieval_used"] is True
    sources = metadata_event["sources"]
    assert sources, "expected at least one retrieved source"
    assert sources[0]["document_id"] == document_id
    for source in sources:
        assert source["document_id"]
        assert source["chunk_id"]
        assert source["source"]
        assert isinstance(source["score"], int | float)

    # G. Verify the streamed answer: token order preserved, done exactly once, no error.
    tokens = [data["text"] for name, data in events if name == "token"]
    assert tokens == list(fake_llm_provider.chunks)
