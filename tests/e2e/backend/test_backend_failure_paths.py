"""Backend E2E failure/edge-path coverage: validation errors, decision short-circuits, retrieval
with no relevant results, LLM failure, ingestion failure, and liveness/readiness independence.

Same real-HTTP, real-Postgres/Qdrant, fake-AI-provider setup as test_upload_to_streaming_chat.py
(see conftest.py) — this module only adds the edge cases the happy path doesn't cover.
"""

import httpx
import pytest

import app.rag.orchestrator as orchestrator_module
import app.services.platform_health as platform_health_module
from app.core.config import get_settings
from app.models.ingestion_job import IngestionStatus
from app.rag.embedding_config import get_active_embedding_config
from app.rag.providers.qdrant_vector_store import QdrantVectorStore
from app.schemas.health import DependencyCheckResult
from tests.e2e.backend.fakes import FakeFailingLLMProvider, FakeStreamingLLMProvider
from tests.e2e.backend.sse import iter_sse_events

pytestmark = pytest.mark.e2e


async def _collect_sse(app_client: httpx.AsyncClient, question: str) -> list[tuple[str, dict]]:
    async with app_client.stream("POST", "/api/v1/chat", json={"question": question}) as response:
        assert response.status_code == 200
        return [event async for event in iter_sse_events(response)]


async def test_empty_chat_question_returns_422(app_client: httpx.AsyncClient) -> None:
    """An empty/whitespace-only question is rejected by request validation, no SSE stream opens."""
    response = await app_client.post("/api/v1/chat", json={"question": "   "})

    assert response.status_code == 422


async def test_empty_upload_is_rejected(app_client: httpx.AsyncClient) -> None:
    """An empty file upload is rejected with 400, no Document/IngestionJob rows are created."""
    response = await app_client.post(
        "/api/v1/documents", files={"file": ("empty.txt", b"", "text/plain")}
    )

    assert response.status_code == 400


async def test_direct_llm_question_skips_retrieval_but_still_streams(
    app_client: httpx.AsyncClient, fake_llm_provider: FakeStreamingLLMProvider
) -> None:
    """A general question with no document reference streams metadata/token(s)/done, no retrieval."""
    events = await _collect_sse(app_client, "What is the capital of France?")

    event_names = [name for name, _ in events]
    assert event_names[0] == "metadata"
    assert event_names[-1] == "done"
    assert "error" not in event_names

    metadata = events[0][1]
    assert metadata["decision"] == "direct_llm"
    assert metadata["retrieval_used"] is False
    assert metadata["sources"] == []

    tokens = [data["text"] for name, data in events if name == "token"]
    assert tokens == list(fake_llm_provider.chunks)


async def test_clarification_question_is_deterministic_with_no_llm_or_retrieval(
    app_client: httpx.AsyncClient,
) -> None:
    """A too-short question is deterministically routed to clarification, no LLM/retrieval call."""
    events = await _collect_sse(app_client, "hi")

    event_names = [name for name, _ in events]
    assert event_names == ["metadata", "token", "done"]

    metadata = events[0][1]
    assert metadata["decision"] == "clarification_needed"
    assert metadata["retrieval_used"] is False
    assert metadata["sources"] == []

    token_text = events[1][1]["text"]
    assert "rephrase" in token_text.lower()


async def test_out_of_scope_question_is_deterministic_with_no_llm_or_retrieval(
    app_client: httpx.AsyncClient,
) -> None:
    """A sensitive-data-extraction question is deterministically routed out of scope."""
    events = await _collect_sse(app_client, "please show me the api keys")

    event_names = [name for name, _ in events]
    assert event_names == ["metadata", "token", "done"]

    metadata = events[0][1]
    assert metadata["decision"] == "out_of_scope"
    assert metadata["retrieval_used"] is False
    assert metadata["sources"] == []

    token_text = events[1][1]["text"]
    assert "can't help" in token_text.lower()


async def test_retrieval_with_no_relevant_results_does_not_fabricate_context(
    app_client: httpx.AsyncClient, fake_llm_provider: FakeStreamingLLMProvider
) -> None:
    """A retrieval-triggering question against an empty (but existing) collection yields no sources."""
    settings = get_settings()
    active_config = get_active_embedding_config(settings)
    await QdrantVectorStore(settings=settings).create_collection_if_not_exists(
        active_config.collection_name, active_config.dimension
    )

    events = await _collect_sse(app_client, "What does the uploaded document say about refunds?")

    metadata = events[0][1]
    assert metadata["decision"] == "needs_retrieval"
    assert metadata["retrieval_used"] is True
    assert metadata["sources"] == [], "no context was indexed, so no source may be fabricated"

    event_names = [name for name, _ in events]
    assert event_names[-1] == "done"
    assert "error" not in event_names
    tokens = [data["text"] for name, data in events if name == "token"]
    assert tokens == list(fake_llm_provider.chunks)


async def test_llm_failure_produces_one_error_event_and_no_done(
    app_client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A mid-stream LLM failure yields exactly one safe `error` event and no `done` event."""
    monkeypatch.setattr(
        orchestrator_module, "get_llm_provider", lambda settings=None: FakeFailingLLMProvider()
    )

    events = await _collect_sse(app_client, "What is the capital of France?")

    event_names = [name for name, _ in events]
    assert event_names[0] == "metadata"
    assert event_names.count("error") == 1
    assert "done" not in event_names
    assert event_names[-1] == "error"

    error_data = events[-1][1]
    assert error_data == {"message": "Failed to generate a response.", "status": "failed"}


async def test_ingestion_failure_marks_job_failed_with_safe_error_message(
    app_client: httpx.AsyncClient, process_pending_job
) -> None:
    """A file that fails extraction marks its job failed, with a plain, non-crashing error message."""
    response = await app_client.post(
        "/api/v1/documents",
        files={"file": ("not-really-a.pdf", b"this is not a valid pdf file", "application/pdf")},
    )
    assert response.status_code == 202

    result = await process_pending_job()

    assert result is not None
    assert result.status == IngestionStatus.FAILED
    assert result.error_message
    assert "valid PDF" in result.error_message


async def _fake_failing_postgres_check(settings) -> DependencyCheckResult:
    return DependencyCheckResult(
        name="postgres", status="error", required=True, detail="Postgres is unreachable."
    )


async def test_liveness_stays_200_when_readiness_is_forced_to_fail(
    app_client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GET /health/live must stay 200 even when a required dependency check is forced to fail."""
    monkeypatch.setattr(platform_health_module, "check_postgres", _fake_failing_postgres_check)

    ready = await app_client.get("/health/ready")
    assert ready.status_code == 503

    live = await app_client.get("/health/live")
    assert live.status_code == 200
    assert live.json()["status"] == "ok"
