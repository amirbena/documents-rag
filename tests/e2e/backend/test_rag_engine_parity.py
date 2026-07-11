"""Backend E2E parity test: the same upload/ingest, then the chat flow run under both RAG_ENGINE
settings ('custom' and 'langchain'), comparing the API/SSE/source contract for equivalence.

The generated answer text may legitimately differ between engines (LangChain's prompt
serialization differs from CustomRagEngine's), but decision routing, retrieval usage, source
attribution/ranking, SSE event ordering, and error/no-results behavior must match exactly — the
frontend must not need to know which engine is configured.
"""

import time

import httpx
import pytest

import app.rag.engines.langchain_engine as langchain_engine_module
import app.rag.orchestrator as orchestrator_module
from app.core.config import get_settings
from app.models.ingestion_job import IngestionStatus
from app.rag.providers.qdrant_vector_store import QdrantVectorStore
from tests.e2e.backend.fakes import FakeFailingLLMProvider, FakeStreamingLLMProvider
from tests.e2e.backend.sse import iter_sse_events

pytestmark = pytest.mark.e2e

_DOCUMENT = (
    b"Topic: vacation policy and vacation days entitlement.\n"
    b"Employees are entitled to 20 vacation days per year.\n"
)

_RETRIEVAL_QUESTION = "What does the uploaded document say about vacation days?"
_DIRECT_QUESTION = "What is the capital of France?"


async def _run_chat_and_capture(
    app_client: httpx.AsyncClient, engine_name: str, question: str, monkeypatch: pytest.MonkeyPatch
) -> dict:
    """Run one POST /api/v1/chat under the given RAG_ENGINE, capturing comparison instrumentation."""
    monkeypatch.setattr(get_settings(), "rag_engine", engine_name)

    started = time.monotonic()
    first_token_at: float | None = None
    events: list[tuple[str, dict]] = []
    async with app_client.stream("POST", "/api/v1/chat", json={"question": question}) as response:
        status_code = response.status_code
        async for event_name, data in iter_sse_events(response):
            if event_name == "token" and first_token_at is None:
                first_token_at = time.monotonic()
            events.append((event_name, data))
    total_duration = time.monotonic() - started

    return {
        "engine": engine_name,
        "status_code": status_code,
        "event_names": [name for name, _ in events],
        "metadata": events[0][1] if events else None,
        "tokens": [data["text"] for name, data in events if name == "token"],
        "chunk_count": sum(1 for name, _ in events if name == "token"),
        "time_to_first_token": (first_token_at - started) if first_token_at is not None else None,
        "total_duration": total_duration,
        "failure_type": next((data.get("status") for name, data in events if name == "error"), None),
    }


async def test_both_engines_produce_an_equivalent_api_and_sse_contract(
    app_client: httpx.AsyncClient,
    process_pending_job,
    fake_llm_provider: FakeStreamingLLMProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Custom and LangChain engines must expose the same API/SSE/source contract end to end."""
    upload = await app_client.post(
        "/api/v1/documents", files={"file": ("vacation-policy.txt", _DOCUMENT, "text/plain")}
    )
    assert upload.status_code == 202
    document_id = upload.json()["document_id"]

    processed = await process_pending_job()
    assert processed is not None
    assert processed.status == IngestionStatus.COMPLETED

    custom_result = await _run_chat_and_capture(app_client, "custom", _RETRIEVAL_QUESTION, monkeypatch)
    langchain_result = await _run_chat_and_capture(
        app_client, "langchain", _RETRIEVAL_QUESTION, monkeypatch
    )

    for result in (custom_result, langchain_result):
        assert result["status_code"] == 200
        assert result["event_names"][0] == "metadata"
        assert result["event_names"][-1] == "done"
        assert result["event_names"].count("done") == 1
        assert "error" not in result["event_names"]
        assert result["failure_type"] is None

    assert custom_result["metadata"]["decision"] == "needs_retrieval"
    assert langchain_result["metadata"]["decision"] == "needs_retrieval"
    assert custom_result["metadata"]["retrieval_used"] is True
    assert langchain_result["metadata"]["retrieval_used"] is True

    custom_sources = custom_result["metadata"]["sources"]
    langchain_sources = langchain_result["metadata"]["sources"]
    assert len(custom_sources) == len(langchain_sources) > 0
    assert [s["document_id"] for s in custom_sources] == [s["document_id"] for s in langchain_sources]
    assert [s["chunk_id"] for s in custom_sources] == [s["chunk_id"] for s in langchain_sources]
    assert [s["source"] for s in custom_sources] == [s["source"] for s in langchain_sources]
    assert [s["score"] for s in custom_sources] == [s["score"] for s in langchain_sources]
    assert custom_sources[0]["document_id"] == document_id

    # Both engines run the same fake LLM, so the streamed chunk sequence must match exactly too.
    assert custom_result["tokens"] == list(fake_llm_provider.chunks)
    assert langchain_result["tokens"] == list(fake_llm_provider.chunks)
    assert custom_result["chunk_count"] == langchain_result["chunk_count"]


async def test_both_engines_agree_on_no_results_behavior(
    app_client: httpx.AsyncClient,
    fake_llm_provider: FakeStreamingLLMProvider,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With nothing indexed, both engines must return empty sources — never fabricated context."""
    settings = get_settings()
    await QdrantVectorStore(settings=settings).create_collection_if_not_exists(
        settings.qdrant_collection_name, settings.vector_size
    )

    custom_result = await _run_chat_and_capture(app_client, "custom", _RETRIEVAL_QUESTION, monkeypatch)
    langchain_result = await _run_chat_and_capture(
        app_client, "langchain", _RETRIEVAL_QUESTION, monkeypatch
    )

    for result in (custom_result, langchain_result):
        assert result["metadata"]["decision"] == "needs_retrieval"
        assert result["metadata"]["retrieval_used"] is True
        assert result["metadata"]["sources"] == []
        assert result["event_names"][-1] == "done"
        assert "error" not in result["event_names"]
        assert result["tokens"] == list(fake_llm_provider.chunks)


async def test_both_engines_agree_on_llm_failure_behavior(
    app_client: httpx.AsyncClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing LLM must yield exactly one `error` event and no `done`, for either engine."""
    monkeypatch.setattr(
        orchestrator_module, "get_llm_provider", lambda settings=None: FakeFailingLLMProvider()
    )
    monkeypatch.setattr(
        langchain_engine_module, "get_llm_provider", lambda settings=None: FakeFailingLLMProvider()
    )

    custom_result = await _run_chat_and_capture(app_client, "custom", _DIRECT_QUESTION, monkeypatch)
    langchain_result = await _run_chat_and_capture(app_client, "langchain", _DIRECT_QUESTION, monkeypatch)

    for result in (custom_result, langchain_result):
        assert result["event_names"][0] == "metadata"
        assert result["event_names"].count("error") == 1
        assert "done" not in result["event_names"]
        assert result["failure_type"] == "failed"
