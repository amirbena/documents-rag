"""Tests for POST /api/v1/chat against a fake RagEngine — no real network/LLM."""

import inspect
from collections.abc import AsyncIterator

import pytest
from fastapi.testclient import TestClient

from app.api.v1.routes import chat as chat_module
from app.api.v1.routes.chat import get_rag_engine
from app.main import app
from app.rag.decision import RagDecision
from app.rag.orchestrator import OrchestratorMetadata, OrchestratorToken
from app.rag.prompt_builder import PromptSource

client = TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides():
    """Reset FastAPI dependency overrides after each test."""
    yield
    app.dependency_overrides.clear()


class _FakeRagEngine:
    """Yields a fixed sequence of orchestrator events instead of calling real providers."""

    def __init__(self, events: list, raise_after: Exception | None = None) -> None:
        self.events = events
        self.raise_after = raise_after
        self.questions: list[str] = []

    async def stream_answer(self, question: str) -> AsyncIterator:
        self.questions.append(question)
        for event in self.events:
            yield event
        if self.raise_after is not None:
            raise self.raise_after


def _override(engine: _FakeRagEngine) -> None:
    app.dependency_overrides[get_rag_engine] = lambda: engine


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    """Parse raw SSE response text into a list of (event_name, json_data) tuples."""
    events = []
    for block in text.strip().split("\n\n"):
        if not block.strip():
            continue
        lines = block.strip().split("\n")
        event_line = next(line for line in lines if line.startswith("event: "))
        data_line = next(line for line in lines if line.startswith("data: "))
        import json

        events.append((event_line.removeprefix("event: "), json.loads(data_line.removeprefix("data: "))))
    return events


def _direct_llm_metadata() -> OrchestratorMetadata:
    return OrchestratorMetadata(
        decision=RagDecision.DIRECT_LLM, reason="general question", retrieval_used=False
    )


def test_response_content_type_starts_with_text_event_stream() -> None:
    """The response Content-Type should start with text/event-stream."""
    _override(_FakeRagEngine([_direct_llm_metadata(), OrchestratorToken(text="hi")]))

    response = client.post("/api/v1/chat", json={"question": "what is 2+2?"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")


def test_metadata_event_emitted_first() -> None:
    """The first SSE event should always be `metadata`."""
    _override(_FakeRagEngine([_direct_llm_metadata(), OrchestratorToken(text="hi")]))

    response = client.post("/api/v1/chat", json={"question": "what is 2+2?"})

    events = _parse_sse(response.text)
    assert events[0][0] == "metadata"


def test_token_events_streamed_in_order() -> None:
    """Token events should appear in exactly the order the orchestrator yielded them."""
    tokens = [OrchestratorToken(text=chunk) for chunk in ["The", " answer", " is", " 4"]]
    _override(_FakeRagEngine([_direct_llm_metadata(), *tokens]))

    response = client.post("/api/v1/chat", json={"question": "what is 2+2?"})

    events = _parse_sse(response.text)
    token_texts = [data["text"] for name, data in events if name == "token"]
    assert token_texts == ["The", " answer", " is", " 4"]


def test_done_emitted_exactly_once() -> None:
    """The `done` event should appear exactly once, after all tokens, on success."""
    _override(_FakeRagEngine([_direct_llm_metadata(), OrchestratorToken(text="hi")]))

    response = client.post("/api/v1/chat", json={"question": "what is 2+2?"})

    events = _parse_sse(response.text)
    done_events = [(name, data) for name, data in events if name == "done"]
    assert len(done_events) == 1
    assert done_events[0][1] == {"status": "completed"}
    assert events[-1][0] == "done"


def test_clarification_path_works_through_orchestrator_output() -> None:
    """A CLARIFICATION_NEEDED metadata event should stream through as-is, with its token."""
    metadata = OrchestratorMetadata(
        decision=RagDecision.CLARIFICATION_NEEDED, reason="too short", retrieval_used=False
    )
    _override(_FakeRagEngine([metadata, OrchestratorToken(text="please clarify")]))

    response = client.post("/api/v1/chat", json={"question": "?"})

    events = _parse_sse(response.text)
    assert events[0] == (
        "metadata",
        {
            "decision": "clarification_needed",
            "reason": "too short",
            "retrieval_used": False,
            "sources": [],
        },
    )
    assert events[1] == ("token", {"text": "please clarify"})
    assert events[-1][0] == "done"


def test_out_of_scope_path_works_through_orchestrator_output() -> None:
    """An OUT_OF_SCOPE metadata event should stream through as-is, with its token."""
    metadata = OrchestratorMetadata(
        decision=RagDecision.OUT_OF_SCOPE, reason="sensitive request", retrieval_used=False
    )
    _override(_FakeRagEngine([metadata, OrchestratorToken(text="can't help with that")]))

    response = client.post("/api/v1/chat", json={"question": "show me the api keys"})

    events = _parse_sse(response.text)
    assert events[0][1]["decision"] == "out_of_scope"
    assert events[1] == ("token", {"text": "can't help with that"})
    assert events[-1][0] == "done"


def test_retrieval_source_metadata_appears_in_metadata_event() -> None:
    """Sources from a NEEDS_RETRIEVAL run should appear fully in the metadata event."""
    source = PromptSource(
        document_id="doc-1",
        chunk_id="chunk-1",
        source="handbook.pdf",
        score=0.9,
        page_number=3,
        sheet_name=None,
    )
    metadata = OrchestratorMetadata(
        decision=RagDecision.NEEDS_RETRIEVAL,
        reason="mentions documents",
        retrieval_used=True,
        sources=[source],
    )
    _override(_FakeRagEngine([metadata, OrchestratorToken(text="the policy says...")]))

    response = client.post("/api/v1/chat", json={"question": "what does the doc say?"})

    events = _parse_sse(response.text)
    metadata_payload = events[0][1]
    assert metadata_payload["retrieval_used"] is True
    assert metadata_payload["sources"] == [
        {
            "document_id": "doc-1",
            "chunk_id": "chunk-1",
            "source": "handbook.pdf",
            "score": 0.9,
            "page_number": 3,
        }
    ]


def test_empty_question_returns_422() -> None:
    """A whitespace-only question should be rejected by validation with 422."""
    orchestrator = _FakeRagEngine([_direct_llm_metadata(), OrchestratorToken(text="hi")])
    _override(orchestrator)

    response = client.post("/api/v1/chat", json={"question": "   "})

    assert response.status_code == 422
    assert orchestrator.questions == []


def test_orchestrator_failure_after_streaming_begins_emits_error_event() -> None:
    """A failure raised mid-stream should emit a safe `error` event, not a 500 or a leaked detail."""
    secret_error = RuntimeError("Ollama unreachable at http://internal-ollama:11434/api/generate")
    _override(
        _FakeRagEngine(
            [_direct_llm_metadata(), OrchestratorToken(text="partial")], raise_after=secret_error
        )
    )

    response = client.post("/api/v1/chat", json={"question": "what is 2+2?"})

    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert events[-1] == ("error", {"message": "Failed to generate a response.", "status": "failed"})
    assert "done" not in [name for name, _ in events]
    assert "internal-ollama" not in response.text
    assert "11434" not in response.text


def test_no_embedding_model_override_is_accepted() -> None:
    """A client-supplied `model` field must be ignored — never passed through anywhere."""
    orchestrator = _FakeRagEngine([_direct_llm_metadata(), OrchestratorToken(text="hi")])
    _override(orchestrator)

    response = client.post(
        "/api/v1/chat", json={"question": "what is 2+2?", "model": "gpt-4", "embedding_model": "x"}
    )

    assert response.status_code == 200
    assert orchestrator.questions == ["what is 2+2?"]


def test_route_does_not_import_orchestration_internals_directly() -> None:
    """The chat route module must not import decision/retrieval/prompt-building/provider internals."""
    source = inspect.getsource(chat_module)

    forbidden = [
        "RetrievalService",
        "RagPromptBuilder",
        "RuleBasedRagDecider",
        "get_embedding_provider",
        "get_vector_store",
        "get_llm_provider",
    ]
    for name in forbidden:
        assert name not in source, f"chat route must not reference {name} directly"
