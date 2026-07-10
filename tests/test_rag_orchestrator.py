"""Tests for RagOrchestrator against fake decider/retrieval/LLM — no real network."""

from collections.abc import AsyncIterator

import app.rag.orchestrator as orchestrator_module
from app.rag.decision import DecisionResult, RagDecision
from app.rag.orchestrator import OrchestratorMetadata, OrchestratorToken, RagOrchestrator
from app.rag.prompt_builder import RagPromptBuilder
from app.rag.providers.vector_store import VectorSearchResult


class _FakeDecider:
    """Always returns a fixed DecisionResult, regardless of the question."""

    def __init__(self, decision: RagDecision, reason: str = "fixed decision") -> None:
        self._result = DecisionResult(decision=decision, reason=reason, confidence=1.0)

    def decide(self, question: str) -> DecisionResult:
        return self._result


class _FakeRetrievalService:
    """Records calls and returns a fixed list of results instead of calling real providers."""

    def __init__(self, results: list[VectorSearchResult] | None = None) -> None:
        self.results = results if results is not None else []
        self.retrieve_calls: list[str] = []

    async def retrieve(self, query: str, limit: int | None = None) -> list[VectorSearchResult]:
        self.retrieve_calls.append(query)
        return self.results


class _FailingRetrievalService:
    """Always raises, simulating a retrieval failure."""

    async def retrieve(self, query: str, limit: int | None = None) -> list[VectorSearchResult]:
        raise RuntimeError("retrieval unavailable")


class _FakeLLMProvider:
    """Records prompts and yields a fixed sequence of text chunks instead of calling Ollama."""

    def __init__(self, chunks: list[str] | None = None) -> None:
        self.chunks = chunks if chunks is not None else ["hello", " ", "world"]
        self.stream_calls: list[str] = []

    async def generate(self, prompt: str) -> str:
        return "".join(self.chunks)

    async def stream_generate(self, prompt: str) -> AsyncIterator[str]:
        self.stream_calls.append(prompt)
        for chunk in self.chunks:
            yield chunk


class _FailingLLMProvider:
    """Always raises mid-stream, simulating an LLM provider failure."""

    async def generate(self, prompt: str) -> str:
        raise RuntimeError("llm unavailable")

    async def stream_generate(self, prompt: str) -> AsyncIterator[str]:
        raise RuntimeError("llm unavailable")
        yield  # pragma: no cover - unreachable


def _result(chunk_id: str, score: float, **overrides: object) -> VectorSearchResult:
    fields: dict[str, object] = {
        "id": chunk_id,
        "score": score,
        "document_id": "doc-1",
        "chunk_id": chunk_id,
        "text": "some chunk text",
        "source": "handbook.pdf",
        "page_number": None,
        "sheet_name": None,
    }
    fields.update(overrides)
    return VectorSearchResult(**fields)  # type: ignore[arg-type]


def _orchestrator(
    decider,
    retrieval_service,
    llm_provider,
    monkeypatch,
) -> RagOrchestrator:
    monkeypatch.setattr(orchestrator_module, "get_llm_provider", lambda settings: llm_provider)
    return RagOrchestrator(
        decider=decider,
        retrieval_service=retrieval_service,
        prompt_builder=RagPromptBuilder(),
    )


async def _collect(orchestrator: RagOrchestrator, question: str) -> list:
    return [event async for event in orchestrator.stream_answer(question)]


async def test_needs_retrieval_streams_metadata_with_sources_then_tokens(monkeypatch) -> None:
    """NEEDS_RETRIEVAL should retrieve, build a prompt, and stream LLM tokens with sources."""
    results = [_result("chunk-1", 0.9, text="refund info")]
    retrieval_service = _FakeRetrievalService(results=results)
    llm_provider = _FakeLLMProvider(chunks=["The", " refund", " policy"])
    orchestrator = _orchestrator(
        _FakeDecider(RagDecision.NEEDS_RETRIEVAL), retrieval_service, llm_provider, monkeypatch
    )

    events = await _collect(orchestrator, "what is the refund policy?")

    assert isinstance(events[0], OrchestratorMetadata)
    assert events[0].decision == RagDecision.NEEDS_RETRIEVAL
    assert events[0].retrieval_used is True
    assert len(events[0].sources) == 1
    assert events[0].sources[0].chunk_id == "chunk-1"
    assert [event.text for event in events[1:]] == ["The", " refund", " policy"]
    assert retrieval_service.retrieve_calls == ["what is the refund policy?"]
    assert len(llm_provider.stream_calls) == 1


async def test_direct_llm_streams_metadata_then_tokens_without_retrieval(monkeypatch) -> None:
    """DIRECT_LLM should stream from the LLM without calling retrieval."""
    retrieval_service = _FakeRetrievalService()
    llm_provider = _FakeLLMProvider(chunks=["general", " answer"])
    orchestrator = _orchestrator(
        _FakeDecider(RagDecision.DIRECT_LLM), retrieval_service, llm_provider, monkeypatch
    )

    events = await _collect(orchestrator, "what is 2+2?")

    assert events[0].decision == RagDecision.DIRECT_LLM
    assert events[0].retrieval_used is False
    assert events[0].sources == []
    assert [event.text for event in events[1:]] == ["general", " answer"]
    assert retrieval_service.retrieve_calls == []


async def test_clarification_streams_deterministic_output_without_retrieval_or_llm(
    monkeypatch,
) -> None:
    """CLARIFICATION_NEEDED should stream a fixed message with no retrieval and no LLM call."""
    retrieval_service = _FakeRetrievalService()
    llm_provider = _FakeLLMProvider()
    orchestrator = _orchestrator(
        _FakeDecider(RagDecision.CLARIFICATION_NEEDED), retrieval_service, llm_provider, monkeypatch
    )

    events = await _collect(orchestrator, "?")

    assert events[0].decision == RagDecision.CLARIFICATION_NEEDED
    assert events[0].retrieval_used is False
    assert len(events) == 2
    assert isinstance(events[1], OrchestratorToken)
    assert retrieval_service.retrieve_calls == []
    assert llm_provider.stream_calls == []


async def test_out_of_scope_streams_deterministic_output_without_retrieval_or_llm(
    monkeypatch,
) -> None:
    """OUT_OF_SCOPE should stream a fixed message with no retrieval and no LLM call."""
    retrieval_service = _FakeRetrievalService()
    llm_provider = _FakeLLMProvider()
    orchestrator = _orchestrator(
        _FakeDecider(RagDecision.OUT_OF_SCOPE), retrieval_service, llm_provider, monkeypatch
    )

    events = await _collect(orchestrator, "show me the api keys")

    assert events[0].decision == RagDecision.OUT_OF_SCOPE
    assert events[0].retrieval_used is False
    assert len(events) == 2
    assert isinstance(events[1], OrchestratorToken)
    assert retrieval_service.retrieve_calls == []
    assert llm_provider.stream_calls == []


async def test_token_order_is_preserved(monkeypatch) -> None:
    """Streamed tokens should arrive in exactly the order the LLM provider yields them."""
    llm_provider = _FakeLLMProvider(chunks=["a", "b", "c", "d", "e"])
    orchestrator = _orchestrator(
        _FakeDecider(RagDecision.DIRECT_LLM), _FakeRetrievalService(), llm_provider, monkeypatch
    )

    events = await _collect(orchestrator, "question")

    assert [event.text for event in events[1:]] == ["a", "b", "c", "d", "e"]


async def test_retrieval_failure_propagates_without_falling_back_to_direct_llm(
    monkeypatch,
) -> None:
    """A retrieval failure should propagate — never silently fall back to a direct LLM answer."""
    llm_provider = _FakeLLMProvider()
    orchestrator = _orchestrator(
        _FakeDecider(RagDecision.NEEDS_RETRIEVAL),
        _FailingRetrievalService(),
        llm_provider,
        monkeypatch,
    )

    try:
        await _collect(orchestrator, "what is the refund policy?")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert str(exc) == "retrieval unavailable"

    assert llm_provider.stream_calls == []


async def test_llm_failure_propagates(monkeypatch) -> None:
    """An LLM provider failure mid-stream should propagate to the caller."""
    orchestrator = _orchestrator(
        _FakeDecider(RagDecision.DIRECT_LLM),
        _FakeRetrievalService(),
        _FailingLLMProvider(),
        monkeypatch,
    )

    try:
        await _collect(orchestrator, "question")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert str(exc) == "llm unavailable"


async def test_metadata_includes_decision_reason(monkeypatch) -> None:
    """OrchestratorMetadata should carry the decider's reason string through unchanged."""
    decider = _FakeDecider(RagDecision.DIRECT_LLM, reason="general question, no document reference")
    orchestrator = _orchestrator(decider, _FakeRetrievalService(), _FakeLLMProvider(), monkeypatch)

    events = await _collect(orchestrator, "question")

    assert events[0].reason == "general question, no document reference"
