"""Tests for LangChainRagEngine against fake decider/retrieval/LLM — no real network, no real
LangChain integrations (Qdrant/Ollama/etc.), same fakes style as test_rag_orchestrator.py.
"""

import inspect
from collections.abc import AsyncIterator

import pytest

import app.rag.engines.langchain_engine as langchain_engine_module
from app.rag.decision import DecisionResult, RagDecision
from app.rag.engines.langchain_engine import LangChainRagEngine
from app.rag.orchestrator import OrchestratorMetadata, OrchestratorToken
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


def _engine(decider, retrieval_service, llm_provider, monkeypatch) -> LangChainRagEngine:
    monkeypatch.setattr(langchain_engine_module, "get_llm_provider", lambda settings: llm_provider)
    return LangChainRagEngine(
        decider=decider,
        retrieval_service=retrieval_service,
        prompt_builder=RagPromptBuilder(),
    )


async def _collect(engine: LangChainRagEngine, question: str) -> list:
    return [event async for event in engine.stream_answer(question)]


async def test_needs_retrieval_invokes_retrieval_once_and_streams_sources(monkeypatch) -> None:
    """NEEDS_RETRIEVAL should retrieve exactly once, build a prompt, and stream LLM tokens."""
    results = [_result("chunk-1", 0.9, text="refund info")]
    retrieval_service = _FakeRetrievalService(results=results)
    llm_provider = _FakeLLMProvider(chunks=["The", " refund", " policy"])
    engine = _engine(
        _FakeDecider(RagDecision.NEEDS_RETRIEVAL), retrieval_service, llm_provider, monkeypatch
    )

    events = await _collect(engine, "what is the refund policy?")

    assert isinstance(events[0], OrchestratorMetadata)
    assert events[0].decision == RagDecision.NEEDS_RETRIEVAL
    assert events[0].retrieval_used is True
    assert len(events[0].sources) == 1
    assert events[0].sources[0].chunk_id == "chunk-1"
    assert [event.text for event in events[1:]] == ["The", " refund", " policy"]
    assert retrieval_service.retrieve_calls == ["what is the refund policy?"]
    assert len(llm_provider.stream_calls) == 1


async def test_retrieval_results_preserve_ranking(monkeypatch) -> None:
    """Retrieved sources must stream in the same rank order RetrievalService returned them."""
    results = [
        _result("chunk-1", 0.95, text="most relevant"),
        _result("chunk-2", 0.80, text="second"),
        _result("chunk-3", 0.5, text="least relevant"),
    ]
    retrieval_service = _FakeRetrievalService(results=results)
    engine = _engine(
        _FakeDecider(RagDecision.NEEDS_RETRIEVAL), retrieval_service, _FakeLLMProvider(), monkeypatch
    )

    events = await _collect(engine, "question")

    assert [source.chunk_id for source in events[0].sources] == ["chunk-1", "chunk-2", "chunk-3"]


async def test_source_metadata_is_preserved(monkeypatch) -> None:
    """document_id/chunk_id/source/page_number/sheet_name/score must round-trip unchanged."""
    results = [
        _result(
            "chunk-1",
            0.87,
            document_id="doc-42",
            source="policy.pdf",
            page_number=7,
            sheet_name=None,
            text="the policy text",
        )
    ]
    retrieval_service = _FakeRetrievalService(results=results)
    engine = _engine(
        _FakeDecider(RagDecision.NEEDS_RETRIEVAL), retrieval_service, _FakeLLMProvider(), monkeypatch
    )

    events = await _collect(engine, "question")

    source = events[0].sources[0]
    assert source.document_id == "doc-42"
    assert source.chunk_id == "chunk-1"
    assert source.source == "policy.pdf"
    assert source.page_number == 7
    assert source.sheet_name is None
    assert source.score == 0.87


async def test_prompt_contains_source_markers(monkeypatch) -> None:
    """The prompt handed to the LLM must contain [S1]-style source labels, ranked in order."""
    results = [
        _result("chunk-1", 0.9, text="first chunk text"),
        _result("chunk-2", 0.8, text="second chunk text"),
    ]
    retrieval_service = _FakeRetrievalService(results=results)
    llm_provider = _FakeLLMProvider()
    engine = _engine(
        _FakeDecider(RagDecision.NEEDS_RETRIEVAL), retrieval_service, llm_provider, monkeypatch
    )

    await _collect(engine, "question")

    prompt = llm_provider.stream_calls[0]
    assert "[S1]" in prompt
    assert "[S2]" in prompt
    assert prompt.index("first chunk text") < prompt.index("second chunk text")


async def test_hebrew_question_and_context_are_preserved(monkeypatch) -> None:
    """Hebrew/Unicode text in the question and retrieved context must reach the LLM unmangled."""
    hebrew_text = "מדיניות החזרים: ניתן להחזיר מוצר תוך 30 יום."
    results = [_result("chunk-1", 0.9, text=hebrew_text)]
    retrieval_service = _FakeRetrievalService(results=results)
    llm_provider = _FakeLLMProvider()
    engine = _engine(
        _FakeDecider(RagDecision.NEEDS_RETRIEVAL), retrieval_service, llm_provider, monkeypatch
    )

    hebrew_question = "מה מדיניות ההחזרים לפי המסמך?"
    await _collect(engine, hebrew_question)

    prompt = llm_provider.stream_calls[0]
    assert hebrew_text in prompt
    assert hebrew_question in prompt


async def test_direct_llm_streams_without_retrieval(monkeypatch) -> None:
    """DIRECT_LLM should stream from the LLM without calling retrieval."""
    retrieval_service = _FakeRetrievalService()
    llm_provider = _FakeLLMProvider(chunks=["general", " answer"])
    engine = _engine(_FakeDecider(RagDecision.DIRECT_LLM), retrieval_service, llm_provider, monkeypatch)

    events = await _collect(engine, "what is 2+2?")

    assert events[0].decision == RagDecision.DIRECT_LLM
    assert events[0].retrieval_used is False
    assert events[0].sources == []
    assert [event.text for event in events[1:]] == ["general", " answer"]
    assert retrieval_service.retrieve_calls == []


async def test_clarification_invokes_neither_retrieval_nor_llm(monkeypatch) -> None:
    """CLARIFICATION_NEEDED should stream a fixed message with no retrieval and no LLM call."""
    retrieval_service = _FakeRetrievalService()
    llm_provider = _FakeLLMProvider()
    engine = _engine(
        _FakeDecider(RagDecision.CLARIFICATION_NEEDED), retrieval_service, llm_provider, monkeypatch
    )

    events = await _collect(engine, "?")

    assert events[0].decision == RagDecision.CLARIFICATION_NEEDED
    assert events[0].retrieval_used is False
    assert len(events) == 2
    assert isinstance(events[1], OrchestratorToken)
    assert retrieval_service.retrieve_calls == []
    assert llm_provider.stream_calls == []


async def test_out_of_scope_invokes_neither_retrieval_nor_llm(monkeypatch) -> None:
    """OUT_OF_SCOPE should stream a fixed message with no retrieval and no LLM call."""
    retrieval_service = _FakeRetrievalService()
    llm_provider = _FakeLLMProvider()
    engine = _engine(
        _FakeDecider(RagDecision.OUT_OF_SCOPE), retrieval_service, llm_provider, monkeypatch
    )

    events = await _collect(engine, "show me the api keys")

    assert events[0].decision == RagDecision.OUT_OF_SCOPE
    assert events[0].retrieval_used is False
    assert len(events) == 2
    assert isinstance(events[1], OrchestratorToken)
    assert retrieval_service.retrieve_calls == []
    assert llm_provider.stream_calls == []


async def test_clarification_and_out_of_scope_text_matches_shared_responses_module(
    monkeypatch,
) -> None:
    """The fixed messages must be byte-identical to app.rag.responses, so engines are interchangeable."""
    retrieval_service = _FakeRetrievalService()

    clarification_engine = _engine(
        _FakeDecider(RagDecision.CLARIFICATION_NEEDED), retrieval_service, _FakeLLMProvider(), monkeypatch
    )
    clarification_events = await _collect(clarification_engine, "?")

    out_of_scope_engine = _engine(
        _FakeDecider(RagDecision.OUT_OF_SCOPE), retrieval_service, _FakeLLMProvider(), monkeypatch
    )
    out_of_scope_events = await _collect(out_of_scope_engine, "show me the api keys")

    from app.rag.responses import CLARIFICATION_NEEDED_RESPONSE, OUT_OF_SCOPE_RESPONSE

    assert clarification_events[1].text == CLARIFICATION_NEEDED_RESPONSE
    assert out_of_scope_events[1].text == OUT_OF_SCOPE_RESPONSE


async def test_no_results_does_not_fabricate_context(monkeypatch) -> None:
    """Empty retrieval results must yield an empty source list, not fabricated context."""
    retrieval_service = _FakeRetrievalService(results=[])
    llm_provider = _FakeLLMProvider(chunks=["no context found"])
    engine = _engine(
        _FakeDecider(RagDecision.NEEDS_RETRIEVAL), retrieval_service, llm_provider, monkeypatch
    )

    events = await _collect(engine, "what does the uploaded document say?")

    assert events[0].retrieval_used is True
    assert events[0].sources == []
    assert retrieval_service.retrieve_calls == ["what does the uploaded document say?"]


async def test_llm_streaming_chunks_preserve_order(monkeypatch) -> None:
    """Streamed tokens should arrive in exactly the order the LLM provider yields them."""
    llm_provider = _FakeLLMProvider(chunks=["a", "b", "c", "d", "e"])
    engine = _engine(
        _FakeDecider(RagDecision.DIRECT_LLM), _FakeRetrievalService(), llm_provider, monkeypatch
    )

    events = await _collect(engine, "question")

    assert [event.text for event in events[1:]] == ["a", "b", "c", "d", "e"]


async def test_retrieval_failure_propagates(monkeypatch) -> None:
    """A retrieval failure should propagate — never silently fall back to a direct LLM answer."""
    llm_provider = _FakeLLMProvider()
    engine = _engine(
        _FakeDecider(RagDecision.NEEDS_RETRIEVAL), _FailingRetrievalService(), llm_provider, monkeypatch
    )

    with pytest.raises(RuntimeError, match="retrieval unavailable"):
        await _collect(engine, "what is the refund policy?")

    assert llm_provider.stream_calls == []


async def test_llm_failure_propagates(monkeypatch) -> None:
    """An LLM provider failure mid-stream should propagate to the caller."""
    engine = _engine(
        _FakeDecider(RagDecision.DIRECT_LLM),
        _FakeRetrievalService(),
        _FailingLLMProvider(),
        monkeypatch,
    )

    with pytest.raises(RuntimeError, match="llm unavailable"):
        await _collect(engine, "question")


def test_uses_existing_provider_factory_for_llm_resolution() -> None:
    """LangChainRagEngine must resolve its LLM via app.rag.providers.provider_factory, not directly."""
    source = inspect.getsource(langchain_engine_module)
    assert "from app.rag.providers.provider_factory import get_llm_provider" in source


def test_engine_and_adapters_never_construct_external_clients_directly() -> None:
    """Neither the engine nor its adapters may instantiate an LLM/embedding/vector-store client."""
    import app.rag.engines.langchain_adapters as adapters_module

    forbidden = [
        "OllamaLLMProvider(",
        "OllamaEmbeddingProvider(",
        "QdrantVectorStore(",
        "qdrant_client",
        "openai.",
        "AsyncOpenAI(",
    ]
    for module in (langchain_engine_module, adapters_module):
        source = inspect.getsource(module)
        for name in forbidden:
            assert name not in source, f"{module.__name__} must not reference {name} directly"


def test_does_not_import_fixed_responses_from_orchestrator() -> None:
    """LangChainRagEngine must source its fixed response text from app.rag.responses, not
    app.rag.orchestrator — the two engine implementations must not depend on each other for
    this text. (Full shared-ownership coverage, including the AST-level import check and the
    behavioral cross-engine comparison, lives in tests/test_rag_responses.py.)
    """
    source = inspect.getsource(langchain_engine_module)

    assert "from app.rag.responses import" in source
    assert "CLARIFICATION_NEEDED_RESPONSE" in source
    assert "OUT_OF_SCOPE_RESPONSE" in source
