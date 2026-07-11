"""Tests for CustomRagEngine: it must behave identically to RagOrchestrator, unmodified."""

from collections.abc import AsyncIterator

from app.rag.decision import DecisionResult, RagDecision
from app.rag.engines.custom_engine import CustomRagEngine
from app.rag.orchestrator import OrchestratorMetadata, OrchestratorToken, RagOrchestrator
from app.rag.prompt_builder import RagPromptBuilder


class _FakeDecider:
    """Always returns a fixed DecisionResult, regardless of the question."""

    def __init__(self, decision: RagDecision, reason: str = "fixed decision") -> None:
        self._result = DecisionResult(decision=decision, reason=reason, confidence=1.0)

    def decide(self, question: str) -> DecisionResult:
        return self._result


class _FakeRetrievalService:
    """Records calls and returns nothing — no document reference in this suite."""

    def __init__(self) -> None:
        self.retrieve_calls: list[str] = []

    async def retrieve(self, query: str, limit: int | None = None) -> list:
        self.retrieve_calls.append(query)
        return []


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


def _engine(decision: RagDecision, llm_provider: _FakeLLMProvider, monkeypatch) -> CustomRagEngine:
    import app.rag.orchestrator as orchestrator_module

    monkeypatch.setattr(orchestrator_module, "get_llm_provider", lambda settings: llm_provider)
    orchestrator = RagOrchestrator(
        decider=_FakeDecider(decision),
        retrieval_service=_FakeRetrievalService(),
        prompt_builder=RagPromptBuilder(),
    )
    return CustomRagEngine(orchestrator=orchestrator)


async def _collect(engine: CustomRagEngine, question: str) -> list:
    return [event async for event in engine.stream_answer(question)]


async def test_streams_metadata_then_tokens_in_order(monkeypatch) -> None:
    """CustomRagEngine should stream metadata first, then tokens in the LLM's order."""
    llm_provider = _FakeLLMProvider(chunks=["a", "b", "c"])
    engine = _engine(RagDecision.DIRECT_LLM, llm_provider, monkeypatch)

    events = await _collect(engine, "what is 2+2?")

    assert isinstance(events[0], OrchestratorMetadata)
    assert [event.text for event in events[1:]] == ["a", "b", "c"]
    assert all(isinstance(event, OrchestratorToken) for event in events[1:])


async def test_delegates_directly_to_the_wrapped_orchestrator(monkeypatch) -> None:
    """CustomRagEngine must not alter RagOrchestrator's events in any way."""
    llm_provider = _FakeLLMProvider(chunks=["x"])
    orchestrator = RagOrchestrator(
        decider=_FakeDecider(RagDecision.DIRECT_LLM),
        retrieval_service=_FakeRetrievalService(),
        prompt_builder=RagPromptBuilder(),
    )
    import app.rag.orchestrator as orchestrator_module

    monkeypatch.setattr(orchestrator_module, "get_llm_provider", lambda settings: llm_provider)
    engine = CustomRagEngine(orchestrator=orchestrator)

    direct_events = [event async for event in orchestrator.stream_answer("question")]
    engine_events = [event async for event in engine.stream_answer("question")]

    assert [type(event) for event in direct_events] == [type(event) for event in engine_events]
    assert [getattr(event, "text", None) for event in direct_events] == [
        getattr(event, "text", None) for event in engine_events
    ]


async def test_all_four_decision_paths_remain_supported(monkeypatch) -> None:
    """Every RagDecision value should stream a metadata event carrying that same decision."""
    for decision in RagDecision:
        llm_provider = _FakeLLMProvider(chunks=["ok"])
        engine = _engine(decision, llm_provider, monkeypatch)

        events = await _collect(engine, "question")

        assert events[0].decision == decision


async def test_answer_collects_full_text_from_stream(monkeypatch) -> None:
    """The base RagEngine.answer() helper should join all streamed tokens in order."""
    llm_provider = _FakeLLMProvider(chunks=["The", " answer", " is", " 4"])
    engine = _engine(RagDecision.DIRECT_LLM, llm_provider, monkeypatch)

    result = await engine.answer("what is 2+2?")

    assert result == "The answer is 4"
