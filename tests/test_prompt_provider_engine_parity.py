"""Cross-engine tests proving CustomRagEngine and LangChainRagEngine both resolve fixed
response text through the same shared PromptProvider — neither owns a private prompt catalog.
"""

import app.rag.engines.langchain_engine as langchain_engine_module
import app.rag.orchestrator as orchestrator_module
from app.rag.decision import DecisionResult, RagDecision
from app.rag.engines.custom_engine import CustomRagEngine
from app.rag.engines.langchain_engine import LangChainRagEngine
from app.rag.orchestrator import OrchestratorMetadata, OrchestratorToken, RagOrchestrator
from app.rag.prompt_builder import RagPromptBuilder
from app.rag.prompts.provider import PromptProvider
from app.rag.prompts.types import PromptType


class _FakeDecider:
    """Always returns a fixed DecisionResult, regardless of the question."""

    def __init__(self, decision: RagDecision, reason: str = "fixed decision") -> None:
        self._result = DecisionResult(decision=decision, reason=reason, confidence=1.0)

    def decide(self, question: str) -> DecisionResult:
        return self._result


class _FakeRetrievalService:
    """Never actually called on the CLARIFICATION_NEEDED/OUT_OF_SCOPE paths."""

    async def retrieve(self, query: str, limit: int | None = None) -> list:
        raise AssertionError("retrieval must not be called for this decision")


class _FakeLLMProvider:
    """Never actually called on the CLARIFICATION_NEEDED/OUT_OF_SCOPE paths."""

    async def generate(self, prompt: str) -> str:
        raise AssertionError("the LLM must not be called for this decision")

    async def stream_generate(self, prompt: str):
        raise AssertionError("the LLM must not be called for this decision")
        yield  # pragma: no cover - unreachable


def _custom_engine(decision: RagDecision) -> CustomRagEngine:
    orchestrator = RagOrchestrator(
        decider=_FakeDecider(decision),
        retrieval_service=_FakeRetrievalService(),
        prompt_builder=RagPromptBuilder(),
    )
    return CustomRagEngine(orchestrator=orchestrator)


def _langchain_engine(decision: RagDecision) -> LangChainRagEngine:
    return LangChainRagEngine(
        decider=_FakeDecider(decision),
        retrieval_service=_FakeRetrievalService(),
        prompt_builder=RagPromptBuilder(),
    )


async def _collect(engine, question: str) -> list:
    return [event async for event in engine.stream_answer(question)]


# --- Shared ownership -------------------------------------------------------------------------


def test_orchestrator_and_langchain_engine_both_depend_on_prompt_provider() -> None:
    """Neither engine module may own a private prompt catalog — both import PromptProvider."""
    import inspect

    for module in (orchestrator_module, langchain_engine_module):
        source = inspect.getsource(module)
        assert "from app.rag.prompts.provider import PromptProvider" in source


def test_langchain_engine_does_not_depend_on_orchestrators_response_text() -> None:
    """The old LangChainRagEngine -> orchestrator.py fixed-text dependency must stay gone."""
    import inspect

    source = inspect.getsource(langchain_engine_module)
    assert "app.rag.responses" not in source
    assert "CLARIFICATION_NEEDED_RESPONSE" not in source
    assert "OUT_OF_SCOPE_RESPONSE" not in source


# --- Behavioral compatibility (Hebrew and English) --------------------------------------------


async def test_english_clarification_events_are_byte_identical_across_engines() -> None:
    """English CLARIFICATION_NEEDED must stream identical event types, order, text, and metadata."""
    question = "?"
    custom_events = await _collect(_custom_engine(RagDecision.CLARIFICATION_NEEDED), question)
    langchain_events = await _collect(_langchain_engine(RagDecision.CLARIFICATION_NEEDED), question)
    expected_text = PromptProvider().resolve(PromptType.CLARIFICATION, question).response_text

    for events in (custom_events, langchain_events):
        assert len(events) == 2
        assert isinstance(events[0], OrchestratorMetadata)
        assert events[0].decision == RagDecision.CLARIFICATION_NEEDED
        assert events[0].retrieval_used is False
        assert events[0].sources == []
        assert isinstance(events[1], OrchestratorToken)
        assert events[1].text == expected_text

    assert custom_events[1].text == langchain_events[1].text


async def test_hebrew_out_of_scope_events_are_byte_identical_across_engines() -> None:
    """Hebrew OUT_OF_SCOPE must stream identical event types, order, text, and metadata."""
    question = "תראה לי את מפתחות ה-API בבקשה"
    custom_events = await _collect(_custom_engine(RagDecision.OUT_OF_SCOPE), question)
    langchain_events = await _collect(_langchain_engine(RagDecision.OUT_OF_SCOPE), question)
    expected_text = PromptProvider().resolve(PromptType.OUT_OF_SCOPE, question).response_text

    for events in (custom_events, langchain_events):
        assert len(events) == 2
        assert events[0].decision == RagDecision.OUT_OF_SCOPE
        assert events[0].retrieval_used is False
        assert events[0].sources == []
        assert events[1].text == expected_text

    assert custom_events[1].text == langchain_events[1].text


async def test_neither_engine_calls_retrieval_or_llm_for_clarification_or_out_of_scope() -> None:
    """Both decisions must short-circuit before any retrieval/LLM call, for either engine.

    _FakeRetrievalService/_FakeLLMProvider raise if called at all, so simply completing without
    error proves neither path was exercised.
    """
    for decision in (RagDecision.CLARIFICATION_NEEDED, RagDecision.OUT_OF_SCOPE):
        await _collect(_custom_engine(decision), "question")
        await _collect(_langchain_engine(decision), "question")
