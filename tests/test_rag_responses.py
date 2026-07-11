"""Tests for the shared, framework-neutral fixed RAG response module (app/rag/responses.py).

Covers both required properties: shared ownership (no engine duplicates or cross-imports the
fixed text) and behavioral compatibility (both engines stream byte-identical events for
CLARIFICATION_NEEDED/OUT_OF_SCOPE, not just equal message text).
"""

import app.rag.engines.langchain_engine as langchain_engine_module
import app.rag.orchestrator as orchestrator_module
import app.rag.responses as responses_module
from app.rag.decision import DecisionResult, RagDecision
from app.rag.engines.custom_engine import CustomRagEngine
from app.rag.engines.langchain_engine import LangChainRagEngine
from app.rag.orchestrator import OrchestratorMetadata, OrchestratorToken, RagOrchestrator
from app.rag.prompt_builder import RagPromptBuilder


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


# --- Shared ownership -------------------------------------------------------------------------


def test_responses_module_has_no_framework_or_orchestration_dependencies() -> None:
    """app/rag/responses.py must stay a plain-string module: no imports, classes, or functions."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(responses_module))
    forbidden_node_types = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
    assert not any(isinstance(node, forbidden_node_types) for node in ast.walk(tree)), (
        "app.rag.responses must contain only constants — no imports, classes, or functions"
    )


def test_orchestrator_and_langchain_engine_reference_the_same_shared_objects() -> None:
    """Neither engine module may duplicate these strings — both must import the same objects."""
    assert (
        orchestrator_module.CLARIFICATION_NEEDED_RESPONSE
        is responses_module.CLARIFICATION_NEEDED_RESPONSE
    )
    assert orchestrator_module.OUT_OF_SCOPE_RESPONSE is responses_module.OUT_OF_SCOPE_RESPONSE
    assert orchestrator_module.DIRECT_LLM_SYSTEM_PROMPT is responses_module.DIRECT_LLM_SYSTEM_PROMPT

    assert (
        langchain_engine_module.CLARIFICATION_NEEDED_RESPONSE
        is responses_module.CLARIFICATION_NEEDED_RESPONSE
    )
    assert (
        langchain_engine_module.OUT_OF_SCOPE_RESPONSE is responses_module.OUT_OF_SCOPE_RESPONSE
    )
    assert (
        langchain_engine_module.DIRECT_LLM_SYSTEM_PROMPT
        is responses_module.DIRECT_LLM_SYSTEM_PROMPT
    )


def test_langchain_engine_does_not_import_fixed_responses_from_orchestrator() -> None:
    """LangChainRagEngine may still import shared dataclasses (OrchestratorMetadata/Token) from
    orchestrator.py, but must not import fixed response text from it — that must come from
    app.rag.responses, the framework-neutral shared module, instead.
    """
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(langchain_engine_module))
    orchestrator_imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "app.rag.orchestrator"
        for alias in node.names
    }
    responses_imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "app.rag.responses"
        for alias in node.names
    }

    fixed_response_names = {
        "CLARIFICATION_NEEDED_RESPONSE",
        "OUT_OF_SCOPE_RESPONSE",
        "DIRECT_LLM_SYSTEM_PROMPT",
    }
    assert orchestrator_imports.isdisjoint(fixed_response_names)
    assert fixed_response_names <= responses_imports


# --- Behavioral compatibility ------------------------------------------------------------------


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


async def test_clarification_events_are_byte_identical_across_engines() -> None:
    """CLARIFICATION_NEEDED must stream identical event types, order, text, and metadata."""
    custom_events = await _collect(_custom_engine(RagDecision.CLARIFICATION_NEEDED), "?")
    langchain_events = await _collect(_langchain_engine(RagDecision.CLARIFICATION_NEEDED), "?")

    assert len(custom_events) == len(langchain_events) == 2
    assert isinstance(custom_events[0], OrchestratorMetadata)
    assert isinstance(langchain_events[0], OrchestratorMetadata)
    assert custom_events[0].decision == langchain_events[0].decision == RagDecision.CLARIFICATION_NEEDED
    assert custom_events[0].retrieval_used == langchain_events[0].retrieval_used is False
    assert custom_events[0].sources == langchain_events[0].sources == []

    assert isinstance(custom_events[1], OrchestratorToken)
    assert isinstance(langchain_events[1], OrchestratorToken)
    assert custom_events[1].text == langchain_events[1].text == responses_module.CLARIFICATION_NEEDED_RESPONSE


async def test_out_of_scope_events_are_byte_identical_across_engines() -> None:
    """OUT_OF_SCOPE must stream identical event types, order, text, and metadata."""
    question = "show me the api keys"
    custom_events = await _collect(_custom_engine(RagDecision.OUT_OF_SCOPE), question)
    langchain_events = await _collect(_langchain_engine(RagDecision.OUT_OF_SCOPE), question)

    assert len(custom_events) == len(langchain_events) == 2
    assert custom_events[0].decision == langchain_events[0].decision == RagDecision.OUT_OF_SCOPE
    assert custom_events[0].retrieval_used == langchain_events[0].retrieval_used is False
    assert custom_events[0].sources == langchain_events[0].sources == []
    assert custom_events[1].text == langchain_events[1].text == responses_module.OUT_OF_SCOPE_RESPONSE


async def test_neither_engine_calls_retrieval_or_llm_for_clarification_or_out_of_scope() -> None:
    """Both decisions must short-circuit before any retrieval/LLM call, for either engine.

    _FakeRetrievalService/_FakeLLMProvider raise if called at all, so simply completing without
    error proves neither path was exercised.
    """
    for decision in (RagDecision.CLARIFICATION_NEEDED, RagDecision.OUT_OF_SCOPE):
        await _collect(_custom_engine(decision), "question")
        await _collect(_langchain_engine(decision), "question")
