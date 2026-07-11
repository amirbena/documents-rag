"""RagEngine implementation that executes retrieval/prompting/generation via LangChain Runnables.

question -> RuleBasedRagDecider.decide() [existing, unmodified — kept outside the LangChain
Runnable so the decision contract is byte-identical to CustomRagEngine's] -> for
CLARIFICATION_NEEDED/OUT_OF_SCOPE, a fixed message streams with no retrieval and no LLM call;
for NEEDS_RETRIEVAL, ProviderBackedRetriever -> the existing, unmodified RagPromptBuilder -> a
LangChain ChatPromptValue piped into ProviderBackedLLM; for DIRECT_LLM, a direct-answer
ChatPromptValue piped into the same LLM adapter. No LangGraph, no agents, no tool calling.

The CLARIFICATION_NEEDED/OUT_OF_SCOPE/DIRECT_LLM message text comes from app.rag.responses — a
framework-neutral shared module, not orchestrator.py — so neither engine implementation depends
on the other's internals, and both stay byte-identical for that text by construction. See
app/rag/responses.py's module docstring for the Phase 2.5 (multilingual Prompt Catalog) boundary
this shared module intentionally stops short of.
"""

from collections.abc import AsyncIterator

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import Runnable, RunnableLambda

from app.core.config import Settings, get_settings
from app.rag.decision import DecisionResult, RagDecision, RuleBasedRagDecider
from app.rag.engine import RagEngine
from app.rag.engines.langchain_adapters import (
    build_provider_backed_llm,
    build_provider_backed_retriever,
    document_to_search_result,
)
from app.rag.orchestrator import OrchestratorMetadata, OrchestratorToken
from app.rag.prompt_builder import PromptSource, RagPromptBuilder
from app.rag.providers.provider_factory import get_llm_provider
from app.rag.responses import (
    CLARIFICATION_NEEDED_RESPONSE,
    DIRECT_LLM_SYSTEM_PROMPT,
    OUT_OF_SCOPE_RESPONSE,
)
from app.rag.retrieval_service import RetrievalService


class LangChainRagEngine(RagEngine):
    """Routes a question through the existing decision layer, then LangChain Runnables."""

    def __init__(
        self,
        settings: Settings | None = None,
        decider: RuleBasedRagDecider | None = None,
        retrieval_service: RetrievalService | None = None,
        prompt_builder: RagPromptBuilder | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._decider = decider or RuleBasedRagDecider()
        self._retrieval_service = retrieval_service or RetrievalService(self._settings)
        self._prompt_builder = prompt_builder or RagPromptBuilder()

    async def stream_answer(
        self, question: str
    ) -> AsyncIterator[OrchestratorMetadata | OrchestratorToken]:
        """Route `question` and stream its answer, mirroring RagOrchestrator's contract exactly.

        CLARIFICATION_NEEDED/OUT_OF_SCOPE decisions stream a fixed message with no retrieval and
        no LLM call — deterministic, matching CustomRagEngine. NEEDS_RETRIEVAL runs
        ProviderBackedRetriever (backed by the real RetrievalService) once, then the existing
        RagPromptBuilder, before streaming from the LLM adapter; DIRECT_LLM streams from the LLM
        adapter without retrieval. A failure in retrieval or the LLM provider propagates to the
        caller — never silently substituted.
        """
        decision_result = self._decider.decide(question)

        if decision_result.decision == RagDecision.CLARIFICATION_NEEDED:
            yield self._metadata(decision_result, retrieval_used=False)
            yield OrchestratorToken(text=CLARIFICATION_NEEDED_RESPONSE)
            return

        if decision_result.decision == RagDecision.OUT_OF_SCOPE:
            yield self._metadata(decision_result, retrieval_used=False)
            yield OrchestratorToken(text=OUT_OF_SCOPE_RESPONSE)
            return

        if decision_result.decision == RagDecision.NEEDS_RETRIEVAL:
            retriever = build_provider_backed_retriever(self._retrieval_service)
            documents = await retriever.ainvoke(question)
            results = [document_to_search_result(document) for document in documents]
            built = self._prompt_builder.build(question, results)
            yield self._metadata(decision_result, retrieval_used=True, sources=built.sources)
            prompt_value = ChatPromptValue(
                messages=[
                    SystemMessage(content=built.system_prompt),
                    HumanMessage(content=built.user_prompt),
                ]
            )
        else:
            yield self._metadata(decision_result, retrieval_used=False)
            prompt_value = ChatPromptValue(
                messages=[
                    SystemMessage(content=DIRECT_LLM_SYSTEM_PROMPT),
                    HumanMessage(content=f"Question: {question}"),
                ]
            )

        llm = build_provider_backed_llm(get_llm_provider(self._settings))
        chain: Runnable = RunnableLambda(lambda _: prompt_value) | llm
        async for chunk in chain.astream({}):
            yield OrchestratorToken(text=chunk)

    @staticmethod
    def _metadata(
        decision_result: DecisionResult,
        retrieval_used: bool,
        sources: list[PromptSource] | None = None,
    ) -> OrchestratorMetadata:
        return OrchestratorMetadata(
            decision=decision_result.decision,
            reason=decision_result.reason,
            retrieval_used=retrieval_used,
            sources=sources or [],
        )
