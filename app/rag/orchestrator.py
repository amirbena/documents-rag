"""Internal RAG orchestrator: composes decision routing, retrieval, prompt building, and LLM
streaming into a single call.

Question -> RuleBasedRagDecider.decide(...) -> (RetrievalService + RagPromptBuilder, only when
retrieval is needed) -> LLMProvider.stream_generate(...). No public endpoint, no conversation
memory, no silent fallback between decisions or providers — a clarification/out-of-scope
decision short-circuits before any retrieval or LLM call, and a failure in retrieval or the LLM
provider propagates as-is rather than being swallowed or substituted.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from app.core.config import Settings, get_settings
from app.rag.decision import DecisionResult, RagDecision, RuleBasedRagDecider
from app.rag.prompt_builder import PromptSource, RagPromptBuilder
from app.rag.providers.provider_factory import get_llm_provider
from app.rag.retrieval_service import RetrievalService

_DIRECT_LLM_SYSTEM_PROMPT = "You are a helpful assistant. Answer the user's question directly."

_CLARIFICATION_MESSAGE = (
    "Could you rephrase or add more detail? Your question is too short or unclear to act on."
)

_OUT_OF_SCOPE_MESSAGE = "I can't help with that request."


@dataclass
class OrchestratorMetadata:
    """First event of a stream_answer() run: routing decision, whether retrieval ran, sources."""

    decision: RagDecision
    reason: str
    retrieval_used: bool
    sources: list[PromptSource] = field(default_factory=list)


@dataclass
class OrchestratorToken:
    """One streamed chunk of answer text, in generation order."""

    text: str


class RagOrchestrator:
    """Routes a question through decision -> (retrieval + prompt building) -> LLM streaming."""

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
        """Route `question` and stream its answer: one OrchestratorMetadata then OrchestratorTokens.

        CLARIFICATION_NEEDED/OUT_OF_SCOPE decisions stream a fixed message with no retrieval and
        no LLM call. NEEDS_RETRIEVAL runs RetrievalService + RagPromptBuilder before streaming
        from the LLM; DIRECT_LLM streams from the LLM without retrieval. A failure in retrieval
        or the LLM provider propagates to the caller — never silently substituted.
        """
        decision_result = self._decider.decide(question)

        if decision_result.decision == RagDecision.CLARIFICATION_NEEDED:
            yield self._metadata(decision_result, retrieval_used=False)
            yield OrchestratorToken(text=_CLARIFICATION_MESSAGE)
            return

        if decision_result.decision == RagDecision.OUT_OF_SCOPE:
            yield self._metadata(decision_result, retrieval_used=False)
            yield OrchestratorToken(text=_OUT_OF_SCOPE_MESSAGE)
            return

        if decision_result.decision == RagDecision.NEEDS_RETRIEVAL:
            results = await self._retrieval_service.retrieve(question)
            built = self._prompt_builder.build(question, results)
            yield self._metadata(decision_result, retrieval_used=True, sources=built.sources)
            prompt = f"{built.system_prompt}\n\n{built.user_prompt}"
        else:
            yield self._metadata(decision_result, retrieval_used=False)
            prompt = f"{_DIRECT_LLM_SYSTEM_PROMPT}\n\nQuestion: {question}"

        llm_provider = get_llm_provider(self._settings)
        async for chunk in llm_provider.stream_generate(prompt):
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
