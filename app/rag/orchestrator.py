"""Internal RAG orchestrator: composes decision routing, retrieval, prompt building, language-
aware prompt resolution, and LLM streaming into a single call.

Question -> RuleBasedRagDecider.decide(...) -> (RetrievalService + RagPromptBuilder, only when
retrieval is needed) -> LLMProvider.stream_generate(...). No public endpoint, no conversation
memory, no silent fallback between decisions or providers — a clarification/out-of-scope/
no-results outcome short-circuits before any LLM call (no-results also skips the LLM, after a
real retrieval attempt found nothing usable), and a failure in retrieval or the LLM provider
propagates as-is rather than being swallowed or substituted.

Fixed/governed response text (clarification, no-results, out-of-scope, and the grounded/direct
answer system instructions) comes from PromptProvider (app/rag/prompts/provider.py) — a
framework-neutral shared module — not from a private, English-only constant. PromptProvider
detects the question's language and returns text in that language, so RagOrchestrator's output is
language-aware without any language-specific logic living here.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from app.core.config import Settings, get_settings
from app.rag.decision import DecisionResult, RagDecision, RuleBasedRagDecider
from app.rag.prompt_builder import PromptSource, RagPromptBuilder
from app.rag.prompts.provider import PromptProvider
from app.rag.prompts.types import PromptType
from app.rag.providers.provider_factory import get_llm_provider
from app.rag.retrieval_service import RetrievalService


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
        prompt_provider: PromptProvider | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._decider = decider or RuleBasedRagDecider()
        self._retrieval_service = retrieval_service or RetrievalService(self._settings)
        self._prompt_builder = prompt_builder or RagPromptBuilder()
        self._prompt_provider = prompt_provider or PromptProvider(self._settings)

    async def stream_answer(
        self, question: str
    ) -> AsyncIterator[OrchestratorMetadata | OrchestratorToken]:
        """Route `question` and stream its answer: one OrchestratorMetadata then OrchestratorTokens.

        CLARIFICATION_NEEDED/OUT_OF_SCOPE stream a fixed, language-appropriate message with no
        retrieval and no LLM call. NEEDS_RETRIEVAL runs RetrievalService + RagPromptBuilder; if
        nothing usable came back, it streams a fixed no-results message with no LLM call either —
        otherwise it streams from the LLM using a language-aware grounded-answer system prompt.
        DIRECT_LLM streams from the LLM without retrieval, using a language-aware direct-answer
        system prompt. A failure in retrieval or the LLM provider propagates to the caller —
        never silently substituted.
        """
        decision_result = self._decider.decide(question)

        if decision_result.decision == RagDecision.CLARIFICATION_NEEDED:
            resolved = self._prompt_provider.resolve(PromptType.CLARIFICATION, question)
            yield self._metadata(decision_result, retrieval_used=False)
            yield OrchestratorToken(text=resolved.response_text or "")
            return

        if decision_result.decision == RagDecision.OUT_OF_SCOPE:
            resolved = self._prompt_provider.resolve(PromptType.OUT_OF_SCOPE, question)
            yield self._metadata(decision_result, retrieval_used=False)
            yield OrchestratorToken(text=resolved.response_text or "")
            return

        if decision_result.decision == RagDecision.NEEDS_RETRIEVAL:
            results = await self._retrieval_service.retrieve(question)
            built = self._prompt_builder.build(question, results)

            if not built.sources:
                resolved = self._prompt_provider.resolve(PromptType.NO_RESULTS, question)
                yield self._metadata(decision_result, retrieval_used=True, sources=[])
                yield OrchestratorToken(text=resolved.response_text or "")
                return

            resolved = self._prompt_provider.resolve(PromptType.GROUNDED_ANSWER, question)
            yield self._metadata(decision_result, retrieval_used=True, sources=built.sources)
            prompt = f"{resolved.system_text}\n\n{built.user_prompt}"
        else:
            resolved = self._prompt_provider.resolve(PromptType.DIRECT_ANSWER, question)
            yield self._metadata(decision_result, retrieval_used=False)
            prompt = f"{resolved.system_text}\n\nQuestion: {question}"

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
