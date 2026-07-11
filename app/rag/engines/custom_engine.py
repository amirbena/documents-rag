"""RagEngine implementation that delegates to the existing RagOrchestrator, unmodified.

This is the platform's default/reference RAG implementation — RuleBasedRagDecider,
RetrievalService, RagPromptBuilder, and LLMProvider.stream_generate() all run exactly as they did
before RagEngine existed. CustomRagEngine only adapts RagOrchestrator to the RagEngine interface;
it adds no logic of its own.
"""

from collections.abc import AsyncIterator

from app.core.config import Settings
from app.rag.engine import RagEngine
from app.rag.orchestrator import OrchestratorMetadata, OrchestratorToken, RagOrchestrator


class CustomRagEngine(RagEngine):
    """Adapts RagOrchestrator to the RagEngine interface — no orchestration logic of its own."""

    def __init__(
        self, orchestrator: RagOrchestrator | None = None, settings: Settings | None = None
    ) -> None:
        self._orchestrator = orchestrator or RagOrchestrator(settings=settings)

    async def stream_answer(self, question: str) -> AsyncIterator[OrchestratorMetadata | OrchestratorToken]:
        """Delegate directly to RagOrchestrator.stream_answer — behavior is unchanged."""
        async for event in self._orchestrator.stream_answer(question):
            yield event
