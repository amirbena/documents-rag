"""Abstract interface for a replaceable RAG execution engine.

RagEngine is the seam between the public chat route and whichever concrete RAG implementation is
configured (see app/rag/engines/engine_factory.py) — CustomRagEngine (wrapping the existing
RagOrchestrator) or LangChainRagEngine. Independent of FastAPI and SSE formatting: it yields the
same OrchestratorMetadata/OrchestratorToken events RagOrchestrator already produces, so
app/api/v1/routes/chat.py's SSE mapping needs no engine-specific branch.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.rag.orchestrator import OrchestratorMetadata, OrchestratorToken


class RagEngine(ABC):
    """Contract for routing a question to a streamed, source-attributed answer."""

    @abstractmethod
    def stream_answer(self, question: str) -> AsyncIterator[OrchestratorMetadata | OrchestratorToken]:
        """Route `question` and stream its answer: one OrchestratorMetadata then OrchestratorTokens."""
        raise NotImplementedError

    async def answer(self, question: str) -> str:
        """Return the full answer text by collecting every streamed token, in order."""
        return "".join(
            [
                event.text
                async for event in self.stream_answer(question)
                if isinstance(event, OrchestratorToken)
            ]
        )
