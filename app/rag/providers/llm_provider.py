"""Abstract interface for chat/completion LLM providers.

See app/rag/providers/ollama_llm_provider.py for the Ollama-backed streaming implementation.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


class LLMProvider(ABC):
    """Contract for generating text completions from a prompt, in full or streamed."""

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Return the model's completion for the given prompt."""
        raise NotImplementedError

    @abstractmethod
    async def stream_generate(self, prompt: str) -> AsyncIterator[str]:
        """Yield the model's completion for the given prompt as text chunks, in order."""
        raise NotImplementedError
        yield  # pragma: no cover - unreachable; makes this an async generator for typing
