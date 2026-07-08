"""Abstract interface for chat/completion LLM providers.

See app/rag/providers/ollama_llm_provider.py for the Ollama-backed streaming implementation.
"""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Contract for generating text completions from a prompt."""

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Return the model's completion for the given prompt."""
        raise NotImplementedError
