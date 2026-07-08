"""Abstract interface for chat/completion LLM providers.

No concrete implementation yet — an Ollama-backed provider (llama3.1)
will be added in a later milestone.
"""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Contract for generating text completions from a prompt."""

    @abstractmethod
    async def generate(self, prompt: str) -> str:
        """Return the model's completion for the given prompt."""
        raise NotImplementedError
