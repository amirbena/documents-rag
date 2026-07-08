"""Abstract interface for text embedding providers.

No concrete implementation yet — an Ollama-backed provider (nomic-embed-text)
will be added in a later milestone.
"""

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Contract for turning text into embedding vectors."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in the same order."""
        raise NotImplementedError
