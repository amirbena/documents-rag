"""Abstract interface for text embedding providers.

See app/rag/providers/ollama_embedding_provider.py for the Ollama-backed implementation.
"""

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Contract for turning text into embedding vectors."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in the same order."""
        raise NotImplementedError
