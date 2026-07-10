"""Shared base for future LLM provider stubs that are recognized but not implemented yet.

Concrete stubs (OpenAIProvider, GeminiProvider, AnthropicProvider) only override
NOT_IMPLEMENTED_MESSAGE — generate()/stream_generate() themselves never call an external API.
"""

from collections.abc import AsyncIterator

from app.rag.providers.errors import ProviderNotImplementedError
from app.rag.providers.llm_provider import LLMProvider


class LLMProviderStub(LLMProvider):
    """Base for future-provider placeholders: every method always raises, never calls out."""

    NOT_IMPLEMENTED_MESSAGE: str = "Provider is not implemented yet."

    async def generate(self, prompt: str) -> str:
        """Always raise; this provider is not implemented yet."""
        raise ProviderNotImplementedError(self.NOT_IMPLEMENTED_MESSAGE)

    async def stream_generate(self, prompt: str) -> AsyncIterator[str]:
        """Always raise; this provider is not implemented yet."""
        raise ProviderNotImplementedError(self.NOT_IMPLEMENTED_MESSAGE)
        yield  # pragma: no cover - unreachable; makes this an async generator for typing
