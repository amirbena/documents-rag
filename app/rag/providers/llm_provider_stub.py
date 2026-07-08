"""Shared base for future LLM provider stubs that are recognized but not implemented yet.

Concrete stubs (OpenAIProvider, GeminiProvider, AnthropicProvider) only override
NOT_IMPLEMENTED_MESSAGE — generate() itself never calls an external API.
"""

from app.rag.providers.errors import ProviderNotImplementedError
from app.rag.providers.llm_provider import LLMProvider


class LLMProviderStub(LLMProvider):
    """Base for future-provider placeholders: generate() always raises, never calls out."""

    NOT_IMPLEMENTED_MESSAGE: str = "Provider is not implemented yet."

    async def generate(self, prompt: str) -> str:
        """Always raise; this provider is not implemented yet."""
        raise ProviderNotImplementedError(self.NOT_IMPLEMENTED_MESSAGE)
