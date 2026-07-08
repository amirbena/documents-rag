"""Tests for the future LLM provider stubs (OpenAI, Gemini, Anthropic).

These stubs implement LLMProvider but make no external API calls — every method must raise
ProviderNotImplementedError explicitly rather than silently doing nothing or delegating to Ollama.
"""

import pytest

from app.rag.providers.anthropic_provider import AnthropicProvider
from app.rag.providers.errors import ProviderNotImplementedError
from app.rag.providers.gemini_provider import GeminiProvider
from app.rag.providers.llm_provider import LLMProvider
from app.rag.providers.openai_provider import OpenAIProvider


@pytest.mark.parametrize(
    "provider_cls",
    [OpenAIProvider, GeminiProvider, AnthropicProvider],
)
async def test_stub_generate_raises_provider_not_implemented(provider_cls: type[LLMProvider]) -> None:
    """Each stub's generate() must raise ProviderNotImplementedError with a clear message."""
    provider = provider_cls()

    with pytest.raises(ProviderNotImplementedError, match="not implemented yet"):
        await provider.generate("hello")


def test_stub_classes_implement_llm_provider_interface() -> None:
    """Each stub must be a concrete LLMProvider — instantiable, not still abstract."""
    assert isinstance(OpenAIProvider(), LLMProvider)
    assert isinstance(GeminiProvider(), LLMProvider)
    assert isinstance(AnthropicProvider(), LLMProvider)
