"""Stub for a future Anthropic-backed LLMProvider.

Explicit placeholder only — makes no HTTP calls, no external API keys are read. Exists so
LLM_PROVIDER=anthropic fails clearly and immediately instead of silently falling back to Ollama.
"""

from app.rag.providers.llm_provider_stub import LLMProviderStub


class AnthropicProvider(LLMProviderStub):
    """Placeholder LLMProvider for Anthropic — not implemented yet."""

    NOT_IMPLEMENTED_MESSAGE = "Anthropic provider is not implemented yet."
