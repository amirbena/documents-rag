"""Stub for a future OpenAI-backed LLMProvider.

Explicit placeholder only — makes no HTTP calls, no external API keys are read. Exists so
LLM_PROVIDER=openai fails clearly and immediately instead of silently falling back to Ollama.
"""

from app.rag.providers.llm_provider_stub import LLMProviderStub


class OpenAIProvider(LLMProviderStub):
    """Placeholder LLMProvider for OpenAI — not implemented yet."""

    NOT_IMPLEMENTED_MESSAGE = "OpenAI provider is not implemented yet."
