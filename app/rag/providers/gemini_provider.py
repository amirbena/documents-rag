"""Stub for a future Gemini-backed LLMProvider.

Explicit placeholder only — makes no HTTP calls, no external API keys are read. Exists so
LLM_PROVIDER=gemini fails clearly and immediately instead of silently falling back to Ollama.
"""

from app.rag.providers.llm_provider_stub import LLMProviderStub


class GeminiProvider(LLMProviderStub):
    """Placeholder LLMProvider for Gemini — not implemented yet."""

    NOT_IMPLEMENTED_MESSAGE = "Gemini provider is not implemented yet."
