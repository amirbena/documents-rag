"""Resolves concrete provider implementations from configuration.

Keeps the rest of the codebase decoupled from any single AI provider — callers ask for an
EmbeddingProvider/LLMProvider/VectorStore by capability via LLM_PROVIDER / EMBEDDING_PROVIDER /
VECTOR_STORE_PROVIDER, and this module decides which concrete class to construct. All
provider-specific code (HTTP calls, error handling) stays inside each provider class; this
module only selects and constructs, it never reimplements provider logic.
"""

from app.core.config import Settings, get_settings
from app.rag.providers.anthropic_provider import AnthropicProvider
from app.rag.providers.embedding_provider import EmbeddingProvider
from app.rag.providers.errors import ProviderNotImplementedError
from app.rag.providers.gemini_provider import GeminiProvider
from app.rag.providers.llm_provider import LLMProvider
from app.rag.providers.llm_provider_stub import LLMProviderStub
from app.rag.providers.ollama_embedding_provider import OllamaEmbeddingProvider
from app.rag.providers.ollama_llm_provider import OllamaLLMProvider
from app.rag.providers.openai_provider import OpenAIProvider
from app.rag.providers.vector_store import VectorStore

# Recognized LLM providers with no real implementation yet. Configuring one of these fails
# explicitly via ProviderNotImplementedError — the backend never silently falls back to Ollama.
_LLM_STUBS: dict[str, type[LLMProviderStub]] = {
    "openai": OpenAIProvider,
    "gemini": GeminiProvider,
    "anthropic": AnthropicProvider,
}


class UnsupportedProviderError(ValueError):
    """Raised when a *_PROVIDER setting names a provider with no registered implementation."""


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Return the EmbeddingProvider implementation configured via EMBEDDING_PROVIDER."""
    settings = settings or get_settings()
    provider = settings.embedding_provider

    if provider == "ollama":
        return OllamaEmbeddingProvider(settings=settings)

    raise UnsupportedProviderError(
        f"Unsupported EMBEDDING_PROVIDER: {provider!r}. Supported providers: 'ollama'."
    )


def get_llm_provider(settings: Settings | None = None) -> LLMProvider:
    """Return the LLMProvider implementation configured via LLM_PROVIDER."""
    settings = settings or get_settings()
    provider = settings.llm_provider

    if provider == "ollama":
        return OllamaLLMProvider(settings=settings)

    if provider in _LLM_STUBS:
        raise ProviderNotImplementedError(_LLM_STUBS[provider].NOT_IMPLEMENTED_MESSAGE)

    raise UnsupportedProviderError(
        f"Unsupported LLM_PROVIDER: {provider!r}. "
        f"Supported providers: 'ollama'. Recognized but not implemented: "
        f"{', '.join(sorted(_LLM_STUBS))}."
    )


def get_vector_store(settings: Settings | None = None) -> VectorStore:
    """Return the VectorStore implementation configured via VECTOR_STORE_PROVIDER."""
    settings = settings or get_settings()
    provider = settings.vector_store_provider

    if provider == "qdrant":
        raise NotImplementedError(
            "VECTOR_STORE_PROVIDER=qdrant is recognized but no concrete VectorStore "
            "implementation exists yet — it will be added in a later milestone."
        )

    raise UnsupportedProviderError(
        f"Unsupported VECTOR_STORE_PROVIDER: {provider!r}. "
        "Supported providers: 'qdrant' (not yet implemented)."
    )
