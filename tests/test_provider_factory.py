"""Tests for the provider factory's configuration-driven resolution."""

import pytest

from app.core.config import Settings
from app.rag.providers.ollama_embedding_provider import OllamaEmbeddingProvider
from app.rag.providers.ollama_llm_provider import OllamaLLMProvider
from app.rag.providers.provider_factory import (
    UnsupportedProviderError,
    get_embedding_provider,
    get_llm_provider,
    get_vector_store,
)


def _settings(**overrides: str) -> Settings:
    """Build a Settings instance, keyed by env-var alias (e.g. EMBEDDING_PROVIDER=...)."""
    return Settings(**overrides)


def test_get_embedding_provider_returns_ollama_when_configured() -> None:
    """EMBEDDING_PROVIDER=ollama should resolve to OllamaEmbeddingProvider."""
    settings = _settings(EMBEDDING_PROVIDER="ollama")

    provider = get_embedding_provider(settings)

    assert isinstance(provider, OllamaEmbeddingProvider)


def test_get_llm_provider_returns_ollama_when_configured() -> None:
    """LLM_PROVIDER=ollama should resolve to OllamaLLMProvider."""
    settings = _settings(LLM_PROVIDER="ollama")

    provider = get_llm_provider(settings)

    assert isinstance(provider, OllamaLLMProvider)


def test_get_embedding_provider_raises_on_unsupported_provider() -> None:
    """An unrecognized EMBEDDING_PROVIDER should raise a clear configuration error."""
    settings = _settings(EMBEDDING_PROVIDER="openai")

    with pytest.raises(UnsupportedProviderError, match="openai"):
        get_embedding_provider(settings)


def test_get_llm_provider_raises_on_unsupported_provider() -> None:
    """An unrecognized LLM_PROVIDER should raise a clear configuration error."""
    settings = _settings(LLM_PROVIDER="anthropic")

    with pytest.raises(UnsupportedProviderError, match="anthropic"):
        get_llm_provider(settings)


def test_get_vector_store_qdrant_raises_not_implemented() -> None:
    """VECTOR_STORE_PROVIDER=qdrant is recognized but has no implementation yet."""
    settings = _settings(VECTOR_STORE_PROVIDER="qdrant")

    with pytest.raises(NotImplementedError):
        get_vector_store(settings)


def test_get_vector_store_raises_on_unsupported_provider() -> None:
    """An unrecognized VECTOR_STORE_PROVIDER should raise a clear configuration error."""
    settings = _settings(VECTOR_STORE_PROVIDER="pinecone")

    with pytest.raises(UnsupportedProviderError, match="pinecone"):
        get_vector_store(settings)
