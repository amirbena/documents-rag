"""Tests for the provider factory's configuration-driven resolution."""

import pytest

from app.core.config import Settings
from app.rag.providers.errors import ProviderNotImplementedError
from app.rag.providers.ollama_embedding_provider import OllamaEmbeddingProvider
from app.rag.providers.ollama_llm_provider import OllamaLLMProvider
from app.rag.providers.provider_factory import (
    UnsupportedProviderError,
    get_embedding_provider,
    get_llm_provider,
    get_vector_store,
)
from app.rag.providers.qdrant_vector_store import QdrantVectorStore


def _settings(**overrides: str) -> Settings:
    """Build a Settings instance, keyed by env-var alias (e.g. EMBEDDING_PROVIDER=...)."""
    return Settings(**overrides)


def _settings_bypassing_provider_validation(**field_overrides: str) -> Settings:
    """Build a Settings instance with a provider name Settings' own validation would reject.

    As of Phase 2.10, Settings validates *_PROVIDER fields against a closed set at construction
    time (the same names the factory itself recognizes), so a truly-unsupported name can no
    longer reach the factory via normal construction — Settings itself now raises first. This
    helper uses `model_construct()` (bypasses Settings' validators) to still exercise the
    factory's own UnsupportedProviderError as defense in depth, keyed by field name (not alias).
    """
    return Settings.model_construct(**{**Settings().model_dump(), **field_overrides})


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
    settings = _settings_bypassing_provider_validation(embedding_provider="openai")

    with pytest.raises(UnsupportedProviderError, match="openai"):
        get_embedding_provider(settings)


def test_get_llm_provider_raises_on_unsupported_provider() -> None:
    """An unrecognized LLM_PROVIDER should raise a clear configuration error."""
    settings = _settings_bypassing_provider_validation(llm_provider="cohere")

    with pytest.raises(UnsupportedProviderError, match="cohere"):
        get_llm_provider(settings)


@pytest.mark.parametrize("provider_name", ["openai", "gemini", "anthropic"])
def test_get_llm_provider_raises_provider_not_implemented_for_future_stubs(provider_name: str) -> None:
    """A recognized-but-unimplemented LLM_PROVIDER should fail explicitly, not fall back to Ollama."""
    settings = _settings(LLM_PROVIDER=provider_name)

    with pytest.raises(ProviderNotImplementedError, match="not implemented yet"):
        get_llm_provider(settings)


def test_get_vector_store_returns_qdrant_when_configured() -> None:
    """VECTOR_STORE_PROVIDER=qdrant should resolve to QdrantVectorStore."""
    settings = _settings(VECTOR_STORE_PROVIDER="qdrant")

    store = get_vector_store(settings)

    assert isinstance(store, QdrantVectorStore)


def test_get_vector_store_raises_on_unsupported_provider() -> None:
    """An unrecognized VECTOR_STORE_PROVIDER should raise a clear configuration error."""
    settings = _settings_bypassing_provider_validation(vector_store_provider="pinecone")

    with pytest.raises(UnsupportedProviderError, match="pinecone"):
        get_vector_store(settings)
