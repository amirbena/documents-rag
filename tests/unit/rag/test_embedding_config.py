"""Tests for EmbeddingIndexConfig — validation and deterministic collection identity."""

import pytest

from app.core.config import Settings
from app.rag.embedding_config import (
    EmbeddingIndexConfig,
    InvalidEmbeddingIndexConfigError,
    get_active_embedding_config,
)


def _config(**overrides: object) -> EmbeddingIndexConfig:
    fields: dict[str, object] = {
        "collection_prefix": "documents",
        "provider": "ollama",
        "model": "nomic-embed-text",
        "dimension": 768,
        "embedding_version": "v1",
        "chunking_version": "v1",
    }
    fields.update(overrides)
    return EmbeddingIndexConfig(**fields)  # type: ignore[arg-type]


def test_valid_configuration_constructs_successfully() -> None:
    """A fully-populated, positive-dimension config should construct without error."""
    config = _config()

    assert config.provider == "ollama"
    assert config.dimension == 768


def test_invalid_dimension_raises() -> None:
    """A zero or negative dimension must raise explicitly."""
    with pytest.raises(InvalidEmbeddingIndexConfigError, match="dimension"):
        _config(dimension=0)
    with pytest.raises(InvalidEmbeddingIndexConfigError, match="dimension"):
        _config(dimension=-1)


@pytest.mark.parametrize(
    "field_name", ["collection_prefix", "provider", "model", "embedding_version", "chunking_version"]
)
def test_empty_string_fields_raise(field_name: str) -> None:
    """Every string field must be non-empty — blank or whitespace-only both fail."""
    with pytest.raises(InvalidEmbeddingIndexConfigError, match=field_name):
        _config(**{field_name: ""})
    with pytest.raises(InvalidEmbeddingIndexConfigError, match=field_name):
        _config(**{field_name: "   "})


def test_collection_name_is_deterministic() -> None:
    """The same config fields must always produce the same collection name."""
    first = _config().collection_name
    second = _config().collection_name

    assert first == second


def test_collection_name_is_sanitized() -> None:
    """Collection names must be lowercase, alnum/-/_ only, regardless of input casing/spacing."""
    config = _config(provider="Ollama", model="Nomic Embed Text!")

    assert config.collection_name == config.collection_name.lower()
    assert " " not in config.collection_name
    assert "!" not in config.collection_name


def test_different_model_produces_different_collection_identity() -> None:
    """Two configs differing only in `model` must never share a collection name."""
    base = _config(model="model-a").collection_name
    other = _config(model="model-b").collection_name

    assert base != other


def test_different_dimension_produces_different_collection_identity() -> None:
    """Two configs differing only in `dimension` must never share a collection name."""
    base = _config(dimension=768).collection_name
    other = _config(dimension=1024).collection_name

    assert base != other


def test_different_embedding_version_produces_different_collection_identity() -> None:
    """Two configs differing only in `embedding_version` must never share a collection name."""
    base = _config(embedding_version="v1").collection_name
    other = _config(embedding_version="v2").collection_name

    assert base != other


def test_different_chunking_version_produces_different_collection_identity() -> None:
    """Two configs differing only in `chunking_version` must never share a collection name."""
    base = _config(chunking_version="v1").collection_name
    other = _config(chunking_version="v2").collection_name

    assert base != other


def test_different_provider_produces_different_collection_identity() -> None:
    """Two configs differing only in `provider` must never share a collection name."""
    base = _config(provider="ollama").collection_name
    other = _config(provider="openai").collection_name

    assert base != other


def test_get_active_embedding_config_reads_from_settings() -> None:
    """get_active_embedding_config() must resolve every field from the given Settings."""
    settings = Settings(
        QDRANT_COLLECTION_NAME="my-docs",
        EMBEDDING_PROVIDER="ollama",
        OLLAMA_EMBEDDING_MODEL="custom-model",
        VECTOR_SIZE=1024,
        EMBEDDING_VERSION="v3",
        CHUNKING_VERSION="v2",
    )

    config = get_active_embedding_config(settings)

    assert config.collection_prefix == "my-docs"
    assert config.provider == "ollama"
    assert config.model == "custom-model"
    assert config.dimension == 1024
    assert config.embedding_version == "v3"
    assert config.chunking_version == "v2"


def test_embedding_model_setting_overrides_ollama_embedding_model() -> None:
    """EMBEDDING_MODEL, when set, must take precedence over OLLAMA_EMBEDDING_MODEL."""
    settings = Settings(OLLAMA_EMBEDDING_MODEL="nomic-embed-text", EMBEDDING_MODEL="bge-m3")

    config = get_active_embedding_config(settings)

    assert config.model == "bge-m3"


def test_ingestion_and_retrieval_resolve_the_same_active_config() -> None:
    """Calling get_active_embedding_config() twice with equivalent settings must be identical."""
    settings = Settings()

    ingestion_side = get_active_embedding_config(settings)
    retrieval_side = get_active_embedding_config(settings)

    assert ingestion_side == retrieval_side
    assert ingestion_side.collection_name == retrieval_side.collection_name
