"""Sanity check that Settings loads and default Ollama model names are correct.

Phase 2.10 configuration-validation coverage (URL format, provider names, MinIO completeness,
positive timeout/pool/retry values, retry cross-field ordering, no secret leakage) lives here too.
"""

import pytest

from app.core.config import Settings, get_settings


def test_settings_load_with_expected_defaults() -> None:
    """Verify Settings loads and the default Ollama model names are correct."""
    settings = get_settings()

    assert isinstance(settings, Settings)
    assert settings.ollama_chat_model == "llama3.1"
    assert settings.ollama_embedding_model == "bge-m3"


def test_resolved_llm_model_falls_back_to_ollama_chat_model() -> None:
    """Without LLM_MODEL set, resolved_llm_model should use OLLAMA_CHAT_MODEL."""
    settings = Settings(OLLAMA_CHAT_MODEL="mistral")

    assert settings.resolved_llm_model == "mistral"


def test_resolved_llm_model_prefers_llm_model_when_set() -> None:
    """LLM_MODEL should take precedence over OLLAMA_CHAT_MODEL when both are set."""
    settings = Settings(LLM_MODEL="llama3.2", OLLAMA_CHAT_MODEL="mistral")

    assert settings.resolved_llm_model == "llama3.2"


def test_generation_llm_provider_and_model_defaults_are_unchanged() -> None:
    """The multilingual-runtime/re-index cleanup work must never change the generation LLM.

    Only the embedding model/dimension/version defaults changed in this milestone — LLM_PROVIDER,
    OLLAMA_CHAT_MODEL, and RAG_ENGINE stay exactly as they were.
    """
    settings = get_settings()

    assert settings.llm_provider == "ollama"
    assert settings.ollama_chat_model == "llama3.1"
    assert settings.rag_engine == "custom"


# --- Phase 2.10: URL format validation --------------------------------------------------------


@pytest.mark.parametrize(
    "field_alias",
    ["DATABASE_URL", "REDIS_URL", "QDRANT_URL", "OLLAMA_BASE_URL"],
)
def test_malformed_url_is_rejected(field_alias: str) -> None:
    """A URL missing a scheme/host must fail Settings construction, not surface later."""
    with pytest.raises(ValueError, match="well-formed URL"):
        Settings(**{field_alias: "not-a-url"})


@pytest.mark.parametrize(
    "field_alias",
    ["DATABASE_URL", "REDIS_URL", "QDRANT_URL", "OLLAMA_BASE_URL"],
)
def test_well_formed_url_is_accepted(field_alias: str) -> None:
    """A structurally valid URL passes, even if unreachable — reachability is a runtime concern."""
    settings = Settings(**{field_alias: "http://example.invalid:1234"})
    assert settings is not None


# --- Phase 2.10: provider/engine name validation ----------------------------------------------


def test_unsupported_embedding_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="EMBEDDING_PROVIDER"):
        Settings(EMBEDDING_PROVIDER="unsupported-provider")


def test_unsupported_vector_store_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="VECTOR_STORE_PROVIDER"):
        Settings(VECTOR_STORE_PROVIDER="unsupported-provider")


def test_unsupported_llm_provider_is_rejected() -> None:
    with pytest.raises(ValueError, match="LLM_PROVIDER"):
        Settings(LLM_PROVIDER="unsupported-provider")


@pytest.mark.parametrize("provider_name", ["openai", "gemini", "anthropic"])
def test_recognized_llm_stub_providers_are_accepted_at_config_time(provider_name: str) -> None:
    """A recognized-but-unimplemented LLM provider is config-valid — it fails later, at the
    factory, via ProviderNotImplementedError, never silently at config time."""
    settings = Settings(LLM_PROVIDER=provider_name)
    assert settings.llm_provider == provider_name


def test_unsupported_rag_engine_is_rejected() -> None:
    with pytest.raises(ValueError, match="RAG_ENGINE"):
        Settings(RAG_ENGINE="unsupported-engine")


# --- Phase 2.10: MinIO cross-field completeness ------------------------------------------------


def test_minio_provider_without_required_fields_is_rejected() -> None:
    with pytest.raises(ValueError, match="FILE_STORAGE_PROVIDER=minio requires"):
        Settings(FILE_STORAGE_PROVIDER="minio")


def test_minio_provider_error_never_echoes_the_secret_key() -> None:
    """The validation error must name missing fields, never a configured secret's value."""
    with pytest.raises(ValueError) as exc_info:
        Settings(FILE_STORAGE_PROVIDER="minio", MINIO_SECRET_KEY="super-secret-value")
    assert "super-secret-value" not in str(exc_info.value)


def test_minio_provider_with_all_required_fields_is_accepted() -> None:
    settings = Settings(
        FILE_STORAGE_PROVIDER="minio",
        MINIO_ENDPOINT="localhost:9000",
        MINIO_ACCESS_KEY="key",
        MINIO_SECRET_KEY="secret",
        MINIO_BUCKET="documents",
    )
    assert settings.file_storage_provider == "minio"


# --- Phase 2.10: timeout / pool / retry settings ------------------------------------------------


@pytest.mark.parametrize(
    "field_alias",
    [
        "OLLAMA_EMBEDDING_TIMEOUT_SECONDS",
        "OLLAMA_LLM_TIMEOUT_SECONDS",
        "OLLAMA_HEALTH_TIMEOUT_SECONDS",
        "QDRANT_TIMEOUT_SECONDS",
        "MINIO_TIMEOUT_SECONDS",
        "DB_POOL_TIMEOUT",
        "PROVIDER_RETRY_BASE_DELAY_SECONDS",
        "PROVIDER_RETRY_MAX_DELAY_SECONDS",
    ],
)
def test_non_positive_timeout_settings_are_rejected(field_alias: str) -> None:
    with pytest.raises(ValueError, match="positive number of seconds"):
        Settings(**{field_alias: 0})


@pytest.mark.parametrize(
    "field_alias",
    ["DB_POOL_SIZE", "DB_MAX_OVERFLOW", "DB_POOL_RECYCLE", "PROVIDER_RETRY_MAX_ATTEMPTS"],
)
def test_non_positive_pool_and_retry_int_settings_are_rejected(field_alias: str) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        Settings(**{field_alias: 0})


def test_retry_max_delay_must_not_be_less_than_base_delay() -> None:
    with pytest.raises(ValueError, match="PROVIDER_RETRY_MAX_DELAY_SECONDS"):
        Settings(PROVIDER_RETRY_BASE_DELAY_SECONDS=5.0, PROVIDER_RETRY_MAX_DELAY_SECONDS=1.0)


def test_retry_max_delay_equal_to_base_delay_is_accepted() -> None:
    settings = Settings(PROVIDER_RETRY_BASE_DELAY_SECONDS=2.0, PROVIDER_RETRY_MAX_DELAY_SECONDS=2.0)
    assert settings.provider_retry_max_delay_seconds == 2.0


def test_new_timeout_pool_retry_settings_have_documented_defaults() -> None:
    """Timeout defaults must match the values previously hardcoded in each provider module."""
    settings = get_settings()

    assert settings.ollama_embedding_timeout_seconds == 30.0
    assert settings.ollama_llm_timeout_seconds == 60.0
    assert settings.ollama_health_timeout_seconds == 5.0
    assert settings.qdrant_timeout_seconds == 30.0
    assert settings.minio_timeout_seconds == 30.0
    assert settings.provider_retry_max_attempts == 3


# --- Phase 2.10: CORS origins -------------------------------------------------------------------


def test_cors_allow_origins_defaults_to_empty_list() -> None:
    settings = get_settings()
    assert settings.cors_allow_origins_list == []


def test_cors_allow_origins_parses_comma_separated_list() -> None:
    settings = Settings(CORS_ALLOW_ORIGINS="http://localhost:3000, http://localhost:5173")
    assert settings.cors_allow_origins_list == ["http://localhost:3000", "http://localhost:5173"]
