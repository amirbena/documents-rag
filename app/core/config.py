"""Application configuration loaded from environment variables / .env file.

Defaults match the docker-compose service names (postgres, redis, qdrant, ollama),
so the app works out of the box inside the Compose network without an .env file.

Validation here is deliberately fail-fast: a malformed URL, an unsupported provider name, an
incomplete MinIO configuration, or a non-positive timeout/pool/retry value all raise at
`Settings()` construction (which happens at process import time via the module-level
`get_settings()` calls in `app/main.py`/`app/db/session.py`) rather than surfacing later on the
first request that happens to exercise the broken path. Error messages name the offending field
only — never a secret value (e.g. `minio_secret_key` is checked for presence, never echoed).
"""

from functools import lru_cache
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

SUPPORTED_RESPONSE_LANGUAGES = ("he", "en")
SUPPORTED_FILE_STORAGE_PROVIDERS = ("local", "minio")
SUPPORTED_EMBEDDING_PROVIDERS = ("ollama",)
SUPPORTED_VECTOR_STORE_PROVIDERS = ("qdrant",)
# 'openai'/'gemini'/'anthropic' are recognized-but-not-yet-implemented LLM providers (see
# app/rag/providers/provider_factory.py's _LLM_STUBS) — a config-time-valid name that still fails
# loudly and explicitly at first use via ProviderNotImplementedError, never a silent Ollama fallback.
SUPPORTED_LLM_PROVIDERS = ("ollama", "openai", "gemini", "anthropic")
SUPPORTED_RAG_ENGINES = ("custom", "langchain")


class Settings(BaseSettings):
    """Typed application settings, one field per environment variable."""

    # hide_input_in_errors: Pydantic's default ValidationError repr embeds the full offending
    # input (the whole constructor kwargs dict, for a model_validator(mode="after") failure) —
    # without this, a validation error on an unrelated field (e.g. an incomplete MinIO
    # configuration) would echo minio_secret_key's raw value into the exception's str().
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", hide_input_in_errors=True
    )

    app_env: str = Field(default="local", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@postgres:5432/rag_db",
        alias="DATABASE_URL",
    )
    redis_url: str = Field(default="redis://redis:6379/0", alias="REDIS_URL")
    qdrant_url: str = Field(default="http://qdrant:6333", alias="QDRANT_URL")

    ollama_base_url: str = Field(default="http://ollama:11434", alias="OLLAMA_BASE_URL")
    ollama_chat_model: str = Field(default="llama3.1", alias="OLLAMA_CHAT_MODEL")
    # bge-m3 is the default: a genuinely multilingual (100+ languages, including Hebrew) Ollama
    # embedding model, 1024-dim. See "Multilingual embedding model" in ARCHITECTURE.md for the
    # selection rationale and the migration note for installations still on nomic-embed-text.
    ollama_embedding_model: str = Field(default="bge-m3", alias="OLLAMA_EMBEDDING_MODEL")

    llm_provider: str = Field(default="ollama", alias="LLM_PROVIDER")
    llm_model: str | None = Field(default=None, alias="LLM_MODEL")
    embedding_provider: str = Field(default="ollama", alias="EMBEDDING_PROVIDER")
    vector_store_provider: str = Field(default="qdrant", alias="VECTOR_STORE_PROVIDER")

    # EMBEDDING_MODEL is the generic, provider-agnostic override — falls back to
    # OLLAMA_EMBEDDING_MODEL, mirroring the LLM_MODEL/OLLAMA_CHAT_MODEL pattern above. Changing
    # either requires bumping EMBEDDING_VERSION (see app/rag/embedding_config.py); the active
    # EmbeddingIndexConfig, not this setting alone, decides which Qdrant collection is used.
    embedding_model: str | None = Field(default=None, alias="EMBEDDING_MODEL")
    # v2: bumped alongside the nomic-embed-text -> bge-m3 default-model change, so installations
    # upgrading from Phase 2.5's v1 land in a new, distinct Qdrant collection rather than
    # silently reusing one built from 768-dim nomic-embed-text vectors under a bge-m3 config.
    embedding_version: str = Field(default="v2", alias="EMBEDDING_VERSION")
    chunking_version: str = Field(default="v1", alias="CHUNKING_VERSION")

    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")

    # Acts as the collection *prefix/namespace* the active EmbeddingIndexConfig derives the real,
    # versioned Qdrant collection name from (see app/rag/embedding_config.py) — not a literal
    # collection name by itself once versioned collections are in play.
    qdrant_collection_name: str = Field(default="documents", alias="QDRANT_COLLECTION_NAME")
    # 1024 matches bge-m3's output dimension (the default embedding model above). Installations
    # pinned to the legacy nomic-embed-text (768-dim) must set VECTOR_SIZE=768 explicitly.
    vector_size: int = Field(default=1024, alias="VECTOR_SIZE")

    retrieval_top_k: int = Field(default=5, alias="RETRIEVAL_TOP_K")
    retrieval_score_threshold: float | None = Field(
        default=None, alias="RETRIEVAL_SCORE_THRESHOLD"
    )

    rag_engine: str = Field(default="custom", alias="RAG_ENGINE")

    default_response_language: str = Field(default="en", alias="DEFAULT_RESPONSE_LANGUAGE")
    # v2: bumped alongside the shared-English-instructions + explicit response-language-directive
    # prompt architecture change — the resolved system prompt's content structure changed even
    # though the catalog's PromptType coverage did not.
    prompt_catalog_version: str = Field(default="v2", alias="PROMPT_CATALOG_VERSION")

    # Storage abstraction (Phase 2.6/2.7) — selects the FileStorage implementation via
    # app/storage/factory.py. Defaults to 'local' so `make verify` and local dev never require
    # MinIO. See "Storage Abstraction" in ARCHITECTURE.md.
    file_storage_provider: str = Field(default="local", alias="FILE_STORAGE_PROVIDER")
    local_storage_root: str = Field(default="storage/documents", alias="LOCAL_STORAGE_ROOT")

    # MinIO-only settings — only read/validated when FILE_STORAGE_PROVIDER=minio.
    minio_endpoint: str | None = Field(default=None, alias="MINIO_ENDPOINT")
    minio_access_key: str | None = Field(default=None, alias="MINIO_ACCESS_KEY")
    minio_secret_key: str | None = Field(default=None, alias="MINIO_SECRET_KEY")
    minio_bucket: str | None = Field(default=None, alias="MINIO_BUCKET")
    minio_secure: bool = Field(default=False, alias="MINIO_SECURE")
    minio_region: str | None = Field(default=None, alias="MINIO_REGION")
    minio_presigned_url_expiry_seconds: int = Field(
        default=3600, alias="MINIO_PRESIGNED_URL_EXPIRY_SECONDS"
    )
    minio_create_bucket_if_missing: bool = Field(default=True, alias="MINIO_CREATE_BUCKET_IF_MISSING")

    # Retry/stale-recovery (Phase 2.8.3) — see app/services/ingestion/retry_service.py and
    # app/services/ingestion/stale_recovery_service.py.
    # A PROCESSING IngestionJob whose updated_at is older than this is treated as an approximate
    # "stale" signal (the job's row hasn't been touched since it was claimed) — not proof the
    # worker died, since a slow-but-alive worker looks identical. 900s (15 minutes) is a
    # deliberately generous default so a normally-slow document is never falsely recovered.
    ingestion_stale_after_seconds: int = Field(default=900, alias="INGESTION_STALE_AFTER_SECONDS")
    # Maximum number of stale PROCESSING jobs recovered per recover_stale_ingestion_jobs() call —
    # bounds how much work one recovery run (script or future scheduler tick) does at once.
    ingestion_recovery_batch_size: int = Field(default=50, alias="INGESTION_RECOVERY_BATCH_SIZE")

    # Provider HTTP timeouts (Phase 2.10) — promoted from what were previously hardcoded literals
    # in each provider module, so they're a documented, validated, tunable surface. Defaults match
    # exactly what those literals already were — no behavior changes until overridden.
    ollama_embedding_timeout_seconds: float = Field(
        default=30.0, alias="OLLAMA_EMBEDDING_TIMEOUT_SECONDS"
    )
    ollama_llm_timeout_seconds: float = Field(default=60.0, alias="OLLAMA_LLM_TIMEOUT_SECONDS")
    ollama_health_timeout_seconds: float = Field(
        default=5.0, alias="OLLAMA_HEALTH_TIMEOUT_SECONDS"
    )
    qdrant_timeout_seconds: float = Field(default=30.0, alias="QDRANT_TIMEOUT_SECONDS")
    minio_timeout_seconds: float = Field(default=30.0, alias="MINIO_TIMEOUT_SECONDS")

    # Postgres connection pool (Phase 2.10) — promoted from SQLAlchemy's implicit defaults (which
    # `create_async_engine` previously received none of) into an explicit, documented, tunable
    # surface. Defaults mirror SQLAlchemy's own defaults for a single-process deployment.
    db_pool_size: int = Field(default=5, alias="DB_POOL_SIZE")
    db_max_overflow: int = Field(default=10, alias="DB_MAX_OVERFLOW")
    db_pool_timeout: float = Field(default=30.0, alias="DB_POOL_TIMEOUT")
    db_pool_recycle: int = Field(default=1800, alias="DB_POOL_RECYCLE")
    db_pool_pre_ping: bool = Field(default=True, alias="DB_POOL_PRE_PING")

    # Provider retry policy (Phase 2.10) — bounded exponential backoff with jitter, applied inside
    # each provider adapter (see app/core/retry.py). Only transient failures are retried; a
    # permanent error (auth, validation, unsupported request) never triggers a retry.
    provider_retry_max_attempts: int = Field(default=3, alias="PROVIDER_RETRY_MAX_ATTEMPTS")
    provider_retry_base_delay_seconds: float = Field(
        default=0.5, alias="PROVIDER_RETRY_BASE_DELAY_SECONDS"
    )
    provider_retry_max_delay_seconds: float = Field(
        default=5.0, alias="PROVIDER_RETRY_MAX_DELAY_SECONDS"
    )

    # CORS (Phase 2.10) — comma-separated list of allowed origins for frontend integration.
    # Empty by default: this backend never allows cross-origin requests until an operator
    # explicitly names the frontend origin(s); "*" is deliberately not a supported shorthand here
    # (see `cors_allow_origins_list` below) — every origin must be spelled out explicitly.
    cors_allow_origins: str = Field(default="", alias="CORS_ALLOW_ORIGINS")

    # Startup dependency validation (Phase 2.10) — off by default so a fresh `docker compose up`
    # (before Ollama models are pulled, or before Alembic migrations run) doesn't prevent the
    # process from starting at all; an operator opts in once the deployment is otherwise ready.
    startup_dependency_check: bool = Field(default=False, alias="STARTUP_DEPENDENCY_CHECK")

    @property
    def cors_allow_origins_list(self) -> list[str]:
        """Parse CORS_ALLOW_ORIGINS into a list of origins; empty string -> empty list."""
        return [origin.strip() for origin in self.cors_allow_origins.split(",") if origin.strip()]

    @field_validator("ingestion_stale_after_seconds", "ingestion_recovery_batch_size")
    @classmethod
    def _validate_positive_ingestion_recovery_settings(cls, value: int, info) -> int:
        """Stale-threshold seconds and recovery batch size must both be positive integers."""
        if value <= 0:
            raise ValueError(f"{info.field_name} must be a positive integer")
        return value

    @field_validator("vector_size")
    @classmethod
    def _validate_vector_size(cls, value: int) -> int:
        """VECTOR_SIZE must be a positive integer — it is a Qdrant vector dimension."""
        if value <= 0:
            raise ValueError("VECTOR_SIZE must be a positive integer")
        return value

    @field_validator(
        "embedding_provider", "embedding_version", "chunking_version", "qdrant_collection_name"
    )
    @classmethod
    def _validate_non_empty(cls, value: str, info) -> str:
        """These fields identify/version the active index — none may be blank."""
        if not value.strip():
            raise ValueError(f"{info.field_name} must not be empty")
        return value

    @field_validator("default_response_language")
    @classmethod
    def _validate_default_response_language(cls, value: str) -> str:
        """DEFAULT_RESPONSE_LANGUAGE must be one of the languages the prompt catalog supports."""
        if value not in SUPPORTED_RESPONSE_LANGUAGES:
            raise ValueError(
                f"DEFAULT_RESPONSE_LANGUAGE must be one of {SUPPORTED_RESPONSE_LANGUAGES}, got {value!r}"
            )
        return value

    @field_validator("file_storage_provider")
    @classmethod
    def _validate_file_storage_provider(cls, value: str) -> str:
        """FILE_STORAGE_PROVIDER must name a storage provider the factory can construct."""
        if value not in SUPPORTED_FILE_STORAGE_PROVIDERS:
            raise ValueError(
                f"FILE_STORAGE_PROVIDER must be one of {SUPPORTED_FILE_STORAGE_PROVIDERS}, got {value!r}"
            )
        return value

    @field_validator("llm_provider")
    @classmethod
    def _validate_llm_provider(cls, value: str) -> str:
        """LLM_PROVIDER must be a real implementation or a recognized (stub) placeholder.

        A recognized stub (openai/gemini/anthropic) still passes here — it fails loudly and
        explicitly at first use via ProviderNotImplementedError (see provider_factory.py), never
        silently at config time and never by falling back to Ollama.
        """
        if value not in SUPPORTED_LLM_PROVIDERS:
            raise ValueError(f"LLM_PROVIDER must be one of {SUPPORTED_LLM_PROVIDERS}, got {value!r}")
        return value

    @field_validator("embedding_provider")
    @classmethod
    def _validate_embedding_provider(cls, value: str) -> str:
        """EMBEDDING_PROVIDER must name a provider the factory can construct."""
        if value not in SUPPORTED_EMBEDDING_PROVIDERS:
            raise ValueError(
                f"EMBEDDING_PROVIDER must be one of {SUPPORTED_EMBEDDING_PROVIDERS}, got {value!r}"
            )
        return value

    @field_validator("vector_store_provider")
    @classmethod
    def _validate_vector_store_provider(cls, value: str) -> str:
        """VECTOR_STORE_PROVIDER must name a provider the factory can construct."""
        if value not in SUPPORTED_VECTOR_STORE_PROVIDERS:
            raise ValueError(
                f"VECTOR_STORE_PROVIDER must be one of {SUPPORTED_VECTOR_STORE_PROVIDERS}, "
                f"got {value!r}"
            )
        return value

    @field_validator("rag_engine")
    @classmethod
    def _validate_rag_engine(cls, value: str) -> str:
        """RAG_ENGINE must name an engine the factory can construct."""
        if value not in SUPPORTED_RAG_ENGINES:
            raise ValueError(f"RAG_ENGINE must be one of {SUPPORTED_RAG_ENGINES}, got {value!r}")
        return value

    @field_validator("database_url", "redis_url", "qdrant_url", "ollama_base_url")
    @classmethod
    def _validate_url_format(cls, value: str, info) -> str:
        """These connection strings must at least be well-formed URLs (scheme + host present).

        This only catches structurally malformed values (typos, missing scheme) — it never
        attempts to connect, so a well-formed but unreachable URL still fails later, at the
        relevant health check or first call, exactly as today.
        """
        parsed = urlparse(value)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError(
                f"{info.field_name} must be a well-formed URL with a scheme and host, got {value!r}"
            )
        return value

    @field_validator(
        "ollama_embedding_timeout_seconds",
        "ollama_llm_timeout_seconds",
        "ollama_health_timeout_seconds",
        "qdrant_timeout_seconds",
        "minio_timeout_seconds",
        "db_pool_timeout",
        "provider_retry_base_delay_seconds",
        "provider_retry_max_delay_seconds",
    )
    @classmethod
    def _validate_positive_float(cls, value: float, info) -> float:
        """Every timeout/delay setting must be a positive number of seconds."""
        if value <= 0:
            raise ValueError(f"{info.field_name} must be a positive number of seconds")
        return value

    @field_validator("db_pool_size", "db_max_overflow", "db_pool_recycle", "provider_retry_max_attempts")
    @classmethod
    def _validate_positive_int(cls, value: int, info) -> int:
        """Pool sizing and retry-attempt settings must be positive integers."""
        if value <= 0:
            raise ValueError(f"{info.field_name} must be a positive integer")
        return value

    @model_validator(mode="after")
    def _validate_retry_delay_ordering(self) -> "Settings":
        """PROVIDER_RETRY_MAX_DELAY_SECONDS must be >= the base delay it backs off from."""
        if self.provider_retry_max_delay_seconds < self.provider_retry_base_delay_seconds:
            raise ValueError(
                "PROVIDER_RETRY_MAX_DELAY_SECONDS must be >= PROVIDER_RETRY_BASE_DELAY_SECONDS "
                f"(got max={self.provider_retry_max_delay_seconds}, "
                f"base={self.provider_retry_base_delay_seconds})"
            )
        return self

    @model_validator(mode="after")
    def _validate_minio_configuration_complete(self) -> "Settings":
        """When FILE_STORAGE_PROVIDER=minio, every required MinIO field must be present.

        Moves what was previously a runtime StorageConfigurationError (raised only when
        create_file_storage() actually constructed a MinioFileStorage) earlier, to config
        construction time — the runtime check in MinioFileStorage.__init__ remains as defense in
        depth for any caller that bypasses Settings validation. Never echoes minio_secret_key.
        """
        if self.file_storage_provider != "minio":
            return self
        missing = [
            name
            for name, value in (
                ("MINIO_ENDPOINT", self.minio_endpoint),
                ("MINIO_ACCESS_KEY", self.minio_access_key),
                ("MINIO_SECRET_KEY", self.minio_secret_key),
                ("MINIO_BUCKET", self.minio_bucket),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "FILE_STORAGE_PROVIDER=minio requires the following settings to be set: "
                f"{', '.join(missing)}"
            )
        return self

    @property
    def resolved_llm_model(self) -> str:
        """Return LLM_MODEL if set, else OLLAMA_CHAT_MODEL for backward compatibility."""
        return self.llm_model or self.ollama_chat_model

    @property
    def resolved_embedding_model(self) -> str:
        """Return EMBEDDING_MODEL if set, else OLLAMA_EMBEDDING_MODEL for backward compatibility."""
        return self.embedding_model or self.ollama_embedding_model


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached Settings instance."""
    return Settings()
