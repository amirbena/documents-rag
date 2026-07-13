"""Application configuration loaded from environment variables / .env file.

Defaults match the docker-compose service names (postgres, redis, qdrant, ollama),
so the app works out of the box inside the Compose network without an .env file.
"""

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

SUPPORTED_RESPONSE_LANGUAGES = ("he", "en")
SUPPORTED_FILE_STORAGE_PROVIDERS = ("local", "minio")


class Settings(BaseSettings):
    """Typed application settings, one field per environment variable."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

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
