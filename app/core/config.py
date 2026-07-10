"""Application configuration loaded from environment variables / .env file.

Defaults match the docker-compose service names (postgres, redis, qdrant, ollama),
so the app works out of the box inside the Compose network without an .env file.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
    ollama_embedding_model: str = Field(default="nomic-embed-text", alias="OLLAMA_EMBEDDING_MODEL")

    llm_provider: str = Field(default="ollama", alias="LLM_PROVIDER")
    llm_model: str | None = Field(default=None, alias="LLM_MODEL")
    embedding_provider: str = Field(default="ollama", alias="EMBEDDING_PROVIDER")
    vector_store_provider: str = Field(default="qdrant", alias="VECTOR_STORE_PROVIDER")

    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")

    qdrant_collection_name: str = Field(default="documents", alias="QDRANT_COLLECTION_NAME")
    vector_size: int = Field(default=768, alias="VECTOR_SIZE")

    retrieval_top_k: int = Field(default=5, alias="RETRIEVAL_TOP_K")
    retrieval_score_threshold: float | None = Field(
        default=None, alias="RETRIEVAL_SCORE_THRESHOLD"
    )

    @property
    def resolved_llm_model(self) -> str:
        """Return LLM_MODEL if set, else OLLAMA_CHAT_MODEL for backward compatibility."""
        return self.llm_model or self.ollama_chat_model


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached Settings instance."""
    return Settings()
