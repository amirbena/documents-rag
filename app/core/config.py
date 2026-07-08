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
    embedding_provider: str = Field(default="ollama", alias="EMBEDDING_PROVIDER")
    vector_store_provider: str = Field(default="qdrant", alias="VECTOR_STORE_PROVIDER")


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide cached Settings instance."""
    return Settings()
