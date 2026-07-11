"""Resolves the configured RagEngine implementation from RAG_ENGINE.

Mirrors app/rag/providers/provider_factory.py's pattern: callers ask for a RagEngine by
configuration name, and this module decides which concrete class to construct. Never silently
falls back to the default engine, and never silently switches which underlying providers a
resolved engine uses.
"""

from app.core.config import Settings, get_settings
from app.rag.engine import RagEngine
from app.rag.engines.custom_engine import CustomRagEngine
from app.rag.engines.langchain_engine import LangChainRagEngine


class UnsupportedRagEngineError(ValueError):
    """Raised when RAG_ENGINE names an engine with no registered implementation."""


def get_rag_engine(settings: Settings | None = None) -> RagEngine:
    """Return the RagEngine implementation configured via RAG_ENGINE ('custom' or 'langchain')."""
    settings = settings or get_settings()
    engine = settings.rag_engine

    if engine == "custom":
        return CustomRagEngine(settings=settings)

    if engine == "langchain":
        return LangChainRagEngine(settings=settings)

    raise UnsupportedRagEngineError(
        f"Unsupported RAG_ENGINE: {engine!r}. Supported engines: 'custom', 'langchain'."
    )
