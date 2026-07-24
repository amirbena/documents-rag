"""Tests for the RagEngine factory's configuration-driven resolution."""

import pytest

from app.core.config import Settings
from app.rag.engines.custom_engine import CustomRagEngine
from app.rag.engines.engine_factory import UnsupportedRagEngineError, get_rag_engine
from app.rag.engines.langchain_engine import LangChainRagEngine


def _settings(**overrides: str) -> Settings:
    """Build a Settings instance, keyed by env-var alias (e.g. RAG_ENGINE=...)."""
    return Settings(**overrides)


def _settings_bypassing_engine_validation(**field_overrides: str) -> Settings:
    """Build a Settings instance with a RAG_ENGINE value Settings' own validation would reject.

    As of Phase 2.10, Settings validates RAG_ENGINE against a closed set at construction time (the
    same names the factory itself recognizes), so a truly-unsupported name can no longer reach the
    factory via normal construction. This bypasses Settings' validators via `model_construct()` to
    still exercise the factory's own UnsupportedRagEngineError as defense in depth.
    """
    return Settings.model_construct(**{**Settings().model_dump(), **field_overrides})


def test_custom_resolves_to_custom_rag_engine() -> None:
    """RAG_ENGINE=custom should resolve to CustomRagEngine."""
    engine = get_rag_engine(_settings(RAG_ENGINE="custom"))

    assert isinstance(engine, CustomRagEngine)


def test_langchain_resolves_to_langchain_rag_engine() -> None:
    """RAG_ENGINE=langchain should resolve to LangChainRagEngine."""
    engine = get_rag_engine(_settings(RAG_ENGINE="langchain"))

    assert isinstance(engine, LangChainRagEngine)


def test_default_engine_is_custom() -> None:
    """With RAG_ENGINE unset, the default must remain custom (existing installs unaffected)."""
    engine = get_rag_engine(_settings())

    assert isinstance(engine, CustomRagEngine)


def test_unknown_engine_raises_explicitly() -> None:
    """An unrecognized RAG_ENGINE must fail explicitly, never silently fall back to custom."""
    with pytest.raises(UnsupportedRagEngineError, match="bogus"):
        get_rag_engine(_settings_bypassing_engine_validation(rag_engine="bogus"))
