"""Sanity check that Settings loads and default Ollama model names are correct."""

from app.core.config import Settings, get_settings


def test_settings_load_with_expected_defaults() -> None:
    """Verify Settings loads and the default Ollama model names are correct."""
    settings = get_settings()

    assert isinstance(settings, Settings)
    assert settings.ollama_chat_model == "llama3.1"
    assert settings.ollama_embedding_model == "nomic-embed-text"
