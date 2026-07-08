"""Sanity check that Settings loads and default Ollama model names are correct."""

from app.core.config import Settings, get_settings


def test_settings_load_with_expected_defaults() -> None:
    """Verify Settings loads and the default Ollama model names are correct."""
    settings = get_settings()

    assert isinstance(settings, Settings)
    assert settings.ollama_chat_model == "llama3.1"
    assert settings.ollama_embedding_model == "nomic-embed-text"


def test_resolved_llm_model_falls_back_to_ollama_chat_model() -> None:
    """Without LLM_MODEL set, resolved_llm_model should use OLLAMA_CHAT_MODEL."""
    settings = Settings(OLLAMA_CHAT_MODEL="mistral")

    assert settings.resolved_llm_model == "mistral"


def test_resolved_llm_model_prefers_llm_model_when_set() -> None:
    """LLM_MODEL should take precedence over OLLAMA_CHAT_MODEL when both are set."""
    settings = Settings(LLM_MODEL="llama3.2", OLLAMA_CHAT_MODEL="mistral")

    assert settings.resolved_llm_model == "llama3.2"
