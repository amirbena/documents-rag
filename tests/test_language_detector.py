"""Tests for ScriptBasedLanguageDetector — deterministic, no ML model, no external service."""

from app.core.config import Settings
from app.rag.language import ScriptBasedLanguageDetector, SupportedLanguage


def _detector(default_response_language: str = "en") -> ScriptBasedLanguageDetector:
    return ScriptBasedLanguageDetector(Settings(DEFAULT_RESPONSE_LANGUAGE=default_response_language))


def test_hebrew_only_query_detects_hebrew() -> None:
    """A purely Hebrew question should resolve to Hebrew."""
    result = _detector().detect("מה מדיניות ההחזרים שלכם?")

    assert result == SupportedLanguage.HE


def test_english_only_query_detects_english() -> None:
    """A purely English question should resolve to English."""
    result = _detector().detect("What is your refund policy?")

    assert result == SupportedLanguage.EN


def test_mixed_query_dominated_by_hebrew_detects_hebrew() -> None:
    """A mostly-Hebrew question with a couple of English words should still resolve to Hebrew."""
    result = _detector().detect("מה קורה כאשר יש בעיה עם ה server ולא ניתן להתחבר אליו כרגע?")

    assert result == SupportedLanguage.HE


def test_mixed_query_dominated_by_english_detects_english() -> None:
    """A mostly-English question with a couple of Hebrew words should resolve to English."""
    result = _detector().detect(
        "Can you please explain what the שלום command actually does in this specific context?"
    )

    assert result == SupportedLanguage.EN


def test_hebrew_with_technical_terms_still_detects_hebrew() -> None:
    """Technical identifiers (Kafka, Kubernetes, Qdrant, LangChain) must not override Hebrew."""
    result = _detector().detect(
        "מה ההבדל בין Kafka לבין Kubernetes בארכיטקטורת מיקרו-שירותים מודרנית שאנחנו בונים?"
    )

    assert result == SupportedLanguage.HE


def test_english_with_hebrew_entity_name_still_detects_english() -> None:
    """A Hebrew proper name embedded in an English sentence must not override English."""
    result = _detector().detect(
        "Please reach out to דוד directly about the pending API integration issue today."
    )

    assert result == SupportedLanguage.EN


def test_punctuation_only_query_falls_back_to_default() -> None:
    """Punctuation/number-only input carries no script signal — must fall back to the default."""
    result = _detector(default_response_language="he").detect("123 !!! ??? ...")

    assert result == SupportedLanguage.HE


def test_empty_query_falls_back_to_default() -> None:
    """An empty question must fall back to the configured default language."""
    result = _detector(default_response_language="en").detect("")

    assert result == SupportedLanguage.EN


def test_exact_tie_falls_back_to_default() -> None:
    """An exact Hebrew/English word-count tie is genuinely ambiguous — falls back to default."""
    result = _detector(default_response_language="he").detect("שלום Hello")

    assert result == SupportedLanguage.HE


def test_default_language_is_configurable_via_settings() -> None:
    """DEFAULT_RESPONSE_LANGUAGE must control the fallback, not a hardcoded constant."""
    assert _detector(default_response_language="en").detect("") == SupportedLanguage.EN
    assert _detector(default_response_language="he").detect("") == SupportedLanguage.HE


def test_detection_never_performed_by_either_engine_directly() -> None:
    """Neither RagOrchestrator nor LangChainRagEngine may import ScriptBasedLanguageDetector
    directly — language detection is reached only through PromptProvider."""
    import inspect

    import app.rag.engines.langchain_engine as langchain_engine_module
    import app.rag.orchestrator as orchestrator_module

    for module in (orchestrator_module, langchain_engine_module):
        source = inspect.getsource(module)
        assert "ScriptBasedLanguageDetector" not in source
