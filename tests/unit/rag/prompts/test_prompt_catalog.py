"""Tests for PromptCatalog/PromptProvider/ResolvedPrompt — no network, no LLM, no engine."""

import pytest

from app.core.config import Settings
from app.rag.language import SupportedLanguage
from app.rag.prompts.catalog import (
    PromptCatalog,
    UnsupportedPromptLanguageError,
    UnsupportedPromptTypeError,
)
from app.rag.prompts.provider import PromptProvider
from app.rag.prompts.types import PromptType, ResolvedPrompt

_FIXED_RESPONSE_TYPES = (PromptType.CLARIFICATION, PromptType.NO_RESULTS, PromptType.OUT_OF_SCOPE)
_SYSTEM_TEXT_TYPES = (PromptType.GROUNDED_ANSWER, PromptType.DIRECT_ANSWER)


@pytest.mark.parametrize("language", [SupportedLanguage.HE, SupportedLanguage.EN])
@pytest.mark.parametrize("prompt_type", _FIXED_RESPONSE_TYPES)
def test_fixed_response_exists_for_every_type_and_language(
    prompt_type: PromptType, language: SupportedLanguage
) -> None:
    """clarification/no_results/out_of_scope must have non-empty text in both he and en."""
    text = PromptCatalog().get_response_text(prompt_type, language)

    assert isinstance(text, str)
    assert text.strip()


@pytest.mark.parametrize("language", [SupportedLanguage.HE, SupportedLanguage.EN])
@pytest.mark.parametrize("prompt_type", _SYSTEM_TEXT_TYPES)
def test_system_text_exists_for_every_type_and_language(
    prompt_type: PromptType, language: SupportedLanguage
) -> None:
    """grounded_answer/direct_answer must have a non-empty system instruction in both languages."""
    text = PromptCatalog().get_system_text(prompt_type, language)

    assert isinstance(text, str)
    assert text.strip()


def test_grounded_answer_instructs_answering_only_from_context() -> None:
    """The grounded_answer system text must forbid inventing unsupported facts."""
    text_en = PromptCatalog().get_system_text(PromptType.GROUNDED_ANSWER, SupportedLanguage.EN)

    assert "only" in text_en.lower()
    assert "invent" in text_en.lower() or "assume" in text_en.lower()


def test_grounded_answer_instructs_preserving_technical_identifiers() -> None:
    """The grounded_answer system text must forbid translating code/API/class/env/command names."""
    text_en = PromptCatalog().get_system_text(PromptType.GROUNDED_ANSWER, SupportedLanguage.EN)
    text_he = PromptCatalog().get_system_text(PromptType.GROUNDED_ANSWER, SupportedLanguage.HE)

    for text in (text_en, text_he):
        assert "API" in text
        assert "translate" in text.lower() or "תתרגם" in text


def test_grounded_answer_system_text_carries_an_explicit_response_language_directive() -> None:
    """The composed system text must include an explicit "respond in <language>" directive."""
    text_en = PromptCatalog().get_system_text(PromptType.GROUNDED_ANSWER, SupportedLanguage.EN)
    text_he = PromptCatalog().get_system_text(PromptType.GROUNDED_ANSWER, SupportedLanguage.HE)

    assert "respond directly and naturally in english (en)" in text_en.lower()
    assert "respond directly and naturally in hebrew (he)" in text_he.lower()


def test_shared_instructions_are_english_only_and_identical_across_languages() -> None:
    """The English-authored shared instruction must never be duplicated or altered per language."""
    catalog = PromptCatalog()

    for prompt_type in _SYSTEM_TEXT_TYPES:
        assert catalog.get_shared_instructions(prompt_type) == catalog.get_shared_instructions(
            prompt_type
        )
        # Same shared instruction underlies both languages' composed system_text.
        text_en = catalog.get_system_text(prompt_type, SupportedLanguage.EN)
        text_he = catalog.get_system_text(prompt_type, SupportedLanguage.HE)
        shared = catalog.get_shared_instructions(prompt_type)
        assert text_en.startswith(shared)
        assert text_he.startswith(shared)


def test_response_language_directive_never_instructs_translate_after_the_fact() -> None:
    """The directive must instruct a direct response, never "answer in English then translate"."""
    catalog = PromptCatalog()

    for language in (SupportedLanguage.EN, SupportedLanguage.HE):
        directive = catalog.get_response_language_directive(language)
        assert "translate" not in directive.lower()
        assert "directly" in directive.lower()


def test_unsupported_language_raises_explicitly() -> None:
    """A language the catalog has no content for at all must fail explicitly."""
    with pytest.raises(UnsupportedPromptLanguageError):
        PromptCatalog().get_response_text(PromptType.CLARIFICATION, "fr")  # type: ignore[arg-type]


def test_missing_prompt_type_behavior_is_explicit() -> None:
    """Requesting a fixed response for a system-text-only type (and vice versa) fails explicitly."""
    with pytest.raises(UnsupportedPromptTypeError):
        PromptCatalog().get_response_text(PromptType.GROUNDED_ANSWER, SupportedLanguage.EN)
    with pytest.raises(UnsupportedPromptTypeError):
        PromptCatalog().get_system_text(PromptType.CLARIFICATION, SupportedLanguage.EN)


def test_resolved_prompt_requires_exactly_one_of_system_or_response_text() -> None:
    """ResolvedPrompt must reject being constructed with both or neither field set."""
    with pytest.raises(ValueError, match="exactly one"):
        ResolvedPrompt(
            prompt_type=PromptType.CLARIFICATION,
            language=SupportedLanguage.EN,
            prompt_version="v1",
            system_text="x",
            response_text="y",
        )
    with pytest.raises(ValueError, match="exactly one"):
        ResolvedPrompt(
            prompt_type=PromptType.CLARIFICATION, language=SupportedLanguage.EN, prompt_version="v1"
        )


def test_provider_resolves_fixed_response_types_with_response_text_only() -> None:
    """CLARIFICATION/NO_RESULTS/OUT_OF_SCOPE must resolve with response_text, no system_text."""
    provider = PromptProvider()

    for prompt_type in _FIXED_RESPONSE_TYPES:
        resolved = provider.resolve(prompt_type, "What is this?")
        assert resolved.response_text is not None
        assert resolved.system_text is None


def test_provider_resolves_generation_types_with_system_text_only() -> None:
    """GROUNDED_ANSWER/DIRECT_ANSWER must resolve with system_text, no response_text."""
    provider = PromptProvider()

    for prompt_type in _SYSTEM_TEXT_TYPES:
        resolved = provider.resolve(prompt_type, "What is this?")
        assert resolved.system_text is not None
        assert resolved.response_text is None


def test_provider_resolves_shared_instructions_and_language_directive_for_generation_types() -> None:
    """Generation types must also expose shared_instructions/language_directive individually."""
    provider = PromptProvider()

    for prompt_type in _SYSTEM_TEXT_TYPES:
        resolved = provider.resolve(prompt_type, "What is this?")
        assert resolved.shared_instructions is not None
        assert resolved.language_directive is not None
        assert resolved.system_text == f"{resolved.shared_instructions}\n\n{resolved.language_directive}"

    for prompt_type in _FIXED_RESPONSE_TYPES:
        resolved = provider.resolve(prompt_type, "What is this?")
        assert resolved.shared_instructions is None
        assert resolved.language_directive is None


def test_resolved_prompt_requires_shared_instructions_and_directive_iff_system_text() -> None:
    """Constructing a ResolvedPrompt with system_text set but not its components must fail."""
    with pytest.raises(ValueError, match="shared_instructions"):
        ResolvedPrompt(
            prompt_type=PromptType.GROUNDED_ANSWER,
            language=SupportedLanguage.EN,
            prompt_version="v2",
            system_text="x",
            language_directive="y",
        )
    with pytest.raises(ValueError, match="language_directive"):
        ResolvedPrompt(
            prompt_type=PromptType.GROUNDED_ANSWER,
            language=SupportedLanguage.EN,
            prompt_version="v2",
            system_text="x",
            shared_instructions="y",
        )


def test_provider_detects_language_before_resolving() -> None:
    """The provider must resolve Hebrew content for a Hebrew question, English for English."""
    provider = PromptProvider()

    hebrew_resolved = provider.resolve(PromptType.OUT_OF_SCOPE, "מה קורה כאן?")
    english_resolved = provider.resolve(PromptType.OUT_OF_SCOPE, "What is happening here?")

    assert hebrew_resolved.language == SupportedLanguage.HE
    assert english_resolved.language == SupportedLanguage.EN
    assert hebrew_resolved.response_text != english_resolved.response_text


def test_prompt_version_reflects_configured_catalog_version() -> None:
    """ResolvedPrompt.prompt_version must reflect PROMPT_CATALOG_VERSION, not a hardcoded value."""
    provider = PromptProvider(settings=Settings(PROMPT_CATALOG_VERSION="v7"))

    resolved = provider.resolve(PromptType.CLARIFICATION, "?")

    assert resolved.prompt_version == "v7"


def test_detect_language_is_exposed_directly() -> None:
    """PromptProvider.detect_language() must be usable independently of resolve()."""
    provider = PromptProvider()

    assert provider.detect_language("שלום עולם") == SupportedLanguage.HE
    assert provider.detect_language("hello world") == SupportedLanguage.EN
