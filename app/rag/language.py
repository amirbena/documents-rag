"""Shared, deterministic language detection for RAG prompt resolution.

Neither RagEngine implementation may detect language on its own — both CustomRagEngine and
LangChainRagEngine reach a LanguageDetector only through PromptProvider (see
app/rag/prompts/provider.py), so the same question always resolves to the same language
regardless of which engine is active. No ML model, no external service, no network call — pure
word-level Unicode-script classification, deterministic and instant.
"""

import re
from abc import ABC, abstractmethod
from enum import StrEnum

from app.core.config import Settings, get_settings

_WORD_PATTERN = re.compile(r"\w+", re.UNICODE)
_HEBREW_CHAR_PATTERN = re.compile(r"[֐-׿]")
_LATIN_CHAR_PATTERN = re.compile(r"[A-Za-z]")


class SupportedLanguage(StrEnum):
    """A response language the platform's prompt catalog has content for."""

    HE = "he"
    EN = "en"


class LanguageDetector(ABC):
    """Contract for resolving the intended response language from a user's question."""

    @abstractmethod
    def detect(self, text: str) -> SupportedLanguage:
        """Return the detected SupportedLanguage for `text`."""
        raise NotImplementedError


def _classify_word(word: str) -> str | None:
    """Return 'he', 'en', or None (e.g. pure digits/punctuation) for one whitespace-split word."""
    if _HEBREW_CHAR_PATTERN.search(word):
        return "he"
    if _LATIN_CHAR_PATTERN.search(word):
        return "en"
    return None


class ScriptBasedLanguageDetector(LanguageDetector):
    """Deterministic script-dominance detector for Hebrew/English.

    Classifies the question word by word (not character by character) so that a handful of
    Latin-script technical identifiers (Kafka, Qdrant, Kubernetes, LangChain, ...) embedded in an
    otherwise-Hebrew sentence never outweigh the surrounding natural-language Hebrew words, and
    vice versa. Resolution rules:

    - No Hebrew and no Latin words at all (empty/punctuation/numbers-only query) -> default.
    - Hebrew word count > Latin word count -> Hebrew.
    - Latin word count > Hebrew word count -> English.
    - An exact tie (both present, equal counts) -> default — a genuinely ambiguous mixed query
      falls back to the configured default rather than guessing.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def detect(self, text: str) -> SupportedLanguage:
        """Return the detected SupportedLanguage for `text`, or the configured default."""
        hebrew_count = 0
        latin_count = 0
        for word in _WORD_PATTERN.findall(text):
            classification = _classify_word(word)
            if classification == "he":
                hebrew_count += 1
            elif classification == "en":
                latin_count += 1

        if hebrew_count == 0 and latin_count == 0:
            return self._default_language()
        if hebrew_count > latin_count:
            return SupportedLanguage.HE
        if latin_count > hebrew_count:
            return SupportedLanguage.EN
        return self._default_language()

    def _default_language(self) -> SupportedLanguage:
        return SupportedLanguage(self._settings.default_response_language)
