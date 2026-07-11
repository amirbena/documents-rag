"""PromptProvider: the single seam both RagEngine implementations use to detect a question's
language and resolve the ResolvedPrompt for it.

CustomRagEngine (via RagOrchestrator) and LangChainRagEngine both depend on this module only —
neither performs its own language detection, and neither owns a private prompt catalog.
"""

from app.core.config import Settings, get_settings
from app.rag.language import LanguageDetector, ScriptBasedLanguageDetector, SupportedLanguage
from app.rag.prompts.catalog import PromptCatalog
from app.rag.prompts.types import PromptType, ResolvedPrompt

_FIXED_RESPONSE_TYPES = frozenset(
    {PromptType.CLARIFICATION, PromptType.NO_RESULTS, PromptType.OUT_OF_SCOPE}
)


class PromptProvider:
    """Detects a question's language, then resolves the catalog content for a given PromptType."""

    def __init__(
        self,
        settings: Settings | None = None,
        language_detector: LanguageDetector | None = None,
        catalog: PromptCatalog | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._language_detector = language_detector or ScriptBasedLanguageDetector(self._settings)
        self._catalog = catalog or PromptCatalog()

    def detect_language(self, question: str) -> SupportedLanguage:
        """Return the detected SupportedLanguage for `question`."""
        return self._language_detector.detect(question)

    def resolve(self, prompt_type: PromptType, question: str) -> ResolvedPrompt:
        """Detect `question`'s language and return the ResolvedPrompt for `prompt_type` in it."""
        language = self.detect_language(question)
        version = self._settings.prompt_catalog_version

        if prompt_type in _FIXED_RESPONSE_TYPES:
            return ResolvedPrompt(
                prompt_type=prompt_type,
                language=language,
                prompt_version=version,
                response_text=self._catalog.get_response_text(prompt_type, language),
            )

        shared_instructions = self._catalog.get_shared_instructions(prompt_type)
        language_directive = self._catalog.get_response_language_directive(language)
        return ResolvedPrompt(
            prompt_type=prompt_type,
            language=language,
            prompt_version=version,
            system_text=f"{shared_instructions}\n\n{language_directive}",
            shared_instructions=shared_instructions,
            language_directive=language_directive,
        )
