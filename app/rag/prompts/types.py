"""Framework-neutral prompt contracts shared by every RagEngine implementation.

Neither CustomRagEngine nor LangChainRagEngine may own a private prompt catalog — both consume
ResolvedPrompt objects produced by PromptProvider (see app/rag/prompts/provider.py).
"""

from dataclasses import dataclass
from enum import StrEnum

from app.rag.language import SupportedLanguage

__all__ = ["PromptType", "ResolvedPrompt", "SupportedLanguage"]


class PromptType(StrEnum):
    """Which fixed/governed prompt content to resolve for one RagEngine turn."""

    GROUNDED_ANSWER = "grounded_answer"
    DIRECT_ANSWER = "direct_answer"
    CLARIFICATION = "clarification"
    NO_RESULTS = "no_results"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass(frozen=True)
class ResolvedPrompt:
    """The fully-resolved, language-specific prompt content for one PromptType.

    `system_text` is the complete model-facing system prompt, set for GROUNDED_ANSWER/
    DIRECT_ANSWER — the engine still calls the LLM, but with this instruction instead of a
    hardcoded English one. It is always `shared_instructions + "\\n\\n" + language_directive`,
    which are also exposed individually: `shared_instructions` is the English-only governance
    text (never duplicated per language), and `language_directive` is the explicit, per-language
    "respond in <language>" instruction — never "answer in English and translate". `response_text`
    is the complete, fixed, deterministic answer text, set for CLARIFICATION/NO_RESULTS/
    OUT_OF_SCOPE — no LLM call is made at all for these, and `shared_instructions`/
    `language_directive` are unset. Exactly one of `system_text`/`response_text` is ever set.
    """

    prompt_type: PromptType
    language: SupportedLanguage
    prompt_version: str
    system_text: str | None = None
    response_text: str | None = None
    shared_instructions: str | None = None
    language_directive: str | None = None

    def __post_init__(self) -> None:
        has_system = self.system_text is not None
        has_response = self.response_text is not None
        if has_system == has_response:
            raise ValueError("ResolvedPrompt must set exactly one of system_text/response_text")
        if has_system != (self.shared_instructions is not None):
            raise ValueError("shared_instructions must be set if and only if system_text is set")
        if has_system != (self.language_directive is not None):
            raise ValueError("language_directive must be set if and only if system_text is set")
