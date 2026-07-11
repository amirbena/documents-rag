"""Language-aware catalog of fixed/governed RAG prompt content.

Supersedes the previous app.rag.responses fixed constants (removed): both RagOrchestrator and
LangChainRagEngine now resolve this text through PromptProvider (app/rag/prompts/provider.py),
never by importing from one another's implementation module.

Generative prompt types (GROUNDED_ANSWER/DIRECT_ANSWER) use a single English-authored governance
instruction — never duplicated per language — plus a short, explicit response-language directive
("Respond directly and naturally in Hebrew (he)."/"...in English (en)."). The model is never told
to think or answer in English and translate; it is told, in English (its most reliable instruction
language), to respond directly in whichever language the question was asked in. Fixed/no-LLM
prompt types (CLARIFICATION/NO_RESULTS/OUT_OF_SCOPE) remain naturally authored per language, since
they bypass the LLM entirely and there is no "instruction vs. output language" distinction to make.

Phase 2.5(+) boundary: this catalog is deliberately a flat, hardcoded dict, not a database-backed
or file-loaded system, and does not perform any language-specific formatting beyond the fixed
strings below. A future milestone may introduce persisted, runtime-editable prompt records if the
platform needs more languages or non-developer prompt edits — do not add more languages,
persistence, or a loader mechanism here ahead of that need.
"""

from app.rag.language import SupportedLanguage
from app.rag.prompts.types import PromptType


class UnsupportedPromptLanguageError(ValueError):
    """Raised when the catalog has no content at all for the requested SupportedLanguage."""


class UnsupportedPromptTypeError(ValueError):
    """Raised when the catalog has no content for the requested PromptType in a given language."""


# Shared, English-only governance instructions for prompt types where the LLM still generates the
# answer — never duplicated per language. The response-language directive (below) is appended
# separately, so this text never mentions a specific target language itself.
_SHARED_SYSTEM_INSTRUCTIONS: dict[PromptType, str] = {
    PromptType.GROUNDED_ANSWER: (
        "You are a helpful assistant that answers questions using only the supplied context. "
        "Do not invent or assume information that is not present in the context. If the answer "
        "is not present in the context, state clearly that the context does not contain enough "
        "information, instead of guessing. Preserve quoted source text in its original language "
        "— never translate a quotation. Keep source titles and attribution labels (such as "
        "[S1], [S2]) intact and untranslated. Never translate code, API names, class names, "
        "filenames, commands, environment variables, or error messages — keep every technical "
        "identifier exactly as written, regardless of the response language."
    ),
    PromptType.DIRECT_ANSWER: (
        "You are a helpful assistant. Answer the user's question directly. Never translate "
        "code, API names, class names, filenames, commands, environment variables, or error "
        "messages — keep every technical identifier exactly as written, regardless of the "
        "response language."
    ),
}

# Explicit output-language directive — never "answer in English and translate", never a claim
# that the model "thinks in English". Appended, in English, after the shared instruction above.
_RESPONSE_LANGUAGE_DIRECTIVES: dict[SupportedLanguage, str] = {
    SupportedLanguage.EN: "Respond directly and naturally in English (en).",
    SupportedLanguage.HE: "Respond directly and naturally in Hebrew (he).",
}

# Complete, fixed, deterministic answers for prompt types that never call the LLM — naturally
# authored per language, since there is no LLM instruction/output-language distinction to make.
_FIXED_RESPONSES: dict[SupportedLanguage, dict[PromptType, str]] = {
    SupportedLanguage.EN: {
        PromptType.CLARIFICATION: (
            "Could you rephrase or add more detail? Your question is too short or unclear to act on."
        ),
        PromptType.NO_RESULTS: (
            "I couldn't find any relevant information in the indexed documents to answer this question."
        ),
        PromptType.OUT_OF_SCOPE: "I can't help with that request.",
    },
    SupportedLanguage.HE: {
        PromptType.CLARIFICATION: (
            "תוכל/י לנסח מחדש או להוסיף פרטים נוספים? השאלה קצרה מדי או לא ברורה מספיק כדי לפעול לפיה."
        ),
        PromptType.NO_RESULTS: "לא מצאתי מידע רלוונטי במסמכים המאונדקסים כדי לענות על שאלה זו.",
        PromptType.OUT_OF_SCOPE: "אינני יכול/ה לסייע בבקשה זו.",
    },
}

_GENERATIVE_PROMPT_TYPES = frozenset(_SHARED_SYSTEM_INSTRUCTIONS)


class PromptCatalog:
    """{grounded_answer, direct_answer} x {he, en} system prompts, plus he/en fixed responses.

    Generative system prompts are composed from one English shared instruction plus a per-language
    response-language directive — never a duplicated per-language system prompt.
    """

    def get_shared_instructions(self, prompt_type: PromptType) -> str:
        """Return the English-only governance instruction for a generation-backed prompt type."""
        if prompt_type not in _SHARED_SYSTEM_INSTRUCTIONS:
            raise UnsupportedPromptTypeError(f"No shared system instructions for {prompt_type!r}")
        return _SHARED_SYSTEM_INSTRUCTIONS[prompt_type]

    def get_response_language_directive(self, language: SupportedLanguage) -> str:
        """Return the explicit output-language directive for `language`."""
        if language not in _RESPONSE_LANGUAGE_DIRECTIVES:
            raise UnsupportedPromptLanguageError(f"No response-language directive for {language!r}")
        return _RESPONSE_LANGUAGE_DIRECTIVES[language]

    def get_system_text(self, prompt_type: PromptType, language: SupportedLanguage) -> str:
        """Return the composed system prompt: shared English instructions + language directive."""
        shared = self.get_shared_instructions(prompt_type)
        directive = self.get_response_language_directive(language)
        return f"{shared}\n\n{directive}"

    def get_response_text(self, prompt_type: PromptType, language: SupportedLanguage) -> str:
        """Return the complete fixed answer for a no-LLM-call prompt type."""
        if language not in _FIXED_RESPONSES:
            raise UnsupportedPromptLanguageError(f"No prompt content for language {language!r}")
        try:
            return _FIXED_RESPONSES[language][prompt_type]
        except KeyError as exc:
            raise UnsupportedPromptTypeError(
                f"No {prompt_type!r} prompt content for language {language!r}"
            ) from exc
