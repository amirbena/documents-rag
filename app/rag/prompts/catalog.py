"""Language-aware catalog of fixed/governed RAG prompt content — he/en, five PromptTypes.

Supersedes the previous app.rag.responses fixed constants (removed): both RagOrchestrator and
LangChainRagEngine now resolve this text through PromptProvider (app/rag/prompts/provider.py),
never by importing from one another's implementation module.

Phase 2.5 boundary: this catalog is deliberately a flat, hardcoded he/en dict, not a
database-backed or file-loaded system, and does not perform any language-specific formatting
beyond the fixed strings below. A future Phase 2.5+ milestone may introduce persisted,
runtime-editable prompt records if the platform needs more languages or non-developer prompt
edits — do not add more languages, persistence, or a loader mechanism here ahead of that need.
"""

from app.rag.language import SupportedLanguage
from app.rag.prompts.types import PromptType


class UnsupportedPromptLanguageError(ValueError):
    """Raised when the catalog has no content at all for the requested SupportedLanguage."""


class UnsupportedPromptTypeError(ValueError):
    """Raised when the catalog has no content for the requested PromptType in a given language."""


# Governance system instructions for prompt types where the LLM still generates the answer.
_SYSTEM_TEXTS: dict[SupportedLanguage, dict[PromptType, str]] = {
    SupportedLanguage.EN: {
        PromptType.GROUNDED_ANSWER: (
            "You are a helpful assistant that answers questions using only the supplied "
            "context. Do not invent or assume information that is not present in the context. "
            "If the answer is not present in the context, say so explicitly instead of "
            "guessing. Answer in the same language as the question. Preserve quoted source "
            "text in its original language — never translate a quotation. Keep source "
            "attribution (labels such as [S1], [S2]) intact. Never translate code, API names, "
            "class names, environment variables, or command names — keep every technical "
            "identifier exactly as written."
        ),
        PromptType.DIRECT_ANSWER: (
            "You are a helpful assistant. Answer the user's question directly, in the same "
            "language as the question. Never translate code, API names, class names, "
            "environment variables, or command names — keep every technical identifier "
            "exactly as written."
        ),
    },
    SupportedLanguage.HE: {
        PromptType.GROUNDED_ANSWER: (
            "אתה עוזר מועיל שעונה על שאלות בהתבסס אך ורק על ההקשר שסופק. אל תמציא ואל תניח "
            "מידע שאינו מופיע בהקשר. אם התשובה אינה מופיעה בהקשר, ציין זאת במפורש במקום לנחש. "
            "ענה באותה שפה שבה נשאלה השאלה. שמור על ציטוטים ממקורות בשפתם המקורית — לעולם אל "
            "תתרגם ציטוט. שמור על ייחוס המקורות (תוויות כגון [S1], [S2]) ללא שינוי. לעולם אל "
            "תתרגם קוד, שמות API, שמות מחלקות, משתני סביבה או שמות פקודות — השאר כל מזהה טכני "
            "בדיוק כפי שהוא."
        ),
        PromptType.DIRECT_ANSWER: (
            "אתה עוזר מועיל. ענה על שאלת המשתמש ישירות, באותה שפה שבה נשאלה השאלה. לעולם אל "
            "תתרגם קוד, שמות API, שמות מחלקות, משתני סביבה או שמות פקודות — השאר כל מזהה טכני "
            "בדיוק כפי שהוא."
        ),
    },
}

# Complete, fixed, deterministic answers for prompt types that never call the LLM.
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


class PromptCatalog:
    """Nested he/en x {grounded_answer, direct_answer, clarification, no_results, out_of_scope}."""

    def get_system_text(self, prompt_type: PromptType, language: SupportedLanguage) -> str:
        """Return the governance system instruction for a generation-backed prompt type."""
        return self._lookup(_SYSTEM_TEXTS, prompt_type, language)

    def get_response_text(self, prompt_type: PromptType, language: SupportedLanguage) -> str:
        """Return the complete fixed answer for a no-LLM-call prompt type."""
        return self._lookup(_FIXED_RESPONSES, prompt_type, language)

    @staticmethod
    def _lookup(
        table: dict[SupportedLanguage, dict[PromptType, str]],
        prompt_type: PromptType,
        language: SupportedLanguage,
    ) -> str:
        if language not in table:
            raise UnsupportedPromptLanguageError(f"No prompt content for language {language!r}")
        try:
            return table[language][prompt_type]
        except KeyError as exc:
            raise UnsupportedPromptTypeError(
                f"No {prompt_type!r} prompt content for language {language!r}"
            ) from exc
