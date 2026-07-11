"""Internal decision layer: routes a user question before any retrieval or generation happens.

Deterministic, rule-based only — no LLM call is made to route. Decides whether a question needs
retrieval, can be answered directly, needs clarification, or is out of scope. Internal-only: not
wired to any public API endpoint, and doesn't perform ingestion, retrieval, or generation itself.

Every pattern below has both an English and a Hebrew form — RuleBasedRagDecider is the one shared
decision service both CustomRagEngine and LangChainRagEngine use; neither engine may keep its own
language-specific decision logic. Hebrew patterns match meaningful intent phrasing (a reference to
a document/file, an extraction verb next to a sensitive noun), never bare Hebrew-script detection
— a Hebrew sentence with no such phrasing is routed exactly like an equivalent English one.
"""

import re
from dataclasses import dataclass
from enum import StrEnum

_MIN_QUESTION_LENGTH = 4

# Verbs that indicate the user wants an actual secret *revealed*, as opposed to a benign action
# verb like "rotate"/"reset"/"open" performed on one's own credential.
_EXTRACTION_VERBS = (
    "show",
    "reveal",
    "extract",
    "list",
    "give",
    "tell",
    "disclose",
    "dump",
    "leak",
    "provide",
    "share",
    "display",
    "expose",
    "print",
    "send",
)

_SENSITIVE_NOUNS = (
    r"ssn",
    r"social security (?:number|no\.?)",
    r"credit card(?:s)?(?: numbers?)?",
    r"password(?:s)?",
    r"api key(?:s)?",
    r"private key(?:s)?",
    r"secret key(?:s)?",
    r"database credentials",
    r"credentials",
    r"bank account(?:s)?",
)

# Hebrew extraction verbs — imperative/request forms of show/reveal/give/send/extract/disclose.
_EXTRACTION_VERBS_HE = (
    "הראה",
    "תראה",
    "תראי",
    "הצג",
    "תציג",
    "חשוף",
    "תחשוף",
    "הוצא",
    "תוציא",
    "שלח",
    "תשלח",
    "תני",
    "תן",
    "גלה",
    "תגלה",
    "פרסם",
    "תפרסם",
)

# Hebrew sensitive nouns — password(s), credit card, ID/SSN-equivalent, credentials, secret/private
# key, bank account. Left permissive on definite-article ("ה") prefixes, common in natural Hebrew.
_SENSITIVE_NOUNS_HE = (
    r"סיסמ(?:ה|אות)",
    r"מספר\s+ה?כרטיס\s+ה?אשראי",
    r"פרטי\s+ה?כרטיס\s+ה?אשראי",
    r"מספר\s+ה?תעודת\s+זהות",
    r"פרטי\s+ה?התחברות",
    r"פרטי\s+ה?גישה",
    r"מפתח(?:ות)?\s+ה?סודי(?:ים)?",
    r"מפתח(?:ות)?\s+ה?פרטי(?:ים)?",
    r"חשבון\s+ה?בנק",
)

# OUT_OF_SCOPE requires an extraction verb followed by a sensitive noun within a few words —
# e.g. "show me the api keys", "extract database credentials" — not just the noun on its own,
# so "how do I rotate an API key?" or "how do I reset a password?" stay in scope. The Hebrew
# pattern mirrors this exactly: an extraction verb followed by a sensitive noun within a few
# words, so "איך מאפסים סיסמה?" ("how do I reset a password?") stays in scope.
_EXTRACTION_INTENT_PATTERN = re.compile(
    rf"\b(?:{'|'.join(_EXTRACTION_VERBS)})\b(?:\s+\S+){{0,4}}\s+\b(?:{'|'.join(_SENSITIVE_NOUNS)})\b"
)
_EXTRACTION_INTENT_PATTERN_HE = re.compile(
    rf"(?:{'|'.join(_EXTRACTION_VERBS_HE)})(?:\s+\S+){{0,4}}\s+ה?(?:{'|'.join(_SENSITIVE_NOUNS_HE)})"
)

_RETRIEVAL_PATTERNS = (
    r"\bdocument(s)?\b",
    r"\buploaded\b",
    r"\battached\b",
    r"\bindexed\b",
    r"\bpdf\b",
    r"\baccording to\b",
    r"\bbased on the\b",
    r"\bknowledge base\b",
)

# Hebrew retrieval-intent patterns — a reference to a document/file (uploaded/attached/indexed/
# "according to the file/document"/"written in the file/document"/"knowledge base"), not bare
# Hebrew-script detection: a Hebrew question with none of these phrasings (e.g. a general "how
# does the system store vectors?") is not routed to retrieval merely for being in Hebrew.
_RETRIEVAL_PATTERNS_HE = (
    r"מסמך|מסמכים",  # document (singular, final-kaf) / documents (plural)
    r"קובץ|קבצים",  # file (singular, final-tsadi) / files (plural)
    r"העל(?:יתי|ה|תה|ינו|ית)",  # uploaded (I/he/she/we/you uploaded)
    r"מצורף|מצורפ(?:ת|ים)",  # attached (masc. singular final-pe / fem./plural)
    r"מאונדקס(?:ים)?",  # indexed
    r"לפי\s+ה?(?:מסמך|קובץ)",  # according to the document/file
    r"בהתבסס\s+על",  # based on
    r"מאגר\s+ה?ידע",  # knowledge base
    r"כתוב\s+ב(?:קובץ|מסמך)",  # written in the file/document
    r"ב(?:מסמך|קובץ)",  # in the document/file
)


class RagDecision(StrEnum):
    """Which strategy should handle a user question."""

    NEEDS_RETRIEVAL = "needs_retrieval"
    DIRECT_LLM = "direct_llm"
    CLARIFICATION_NEEDED = "clarification_needed"
    OUT_OF_SCOPE = "out_of_scope"


@dataclass
class DecisionResult:
    """Outcome of routing a question: which decision, why, and how confident."""

    decision: RagDecision
    reason: str
    confidence: float | None = None


class RuleBasedRagDecider:
    """Deterministic, keyword/pattern-based decider — no LLM call is made to route."""

    def decide(self, question: str) -> DecisionResult:
        """Route a question to NEEDS_RETRIEVAL, DIRECT_LLM, CLARIFICATION_NEEDED, or OUT_OF_SCOPE."""
        stripped = question.strip()

        if not stripped:
            return DecisionResult(
                decision=RagDecision.CLARIFICATION_NEEDED,
                reason="Question is empty.",
                confidence=1.0,
            )

        if len(stripped) < _MIN_QUESTION_LENGTH:
            return DecisionResult(
                decision=RagDecision.CLARIFICATION_NEEDED,
                reason="Question is too short to act on.",
                confidence=0.9,
            )

        # Hebrew has no case distinction, so .lower() only affects the Latin-script portion of a
        # mixed-language question (e.g. "Qdrant"/"Kafka" embedded in an otherwise-Hebrew sentence).
        lowered = stripped.lower()

        if _EXTRACTION_INTENT_PATTERN.search(lowered) or _EXTRACTION_INTENT_PATTERN_HE.search(stripped):
            return DecisionResult(
                decision=RagDecision.OUT_OF_SCOPE,
                reason="Question appears to request sensitive or private data extraction.",
                confidence=0.9,
            )

        if any(re.search(pattern, lowered) for pattern in _RETRIEVAL_PATTERNS) or any(
            re.search(pattern, stripped) for pattern in _RETRIEVAL_PATTERNS_HE
        ):
            return DecisionResult(
                decision=RagDecision.NEEDS_RETRIEVAL,
                reason="Question references uploaded or indexed documents.",
                confidence=0.8,
            )

        return DecisionResult(
            decision=RagDecision.DIRECT_LLM,
            reason="General question with no document reference or sensitive-data request.",
            confidence=0.6,
        )
