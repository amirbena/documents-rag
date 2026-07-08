"""Internal decision layer: routes a user question before any retrieval or generation happens.

Deterministic, rule-based only — no LLM call is made to route. Decides whether a question needs
retrieval, can be answered directly, needs clarification, or is out of scope. Internal-only: not
wired to any public API endpoint, and doesn't perform ingestion, retrieval, or generation itself.
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

# OUT_OF_SCOPE requires an extraction verb followed by a sensitive noun within a few words —
# e.g. "show me the api keys", "extract database credentials" — not just the noun on its own,
# so "how do I rotate an API key?" or "how do I reset a password?" stay in scope.
_EXTRACTION_INTENT_PATTERN = re.compile(
    rf"\b(?:{'|'.join(_EXTRACTION_VERBS)})\b(?:\s+\S+){{0,4}}\s+\b(?:{'|'.join(_SENSITIVE_NOUNS)})\b"
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

        lowered = stripped.lower()

        if _EXTRACTION_INTENT_PATTERN.search(lowered):
            return DecisionResult(
                decision=RagDecision.OUT_OF_SCOPE,
                reason="Question appears to request sensitive or private data extraction.",
                confidence=0.9,
            )

        if any(re.search(pattern, lowered) for pattern in _RETRIEVAL_PATTERNS):
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
