"""Internal decision layer: routes a user question before any retrieval or generation happens.

Deterministic, rule-based only — no LLM call is made to route. Decides whether a question needs
retrieval, can be answered directly, needs clarification, or is out of scope. Internal-only: not
wired to any public API endpoint, and doesn't perform ingestion, retrieval, or generation itself.
"""

import re
from dataclasses import dataclass
from enum import StrEnum

_MIN_QUESTION_LENGTH = 4

_OUT_OF_SCOPE_PATTERNS = (
    r"\bssn\b",
    r"\bsocial security (number|no\.?)\b",
    r"\bcredit card\b",
    r"\bpassword(s)?\b",
    r"\bapi key(s)?\b",
    r"\bprivate key(s)?\b",
    r"\bdatabase credentials\b",
    r"\bsecret key(s)?\b",
    r"\bbank account\b",
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

        if any(re.search(pattern, lowered) for pattern in _OUT_OF_SCOPE_PATTERNS):
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
