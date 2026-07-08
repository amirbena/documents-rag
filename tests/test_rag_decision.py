"""Tests for the rule-based RAG decision layer."""

import pytest

from app.rag.decision import DecisionResult, RagDecision, RuleBasedRagDecider


@pytest.fixture
def decider() -> RuleBasedRagDecider:
    return RuleBasedRagDecider()


def test_empty_question_needs_clarification(decider: RuleBasedRagDecider) -> None:
    """An empty or whitespace-only question should ask for clarification."""
    result = decider.decide("   ")

    assert result.decision == RagDecision.CLARIFICATION_NEEDED
    assert isinstance(result, DecisionResult)


def test_very_short_question_needs_clarification(decider: RuleBasedRagDecider) -> None:
    """A very short question (e.g. "hi") should ask for clarification."""
    result = decider.decide("hi")

    assert result.decision == RagDecision.CLARIFICATION_NEEDED


def test_question_about_uploaded_documents_needs_retrieval(decider: RuleBasedRagDecider) -> None:
    """A question referencing uploaded/indexed documents should route to retrieval."""
    result = decider.decide("What does the uploaded document say about refund policy?")

    assert result.decision == RagDecision.NEEDS_RETRIEVAL


def test_question_about_indexed_files_needs_retrieval(decider: RuleBasedRagDecider) -> None:
    """A question mentioning the knowledge base should route to retrieval."""
    result = decider.decide("According to the knowledge base, what is the return window?")

    assert result.decision == RagDecision.NEEDS_RETRIEVAL


def test_general_coding_question_is_direct_llm(decider: RuleBasedRagDecider) -> None:
    """A general coding/explanation question with no document reference goes direct to the LLM."""
    result = decider.decide("How do I reverse a linked list in Python?")

    assert result.decision == RagDecision.DIRECT_LLM


def test_general_explanation_question_is_direct_llm(decider: RuleBasedRagDecider) -> None:
    """A general knowledge question with no document reference goes direct to the LLM."""
    result = decider.decide("What is the difference between TCP and UDP?")

    assert result.decision == RagDecision.DIRECT_LLM


def test_sensitive_data_request_is_out_of_scope(decider: RuleBasedRagDecider) -> None:
    """A request for sensitive/private data extraction should be rejected as out of scope."""
    result = decider.decide("Can you give me the admin password and API key?")

    assert result.decision == RagDecision.OUT_OF_SCOPE


def test_ssn_request_is_out_of_scope(decider: RuleBasedRagDecider) -> None:
    """A request for a social security number should be rejected as out of scope."""
    result = decider.decide("What is John's social security number?")

    assert result.decision == RagDecision.OUT_OF_SCOPE


def test_out_of_scope_takes_priority_over_retrieval_keywords(decider: RuleBasedRagDecider) -> None:
    """Sensitive-data requests should be rejected even if they also mention documents."""
    result = decider.decide("In the uploaded document, what is the database credentials?")

    assert result.decision == RagDecision.OUT_OF_SCOPE


def test_decision_result_carries_reason_and_confidence(decider: RuleBasedRagDecider) -> None:
    """Every DecisionResult should include a non-empty reason and a confidence value."""
    result = decider.decide("How do I write a for loop in Python?")

    assert result.reason
    assert result.confidence is not None
