"""Tests for RagPromptBuilder — deterministic prompt shaping, no LLM/network involved."""

import copy

from app.rag.prompt_builder import RagPromptBuilder
from app.rag.providers.vector_store import VectorSearchResult


def _result(chunk_id: str, score: float, **overrides: object) -> VectorSearchResult:
    fields: dict[str, object] = {
        "id": chunk_id,
        "score": score,
        "document_id": "doc-1",
        "chunk_id": chunk_id,
        "text": "some chunk text",
        "source": "handbook.pdf",
        "page_number": None,
        "sheet_name": None,
    }
    fields.update(overrides)
    return VectorSearchResult(**fields)  # type: ignore[arg-type]


def test_single_source_prompt() -> None:
    """A single result should produce one labeled source block and one PromptSource."""
    result = _result("chunk-1", 0.9, text="refunds are processed within 14 days")

    built = RagPromptBuilder().build("what is the refund policy?", [result])

    assert "[S1]" in built.context
    assert "refunds are processed within 14 days" in built.context
    assert "what is the refund policy?" in built.user_prompt
    assert len(built.sources) == 1
    assert built.sources[0].chunk_id == "chunk-1"
    assert built.sources[0].score == 0.9


def test_multiple_ranked_sources() -> None:
    """Multiple results should get sequential labels in the given (already-ranked) order."""
    results = [
        _result("chunk-1", 0.95, text="first fact"),
        _result("chunk-2", 0.80, text="second fact"),
        _result("chunk-3", 0.60, text="third fact"),
    ]

    built = RagPromptBuilder().build("question", results)

    assert built.context.index("[S1]") < built.context.index("[S2]") < built.context.index("[S3]")
    assert built.context.index("first fact") < built.context.index("second fact")
    assert built.context.index("second fact") < built.context.index("third fact")
    assert [source.chunk_id for source in built.sources] == ["chunk-1", "chunk-2", "chunk-3"]


def test_page_metadata_formatting() -> None:
    """A result with page_number should include 'page N' in its context block."""
    result = _result("chunk-1", 0.9, source="handbook.pdf", page_number=3, text="page content")

    built = RagPromptBuilder().build("question", [result])

    assert "page 3" in built.context
    assert built.sources[0].page_number == 3


def test_sheet_metadata_formatting() -> None:
    """A result with sheet_name should include 'sheet <name>' in its context block."""
    result = _result("chunk-1", 0.9, source="report.xlsx", sheet_name="Sheet1", text="row data")

    built = RagPromptBuilder().build("question", [result])

    assert "sheet Sheet1" in built.context
    assert built.sources[0].sheet_name == "Sheet1"


def test_source_labels_deterministic() -> None:
    """Building the same input twice should produce identical labels/context/sources."""
    results = [_result("chunk-1", 0.9, text="a"), _result("chunk-2", 0.5, text="b")]

    first = RagPromptBuilder().build("question", results)
    second = RagPromptBuilder().build("question", results)

    assert first.context == second.context
    assert first.user_prompt == second.user_prompt
    assert first.sources == second.sources


def test_empty_chunk_text_ignored() -> None:
    """Results with empty/whitespace-only text should be dropped, not shown as a source."""
    results = [
        _result("chunk-1", 0.9, text=""),
        _result("chunk-2", 0.8, text="   "),
        _result("chunk-3", 0.7, text="real content"),
    ]

    built = RagPromptBuilder().build("question", results)

    assert len(built.sources) == 1
    assert built.sources[0].chunk_id == "chunk-3"
    assert "[S1]" in built.context
    assert "[S2]" not in built.context
    assert "real content" in built.context


def test_no_results_behavior() -> None:
    """No results (or all-empty-text results) should produce a clear no-context prompt."""
    built = RagPromptBuilder().build("what is the refund policy?", [])

    assert built.sources == []
    assert "no relevant context" in built.context.lower()
    assert "what is the refund policy?" in built.user_prompt
    assert "no relevant context" in built.user_prompt.lower()


def test_no_results_behavior_is_deterministic_and_no_fabrication() -> None:
    """Repeated no-results builds should be identical, and never invent fallback context."""
    first = RagPromptBuilder().build("question", [])
    second = RagPromptBuilder().build("question", [])

    assert first.context == second.context
    assert first.user_prompt == second.user_prompt
    assert "handbook" not in first.context.lower()


def test_hebrew_question_and_context_preserved() -> None:
    """Hebrew text in the question and chunk content should pass through unmodified."""
    result = _result(
        "chunk-1", 0.9, source="מדריך.pdf", text="ההחזר הכספי מתבצע תוך 14 יום"
    )

    built = RagPromptBuilder().build("מה מדיניות ההחזרים?", [result])

    assert "מה מדיניות ההחזרים?" in built.user_prompt
    assert "ההחזר הכספי מתבצע תוך 14 יום" in built.context
    assert "מדריך.pdf" in built.context


def test_no_llm_provider_is_called(monkeypatch) -> None:
    """build() must never invoke an LLM provider — prompt building only shapes text."""

    def _fail_if_called(settings=None):
        raise AssertionError("get_llm_provider must never be called while building a prompt")

    monkeypatch.setattr("app.rag.providers.provider_factory.get_llm_provider", _fail_if_called)

    RagPromptBuilder().build("question", [_result("chunk-1", 0.9, text="content")])


def test_retrieval_results_are_not_mutated() -> None:
    """build() must not mutate the VectorSearchResult objects or list passed in."""
    results = [_result("chunk-1", 0.9, text="content"), _result("chunk-2", 0.5, text="more")]
    snapshot = copy.deepcopy(results)

    RagPromptBuilder().build("question", results)

    assert results == snapshot
