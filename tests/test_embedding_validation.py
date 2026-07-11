"""Tests for app.rag.embedding_validation.validate_embeddings."""

import pytest

from app.rag.embedding_validation import (
    EmbeddingDimensionMismatchError,
    EmbeddingResultCountMismatchError,
    validate_embeddings,
)


def test_validate_embeddings_passes_for_matching_count_and_dimension() -> None:
    """Correct vectors (count and dimension both matching) raise nothing."""
    validate_embeddings([[0.1, 0.2], [0.3, 0.4]], expected_count=2, expected_dimension=2)


def test_validate_embeddings_rejects_wrong_count() -> None:
    """Fewer/more vectors than inputs is rejected before any write/search happens."""
    with pytest.raises(EmbeddingResultCountMismatchError):
        validate_embeddings([[0.1, 0.2]], expected_count=2, expected_dimension=2)


def test_validate_embeddings_rejects_one_malformed_vector_in_a_batch() -> None:
    """A single wrong-dimension vector rejects the entire batch, not just that item."""
    with pytest.raises(EmbeddingDimensionMismatchError):
        validate_embeddings([[0.1, 0.2], [0.3]], expected_count=2, expected_dimension=2)


def test_validate_embeddings_rejects_empty_vector() -> None:
    """An empty vector (malformed provider output) is rejected, not treated as dimension 0 == 0."""
    with pytest.raises(EmbeddingDimensionMismatchError):
        validate_embeddings([[]], expected_count=1, expected_dimension=2)


def test_validate_embeddings_rejects_query_vector_dimension_mismatch() -> None:
    """A single query vector (retrieval path, expected_count=1) is validated the same way."""
    with pytest.raises(EmbeddingDimensionMismatchError):
        validate_embeddings([[0.1, 0.2, 0.3]], expected_count=1, expected_dimension=2)
