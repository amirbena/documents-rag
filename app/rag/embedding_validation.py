"""Shared validation for embedding-provider output against the active EmbeddingIndexConfig.

Configuration alone (EMBEDDING_MODEL/VECTOR_SIZE) cannot catch a misconfigured pair — an operator
setting VECTOR_SIZE to the wrong value for the configured model would otherwise only be caught
later, if at all, by Qdrant's own dimension check on an *existing* collection. This module
validates the embedding provider's actual output before any point is written or any document is
marked indexed, and before a query vector is used to search — the same two checks apply to both
ingestion/re-index (a batch of chunk vectors) and retrieval (a single query vector), and both
RagEngine implementations reach them only via IngestionWorker/ReindexService/RetrievalService, not
by validating independently.
"""


class EmbeddingResultCountMismatchError(ValueError):
    """Raised when an embedding provider returns a different number of vectors than inputs."""


class EmbeddingDimensionMismatchError(ValueError):
    """Raised when an embedding vector's length doesn't match the active configuration's dimension."""


def validate_embeddings(
    vectors: list[list[float]], expected_count: int, expected_dimension: int
) -> None:
    """Validate a batch of embedding vectors before they are written or searched with.

    Raises EmbeddingResultCountMismatchError if the provider returned the wrong number of
    vectors, or EmbeddingDimensionMismatchError if any vector's length doesn't match
    `expected_dimension` (including an empty/malformed vector). Raises before the caller performs
    any Qdrant write or search, and before any document is marked indexed.
    """
    if len(vectors) != expected_count:
        raise EmbeddingResultCountMismatchError(
            f"Embedding provider returned {len(vectors)} vector(s) for {expected_count} input(s)."
        )
    for index, vector in enumerate(vectors):
        if not vector or len(vector) != expected_dimension:
            raise EmbeddingDimensionMismatchError(
                f"Embedding vector at index {index} has dimension "
                f"{len(vector) if vector else 0}, expected {expected_dimension}."
            )
