"""Tests for RetrievalService against fake embedding/vector-store providers — no real network."""

import app.rag.retrieval_service as retrieval_service_module
from app.core.config import Settings, get_settings
from app.rag.embedding_config import get_active_embedding_config
from app.rag.providers.vector_store import VectorSearchResult
from app.rag.retrieval_service import EmptyQueryError, RetrievalService


class _FakeEmbeddingProvider:
    """Records the texts it's asked to embed and returns one fixed-length vector per text."""

    def __init__(self, vector: list[float] | None = None) -> None:
        self.vector = vector or [0.1, 0.2, 0.3]
        self.embed_calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(texts)
        return [self.vector for _ in texts]


class _FailingEmbeddingProvider:
    """Always raises, simulating an embedding provider failure."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding provider unavailable")


def _result(chunk_id: str, score: float, **overrides: object) -> VectorSearchResult:
    fields: dict[str, object] = {
        "id": chunk_id,
        "score": score,
        "document_id": "doc-1",
        "chunk_id": chunk_id,
        "text": "some chunk text",
        "source": "handbook.pdf",
        "page_number": 2,
        "sheet_name": None,
    }
    fields.update(overrides)
    return VectorSearchResult(**fields)  # type: ignore[arg-type]


class _FakeVectorStore:
    """Records search calls and returns a fixed list of results instead of calling real Qdrant."""

    def __init__(self, results: list[VectorSearchResult] | None = None) -> None:
        self.results = results if results is not None else [_result("chunk-1", 0.9)]
        self.search_calls: list[tuple[str, list[float], int]] = []

    async def search_similar(
        self, collection_name: str, query_vector: list[float], limit: int = 5
    ) -> list[VectorSearchResult]:
        self.search_calls.append((collection_name, query_vector, limit))
        return self.results


class _FailingVectorStore:
    """Always raises, simulating a vector-store failure."""

    async def search_similar(
        self, collection_name: str, query_vector: list[float], limit: int = 5
    ) -> list[VectorSearchResult]:
        raise RuntimeError("vector store unavailable")


def _patch_providers(monkeypatch, embedding_provider, vector_store) -> None:
    monkeypatch.setattr(
        retrieval_service_module, "get_embedding_provider", lambda settings: embedding_provider
    )
    monkeypatch.setattr(retrieval_service_module, "get_vector_store", lambda settings: vector_store)


async def test_query_is_embedded_exactly_once(monkeypatch) -> None:
    """retrieve() should call the embedding provider exactly once, with the query text."""
    embedding_provider = _FakeEmbeddingProvider()
    vector_store = _FakeVectorStore()
    _patch_providers(monkeypatch, embedding_provider, vector_store)

    await RetrievalService().retrieve("what is the refund policy?")

    assert embedding_provider.embed_calls == [["what is the refund policy?"]]


async def test_configured_collection_is_searched(monkeypatch) -> None:
    """retrieve() should search the active versioned collection (see app.rag.embedding_config)."""
    vector_store = _FakeVectorStore()
    _patch_providers(monkeypatch, _FakeEmbeddingProvider(), vector_store)

    await RetrievalService().retrieve("policy question")

    assert vector_store.search_calls[0][0] == get_active_embedding_config(get_settings()).collection_name


async def test_default_top_k_is_used(monkeypatch) -> None:
    """retrieve() without an explicit limit should search with RETRIEVAL_TOP_K."""
    vector_store = _FakeVectorStore()
    _patch_providers(monkeypatch, _FakeEmbeddingProvider(), vector_store)
    settings = Settings(RETRIEVAL_TOP_K=7)

    await RetrievalService(settings=settings).retrieve("policy question")

    assert vector_store.search_calls[0][2] == 7


async def test_explicit_limit_overrides_default(monkeypatch) -> None:
    """A limit passed to retrieve() should override RETRIEVAL_TOP_K."""
    vector_store = _FakeVectorStore()
    _patch_providers(monkeypatch, _FakeEmbeddingProvider(), vector_store)
    settings = Settings(RETRIEVAL_TOP_K=7)

    await RetrievalService(settings=settings).retrieve("policy question", limit=2)

    assert vector_store.search_calls[0][2] == 2


async def test_results_remain_ranked(monkeypatch) -> None:
    """retrieve() should preserve the order returned by the vector store (already ranked by score)."""
    results = [_result("chunk-1", 0.95), _result("chunk-2", 0.8), _result("chunk-3", 0.6)]
    vector_store = _FakeVectorStore(results=results)
    _patch_providers(monkeypatch, _FakeEmbeddingProvider(), vector_store)

    returned = await RetrievalService().retrieve("policy question")

    assert [result.chunk_id for result in returned] == ["chunk-1", "chunk-2", "chunk-3"]


async def test_metadata_is_preserved(monkeypatch) -> None:
    """retrieve() should preserve document_id, chunk_id, text, source, page_number, sheet_name, score."""
    result = _result(
        "chunk-9",
        0.77,
        document_id="doc-9",
        text="row content",
        source="report.xlsx",
        page_number=None,
        sheet_name="Sheet1",
    )
    vector_store = _FakeVectorStore(results=[result])
    _patch_providers(monkeypatch, _FakeEmbeddingProvider(), vector_store)

    returned = await RetrievalService().retrieve("policy question")

    assert len(returned) == 1
    got = returned[0]
    assert got.document_id == "doc-9"
    assert got.chunk_id == "chunk-9"
    assert got.text == "row content"
    assert got.source == "report.xlsx"
    assert got.page_number is None
    assert got.sheet_name == "Sheet1"
    assert got.score == 0.77


async def test_threshold_filters_low_score_results(monkeypatch) -> None:
    """RETRIEVAL_SCORE_THRESHOLD should filter out results scoring below it."""
    results = [_result("chunk-1", 0.9), _result("chunk-2", 0.3)]
    vector_store = _FakeVectorStore(results=results)
    _patch_providers(monkeypatch, _FakeEmbeddingProvider(), vector_store)
    settings = Settings(RETRIEVAL_SCORE_THRESHOLD=0.5)

    returned = await RetrievalService(settings=settings).retrieve("policy question")

    assert [result.chunk_id for result in returned] == ["chunk-1"]


async def test_threshold_disabled_by_default_keeps_all_results(monkeypatch) -> None:
    """With RETRIEVAL_SCORE_THRESHOLD unset, no results should be filtered by score."""
    results = [_result("chunk-1", 0.9), _result("chunk-2", 0.01)]
    vector_store = _FakeVectorStore(results=results)
    _patch_providers(monkeypatch, _FakeEmbeddingProvider(), vector_store)

    returned = await RetrievalService().retrieve("policy question")

    assert [result.chunk_id for result in returned] == ["chunk-1", "chunk-2"]


async def test_no_relevant_results_returns_empty_list(monkeypatch) -> None:
    """When the vector store returns nothing, retrieve() should return an empty list."""
    vector_store = _FakeVectorStore(results=[])
    _patch_providers(monkeypatch, _FakeEmbeddingProvider(), vector_store)

    returned = await RetrievalService().retrieve("policy question")

    assert returned == []


async def test_empty_query_rejected_before_provider_calls(monkeypatch) -> None:
    """An empty/whitespace-only query should raise before any provider is called."""
    embedding_provider = _FakeEmbeddingProvider()
    vector_store = _FakeVectorStore()
    _patch_providers(monkeypatch, embedding_provider, vector_store)

    for query in ("", "   ", "\n\t"):
        try:
            await RetrievalService().retrieve(query)
            raise AssertionError("expected EmptyQueryError")
        except EmptyQueryError:
            pass

    assert embedding_provider.embed_calls == []
    assert vector_store.search_calls == []


async def test_embedding_failure_propagates_clearly(monkeypatch) -> None:
    """An embedding provider failure should propagate, not be swallowed."""
    _patch_providers(monkeypatch, _FailingEmbeddingProvider(), _FakeVectorStore())

    try:
        await RetrievalService().retrieve("policy question")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert str(exc) == "embedding provider unavailable"


async def test_vector_store_failure_propagates_clearly(monkeypatch) -> None:
    """A vector-store failure should propagate, not be swallowed."""
    _patch_providers(monkeypatch, _FakeEmbeddingProvider(), _FailingVectorStore())

    try:
        await RetrievalService().retrieve("policy question")
        raise AssertionError("expected RuntimeError")
    except RuntimeError as exc:
        assert str(exc) == "vector store unavailable"


async def test_no_llm_provider_is_called(monkeypatch) -> None:
    """retrieve() must never invoke an LLM provider — retrieval only embeds and searches."""

    def _fail_if_called(settings=None):
        raise AssertionError("get_llm_provider must never be called during retrieval")

    _patch_providers(monkeypatch, _FakeEmbeddingProvider(), _FakeVectorStore())
    monkeypatch.setattr("app.rag.providers.provider_factory.get_llm_provider", _fail_if_called)

    await RetrievalService().retrieve("policy question")
