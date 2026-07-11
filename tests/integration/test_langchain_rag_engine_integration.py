"""Integration tests for the LangChain compatibility layer against a real, ephemeral Qdrant
container.

Verifies that ProviderBackedRetriever (backed by the real RetrievalService/QdrantVectorStore)
can search vectors indexed exactly like the existing ingestion pipeline does — same
VectorPoint shape, same collection, same embedding provider seam — with only the embedding model
faked. Never a real Ollama call, never a second Qdrant SDK path, never a separate collection.
"""

import uuid

import httpx

import app.rag.retrieval_service as retrieval_service_module
from app.core.config import get_settings
from app.rag.embedding_config import get_active_embedding_config
from app.rag.engines.langchain_adapters import build_provider_backed_retriever
from app.rag.providers.qdrant_vector_store import QdrantVectorStore
from app.rag.providers.vector_store import VectorPoint
from app.rag.retrieval_service import RetrievalService


class _FakeEmbeddingProvider:
    """Returns a fixed, deterministic vector per text — no real Ollama call."""

    def __init__(self, vector: list[float]) -> None:
        self._vector = vector
        self.embed_calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(texts)
        return [self._vector for _ in texts]


def _unique_collection_name() -> str:
    return f"integration-langchain-{uuid.uuid4().hex}"


async def _collection_names(qdrant_url: str) -> set[str]:
    async with httpx.AsyncClient(base_url=qdrant_url, timeout=10.0) as client:
        response = await client.get("/collections")
        response.raise_for_status()
        return {item["name"] for item in response.json()["result"]["collections"]}


async def test_existing_vectors_are_searchable_through_the_langchain_adapter(
    qdrant_url: str, monkeypatch
) -> None:
    """A collection indexed exactly like ingestion does must be searchable via ProviderBackedRetriever."""
    settings = get_settings()
    monkeypatch.setattr(settings, "vector_size", 3)
    collection_name = _unique_collection_name()
    monkeypatch.setattr(settings, "qdrant_collection_name", collection_name)
    collection_name = get_active_embedding_config(settings).collection_name

    vector_store = QdrantVectorStore(settings=settings)
    await vector_store.create_collection_if_not_exists(collection_name, vector_size=3)

    point = VectorPoint(
        id=str(uuid.uuid4()),
        vector=[1.0, 0.0, 0.0],
        document_id="doc-1",
        chunk_id="chunk-1",
        text="the refund policy allows 30 days",
        source="handbook.pdf",
        page_number=4,
        sheet_name=None,
    )
    await vector_store.upsert_vectors(collection_name, [point])

    fake_embedding_provider = _FakeEmbeddingProvider(vector=[1.0, 0.0, 0.0])
    monkeypatch.setattr(
        retrieval_service_module, "get_embedding_provider", lambda settings=None: fake_embedding_provider
    )

    retrieval_service = RetrievalService(settings)
    retriever = build_provider_backed_retriever(retrieval_service)

    documents = await retriever.ainvoke("what is the refund policy?")

    assert len(documents) == 1
    assert documents[0].page_content == "the refund policy allows 30 days"
    assert fake_embedding_provider.embed_calls == [["what is the refund policy?"]]


async def test_metadata_round_trips_correctly_through_real_qdrant(qdrant_url: str, monkeypatch) -> None:
    """document_id/chunk_id/source/page_number/sheet_name/score must survive a real round trip."""
    settings = get_settings()
    monkeypatch.setattr(settings, "vector_size", 2)
    collection_name = _unique_collection_name()
    monkeypatch.setattr(settings, "qdrant_collection_name", collection_name)
    collection_name = get_active_embedding_config(settings).collection_name

    vector_store = QdrantVectorStore(settings=settings)
    await vector_store.create_collection_if_not_exists(collection_name, vector_size=2)

    point = VectorPoint(
        id=str(uuid.uuid4()),
        vector=[0.5, 0.5],
        document_id="doc-42",
        chunk_id="chunk-7",
        text="sheet contents",
        source="report.xlsx",
        page_number=None,
        sheet_name="Sheet1",
    )
    await vector_store.upsert_vectors(collection_name, [point])

    fake_embedding_provider = _FakeEmbeddingProvider(vector=[0.5, 0.5])
    monkeypatch.setattr(
        retrieval_service_module, "get_embedding_provider", lambda settings=None: fake_embedding_provider
    )

    retriever = build_provider_backed_retriever(RetrievalService(settings))
    documents = await retriever.ainvoke("query")

    metadata = documents[0].metadata
    assert metadata["document_id"] == "doc-42"
    assert metadata["chunk_id"] == "chunk-7"
    assert metadata["source"] == "report.xlsx"
    assert metadata["sheet_name"] == "Sheet1"
    assert metadata["page_number"] is None
    assert isinstance(metadata["score"], int | float)


async def test_no_separate_collection_is_created(qdrant_url: str, monkeypatch) -> None:
    """Retrieving through the LangChain adapter must not create any collection of its own."""
    settings = get_settings()
    monkeypatch.setattr(settings, "vector_size", 2)
    collection_name = _unique_collection_name()
    monkeypatch.setattr(settings, "qdrant_collection_name", collection_name)
    collection_name = get_active_embedding_config(settings).collection_name

    vector_store = QdrantVectorStore(settings=settings)
    await vector_store.create_collection_if_not_exists(collection_name, vector_size=2)

    point = VectorPoint(
        id=str(uuid.uuid4()),
        vector=[0.1, 0.2],
        document_id="doc-1",
        chunk_id="chunk-1",
        text="text",
        source="a.txt",
    )
    await vector_store.upsert_vectors(collection_name, [point])

    before = await _collection_names(qdrant_url)
    assert collection_name in before

    fake_embedding_provider = _FakeEmbeddingProvider(vector=[0.1, 0.2])
    monkeypatch.setattr(
        retrieval_service_module, "get_embedding_provider", lambda settings=None: fake_embedding_provider
    )
    retriever = build_provider_backed_retriever(RetrievalService(settings))
    await retriever.ainvoke("query")

    after = await _collection_names(qdrant_url)
    assert after == before, "the LangChain adapter must search the existing collection, never a new one"


async def test_configured_embedding_provider_is_used(qdrant_url: str, monkeypatch) -> None:
    """The retriever must embed the query via whichever EmbeddingProvider is configured."""
    settings = get_settings()
    monkeypatch.setattr(settings, "vector_size", 2)
    collection_name = _unique_collection_name()
    monkeypatch.setattr(settings, "qdrant_collection_name", collection_name)
    collection_name = get_active_embedding_config(settings).collection_name

    vector_store = QdrantVectorStore(settings=settings)
    await vector_store.create_collection_if_not_exists(collection_name, vector_size=2)
    await vector_store.upsert_vectors(
        collection_name,
        [
            VectorPoint(
                id=str(uuid.uuid4()),
                vector=[0.2, 0.8],
                document_id="doc-1",
                chunk_id="chunk-1",
                text="text",
                source="a.txt",
            )
        ],
    )

    fake_embedding_provider = _FakeEmbeddingProvider(vector=[0.2, 0.8])
    monkeypatch.setattr(
        retrieval_service_module, "get_embedding_provider", lambda settings=None: fake_embedding_provider
    )

    retriever = build_provider_backed_retriever(RetrievalService(settings))
    await retriever.ainvoke("a specific query")

    assert fake_embedding_provider.embed_calls == [["a specific query"]]


async def test_retrieval_result_returns_no_relevant_results_without_fabrication(
    qdrant_url: str, monkeypatch
) -> None:
    """An empty collection must return no documents — no fabricated results."""
    settings = get_settings()
    monkeypatch.setattr(settings, "vector_size", 2)
    collection_name = _unique_collection_name()
    monkeypatch.setattr(settings, "qdrant_collection_name", collection_name)
    collection_name = get_active_embedding_config(settings).collection_name

    vector_store = QdrantVectorStore(settings=settings)
    await vector_store.create_collection_if_not_exists(collection_name, vector_size=2)

    fake_embedding_provider = _FakeEmbeddingProvider(vector=[0.0, 0.0])
    monkeypatch.setattr(
        retrieval_service_module, "get_embedding_provider", lambda settings=None: fake_embedding_provider
    )

    retriever = build_provider_backed_retriever(RetrievalService(settings))
    documents = await retriever.ainvoke("query")

    assert documents == []
