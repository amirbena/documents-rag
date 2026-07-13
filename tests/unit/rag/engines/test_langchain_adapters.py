"""Tests for the LangChain adapters — each wraps an existing provider interface, no network calls."""

from collections.abc import AsyncIterator

import pytest

from app.rag.engines.langchain_adapters import (
    ProviderBackedEmbeddings,
    ProviderBackedLLM,
    ProviderBackedRetriever,
    build_provider_backed_llm,
    build_provider_backed_retriever,
    document_to_search_result,
)
from app.rag.providers.vector_store import VectorSearchResult


class _FakeLLMProvider:
    """Records prompts and yields a fixed sequence of text chunks instead of calling Ollama."""

    def __init__(self, chunks: list[str] | None = None) -> None:
        self.chunks = chunks if chunks is not None else ["a", "b"]
        self.generate_calls: list[str] = []
        self.stream_calls: list[str] = []

    async def generate(self, prompt: str) -> str:
        self.generate_calls.append(prompt)
        return "".join(self.chunks)

    async def stream_generate(self, prompt: str) -> AsyncIterator[str]:
        self.stream_calls.append(prompt)
        for chunk in self.chunks:
            yield chunk


class _FakeEmbeddingProvider:
    """Returns fixed vectors instead of calling a real embedding model."""

    def __init__(self) -> None:
        self.embed_calls: list[list[str]] = []

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self.embed_calls.append(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]


class _FakeRetrievalService:
    """Returns a fixed list of VectorSearchResult instead of calling a real RetrievalService."""

    def __init__(self, results: list[VectorSearchResult]) -> None:
        self.results = results
        self.retrieve_calls: list[str] = []

    async def retrieve(self, query: str, limit: int | None = None) -> list[VectorSearchResult]:
        self.retrieve_calls.append(query)
        return self.results


def _result(**overrides: object) -> VectorSearchResult:
    fields: dict[str, object] = {
        "id": "chunk-1",
        "score": 0.9,
        "document_id": "doc-1",
        "chunk_id": "chunk-1",
        "text": "some chunk text",
        "source": "handbook.pdf",
        "page_number": None,
        "sheet_name": None,
    }
    fields.update(overrides)
    return VectorSearchResult(**fields)  # type: ignore[arg-type]


async def test_llm_adapter_astream_delegates_to_provider_stream_generate() -> None:
    """ProviderBackedLLM.astream should yield the wrapped LLMProvider's chunks, in order."""
    provider = _FakeLLMProvider(chunks=["Hello", " ", "World"])
    llm = build_provider_backed_llm(provider)

    chunks = [chunk async for chunk in llm.astream("some prompt")]

    assert chunks == ["Hello", " ", "World"]
    assert provider.stream_calls == ["some prompt"]


async def test_llm_adapter_ainvoke_delegates_to_provider_generate() -> None:
    """ProviderBackedLLM.ainvoke should return the wrapped LLMProvider's full generate() output."""
    provider = _FakeLLMProvider(chunks=["full", " answer"])
    llm = build_provider_backed_llm(provider)

    result = await llm.ainvoke("some prompt")

    assert result == "full answer"
    assert provider.generate_calls == ["some prompt"]


def test_llm_adapter_sync_call_is_not_supported() -> None:
    """The sync _call path must fail explicitly — this app is async end to end."""
    llm = ProviderBackedLLM(provider=_FakeLLMProvider())

    with pytest.raises(NotImplementedError):
        llm._call("prompt")


async def test_embeddings_adapter_aembed_query_delegates_to_provider() -> None:
    """ProviderBackedEmbeddings.aembed_query should return the wrapped EmbeddingProvider's vector."""
    provider = _FakeEmbeddingProvider()
    embeddings = ProviderBackedEmbeddings(provider)

    vector = await embeddings.aembed_query("hello")

    assert vector == [0.1, 0.2, 0.3]
    assert provider.embed_calls == [["hello"]]


async def test_embeddings_adapter_aembed_documents_delegates_to_provider() -> None:
    """ProviderBackedEmbeddings.aembed_documents should embed every text via the same provider."""
    provider = _FakeEmbeddingProvider()
    embeddings = ProviderBackedEmbeddings(provider)

    vectors = await embeddings.aembed_documents(["a", "b"])

    assert vectors == [[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]
    assert provider.embed_calls == [["a", "b"]]


def test_embeddings_adapter_sync_methods_are_not_supported() -> None:
    """The sync embed_query/embed_documents paths must fail explicitly."""
    embeddings = ProviderBackedEmbeddings(_FakeEmbeddingProvider())

    with pytest.raises(NotImplementedError):
        embeddings.embed_query("hello")
    with pytest.raises(NotImplementedError):
        embeddings.embed_documents(["hello"])


async def test_retriever_adapter_delegates_to_retrieval_service_once() -> None:
    """ProviderBackedRetriever.ainvoke should call RetrievalService.retrieve exactly once."""
    results = [_result(chunk_id="c1", score=0.9), _result(chunk_id="c2", score=0.5)]
    retrieval_service = _FakeRetrievalService(results)
    retriever = build_provider_backed_retriever(retrieval_service)

    documents = await retriever.ainvoke("question")

    assert retrieval_service.retrieve_calls == ["question"]
    assert [document.metadata["chunk_id"] for document in documents] == ["c1", "c2"]


async def test_retriever_adapter_preserves_all_metadata_fields() -> None:
    """Every VectorSearchResult field must round-trip through Document.metadata unchanged."""
    result = _result(
        document_id="doc-42",
        chunk_id="chunk-7",
        source="report.xlsx",
        page_number=None,
        sheet_name="Sheet1",
        score=0.73,
        text="the cell contents",
    )
    retriever = build_provider_backed_retriever(_FakeRetrievalService([result]))

    documents = await retriever.ainvoke("question")

    document = documents[0]
    assert document.page_content == "the cell contents"
    assert document.metadata["document_id"] == "doc-42"
    assert document.metadata["chunk_id"] == "chunk-7"
    assert document.metadata["source"] == "report.xlsx"
    assert document.metadata["sheet_name"] == "Sheet1"
    assert document.metadata["page_number"] is None
    assert document.metadata["score"] == 0.73


async def test_document_to_search_result_is_the_exact_inverse_of_retrieval() -> None:
    """Converting a retrieved Document back to VectorSearchResult must reproduce the original."""
    original = _result(
        document_id="doc-1", chunk_id="c1", source="a.pdf", page_number=3, sheet_name=None, score=0.42
    )
    retriever = build_provider_backed_retriever(_FakeRetrievalService([original]))

    documents = await retriever.ainvoke("question")
    reconstructed = document_to_search_result(documents[0])

    assert reconstructed == original


def test_retriever_adapter_sync_path_is_not_supported() -> None:
    """The sync _get_relevant_documents path must fail explicitly — async only."""
    retriever = ProviderBackedRetriever(retrieval_service=_FakeRetrievalService([]))

    with pytest.raises(NotImplementedError):
        retriever._get_relevant_documents("question")
