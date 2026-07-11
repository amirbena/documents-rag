"""LangChain-compatible adapters wrapping the platform's existing provider interfaces.

Every adapter here delegates to an already-resolved EmbeddingProvider / LLMProvider /
RetrievalService instance (obtained via app.rag.providers.provider_factory, exactly like
RagOrchestrator does) — none of them construct an Ollama, OpenAI, Gemini, Anthropic, or Qdrant
client directly, and none of them select a different embedding model or collection. This is the
only place LangChainRagEngine touches LangChain's provider-facing base classes; the rest of the
platform (ingestion, RetrievalService, RagPromptBuilder, provider factory) is untouched.
"""

from collections.abc import AsyncIterator
from typing import Any

from langchain_core.callbacks import AsyncCallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_core.outputs import GenerationChunk
from langchain_core.retrievers import BaseRetriever

from app.rag.providers.embedding_provider import EmbeddingProvider
from app.rag.providers.llm_provider import LLMProvider
from app.rag.providers.vector_store import VectorSearchResult
from app.rag.retrieval_service import RetrievalService

_ASYNC_ONLY_MESSAGE = "This adapter supports async invocation only — the app is async end to end."


class ProviderBackedLLM(LLM):
    """Adapts the platform's LLMProvider to LangChain's LLM Runnable interface.

    Wraps whatever LLMProvider app.rag.providers.provider_factory.get_llm_provider() resolved —
    it never instantiates a provider client itself, so LLM_PROVIDER selection stays centralized
    in the existing factory regardless of which RagEngine is configured.
    """

    provider: Any

    @property
    def _llm_type(self) -> str:
        return "provider-backed"

    def _call(
        self, prompt: str, stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any
    ) -> str:
        """Not supported — this app is async end to end; use ainvoke/astream instead."""
        raise NotImplementedError(_ASYNC_ONLY_MESSAGE)

    async def _acall(
        self, prompt: str, stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any
    ) -> str:
        """Return the full completion via LLMProvider.generate()."""
        return await self.provider.generate(prompt)

    async def _astream(
        self, prompt: str, stop: list[str] | None = None, run_manager: Any = None, **kwargs: Any
    ) -> AsyncIterator[GenerationChunk]:
        """Yield text chunks via LLMProvider.stream_generate(), in order."""
        async for chunk in self.provider.stream_generate(prompt):
            yield GenerationChunk(text=chunk)


def build_provider_backed_llm(provider: LLMProvider) -> ProviderBackedLLM:
    """Wrap an already-resolved LLMProvider as a LangChain-compatible streaming model."""
    return ProviderBackedLLM(provider=provider)


class ProviderBackedEmbeddings(Embeddings):
    """Adapts the platform's EmbeddingProvider to LangChain's Embeddings interface.

    Wraps whatever EmbeddingProvider app.rag.providers.provider_factory.get_embedding_provider()
    resolved — same embedding model, same vectors, as every other caller of that provider.
    """

    def __init__(self, provider: EmbeddingProvider) -> None:
        self._provider = provider

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Not supported — this app is async end to end; use aembed_documents instead."""
        raise NotImplementedError(_ASYNC_ONLY_MESSAGE)

    def embed_query(self, text: str) -> list[float]:
        """Not supported — this app is async end to end; use aembed_query instead."""
        raise NotImplementedError(_ASYNC_ONLY_MESSAGE)

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts via the wrapped EmbeddingProvider."""
        return await self._provider.embed(texts)

    async def aembed_query(self, text: str) -> list[float]:
        """Embed a single query text via the wrapped EmbeddingProvider."""
        return (await self._provider.embed([text]))[0]


def _search_result_to_document(result: VectorSearchResult) -> Document:
    """Build a LangChain Document from a VectorSearchResult, preserving every metadata field."""
    return Document(
        page_content=result.text,
        metadata={
            "id": result.id,
            "score": result.score,
            "document_id": result.document_id,
            "chunk_id": result.chunk_id,
            "source": result.source,
            "page_number": result.page_number,
            "sheet_name": result.sheet_name,
        },
    )


def document_to_search_result(document: Document) -> VectorSearchResult:
    """Reconstruct a VectorSearchResult from a Document produced by ProviderBackedRetriever.

    The inverse of _search_result_to_document — used by LangChainRagEngine to feed retrieved
    Documents back into the existing, unmodified RagPromptBuilder, which expects
    VectorSearchResult, not a generic Document.
    """
    metadata = document.metadata
    return VectorSearchResult(
        id=metadata["id"],
        score=metadata["score"],
        document_id=metadata["document_id"],
        chunk_id=metadata["chunk_id"],
        text=document.page_content,
        source=metadata["source"],
        page_number=metadata.get("page_number"),
        sheet_name=metadata.get("sheet_name"),
    )


class ProviderBackedRetriever(BaseRetriever):
    """Adapts the platform's RetrievalService to LangChain's BaseRetriever interface.

    Delegates every search to the existing RetrievalService — same embedding provider, same
    VectorStore/collection, same RETRIEVAL_TOP_K/RETRIEVAL_SCORE_THRESHOLD — so retrieval behavior
    is identical to CustomRagEngine's. Never constructs a Qdrant client directly.
    """

    retrieval_service: Any

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        results = await self.retrieval_service.retrieve(query)
        return [_search_result_to_document(result) for result in results]

    def _get_relevant_documents(self, query: str, *, run_manager: Any = None) -> list[Document]:
        """Not supported — this app is async end to end; use ainvoke instead."""
        raise NotImplementedError(_ASYNC_ONLY_MESSAGE)


def build_provider_backed_retriever(retrieval_service: RetrievalService) -> ProviderBackedRetriever:
    """Wrap an already-constructed RetrievalService as a LangChain-compatible retriever."""
    return ProviderBackedRetriever(retrieval_service=retrieval_service)
