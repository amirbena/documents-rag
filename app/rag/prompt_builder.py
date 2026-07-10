"""Deterministic RAG prompt builder: turns a question and ranked retrieval results into a
structured prompt with source attribution.

Document -> RetrievalService.retrieve(...) -> RagPromptBuilder.build(...) -> BuiltRagPrompt.
No LLM call, no chat/SSE endpoint, no retrieval, no conversation memory — this only shapes a
prompt from already-ranked VectorSearchResults, mirroring how RetrievalService composes the read
side of the RAG pipeline in isolation.
"""

from dataclasses import dataclass

from app.rag.providers.vector_store import VectorSearchResult

_SYSTEM_PROMPT = (
    "You are a helpful assistant that answers questions using only the supplied context. "
    "Do not invent or assume information that is not present in the context. If the answer is "
    "not present in the context, say so explicitly instead of guessing."
)

_NO_RESULTS_CONTEXT = "No relevant context was found for this question."


@dataclass
class PromptSource:
    """One context source's attribution metadata, aligned with its label in the prompt context."""

    document_id: str
    chunk_id: str
    source: str
    score: float
    page_number: int | None = None
    sheet_name: str | None = None


@dataclass
class BuiltRagPrompt:
    """A deterministic, structured prompt built from a question and ranked retrieval results."""

    system_prompt: str
    user_prompt: str
    context: str
    sources: list[PromptSource]


class RagPromptBuilder:
    """Builds a deterministic BuiltRagPrompt from a question and ranked VectorSearchResults."""

    def build(self, question: str, results: list[VectorSearchResult]) -> BuiltRagPrompt:
        """Build system/user prompts, labeled context, and source metadata, ranked and attributed.

        Chunks with empty/whitespace-only text are ignored. If nothing remains, the context
        states plainly that no relevant context was found — no fallback content is fabricated.
        """
        non_empty = [result for result in results if result.text.strip()]

        if not non_empty:
            return BuiltRagPrompt(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=self._user_prompt(question, _NO_RESULTS_CONTEXT),
                context=_NO_RESULTS_CONTEXT,
                sources=[],
            )

        context = "\n\n".join(
            self._format_source_block(f"[S{index}]", result)
            for index, result in enumerate(non_empty, start=1)
        )
        sources = [self._to_prompt_source(result) for result in non_empty]

        return BuiltRagPrompt(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=self._user_prompt(question, context),
            context=context,
            sources=sources,
        )

    @staticmethod
    def _to_prompt_source(result: VectorSearchResult) -> PromptSource:
        return PromptSource(
            document_id=result.document_id,
            chunk_id=result.chunk_id,
            source=result.source,
            score=result.score,
            page_number=result.page_number,
            sheet_name=result.sheet_name,
        )

    @staticmethod
    def _format_source_block(label: str, result: VectorSearchResult) -> str:
        header_parts = [label, result.source]
        if result.page_number is not None:
            header_parts.append(f"page {result.page_number}")
        if result.sheet_name is not None:
            header_parts.append(f"sheet {result.sheet_name}")
        return f"{' '.join(header_parts)}\n{result.text}"

    @staticmethod
    def _user_prompt(question: str, context: str) -> str:
        return (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer using only the context above. If the answer isn't in the context, say so."
        )
