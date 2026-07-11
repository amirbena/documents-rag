"""Streaming SSE chat endpoint: reuses whichever RagEngine is configured, no orchestration logic
of its own.

Formats RagEngine.stream_answer(question) output as Server-Sent Events (metadata, token, done,
error) and returns it via a FastAPI StreamingResponse. Contains no decision, retrieval, or
prompt-building logic, makes no direct provider calls, and does not know or care whether
CustomRagEngine or LangChainRagEngine is configured (see RAG_ENGINE) — it only serializes events
the engine already produced.
"""

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.rag.engine import RagEngine
from app.rag.engines.engine_factory import get_rag_engine as resolve_rag_engine
from app.rag.orchestrator import OrchestratorMetadata, OrchestratorToken
from app.rag.prompt_builder import PromptSource
from app.schemas.chat import ChatRequest

router = APIRouter()

_SAFE_ERROR_MESSAGE = "Failed to generate a response."


def get_rag_engine() -> RagEngine:
    """Build the configured RagEngine (see RAG_ENGINE) instance."""
    return resolve_rag_engine()


def _sse_event(event: str, data: dict) -> str:
    """Format one SSE event: `event: <name>`, `data: <JSON>`, then a blank line."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _source_payload(source: PromptSource) -> dict:
    """Build a JSON-safe dict for one PromptSource, omitting unset optional fields."""
    payload: dict = {
        "document_id": source.document_id,
        "chunk_id": source.chunk_id,
        "source": source.source,
        "score": source.score,
    }
    if source.page_number is not None:
        payload["page_number"] = source.page_number
    if source.sheet_name is not None:
        payload["sheet_name"] = source.sheet_name
    return payload


def _metadata_payload(metadata: OrchestratorMetadata) -> dict:
    """Build the JSON payload for a `metadata` SSE event from an OrchestratorMetadata."""
    return {
        "decision": metadata.decision.value,
        "reason": metadata.reason,
        "retrieval_used": metadata.retrieval_used,
        "sources": [_source_payload(source) for source in metadata.sources],
    }


async def _stream_chat_events(question: str, engine: RagEngine) -> AsyncIterator[str]:
    """Consume RagEngine.stream_answer(question) and yield it as formatted SSE events.

    Emits `metadata` before any `token`, then `done` exactly once on normal completion. A
    failure raised after streaming has started is emitted as a single `error` event with a
    fixed, safe message — no stack trace, prompt, or provider detail is ever included — and
    streaming stops there without a `done` event. Client cancellation propagates normally.
    """
    try:
        async for event in engine.stream_answer(question):
            if isinstance(event, OrchestratorMetadata):
                yield _sse_event("metadata", _metadata_payload(event))
            elif isinstance(event, OrchestratorToken):
                yield _sse_event("token", {"text": event.text})
    except Exception:
        yield _sse_event("error", {"message": _SAFE_ERROR_MESSAGE, "status": "failed"})
        return

    yield _sse_event("done", {"status": "completed"})


@router.post("/chat")
async def chat(
    request: ChatRequest,
    engine: RagEngine = Depends(get_rag_engine),
) -> StreamingResponse:
    """Stream the configured RagEngine's answer to `request.question` as Server-Sent Events."""
    return StreamingResponse(
        _stream_chat_events(request.question, engine),
        media_type="text/event-stream",
    )
