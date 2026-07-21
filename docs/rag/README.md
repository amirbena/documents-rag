# RAG (Retrieval-Augmented Generation)

Ingestion-to-retrieval data flow, RAG engine ownership, and current capabilities/limitations.
For LangChain-specific detail see [docs/langchain/](../langchain/README.md); for language/prompt
handling see [docs/multilingual/](../multilingual/README.md); for document lifecycle state see
[docs/document-lifecycle/](../document-lifecycle/README.md).

## Ingestion-to-retrieval data flow

```
Document → DocumentTextExtractor → DocumentChunker → EmbeddingProvider → Qdrant upsert
                                                                              │
Question → LanguageDetector → PromptProvider ─────────┐                      │
                                                        ▼                      ▼
                                    RetrievalService.retrieve() ──── Qdrant search
                                                        │
                                    RagPromptBuilder.build() (label + attribute)
                                                        │
                                    LLMProvider.stream_generate()
                                                        │
                                    SSE: metadata → token(s) → done | error
```

Write side (ingestion) and read side (retrieval) share the same active `EmbeddingIndexConfig` —
same provider, model, dimension, embedding version, chunking version — resolved via
`get_active_embedding_config(settings)`, the single place either side reads indexing configuration
from. Neither reads `EMBEDDING_MODEL`/`VECTOR_SIZE` directly.

## Component ownership

| Component | File | Owns | Never does |
|---|---|---|---|
| `RuleBasedRagDecider` | `app/rag/decision.py` | Classifies a question into `NEEDS_RETRIEVAL` / `DIRECT_LLM` / `CLARIFICATION_NEEDED` / `OUT_OF_SCOPE` via deterministic keyword rules | Any retrieval, generation, or LLM call |
| `RetrievalService` | `app/rag/retrieval_service.py` | Embeds a query, searches Qdrant, applies score threshold | Prompt building, LLM calls |
| `RagPromptBuilder` | `app/rag/prompt_builder.py` | Deterministic, pure function: ranked results → labeled/attributed prompt | Retrieval, LLM calls, mutating its inputs |
| `RagOrchestrator` | `app/rag/orchestrator.py` | Composes decider + retrieval + prompt builder + LLM provider into one streamed call | Conversation memory, silent fallback between decisions/providers |
| `RagEngine` | `app/rag/engine.py` | Abstraction letting `POST /chat` be engine-agnostic | — |
| `CustomRagEngine` | `app/rag/engines/custom_engine.py` | Default; thin wrapper delegating to `RagOrchestrator` unchanged | Any logic of its own |
| `LangChainRagEngine` | `app/rag/engines/langchain_engine.py` | Same 4-way decision flow via LangChain `Runnable`s | See [docs/langchain/](../langchain/README.md) |

`RagOrchestrator` remains the platform's reference implementation — every other engine is judged
against its behavior.

## `stream_answer()` decision paths

| Decision | Retrieval called? | LLM called? | Output |
|---|---|---|---|
| `CLARIFICATION_NEEDED` | No | No | One fixed, language-appropriate message |
| `OUT_OF_SCOPE` | No | No | One fixed, language-appropriate message |
| `NEEDS_RETRIEVAL` (results found) | Yes | Yes | Streamed answer + `sources` |
| `NEEDS_RETRIEVAL` (nothing attributable) | Yes | **No** | Fixed `no_results` message, `sources: []` |
| `DIRECT_LLM` | No | Yes | Streamed answer, no sources |

A failure in `RetrievalService` or the LLM provider propagates unchanged — the orchestrator never
substitutes a direct answer for a failed retrieval, and never silently retries a different
provider. See [docs/document-lifecycle/README.md#chat](../document-lifecycle/README.md#api-contracts)
for the full `POST /chat` API contract and SSE event shape.

## Test ownership

See [docs/testing/](../testing/README.md) for the full taxonomy. RAG-specific test entrypoints:

```bash
make test-rag-engines      # unit + integration + E2E-parity across both engines (needs Docker)
make verify-rag-engines
```

## Current Limitations

- No standalone public retrieval endpoint — `RetrievalService` is reachable only indirectly via
  `POST /chat`'s `NEEDS_RETRIEVAL` path.
- No conversation memory / multi-turn context.
- The rule-based decider is deterministic keyword matching, not an LLM-based router.
- No client-selectable model override on `POST /chat`.

## Deferred Behavior

- An LLM-based question router (a future milestone may replace/augment `RuleBasedRagDecider`; the
  rule-based version exists first so the decision contract is fixed and testable).
- Conversation memory / multi-turn context in prompt building or the orchestrator.
- A standalone retrieval-only public endpoint.
- A per-request model override on `POST /chat`.

None of the above exists today — do not describe them as implemented, and this documentation pass
does not implement them.
