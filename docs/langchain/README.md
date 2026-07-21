# LangChain Integration

`LangChainRagEngine` is an optional, parity-tested alternative `RagEngine` implementation — never
the platform's reference behavior. See [docs/rag/](../rag/README.md) for the shared decision/
retrieval/prompt architecture both engines sit on top of.

## Selection

```bash
RAG_ENGINE=custom      # default
RAG_ENGINE=langchain   # optional
```

`get_rag_engine(settings)` (`app/rag/engines/engine_factory.py`) resolves the configured engine —
an unrecognized value raises `UnsupportedRagEngineError` immediately; there is no silent fallback
to `custom`. `RAG_ENGINE` is a server-side deployment setting, never a per-request parameter.

## Ownership boundaries

| Component | File | Role |
|---|---|---|
| `LangChainRagEngine` | `app/rag/engines/langchain_engine.py` | Runs the same 4-way decision routing through LangChain `Runnable`s/prompt values |
| `ProviderBackedLLM` | `app/rag/engines/langchain_adapters.py` | LangChain `LLM` streaming from whatever `LLMProvider` the factory resolved |
| `ProviderBackedEmbeddings` | same | LangChain `Embeddings` wrapping the configured `EmbeddingProvider` |
| `ProviderBackedRetriever` | same | LangChain `BaseRetriever` wrapping the existing `RetrievalService` |

None of the three adapters construct an Ollama/OpenAI/Gemini/Anthropic/Qdrant client directly —
each is handed an already-resolved provider/service instance and only adapts its interface. There
is no `langchain-community` Qdrant vector-store integration and no second Qdrant SDK path —
`QdrantVectorStore`'s own `httpx`-based HTTP calls are the only thing that ever talks to Qdrant, in
either engine.

## Parity expectations (differences from the custom engine)

Guaranteed identical between engines:
- Decision routing (`RuleBasedRagDecider.decide()` is reused unmodified, outside any LangChain
  `Runnable`).
- Embedding model, Qdrant collection, vector payload, chunk/point IDs (no re-embedding, no second
  collection).
- Source labels (`[S1]`/`[S2]`...), rank order, and attribution metadata.
- Fixed governance instructions ("answer only from context", "say so if the answer isn't
  present") and their Hebrew/English resolution via `PromptProvider`.
- `no_results` behavior — no LLM call, `sources: []`, identical fixed message.
- The public SSE contract (`metadata`/`token`/`done`/`error`) — the route has no engine-specific
  branch.

**Legitimately different:** the generated answer text itself may differ (LangChain's own prompt
serialization differs from `RagOrchestrator`'s plain string concatenation). This is expected, not a
bug.

`ChatPromptValue` is built from literal `SystemMessage`/`HumanMessage` content — never an
interpolated `ChatPromptTemplate` — so arbitrary document text containing `{`/`}` can never be
misparsed as a template variable.

## Why LangGraph is deferred

LangChain's `Runnable`/prompt/retriever primitives already express this platform's 4-way decision
routing plus a single retrieval-then-generate step. There is no multi-step agent loop, no tool
calling, and no conversation memory for LangGraph's graph/state machinery to add value to.
Introducing it now would add a second orchestration paradigm with nothing for it to orchestrate.

## Test ownership

```bash
make test-rag-engines      # unit (factory, custom/LangChain engines, adapters, prompt parity)
                            # + integration (real ephemeral Qdrant, fake embeddings)
                            # + E2E parity (both engines compared: sources/ranking/SSE ordering)
make verify-rag-engines
```

## Current Limitations

- No agents, tool calling, or LangGraph — both engines are a fixed decide-then-generate flow.
- No LangChain-specific ingestion path — ingestion is engine-independent.

## Deferred Behavior

- LangGraph-based agentic workflows — deferred until a real multi-step, conditional workflow
  actually needs it (see "Why LangGraph is deferred" above). Do not introduce LangGraph as part
  of documentation or lifecycle work.
- Tool calling / agent packages of any kind.
