# Providers

Provider abstraction for embeddings, LLM generation, and vector storage. See
[docs/storage/](../storage/README.md) for the object-storage provider abstraction (a separate,
parallel pattern) and [docs/configuration/](../configuration/README.md) for environment variables.

## Provider factory

`app/rag/providers/provider_factory.py` resolves which concrete class to construct, based on
three independent settings, so the rest of the codebase depends on the abstract interfaces
(`EmbeddingProvider`/`LLMProvider`/`VectorStore`) rather than a specific backend:

| Function | Setting | Recognized values |
|---|---|---|
| `get_embedding_provider()` | `EMBEDDING_PROVIDER` | `ollama` → `OllamaEmbeddingProvider` |
| `get_llm_provider()` | `LLM_PROVIDER` | `ollama` → `OllamaLLMProvider`; `openai`/`gemini`/`anthropic` → stub (raises `ProviderNotImplementedError`) |
| `get_vector_store()` | `VECTOR_STORE_PROVIDER` | `qdrant` → `QdrantVectorStore` |

**No silent fallback, ever.** An unrecognized value raises `UnsupportedProviderError` naming the
offending value. A recognized-but-unimplemented provider (`openai`/`gemini`/`anthropic`) fails
loudly at resolution time — never silently defaults to Ollama.

## Supported providers

| Provider | Embedding | LLM | Vector store | Status |
|---|---|---|---|---|
| Ollama | ✅ (`bge-m3` default) | ✅ (`llama3.1` default, streaming) | — | Fully implemented |
| Qdrant | — | — | ✅ | Fully implemented |
| OpenAI | — | Stub only | — | `ProviderNotImplementedError` |
| Gemini | — | Stub only | — | `ProviderNotImplementedError` |
| Anthropic | — | Stub only | — | `ProviderNotImplementedError` |

## Provider selection

Set via environment variables (`EMBEDDING_PROVIDER`/`LLM_PROVIDER`/`VECTOR_STORE_PROVIDER`) — a
deployment-time choice, never a per-request parameter.

`LLM_PROVIDER` (which backend) and `LLM_MODEL` (which model that backend uses) are deliberately
separate — changing the model never requires touching provider selection. `Settings.resolved_llm_model`
is the single place that decides the effective model: `LLM_MODEL` if set, else `OLLAMA_CHAT_MODEL`.

`OLLAMA_EMBEDDING_MODEL` is intentionally **not** part of this fallback mechanism — embeddings use
a fixed model, independent of `LLM_MODEL`, since swapping it silently invalidates previously
computed vectors. Changing the embedding model deliberately requires bumping `EMBEDDING_VERSION`
and re-indexing — see [docs/multilingual/](../multilingual/README.md).

## Provider-specific configuration

| Provider | Required variables |
|---|---|
| Ollama | `OLLAMA_BASE_URL`, `OLLAMA_CHAT_MODEL`, `OLLAMA_EMBEDDING_MODEL` |
| Qdrant | `QDRANT_URL`, `QDRANT_COLLECTION_NAME` (prefix, not literal name — see [docs/storage/](../storage/README.md)), `VECTOR_SIZE` |

Full variable list, defaults, and required-vs-optional: [docs/configuration/](../configuration/README.md).

## Timeout and retry policy (Phase 2.10)

Every provider HTTP client (Ollama embedding, Ollama LLM, Ollama health, Qdrant, MinIO) has a
settings-backed timeout — no unbounded-wait literal remains anywhere in `app/rag/providers/` or
`app/storage/minio_storage.py`. Exact variables/defaults:
[docs/configuration/](../configuration/README.md#provider-http-timeouts-phase-210).

Ollama embedding, Qdrant, and MinIO calls additionally retry with bounded exponential backoff and
full jitter (`app/core/retry.py`'s `retry_async`, `PROVIDER_RETRY_MAX_ATTEMPTS`/
`PROVIDER_RETRY_BASE_DELAY_SECONDS`/`PROVIDER_RETRY_MAX_DELAY_SECONDS` — see
[docs/configuration/](../configuration/README.md#provider-retry-policy-phase-210)). Classification
(`app/rag/providers/http_retry_policy.py` for the raw-httpx providers):

- **Transient (retried):** connection/timeout failures (any `httpx.HTTPError` that isn't a status
  error), and HTTP 429/502/503/504.
- **Permanent (never retried):** every other 4xx/5xx status (400/401/403/404/etc.), and any
  malformed-response error (`ValueError`/`KeyError` from response-parsing code).

MinIO uses its own classifier against the `minio` SDK's exception types: `MaxRetryError`
(connection-level) is always transient; an `S3Error` is transient only for
`ServiceUnavailable`/`SlowDown`/`InternalError` — every other `S3Error` code (`NoSuchKey`, auth
failures, `BucketAlreadyOwnedByYou`, etc.) is permanent.

**Retry exhaustion** re-raises the *last* transient exception unchanged — never a generic
"retries exhausted" wrapper, and never swallowed — so a caller catching that provider's own error
type sees no difference between a call that needed a retry and one that didn't.

**Explicit exclusions — no retry attempted at all:**

- `OllamaLLMProvider.stream_generate()` (streaming chat generation) — retrying would risk
  re-emitting already-yielded tokens to a caller mid-stream.
- `MinioFileStorage.read()`'s `response.read()` step — only the `get_object()` call that opens the
  connection is retried; a partially-consumed response stream cannot be safely re-read from the
  start on the same object.

## Extension points

Adding a new LLM provider stub: create a class inheriting `LLMProviderStub`
(`app/rag/providers/llm_provider_stub.py`) with its own `NOT_IMPLEMENTED_MESSAGE`, then register it
in `_LLM_STUBS` in `provider_factory.py`. `LLMProviderStub.generate()`/`stream_generate()` both
always raise `ProviderNotImplementedError` — a stub makes no HTTP calls and reads no external API
keys.

`LLMProvider` (`app/rag/providers/llm_provider.py`) declares both `generate(prompt) -> str` and
`stream_generate(prompt) -> AsyncIterator[str]` as abstract — every implementation, including
stubs, must implement both.

## Implementation notes

- Ollama and Qdrant providers call the raw HTTP API directly (`httpx`), never an official SDK —
  deliberate, to keep the dependency surface small and behavior fully inspectable.
- `OllamaClient` (health/reachability checks) is kept separate from the generation/embedding
  provider interfaces so health checks don't get entangled with the generation contract.

## Test ownership

Provider tests live under `tests/unit/rag/providers/` (mocked `httpx` transports — no real
network). Real-Ollama/real-model checks are deliberately out of every automated tier — see
[docs/testing/](../testing/README.md)'s AI-provider policy.

## Current Limitations

- Only one real embedding provider (Ollama) and one real vector store (Qdrant) exist.
- No client-selectable provider override per request — provider selection is server-side only.

## Deferred Behavior

- Real `OpenAIProvider`/`GeminiProvider`/`AnthropicProvider` implementations — the stubs exist so
  future provider support has a place to land without changing the factory or callers, but no real
  implementation exists yet.
- A second `VectorStore` implementation (e.g. pgvector, Pinecone) — the abstraction supports one,
  but only Qdrant is implemented.
