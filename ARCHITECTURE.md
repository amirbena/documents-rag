# Architecture

## System overview

`documents-rag` is a local-first RAG (Retrieval-Augmented Generation) platform. Everything runs on
the user's machine via Docker Compose, with Ollama providing local LLM and embedding inference —
no external API calls or cloud dependencies.

This milestone is infrastructure plus ten vertical slices: a FastAPI app wired to Postgres,
Redis, Qdrant, and Ollama, with a health endpoint, an Ollama health/model-availability check, a
concrete Ollama-backed embedding provider, a concrete streaming Ollama-backed LLM provider, a
concrete Qdrant-backed vector store, a rule-based RAG decision layer, a document upload +
ingestion job skeleton, an async ingestion worker that claims and resolves those jobs, a
document text extractor (`.txt`/`.md`/`.pdf`/`.docx`/`.xlsx`), a document chunker that splits
extracted text into fixed-size, overlapping, word-boundary-aware chunks, and chunk
embedding/Qdrant indexing — the worker's pipeline is now Document → extraction → chunking →
embedding → Qdrant upsert. No retrieval or chat endpoint happens yet — a claimed job's chunks
are embedded and their vectors upserted into Qdrant, but nothing yet reads them back out for
retrieval; the job resolves to `completed` only if extraction, chunking, embedding, and the
Qdrant upsert all succeed.

## Services

| Service    | Image                    | Purpose (current)                                             | Purpose (future) |
|------------|--------------------------|------------------------------------------------------------------|-------------------|
| `app`      | built from `Dockerfile`  | FastAPI process: `/api/v1/health`, `/api/v1/providers/ollama/health`, `POST /api/v1/documents` | RAG API: ingestion processing, retrieval, chat |
| `postgres` | `postgres:16-alpine`     | Stores `documents`/`ingestion_jobs` rows via async SQLAlchemy     | Session/metadata storage |
| `redis`    | `redis:7-alpine`         | Available on the network                                         | Caching, task queues |
| `qdrant`   | `qdrant/qdrant:latest`   | Collection create/upsert/search via `QdrantVectorStore`           | Backing document retrieval in a future RAG flow |
| `ollama`   | `ollama/ollama:latest`   | Health/model checks + embeddings (`nomic-embed-text`) + streaming generation (`llama3.1`) | Backing a future public chat endpoint |

The app queries Ollama's `/api/tags` endpoint (via `app/services/ollama_client.py`) to check
reachability and whether the configured models are pulled, calls `/api/embeddings` (via
`app/rag/providers/ollama_embedding_provider.py`) to embed text with `OLLAMA_EMBEDDING_MODEL`,
and calls `/api/generate` with `stream=true` (via `app/rag/providers/ollama_llm_provider.py`) to
stream completions from the configured chat model (`LLM_MODEL`, falling back to
`OLLAMA_CHAT_MODEL` — see "LLM provider vs. model" below). The LLM provider is internal-only —
there is no public chat or SSE endpoint yet. The app also talks to Qdrant's HTTP API under
`QDRANT_URL` (via `app/rag/providers/qdrant_vector_store.py`) to create collections, upsert
vectors, and run similarity search — see "Vector store" below. Callers resolve all of these
providers through `app/rag/providers/provider_factory.py` rather than importing Ollama/Qdrant
classes directly — see "Provider factory" below. `POST /api/v1/documents` stores an uploaded
file and creates `Document`/`IngestionJob` rows in Postgres — see "Document upload and ingestion
job skeleton" below. Nothing yet processes an `IngestionJob`; Redis is still unused beyond
connection configuration.

## Provider factory

`app/rag/providers/provider_factory.py` resolves which concrete provider class to construct,
based on three config variables, so the rest of the codebase depends on the `EmbeddingProvider` /
`LLMProvider` / `VectorStore` interfaces rather than being coupled to Ollama or Qdrant directly:

- `get_embedding_provider()` — `EMBEDDING_PROVIDER` (`"ollama"` → `OllamaEmbeddingProvider`)
- `get_llm_provider()` — `LLM_PROVIDER` (`"ollama"` → `OllamaLLMProvider`; `"openai"`, `"gemini"`,
  `"anthropic"` are recognized but raise `ProviderNotImplementedError` — see "Future LLM provider
  stubs" below)
- `get_vector_store()` — `VECTOR_STORE_PROVIDER` (`"qdrant"` → `QdrantVectorStore`)

An unrecognized provider name raises `UnsupportedProviderError` (a `ValueError`) with a message
naming the offending value and the supported provider(s). All Ollama-specific logic (HTTP calls,
error handling) stays inside the Ollama provider classes — the factory only selects and
constructs; it never reimplements provider behavior, and business/service code should resolve
providers through it rather than importing `OllamaEmbeddingProvider`/`OllamaLLMProvider` directly.
The factory never falls back to Ollama for a misconfigured or unimplemented provider — every
non-`ollama` value either resolves to its own explicit failure or a real alternative
implementation.

## LLM provider vs. model

`LLM_PROVIDER` (which backend to use, e.g. `ollama`) and `LLM_MODEL` (which model that backend
should use, e.g. `llama3.1`) are deliberately separate settings — changing the model doesn't
require touching provider selection, and vice versa. `Settings.resolved_llm_model`
(`app/core/config.py`) is the single place that decides the effective model: it returns
`LLM_MODEL` if set, otherwise falls back to `OLLAMA_CHAT_MODEL` for backward compatibility.
`OllamaLLMProvider` calls `resolved_llm_model`, never `ollama_chat_model` directly, when building
its `/api/generate` request.

`OLLAMA_EMBEDDING_MODEL` is intentionally **not** part of this model-selection mechanism —
embeddings use a fixed model, independent of `LLM_MODEL`, since swapping the embedding model
would silently invalidate any previously-computed vectors. `OllamaEmbeddingProvider` always reads
`ollama_embedding_model` directly.

## Vector store

`app/rag/providers/vector_store.py` defines the abstract `VectorStore` contract plus its shared
data types:

- `VectorPoint` — one embedding vector to upsert, with `id`, `vector`, and payload metadata
  (`document_id`, `chunk_id`, `text`, `source`, optional `page_number`, optional `sheet_name`).
- `VectorSearchResult` — one nearest-neighbor match: `id`, `score`, plus the same payload fields.

`VectorStore` methods: `create_collection_if_not_exists(collection_name, vector_size)`,
`upsert_vectors(collection_name, points)`, `search_similar(collection_name, query_vector, limit)`.

`QdrantVectorStore` (`app/rag/providers/qdrant_vector_store.py`) is the concrete implementation.
It talks to Qdrant's REST API directly under `QDRANT_URL` via async `httpx` — **no official
Qdrant SDK is used**, matching the same pattern as the Ollama providers. It calls:

- `GET /collections/{name}` + `PUT /collections/{name}` — check-then-create a collection
  (skips creation if the collection already exists).
- `PUT /collections/{name}/points?wait=true` — upsert points (id, vector, payload).
- `POST /collections/{name}/points/search` — similarity search, returning parsed
  `VectorSearchResult` objects.

`QdrantVectorStoreError` is raised on an unreachable server, a non-200 response, or a malformed
response (e.g. missing payload fields). `IngestionWorker` is now a caller (see "Chunk embedding
and Qdrant indexing" below) — no other caller wires it into a retrieval pipeline yet.

## RAG decision layer

`app/rag/decision.py` is a small internal decision/orchestration layer that classifies a user
question *before* any retrieval or generation happens — it does not itself perform retrieval,
generation, ingestion, or document upload, and is not wired to any public API endpoint.

- `RagDecision` (a `StrEnum`) — one of `NEEDS_RETRIEVAL`, `DIRECT_LLM`,
  `CLARIFICATION_NEEDED`, `OUT_OF_SCOPE`.
- `DecisionResult` — a dataclass with `decision`, `reason`, and an optional `confidence`.
- `RuleBasedRagDecider.decide(question) -> DecisionResult` — deterministic, keyword/pattern-based
  routing. **No LLM call is made to route** — rules are checked in order:
  1. Empty or very short question → `CLARIFICATION_NEEDED`.
  2. Sensitive/private data extraction requests (SSN, passwords, API keys, credentials, etc.) →
     `OUT_OF_SCOPE` — checked *before* the retrieval keywords, so a request that mentions both
     documents and sensitive data is still rejected.
  3. Question references uploaded/indexed documents (`document`, `uploaded`, `pdf`, `knowledge
     base`, etc.) → `NEEDS_RETRIEVAL`.
  4. Otherwise → `DIRECT_LLM` (general question, no document reference, nothing sensitive).

This is deliberately the simplest possible decider — a future milestone may replace or augment it
with an LLM-based router, but the rule-based version exists first so the decision *contract*
(`RagDecision`/`DecisionResult`) is fixed and testable before anything calls out to a model for
routing.

## Document upload and ingestion job skeleton

`POST /api/v1/documents` (`app/api/v1/routes/documents.py`) is the first public endpoint that
touches the database. It accepts a multipart file upload and does exactly three things, all
inside one request:

1. Saves the file via `LocalFileStorage` (`app/services/local_file_storage.py`) under
   `storage/documents/`, using a **generated, filesystem-safe stored filename** (a UUID plus a
   sanitized extension) — never the raw original filename, which may contain Unicode, spaces, or
   path-unsafe characters. No S3 or other remote backend exists yet.
2. Inserts a `Document` row (`app/models/document.py`): `original_filename` (stored exactly as
   received — Hebrew or any other Unicode text is preserved verbatim, since Postgres/SQLAlchemy
   `String` columns are UTF-8 natively), `stored_filename`, `content_type`, `file_size`,
   `stored_path`.
3. Inserts an `IngestionJob` row (`app/models/ingestion_job.py`) with `status=PENDING`,
   referencing the `Document` via `document_id`. `IngestionStatus` (a `StrEnum`): `PENDING`,
   `PROCESSING`, `COMPLETED`, `FAILED` — stored in Postgres as their lowercase `.value`
   (`pending`, `processing`, ...), not the enum member name.

The endpoint returns `202 Accepted` with `{document_id, job_id, status}` — **it does not parse,
chunk, embed, or index the document inside the request.** An empty (zero-byte) upload is
rejected with `400` before any row is created. `IngestionWorker` (below) is what eventually
picks up and resolves the `pending` job it creates.

## Ingestion worker

`IngestionWorker` (`app/services/ingestion_worker.py`) is an internal service — **no public API**
— that claims and resolves one pending `IngestionJob` at a time via `process_next_job(session)`:

1. **Claim**: `SELECT ... WHERE status='pending' ORDER BY created_at LIMIT 1 FOR UPDATE SKIP
   LOCKED` — Postgres row-level locking so multiple worker instances never claim the same job.
   Returns `None` if there is no pending job.
2. Flips the claimed job to `PROCESSING` and **commits immediately** — a separate transaction
   boundary from the outcome below, so the claim is durable before any processing is attempted.
3. Looks up the associated `Document` and calls the injected processing step (default:
   `_default_process_document`, which runs Document → extraction (`DocumentTextExtractor` —
   see "Document text extraction" below) → chunking (`DocumentChunker` — see "Document
   chunking" below) → embedding → Qdrant upsert (see "Chunk embedding and Qdrant indexing"
   below)) with `(document, job)`.
4. On success: `PROCESSING` → `COMPLETED`, committed.
   On any exception (extraction, chunking, embedding, or Qdrant upsert): `PROCESSING` → `FAILED`,
   `error_message` set to `str(exception)`, committed.

The processing step is injected via the constructor (`IngestionWorker(process_document=...)`) so
tests can substitute a fake pipeline without changing the claim/lock/transition logic.
`IngestionWorker` never imports or calls `LLMProvider` itself — ingestion embeds and indexes
chunks, it never generates text.

**Idempotent by construction**: once a job leaves `PENDING` (to `PROCESSING`, then `COMPLETED`
or `FAILED`), the claim query's `WHERE status='pending'` filter can never select it again —
calling `process_next_job()` repeatedly does not require any separate "already processed" check.

`with_for_update(skip_locked=True)` is Postgres-specific row-locking syntax; SQLite does not
represent it correctly even if it accepts the same SQLAlchemy call, so this project deliberately
does not add SQLite/`aiosqlite` for testing this worker. Its tests use a fake `AsyncSession`
double that faithfully simulates the pending-job filter and `Document` lookup instead.

## Document text extraction

`DocumentTextExtractor` (`app/services/document_text_extractor.py`) is the ingestion worker's
default processing step: it loads a `Document`'s `stored_path` file and extracts its raw text —
**no chunking, embedding, or Qdrant upsert**. It routes by file extension, then validates the
file's basic structure/content against what that extension claims before attempting to parse it
(see "Routing and validation" below). It supports exactly five file types:

- `.txt` / `.md` — read as UTF-8 text and returned as a single `ExtractedPage` with
  `page_number=None`, `sheet_name=None`. Hebrew and other non-ASCII Unicode content is preserved
  exactly.
- `.pdf` — extracted page by page via `pypdf` (`PdfReader`), producing one `ExtractedPage` per
  page with a 1-indexed `page_number`, so downstream chunking/citation can reference the
  original page a piece of text came from.
- `.docx` — extracted via `python-docx`: all paragraph text joined into a single `ExtractedPage`
  (`page_number=None`, `sheet_name=None`) — plain text only, no tables, headers/footers, or
  pagination.
- `.xlsx` — extracted sheet by sheet via `openpyxl` (`load_workbook(..., read_only=True,
  data_only=True)`), producing one `ExtractedPage` per worksheet with `sheet_name` set to the
  worksheet's title and `page_number=None`; each row's non-empty cell values are tab-joined.

Any other extension raises `DocumentTextExtractionError("Unsupported file type: ...")` — there
is no fallback or content-based detection.

Data types:

- `ExtractedPage` — `text: str`, `page_number: int | None` (PDF only), `sheet_name: str | None`
  (XLSX only) — both `None` for `.txt`/`.md`/`.docx`, which have no natural pagination.
- `ExtractedDocument` — `document_id: str`, `pages: list[ExtractedPage]`, plus a `full_text`
  property that joins all pages' text.

`extract(document)` runs the actual file I/O and PDF parsing off the event loop via
`asyncio.to_thread` (both are blocking operations). `DocumentTextExtractionError` is raised for:
a missing `stored_path` file, an unsupported extension, or a file whose extracted text is empty
or whitespace-only. Any of these propagate up through `IngestionWorker.process_next_job()` and
resolve the job to `failed` with the error message stored — extraction never crashes the worker
process itself.

### Routing and validation

`DocumentTextExtractor` decides how to parse a file from `Path(stored_path).suffix`, then
validates the file's basic structure/content against what that extension claims **before**
handing it to the corresponding parser (`_validate_file_type`, called from `_extract_sync`
ahead of any extraction call):

| Extension | Handler | Validation before parsing |
|-----------|---------|----------------------------|
| `.txt`    | UTF-8 plain text (`path.read_text(encoding="utf-8")`) | Readable as UTF-8 (a `UnicodeDecodeError` is caught and re-raised as `DocumentTextExtractionError`) |
| `.md`     | UTF-8 markdown/plain text (same as `.txt` — no Markdown parsing) | Same as `.txt` |
| `.pdf`    | `pypdf` (`PdfReader`), page by page | First 4 bytes equal the PDF header `%PDF` |
| `.docx`   | `python-docx` (`docx.Document`), paragraph text | Valid ZIP archive (`zipfile.is_zipfile`) containing `word/document.xml` |
| `.xlsx`   | `openpyxl` (`load_workbook`), sheet by sheet | Valid ZIP archive containing `xl/workbook.xml` |

This is still lightweight, structural validation, not full content sanitization — it catches a
mismatched or corrupt file before wasting effort on the wrong parser (e.g. an `.xlsx` renamed to
`.pdf`, or arbitrary bytes given a document extension), each raising a specific
`DocumentTextExtractionError`. It does not validate the upload's `content_type` header, do deep
MIME/magic-byte sniffing beyond the checks above, or scan file contents for malicious payloads —
those remain future hardening if ever needed.

The extracted result is passed directly into chunking (see "Document chunking" below) — nothing
persists the raw extracted text itself.

## Document chunking

`DocumentChunker` (`app/services/document_chunker.py`) is the ingestion worker's second
processing step: it takes the `ExtractedDocument` that `DocumentTextExtractor` produced and
splits each page's text into fixed-size, overlapping chunks — **no embedding generation, no
Qdrant upsert, no retrieval**.

- Input: `ExtractedDocument` (one `ExtractedDocument` covering all of a document's pages/sheets).
- Output: `list[DocumentChunk]`.
- `DocumentChunk` — `document_id: str`, `chunk_id: str`, `text: str`, `chunk_index: int`,
  `page_number: int | None`, `sheet_name: str | None`. `page_number`/`sheet_name` are copied
  straight from the source `ExtractedPage` a chunk came from — `page_number` set for chunks from
  a PDF page, `sheet_name` set for chunks from an XLSX sheet, both `None` for `.txt`/`.md`/`.docx`
  chunks (which have no natural pagination).

Chunking rules:

- **Fixed target size** (`chunk_size`, in characters) and **configurable overlap**
  (`chunk_overlap`, in characters) — both configured via `CHUNK_SIZE`/`CHUNK_OVERLAP` (see
  "Environment variables" below), and both also settable directly via
  `DocumentChunker(chunk_size=..., chunk_overlap=...)`. The constructor raises `ValueError` if
  `chunk_overlap >= chunk_size` or either value is non-positive/negative.
- **Word-boundary-aware**: chunks are built by accumulating whole words up to `chunk_size`
  characters — a chunk never ends or starts mid-word. Overlap is built from the trailing whole
  words of the previous chunk, up to `chunk_overlap` characters.
- **Empty chunks are ignored**: a page whose text is empty or whitespace-only (after
  `str.split()`) produces zero chunks, not an empty-text chunk.
- **Deterministic**: `chunk()` is a pure function of its input — the same `ExtractedDocument`
  always produces the same chunks, in the same order, with the same `chunk_id`s
  (`f"{document_id}-{chunk_index}"`, where `chunk_index` increments continuously across all
  pages of the document, not reset per page).

The chunker's output feeds directly into embedding (see "Chunk embedding and Qdrant indexing"
below) — there is still no table or field that persists chunk text itself in Postgres; Qdrant is
the system of record for chunk vectors and their metadata.

## Chunk embedding and Qdrant indexing

The ingestion worker's third and fourth processing steps (`app/services/ingestion_worker.py`)
turn each `DocumentChunk` into an indexed vector — **no retrieval, chat, or SSE endpoint reads
these back out yet**:

1. **Embed**: `get_embedding_provider()` (reads `EMBEDDING_PROVIDER`) embeds every chunk's text
   in one call — `embedding_provider.embed([chunk.text for chunk in chunks])` — returning one
   vector per chunk, in the same order.
2. **Build `VectorPoint`s**: each chunk + its vector becomes a `VectorPoint` with `id` set to a
   deterministic `uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)` (so re-processing a document
   overwrites the same Qdrant points instead of duplicating them, and the id is always a
   Qdrant-valid UUID regardless of `chunk_id`'s own format), and payload fields `document_id`,
   `chunk_id`, `text`, `source` (the `Document.original_filename`), `page_number`, and
   `sheet_name` carried straight over from the chunk.
3. **Index**: `get_vector_store()` (reads `VECTOR_STORE_PROVIDER`) creates the collection named
   `QDRANT_COLLECTION_NAME` with dimensionality `VECTOR_SIZE` if it doesn't already exist
   (cheap no-op if it does), then upserts all points in one `upsert_vectors` call.

A document that produces zero chunks (e.g. empty/whitespace-only extracted text) short-circuits
before calling the embedding provider or vector store at all. A failure at either step
(`EmbeddingProvider.embed` or `VectorStore.create_collection_if_not_exists`/`upsert_vectors`
raising) propagates up through `_default_process_document` exactly like an extraction or
chunking failure — `IngestionWorker` catches it and marks the job `FAILED` with
`error_message = str(exception)`. `IngestionWorker` still never imports or calls `LLMProvider` —
this pipeline embeds and indexes, it never generates text.

## Retrieval service

`RetrievalService` (`app/rag/retrieval_service.py`) is the internal read-side counterpart to
chunk embedding/indexing: given a query, it embeds it and searches Qdrant for relevant chunks —
**no public retrieval/chat/SSE endpoint exists yet, and no LLM call is made**.

`retrieve(query: str, limit: int | None = None) -> list[VectorSearchResult]`:

1. **Validate**: an empty/whitespace-only `query` raises `EmptyQueryError` before any provider is
   called.
2. **Embed**: `get_embedding_provider()` (reads `EMBEDDING_PROVIDER`) embeds the query text —
   `embedding_provider.embed([query])[0]` — using the same fixed embedding model ingestion used,
   so query and chunk vectors stay comparable.
3. **Search**: `get_vector_store()` (reads `VECTOR_STORE_PROVIDER`) runs
   `search_similar(QDRANT_COLLECTION_NAME, query_vector, limit)`, where `limit` is the caller's
   explicit `limit` if given, else `RETRIEVAL_TOP_K`. Qdrant returns results already ordered by
   score, and `RetrievalService` preserves that order.
4. **Threshold filter**: if `RETRIEVAL_SCORE_THRESHOLD` is set, results scoring below it are
   dropped; left unset (`None`), no score filtering happens.

Each returned `VectorSearchResult` preserves `document_id`, `chunk_id`, `text`, `source`,
`page_number`, `sheet_name`, and `score`. A failure in either the embedding provider or the
vector store propagates unchanged — `RetrievalService` does not catch or wrap it — and zero
matching results (or all filtered out by the threshold) simply return an empty list rather than
fabricating context. `RetrievalService` never imports or calls `LLMProvider`.

## RAG prompt builder

`RagPromptBuilder` (`app/rag/prompt_builder.py`) is a pure, synchronous, deterministic function
of its inputs: given a question and a list of `VectorSearchResult`s (from `RetrievalService`), it
builds a `BuiltRagPrompt` — **no LLM call, no public chat/SSE endpoint, no retrieval of its own,
no conversation memory**.

`build(question: str, results: list[VectorSearchResult]) -> BuiltRagPrompt`:

1. **Filter**: results with empty/whitespace-only `text` are dropped before anything else — they
   never appear in the context or in `sources`.
2. **No-results path**: if nothing remains after filtering, `context` is set to a fixed sentence
   stating no relevant context was found, `sources` is `[]`, and `user_prompt` is built from that
   same fixed context — deterministic, and no fallback content is fabricated.
3. **Label and format**: otherwise, each remaining result is processed **in the order given**
   (the caller's retrieval rank is preserved, never re-sorted) and assigned a stable label —
   `[S1]`, `[S2]`, ... — used as both the context block's marker and the implicit index into
   `sources`. Each context block is `"{label} {source}[ page {page_number}][ sheet
   {sheet_name}]\n{text}"`, joined with blank lines.
4. **Attribution**: each context block has a matching `PromptSource` (`document_id`, `chunk_id`,
   `source`, `score`, `page_number`, `sheet_name`) appended to `sources` in the same order.

`BuiltRagPrompt` fields:

| Field | Type | Contents |
|---|---|---|
| `system_prompt` | `str` | Fixed instruction: answer only from the supplied context, never invent missing information, say explicitly when the answer isn't present |
| `user_prompt` | `str` | The question plus the formatted `context`, with a closing reminder to answer only from context |
| `context` | `str` | The joined, labeled context blocks (or the fixed no-results sentence) |
| `sources` | `list[PromptSource]` | Attribution metadata per context block, in context order |

`RagPromptBuilder` never mutates the `VectorSearchResult`s or the list passed to `build()`, and
never imports or calls `LLMProvider` or `RetrievalService` itself — it only shapes already-ranked
results into prompt text, leaving the actual retrieval call to whoever composes it with
`RetrievalService`.

## RAG orchestrator

`RagOrchestrator` (`app/rag/orchestrator.py`) is the single component that composes
`RuleBasedRagDecider`, `RetrievalService`, `RagPromptBuilder`, and `LLMProvider` (via
`get_llm_provider()`) into one call — **no conversation memory, and no silent fallback between
decisions or providers**. It's exposed publicly via `POST /api/v1/chat` (see "Streaming chat
endpoint" below).

`stream_answer(question: str) -> AsyncIterator[OrchestratorMetadata | OrchestratorToken]`:

1. **Decide**: `RuleBasedRagDecider.decide(question)` runs first, exactly as it does standalone
   (see "RAG decision layer" above) — no LLM call is made to route.
2. **`CLARIFICATION_NEEDED` / `OUT_OF_SCOPE`**: yields one `OrchestratorMetadata`
   (`retrieval_used=False`, `sources=[]`) then a single fixed `OrchestratorToken` — neither
   `RetrievalService` nor any `LLMProvider` method is called on this path.
3. **`NEEDS_RETRIEVAL`**: calls `RetrievalService.retrieve(question)`, passes the results to
   `RagPromptBuilder.build(question, results)`, yields one `OrchestratorMetadata`
   (`retrieval_used=True`, `sources` from the built prompt), then streams
   `LLMProvider.stream_generate(f"{system_prompt}\n\n{user_prompt}")` chunk by chunk as
   `OrchestratorToken`s.
4. **`DIRECT_LLM`**: yields one `OrchestratorMetadata` (`retrieval_used=False`, `sources=[]`),
   then streams `LLMProvider.stream_generate(...)` directly from a fixed system prompt plus the
   question — no retrieval call.

`OrchestratorMetadata` (`decision`, `reason`, `retrieval_used`, `sources: list[PromptSource]`) is
always the first event of a `stream_answer()` run; every subsequent event is an
`OrchestratorToken(text: str)`, in the exact order the LLM (or the fixed message) produced them.
A failure raised by `RetrievalService.retrieve()` or `LLMProvider.stream_generate()` propagates
unchanged out of the async generator — `RagOrchestrator` does not catch it to substitute a
direct-LLM answer for a failed retrieval, and does not catch a provider failure to retry with a
different provider; `get_llm_provider()`'s existing no-silent-fallback guarantee (see "Provider
factory" above) is preserved end-to-end.

This required extending `LLMProvider`'s abstract contract with `stream_generate(prompt) ->
AsyncIterator[str]` alongside the existing `generate(prompt) -> str` (previously only
`OllamaLLMProvider` exposed streaming). `LLMProviderStub` — the shared base for
`OpenAIProvider`/`GeminiProvider`/`AnthropicProvider` — now raises
`ProviderNotImplementedError` from `stream_generate()` too, keeping the "stub never calls out"
guarantee for both methods.

## RAG Engine Compatibility Layer

`RagOrchestrator` remains the platform's single reference RAG implementation, but
`POST /api/v1/chat` no longer depends on it directly — it depends on a small `RagEngine`
abstraction (`app/rag/engine.py`), so a second, LangChain-backed execution engine can be
selected without touching the public API, the SSE contract, or any existing
provider/retrieval/prompt/orchestration code:

```
RagEngine (app/rag/engine.py)
├── CustomRagEngine    (app/rag/engines/custom_engine.py)   — default
└── LangChainRagEngine (app/rag/engines/langchain_engine.py) — optional
```

**Contract**: `stream_answer(question: str) -> AsyncIterator[OrchestratorMetadata |
OrchestratorToken]`, plus an `answer(question) -> str` default that collects the streamed
tokens. Both concrete engines yield the exact same `OrchestratorMetadata`/`OrchestratorToken`
dataclasses `RagOrchestrator` already defines — `RagEngine` is independent of FastAPI and SSE
formatting, so `app/api/v1/routes/chat.py`'s SSE mapping needs no engine-specific branch.

**Engine-selection flow**: `get_rag_engine(settings)`
(`app/rag/engines/engine_factory.py`) reads `RAG_ENGINE` and resolves the concrete engine — the
same "resolve, don't branch" shape as `provider_factory.py`. `RAG_ENGINE=custom` (the default)
resolves to `CustomRagEngine`; `RAG_ENGINE=langchain` resolves to `LangChainRagEngine`; any other
value raises `UnsupportedRagEngineError` immediately. There is no silent fallback to `custom` and
no silent provider switch — mirroring the **no-fallback rule** `provider_factory.py` already
established for `LLM_PROVIDER`/`EMBEDDING_PROVIDER`/`VECTOR_STORE_PROVIDER`.

**`CustomRagEngine`** adds no logic: it constructs (or accepts) a `RagOrchestrator` and delegates
`stream_answer()` straight to it. `RuleBasedRagDecider`, `RetrievalService`, `RagPromptBuilder`,
`LLMProvider.stream_generate()`, source metadata, and failure propagation are byte-for-byte the
same as before this layer existed — this is why `RagOrchestrator` remains the reference
implementation: every other engine is judged against its behavior, not the other way around.

**`LangChainRagEngine`** (`app/rag/engines/langchain_engine.py`) reuses
`RuleBasedRagDecider.decide(question)` directly, unmodified, *outside* any LangChain `Runnable` —
this keeps the decision contract (which four `RagDecision` values exist, and their `reason`
strings) identical to `CustomRagEngine`'s without needing LangChain to model routing at all.
`CLARIFICATION_NEEDED`/`OUT_OF_SCOPE` stream the same fixed message text `RagOrchestrator` uses
(imported directly from `app.rag.orchestrator`, not duplicated, so the two engines can never
drift apart on that text) with no retrieval and no LLM call. For `NEEDS_RETRIEVAL`/`DIRECT_LLM`,
it builds a LangChain `ChatPromptValue` (from literal `SystemMessage`/`HumanMessage` content —
never an interpolated `ChatPromptTemplate`, so arbitrary document text containing `{`/`}`
characters can never be misparsed as a template variable) and pipes it through a LangChain
`RunnableLambda(...) | ProviderBackedLLM` chain, streaming the result chunk by chunk.

**Adapter boundaries and provider-factory reuse** (`app/rag/engines/langchain_adapters.py`) —
the *only* place `LangChainRagEngine` touches a LangChain provider-facing base class:

- `ProviderBackedLLM` (`langchain_core.language_models.llms.LLM`) streams from whatever
  `LLMProvider` `app.rag.providers.provider_factory.get_llm_provider()` resolved.
- `ProviderBackedEmbeddings` (`langchain_core.embeddings.Embeddings`) wraps whatever
  `EmbeddingProvider` `get_embedding_provider()` resolved.
- `ProviderBackedRetriever` (`langchain_core.retrievers.BaseRetriever`) wraps the existing
  `RetrievalService` — the same `QdrantVectorStore`, `QDRANT_COLLECTION_NAME`,
  `RETRIEVAL_TOP_K`/`RETRIEVAL_SCORE_THRESHOLD` filtering, and embedding provider every other
  caller of `RetrievalService` gets. There is no `langchain-community` Qdrant vector-store
  integration and no second Qdrant SDK path — `QdrantVectorStore`'s own `httpx`-based HTTP calls
  are the only thing that ever talks to Qdrant, in either engine.

None of the three adapters construct an Ollama, OpenAI, Gemini, Anthropic, or Qdrant client
directly — each one is handed an already-resolved provider/service instance and only adapts its
interface, never its configuration or selection.

**Shared Qdrant/embedding contract**: `ProviderBackedRetriever` returns LangChain `Document`s
built from `VectorSearchResult` (`document_to_search_result()`/`_search_result_to_document()` in
`langchain_adapters.py` are exact inverses of each other), and `LangChainRagEngine` converts
retrieved `Document`s straight back into `VectorSearchResult`s before handing them to the
existing, unmodified `RagPromptBuilder`. This means: same embedding model, same `VECTOR_SIZE`,
same `QDRANT_COLLECTION_NAME`, same vectors and payload metadata, same `[S1]`/`[S2]` source
labels and rank order, same governance instructions ("answer only from context", "say so if the
answer isn't present"), and the same Hebrew/Unicode handling as `CustomRagEngine` — nothing about
switching engines re-embeds a document, creates a new collection, or changes a chunk/point ID.

**API/SSE independence**: `app/api/v1/routes/chat.py` depends on `RagEngine` (via
`get_rag_engine()`, a route-local dependency wrapping the factory), not `RagOrchestrator` or
`LangChainRagEngine` — the route has no knowledge of which concrete engine is configured, no
`RAG_ENGINE` branch, and (per "Route Layer Style" in CLAUDE.md) no decision/retrieval/prompt
logic of its own either way.

**No-results behavior is identical across engines**: when `RetrievalService.retrieve()` returns
no results, `RagPromptBuilder` already emits its fixed "no relevant context was found" content
with `sources=[]` — `LangChainRagEngine` never substitutes a `DIRECT_LLM` answer or fabricates a
source in that case, matching `CustomRagEngine` exactly.

**Why LangGraph is intentionally deferred**: LangChain's `Runnable`/prompt/retriever primitives
are sufficient to express this platform's existing four-way decision routing plus a single
retrieval-then-generate step — there is no multi-step agent loop, no tool calling, and no
conversation memory for LangGraph's graph/state machinery to add value to. Introducing LangGraph
now would add a second orchestration paradigm with nothing for it to orchestrate; it belongs in a
future milestone only once a real agentic workflow (multi-step tool use, conditional branching
driven by intermediate LLM output, etc.) actually requires it.

## Streaming chat endpoint

`POST /api/v1/chat` (`app/api/v1/routes/chat.py`) is the first public endpoint that produces an
end-to-end RAG answer. It is deliberately a **thin route**: it validates the request, resolves a
`RagEngine` via a FastAPI dependency (`get_rag_engine()` in the route module, wrapping
`app.rag.engines.engine_factory.get_rag_engine()` — see "RAG Engine Compatibility Layer" above),
and formats `stream_answer()`'s output as Server-Sent Events — it contains no decision,
retrieval, or prompt-building logic, makes no direct call to any provider factory
(`get_embedding_provider()`/`get_vector_store()`/`get_llm_provider()`) or to
`RuleBasedRagDecider`/`RetrievalService`/`RagPromptBuilder`, and does not know or branch on
whether `CustomRagEngine` or `LangChainRagEngine` is configured — those all live inside whichever
`RagEngine` is resolved, which the route only consumes.

- **Request**: `ChatRequest` (`app/schemas/chat.py`) — `{"question": str}` only. A
  `field_validator` rejects an empty/whitespace-only `question`, which FastAPI turns into a
  standard `422` response before the route body ever runs. There is no `model` field: neither
  `RagEngine.stream_answer()` nor `LLMProvider` currently accepts a validated per-request
  model override, so none is exposed — adding one is future work, not a silent gap. The embedding
  model is never client-selectable, matching the existing `OLLAMA_EMBEDDING_MODEL`-is-fixed rule
  (see "LLM provider vs. model" above).
- **Response**: `StreamingResponse(..., media_type="text/event-stream")`, wrapping an async
  generator (`_stream_chat_events`) that iterates `RagOrchestrator.stream_answer(question)` and
  yields each event already SSE-formatted (`event: <name>\ndata: <JSON>\n\n`) — tokens are
  written to the response as the orchestrator produces them, never buffered into one full-text
  response first.
- **Event mapping**: `OrchestratorMetadata` → `metadata` (`decision.value`, `reason`,
  `retrieval_used`, `sources` — each `PromptSource` becomes `document_id`, `chunk_id`, `source`,
  `score`, plus `page_number`/`sheet_name` only when not `None`); `OrchestratorToken` → `token`
  (`text`); normal generator completion → `done` (`status: "completed"`), emitted exactly once
  and only after every token; an exception raised while consuming `stream_answer()` → `error`
  (`message` — a single fixed string, `status: "failed"`), and the stream ends there with no
  `done` event.
- **Error safety**: the `error` event's `message` is always the fixed string `"Failed to
  generate a response."` — the route never serializes an exception's `str()`, so no stack trace,
  prompt text, secret, credential, internal URL (e.g. `OLLAMA_BASE_URL`, `QDRANT_URL`), or raw
  provider response body can leak into a client-visible event.
- **No silent fallback**: because the route does not catch exceptions from individual pipeline
  stages separately — only from the orchestrator's combined stream — a `RetrievalService`
  failure can never be swallowed and silently answered via `DIRECT_LLM`, and a provider failure
  can never be silently retried against a different provider; both simply become the one
  `error` event.
- **Cancellation**: the route catches `Exception`, not `BaseException`, so
  `asyncio.CancelledError` (raised into the generator when a client disconnects mid-stream) is
  never caught as a normal failure — it propagates and lets the ASGI server clean up the
  connection normally, instead of the route trying to write an `error` event to an already-closed
  socket.

```bash
curl -N -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"question":"What does the uploaded document say?"}'
```

## Test architecture

Tests are split into tiers by what they need to run, each with a different speed/fidelity
trade-off:

- **Unit tests** (`tests/*.py`, unmarked/default) — fakes and mocks only (fake sessions, fake
  providers, mocked `httpx` transports); no real database, no real Qdrant, no real network, no
  Docker. Fast (the whole suite runs in well under a second) and always run by `make test`/
  `make verify`.
- **Integration tests** (`tests/integration/*.py`, `@pytest.mark.integration`, auto-applied by
  `tests/integration/conftest.py`) — real, ephemeral Postgres and Qdrant containers started via
  [Testcontainers for Python](https://testcontainers-python.readthedocs.io/), never the
  repository's `docker-compose.yml`, on dynamically assigned ports with no persistent volumes.
  Covers behavior a mock/fake cannot faithfully represent: Alembic migrations against a real
  schema, `IngestionWorker`'s `SELECT ... FOR UPDATE SKIP LOCKED` claim semantics under genuine
  Postgres transaction locking, and Qdrant's actual HTTP request/response contract. AI providers
  stay fake and deterministic even here — no real Ollama container, no model pulled — see
  "AI-provider policy" below. Run via `make test-integration`/`make verify-integration`, never as
  part of `make test`/`make verify`.
- **Backend E2E tests** (`tests/e2e/backend/*.py`, `@pytest.mark.e2e`, auto-applied by
  `tests/e2e/backend/conftest.py`) — exercises the complete backend user flow through real HTTP:
  document upload → real `IngestionWorker` processing (extraction, chunking, Qdrant upsert) →
  retrieval/orchestration → the streaming chat SSE endpoint, consumed incrementally so event
  order/timing is genuinely exercised rather than inspected as one buffered string. Runs the real
  FastAPI app behind a real ASGI HTTP client (`httpx.AsyncClient` + `ASGITransport`), against its
  own ephemeral Testcontainers-managed Postgres and Qdrant — never `docker-compose.yml`, never
  fixed ports, with an isolated database and Qdrant collection per test. AI providers stay fake
  and deterministic here too — no real Ollama container, no model pulled — see "AI-provider
  policy" below. Run via `make test-e2e-backend`/`make verify-e2e-backend`, never as part of
  `make test`/`make verify`, and not added to the pre-commit hook.
- **Frontend E2E tests** — future milestone; no frontend exists yet in this repository.
- **Real-AI smoke tests** — future milestone, kept deliberately separate from the unit,
  integration, and backend E2E suites: a small, manual/nightly suite that runs against a real
  Ollama container with real models pulled, to catch drift in actual model behavior/output shape
  without paying that cost (container pull time, model pull time, non-determinism) on every
  commit.

**Local development** (running the app, trying it end-to-end by hand) continues to use
`docker-compose.yml` exactly as before — nothing about that workflow changes. Tests, in any tier,
must never depend on `docker-compose.yml` being up or on any state it created; the integration and
backend E2E suites' fixtures (`tests/integration/conftest.py`, `tests/e2e/backend/conftest.py`)
start their own containers from scratch every session and guard against ever pointing at a
production `APP_ENV`/`DATABASE_URL`/`QDRANT_URL`.

The RAG engine compatibility layer's tests span all three tiers above rather than forming a
separate tier of their own: `tests/test_rag_engine_factory.py`/`test_custom_rag_engine.py`/
`test_langchain_rag_engine.py`/`test_langchain_adapters.py` are unit tests,
`tests/integration/test_langchain_rag_engine_integration.py` is an integration test (real
ephemeral Qdrant, existing `VectorPoint` format, fake embeddings), and
`tests/e2e/backend/test_rag_engine_parity.py` runs the full backend E2E flow under both
`RAG_ENGINE=custom` and `RAG_ENGINE=langchain`, comparing source IDs/ranking/metadata/SSE
ordering/error/no-results behavior for equivalence (the generated answer text may legitimately
differ). `make test-rag-engines`/`make verify-rag-engines` run just these files across all three
tiers as a convenience — they are not a substitute for `make verify`/`verify-integration`/
`verify-e2e-backend`, which still cover everything including this layer.

### AI-provider policy in tests

No tier pulls or calls a real LLM/embedding model. Unit tests use hand-written fake provider
doubles (see e.g. `tests/test_retrieval_service.py`, `tests/test_rag_orchestrator.py`). The
integration suite's one end-to-end pipeline test
(`tests/integration/test_ingestion_worker_postgres.py`) runs the real `IngestionWorker` default
pipeline against real Postgres and real Qdrant, but with `get_embedding_provider` monkeypatched
to a small fixed-vector fake. The backend E2E suite goes one step further and exercises the real
HTTP/chat surface too, with `FakeEmbeddingProvider` (deterministic bag-of-words hashing, so a
query genuinely matches its relevant indexed chunks under Qdrant's real cosine search) and
`FakeStreamingLLMProvider`/`FakeFailingLLMProvider` (`tests/e2e/backend/fakes.py`) swapped in by
monkeypatching the provider-factory function each consuming module already imports — never a
branch on `APP_ENV` in production code. Real Ollama stays entirely outside all three suites,
reserved for the future real-AI smoke suite described above.

## Operational Health Contract

`app/api/routes/health.py` exposes four **unversioned** endpoints — `GET /health`,
`/health/live`, `/health/ready`, `/health/dependencies` — registered on `app` with **no
`/api/v1` prefix**. This is deliberate: business API versioning (`/api/v1`, and any future
`/api/v2`) is about the shape of request/response contracts for clients of the RAG features;
operational health is a different, version-independent contract consumed by infrastructure —
Kubernetes probes, load balancers, ArgoCD, monitoring/alerting — that must never need to change
just because the business API moved to a new version. Moving these endpoints under a versioned
prefix later would break every external prober pointed at them; see the standing rule in
[CLAUDE.md](CLAUDE.md).

**Why four separate endpoints, not one**: each answers a different operational question and is
polled at different rates by different consumers.

- **`GET /health`** — "is the process up." A static, zero-dependency summary
  (`status`/`service`/`version`), for a human or a very cheap uptime check.
- **`GET /health/live`** (liveness) — "is the process alive and not deadlocked." Never calls
  Postgres, Redis, Qdrant, or Ollama. This is what a Kubernetes `livenessProbe` should point at:
  if it ever returns non-200 or times out, the pod should be restarted — but a downstream
  dependency being temporarily down is *not* a reason to restart this process, so liveness must
  stay independent of every external service.
- **`GET /health/ready`** (readiness) — "can this instance actually serve traffic right now."
  Calls `app/services/platform_health.get_readiness_result()`, which runs every check and
  returns `200` only if every **required** check passes, else `503`. This is what a Kubernetes
  `readinessProbe` and a load balancer's health check should point at: `503` here means "stop
  routing traffic to this instance," not "restart it" — `live` can (and often will) stay `200`
  while `ready` is `503` (e.g. Qdrant is temporarily unreachable but the process itself is fine).
- **`GET /health/dependencies`** — the same checks as readiness, but always returns `200` with
  the full per-dependency detail in the body (`status` per check, `required` per check, a safe
  `detail` string on failure). Intended for monitoring/alerting dashboards and human debugging,
  not for gating traffic — that's what `/health/ready`'s HTTP status code is for.

**Dependency/readiness semantics** — `app/services/platform_health.py`:

| Check | Method | Required for readiness? |
|---|---|---|
| `postgres` | `SELECT 1` via a short-lived async engine | Yes |
| `redis` | `PING` via `redis.asyncio` | No — no application code path reads/writes Redis yet |
| `qdrant` | `GET /collections` (same reachability check `create_collection_if_not_exists` uses) | Yes |
| `ollama` | Reuses `OllamaClient.check_health()` (reachability) | Yes |
| `ollama_chat_model` | Same call, `chat_model_available` | Yes |
| `ollama_embedding_model` | Same call, `embedding_model_available` | Yes |

Every check runs concurrently (`asyncio.gather`) with its own `CHECK_TIMEOUT_SECONDS` (3s)
timeout, wrapping `asyncio.timeout(...)` around the actual I/O — no automatic retries beyond
that timeout, and no check ever mutates or restarts the dependency it's probing (a `SELECT 1`, a
`PING`, a `GET`, nothing else). A failed check is never silently dropped: `run_all_checks()`
always returns one `DependencyCheckResult` per check, and both `/health/ready` and
`/health/dependencies` surface every result. `redis` is checked and reported everywhere (so
observability doesn't lose it) but is `required=False`, so a down Redis alone can never flip
readiness to `503` — reflecting that nothing in this codebase actually depends on it yet (see the
environment variable table below); this is a deliberate, documented choice, not an oversight, and
should be revisited the moment any code path starts using `REDIS_URL`.

Every `DependencyCheckResult`'s `detail` is a fixed, generic string per failure mode (e.g.
`"Postgres is unreachable."`) — none of the checks ever return a raw exception message, a
connection string, a credential, or a provider's raw response body to the client.

**Thin-controller route, aggregation in the service layer**: `app/api/routes/health.py`'s
`readiness`/`dependencies` handlers do only three things — resolve `Settings` via `Depends`, call
one function in `app/services/platform_health.py`, and apply the status code / return the body
that function already produced. All required-check filtering, failed-check calculation, overall
status calculation, and safe error-summary construction live in the service module as pure,
synchronous, directly-unit-testable functions:

- `build_readiness_result(checks) -> ReadinessResult` — `ReadinessResult` is a small dataclass
  (`response: ReadinessResponse`, `status_code: int`) so the route never computes `200`/`503`
  itself, it only copies a value the service already decided.
- `build_dependencies_response(checks) -> DependenciesResponse`
- `get_readiness_result(settings)`/`get_dependencies_response(settings)` — thin async wrappers
  that call `run_all_checks(settings)` then delegate to the two functions above; these are what
  the route actually calls.

This mirrors how `POST /api/v1/chat` (see "Streaming chat endpoint" above) stays a thin route
over `RagOrchestrator` — routes handle HTTP concerns (validation, dependency injection, status
codes, response shape) and delegate everything else to a service; see the standing rule in
[CLAUDE.md](CLAUDE.md).

**Future DevOps consumers** this contract is designed for (none wired up yet in this repository):
Kubernetes liveness/readiness probes, load balancer health checks, ArgoCD rollout health checks
(a rollout can gate on `/health/ready` before shifting traffic to a new revision),
monitoring/alerting systems polling `/health/dependencies` for per-dependency status, and a
future backend E2E suite's own startup check (poll `/health/ready` before running E2E tests
against a freshly started stack, instead of a fixed sleep).

**Legacy**: `GET /api/v1/health` (`app/api/v1/routes/health.py`, `HealthResponse`) is unchanged
and still works — kept for backward compatibility with any existing client — but is superseded by
`GET /health` for anything operational going forward.

## Future LLM provider stubs

`OpenAIProvider`, `GeminiProvider`, and `AnthropicProvider`
(`app/rag/providers/{openai,gemini,anthropic}_provider.py`) are explicit placeholders for
providers with no real implementation yet. Each implements `LLMProvider` via a shared base,
`LLMProviderStub` (`app/rag/providers/llm_provider_stub.py`), whose `generate()` always raises
`ProviderNotImplementedError` (`app/rag/providers/errors.py`) with a message naming the provider
— they make no HTTP calls and read no external API keys. `get_llm_provider()` raises the same
error immediately when `LLM_PROVIDER` names one of these, before any provider object does
anything, so a misconfigured provider fails loudly at resolution time rather than silently
falling back to Ollama or failing later on first use.

Adding a new stub for another future provider means: create a class inheriting
`LLMProviderStub` with its own `NOT_IMPLEMENTED_MESSAGE`, and add it to `_LLM_STUBS` in
`provider_factory.py`. See [CLAUDE.md](CLAUDE.md) for the standing rule on how stubs must behave.

## Docker Compose topology

All services join the default Compose network (`documents-rag_default`) and address each other by
service name (Docker's embedded DNS). The app reaches its dependencies at:

- `postgres:5432`
- `redis:6379`
- `qdrant:6333` (HTTP)
- `ollama:11434` (HTTP)

Only the ports needed for host-side debugging are published (`8000`, `5432`, `6379`, `6333`,
`11434`). In a production deployment, only `app`'s port would typically be exposed.

```
host:8000 ──► app ──► postgres:5432
                 ├──► redis:6379
                 ├──► qdrant:6333
                 └──► ollama:11434
```

## Environment variables

Set via `docker-compose.yml` for containers, or `.env` (copy from `.env.example`) for local runs
outside Docker. `app/core/config.py` (`Settings`) is the single source of truth for defaults.

| Variable                  | Default                                                              | Notes |
|----------------------------|-----------------------------------------------------------------------|-------|
| `APP_ENV`                 | `local`                                                                | Echoed by the legacy `GET /api/v1/health`'s `environment` field |
| `LOG_LEVEL`                | `INFO`                                                                 | Not yet wired to a logger |
| `DATABASE_URL`             | `postgresql+asyncpg://postgres:postgres@postgres:5432/rag_db`         | Async SQLAlchemy engine; stores `documents`/`ingestion_jobs`; also `SELECT 1`-checked by `GET /health/ready`/`/health/dependencies` |
| `REDIS_URL`                | `redis://redis:6379/0`                                                | Not yet consumed by any business code path; `PING`-checked (not required) by `GET /health/ready`/`/health/dependencies` |
| `QDRANT_URL`               | `http://qdrant:6333`                                                  | Used by `QdrantVectorStore` for collection/upsert/search; also checked by `GET /health/ready`/`/health/dependencies` |
| `OLLAMA_BASE_URL`          | `http://ollama:11434`                                                 | Used by `OllamaClient` for health/model checks (also reused by `GET /health/ready`/`/health/dependencies`) |
| `OLLAMA_CHAT_MODEL`        | `llama3.1`                                                             | Checked for availability; backward-compatible fallback for `LLM_MODEL` if unset |
| `OLLAMA_EMBEDDING_MODEL`   | `nomic-embed-text`                                                     | Checked for availability; always used by `OllamaEmbeddingProvider` — fixed, not selectable via `LLM_MODEL` |
| `LLM_PROVIDER`             | `ollama`                                                               | Selects the `LLMProvider` implementation; `openai`/`gemini`/`anthropic` are recognized stubs |
| `LLM_MODEL`                | *(unset)*                                                              | Selects the model `OllamaLLMProvider` uses; falls back to `OLLAMA_CHAT_MODEL` if unset (see "LLM provider vs. model") |
| `EMBEDDING_PROVIDER`       | `ollama`                                                               | Selects the `EmbeddingProvider` implementation via the provider factory |
| `VECTOR_STORE_PROVIDER`    | `qdrant`                                                               | Selects the `VectorStore` implementation via the provider factory |
| `CHUNK_SIZE`               | `1000`                                                                 | Target chunk size in characters, used by `DocumentChunker` |
| `CHUNK_OVERLAP`            | `200`                                                                  | Overlap between consecutive chunks in characters, used by `DocumentChunker` |
| `QDRANT_COLLECTION_NAME`   | `documents`                                                            | Collection `IngestionWorker` creates (if missing) and upserts chunk vectors into |
| `VECTOR_SIZE`              | `768`                                                                  | Vector dimensionality passed to `create_collection_if_not_exists` — must match the embedding provider's output size (`nomic-embed-text` produces 768-dim vectors) |
| `RETRIEVAL_TOP_K`          | `5`                                                                    | Default number of results `RetrievalService.retrieve()` asks Qdrant for, when no explicit `limit` is passed |
| `RETRIEVAL_SCORE_THRESHOLD`| *(unset)*                                                              | Minimum Qdrant score a result must meet to be returned; unset/`null` disables score filtering |
| `RAG_ENGINE`               | `custom`                                                               | Selects the `RagEngine` implementation via `get_rag_engine()` (see "RAG Engine Compatibility Layer"); `langchain` is the only other recognized value — anything else raises `UnsupportedRagEngineError` |

## Current boundaries

- `app/api/routes` — **unversioned** operational routes: `GET /health`, `/health/live`,
  `/health/ready`, `/health/dependencies` (see "Operational Health Contract" above). Registered
  on `app` with no prefix — never move these under `app/api/v1`.
- `app/api/v1/routes` — versioned business API routers: `GET /health` (legacy, see "Operational
  Health Contract" above), `GET /providers/ollama/health`, `POST /documents` (see "Document
  upload and ingestion job skeleton" above), and `POST /chat` (see "Streaming chat endpoint"
  above) — the only router that depends on `app/rag`.
- `app/core` — configuration and cross-cutting concerns, plus `version.py` (`SERVICE_NAME`/
  `SERVICE_VERSION` — the single source of truth for both the FastAPI app's own metadata and the
  unversioned platform health responses).
- `app/db` — SQLAlchemy async engine/session setup.
- `app/models` — ORM models: `Document`, `IngestionJob`/`IngestionStatus` (see "Document upload
  and ingestion job skeleton" above).
- `app/schemas` — Pydantic request/response schemas.
- `app/services` — business logic layer: `OllamaClient` (`app/services/ollama_client.py`), a thin
  async HTTP client scoped strictly to reachability and model-availability checks — it
  intentionally does not call generation or embedding endpoints; `LocalFileStorage`
  (`app/services/local_file_storage.py`), which saves uploaded files to local disk under a
  generated safe filename (see "Document upload and ingestion job skeleton" above); and
  `IngestionWorker` (`app/services/ingestion_worker.py`), which claims and resolves pending
  ingestion jobs (see "Ingestion worker" above) — no public API — and whose default pipeline now
  calls the embedding/vector-store providers (see "Chunk embedding and Qdrant indexing" above),
  the only place in this layer that does; `DocumentTextExtractor`
  (`app/services/document_text_extractor.py`), which extracts text from a document's stored
  `.txt`/`.md`/`.pdf`/`.docx`/`.xlsx` file (see "Document text extraction" above); and
  `DocumentChunker` (`app/services/document_chunker.py`), which splits an `ExtractedDocument`
  into `DocumentChunk`s (see "Document chunking" above); and `platform_health.py`, the dependency
  checks backing `GET /health/ready`/`/health/dependencies` (see "Operational Health Contract"
  above) — reuses `OllamaClient` for the Ollama check rather than duplicating it.
- `app/rag/retrieval_service.py` — `RetrievalService`, the internal read-side counterpart to
  ingestion's embed/upsert steps (see "Retrieval service" above). It is the second caller of
  `get_embedding_provider()`/`get_vector_store()` alongside `IngestionWorker`, and it never calls
  `LLMProvider`.
- `app/rag/prompt_builder.py` — `RagPromptBuilder`, `BuiltRagPrompt`, `PromptSource` (see "RAG
  prompt builder" above). Pure and synchronous — it calls no provider at all (not even
  `get_embedding_provider()`/`get_vector_store()`), consuming only the `VectorSearchResult`s a
  caller already obtained from `RetrievalService`.
- `app/rag/providers` — abstract interfaces for embedding, LLM, and vector store providers, a
  `provider_factory.py` that resolves the configured implementation for each (see "Provider
  factory" above), and three concrete implementations:
  - `OllamaEmbeddingProvider` (`app/rag/providers/ollama_embedding_provider.py`) — calls
    `POST /api/embeddings` for `OLLAMA_EMBEDDING_MODEL` only.
  - `OllamaLLMProvider` (`app/rag/providers/ollama_llm_provider.py`) — calls
    `POST /api/generate` with `stream=true` for `Settings.resolved_llm_model`
    (`LLM_MODEL`, falling back to `OLLAMA_CHAT_MODEL`), implementing both abstract `LLMProvider`
    methods: `stream_generate(prompt) -> AsyncIterator[str]` (yields text chunks as Ollama
    streams them) and `generate(prompt) -> str` (joins the streamed chunks). Internal-only — no
    ingestion, no Qdrant writes, no public chat/SSE endpoint.
  - `QdrantVectorStore` (`app/rag/providers/qdrant_vector_store.py`) — calls Qdrant's HTTP API
    for collection create/upsert/search only (see "Vector store" above). Internal-only — no
    document upload, no chat/SSE endpoint, no full RAG flow; `IngestionWorker` (write side) and
    `RetrievalService` (read side) are its only callers so far.

  `LLMProvider` (`app/rag/providers/llm_provider.py`) declares both `generate(prompt) -> str` and
  `stream_generate(prompt) -> AsyncIterator[str]` as abstract methods — every implementation,
  including the future-provider stubs, must implement both.

  Three future-provider stubs also exist — `OpenAIProvider`, `GeminiProvider`,
  `AnthropicProvider` (`app/rag/providers/{openai,gemini,anthropic}_provider.py`) — which
  implement `LLMProvider` but always raise `ProviderNotImplementedError` from both `generate()`
  and `stream_generate()` (see "Future LLM provider stubs" above).

  `OllamaClient` (health checks) is deliberately kept separate from these provider interfaces so
  health checks don't get entangled with the generation/embedding/storage contracts.
- `app/rag/decision.py` — the RAG decision layer (see "RAG decision layer" above): `RagDecision`,
  `DecisionResult`, `RuleBasedRagDecider`. Separate from `app/rag/providers` since it doesn't call
  any provider itself — it only classifies a question.
- `app/rag/orchestrator.py` — `RagOrchestrator`, `OrchestratorMetadata`, `OrchestratorToken` (see
  "RAG orchestrator" above). The only component that composes the decision layer, retrieval
  service, prompt builder, and LLM provider together — no other module in `app/rag` calls more
  than one of them.
- `app/rag/engine.py` — the `RagEngine` abstraction (see "RAG Engine Compatibility Layer" above).
- `app/rag/engines/` — concrete `RagEngine` implementations: `custom_engine.py`
  (`CustomRagEngine`, the default, wrapping `RagOrchestrator`), `langchain_engine.py`
  (`LangChainRagEngine`, optional), `langchain_adapters.py` (`ProviderBackedLLM`/
  `ProviderBackedEmbeddings`/`ProviderBackedRetriever`), and `engine_factory.py`
  (`get_rag_engine()`, `UnsupportedRagEngineError`).
- `app/workers` — background job placeholders.
- `tests/integration/` — the Testcontainers-based integration suite (see "Test architecture"
  above): `conftest.py` (ephemeral Postgres/Qdrant fixtures, the production-environment guard,
  the Alembic-migration helpers), `test_alembic_migrations.py`, `test_ingestion_worker_postgres.py`,
  `test_qdrant_vector_store_integration.py`, `test_langchain_rag_engine_integration.py`. Entirely
  separate from `tests/*.py` (unit tests); auto-marked `@pytest.mark.integration` and excluded
  from `make test`/`make verify`.

## What is intentionally not implemented yet

- A standalone public retrieval endpoint — `RetrievalService` is only reachable indirectly, via
  `POST /api/v1/chat`'s `NEEDS_RETRIEVAL` path; there's no endpoint that returns raw
  `VectorSearchResult`s on their own
- Persisting extracted text or chunks in Postgres — `DocumentChunker`'s output is only persisted
  as vectors in Qdrant (via the embedding/upsert step); there's no relational table for chunk text
- Anything that continuously runs `IngestionWorker.process_next_job()` in a loop (no scheduler
  or long-running process invokes it yet — it's called directly, one job at a time)
- A public API endpoint for embeddings, vector store, chunking, prompt-building, or
  decision-layer operations on their own (all internal-only; only reachable indirectly through
  `POST /api/v1/chat`)
- An LLM-based (as opposed to rule-based) question router
- Conversation memory / multi-turn context (in prompt building or the orchestrator)
- A client-selectable model override on `POST /api/v1/chat` — `ChatRequest` has no `model` field
- MinIO / any S3-compatible object storage, and a `FileStorage` provider factory —
  `LocalFileStorage` remains the only implementation
- Frontend E2E tests — no frontend exists yet in this repository
- A real-Ollama smoke suite — deliberately kept separate/manual/nightly, not part of the default
  integration run (see "Test architecture" above)
- LangGraph — the LangChain compatibility layer uses only `langchain-core`'s
  `Runnable`/prompt/retriever primitives; see "Why LangGraph is intentionally deferred" under "RAG
  Engine Compatibility Layer" above
- Agents, tool calling, and any LangChain/LangGraph agent packages — neither `RagEngine`
  implementation exposes tool use; both are a fixed decide-then-generate flow
- A LangChain-specific ingestion path, a second Qdrant SDK/collection, or a client-selectable
  `RAG_ENGINE` override — `RAG_ENGINE` is a server-side deployment setting, never a per-request
  parameter
- Kubernetes manifests, Helm charts, ArgoCD Application/Rollout resources, or any
  monitoring/alerting configuration that actually consumes the new `/health/*` endpoints — this
  milestone only establishes the operational health *contract* those future consumers would use
- Auth, rate limiting, an observability/logging pipeline, and Redis actually being used for
  anything (it is only `PING`-checked, not read from or written to, by any code path)

These land in later milestones once the infrastructure is confirmed stable.
