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
| `APP_ENV`                 | `local`                                                                | Echoed by `/health` |
| `LOG_LEVEL`                | `INFO`                                                                 | Not yet wired to a logger |
| `DATABASE_URL`             | `postgresql+asyncpg://postgres:postgres@postgres:5432/rag_db`         | Async SQLAlchemy engine; stores `documents`/`ingestion_jobs` |
| `REDIS_URL`                | `redis://redis:6379/0`                                                | Not yet consumed |
| `QDRANT_URL`               | `http://qdrant:6333`                                                  | Used by `QdrantVectorStore` for collection/upsert/search |
| `OLLAMA_BASE_URL`          | `http://ollama:11434`                                                 | Used by `OllamaClient` for health/model checks |
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

## Current boundaries

- `app/api` — FastAPI routers: `/health`, `/providers/ollama/health`, and `POST /documents`
  (see "Document upload and ingestion job skeleton" above).
- `app/core` — configuration and cross-cutting concerns.
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
  into `DocumentChunk`s (see "Document chunking" above).
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
    (`LLM_MODEL`, falling back to `OLLAMA_CHAT_MODEL`), exposing
    `stream_generate(prompt) -> AsyncIterator[str]` (yields text chunks as Ollama streams them)
    and `generate(prompt) -> str` (joins the streamed chunks, satisfying the abstract
    `LLMProvider` contract). Internal-only — no ingestion, no Qdrant writes, no public chat/SSE
    endpoint.
  - `QdrantVectorStore` (`app/rag/providers/qdrant_vector_store.py`) — calls Qdrant's HTTP API
    for collection create/upsert/search only (see "Vector store" above). Internal-only — no
    document upload, no chat/SSE endpoint, no full RAG flow; `IngestionWorker` (write side) and
    `RetrievalService` (read side) are its only callers so far.

  Three future-provider stubs also exist — `OpenAIProvider`, `GeminiProvider`,
  `AnthropicProvider` (`app/rag/providers/{openai,gemini,anthropic}_provider.py`) — which
  implement `LLMProvider` but always raise `ProviderNotImplementedError` (see "Future LLM
  provider stubs" above).

  `OllamaClient` (health checks) is deliberately kept separate from these provider interfaces so
  health checks don't get entangled with the generation/embedding/storage contracts.
- `app/rag/decision.py` — the RAG decision layer (see "RAG decision layer" above): `RagDecision`,
  `DecisionResult`, `RuleBasedRagDecider`. Separate from `app/rag/providers` since it doesn't call
  any provider itself — it only classifies a question.
- `app/workers` — background job placeholders.

## What is intentionally not implemented yet

- A public retrieval endpoint — `RetrievalService` can embed a query and search Qdrant
  internally, but nothing exposes it over the API yet
- Persisting extracted text or chunks in Postgres — `DocumentChunker`'s output is only persisted
  as vectors in Qdrant (via the embedding/upsert step); there's no relational table for chunk text
- Anything that continuously runs `IngestionWorker.process_next_job()` in a loop (no scheduler
  or long-running process invokes it yet — it's called directly, one job at a time)
- A public chat/query endpoint (including SSE streaming to clients)
- A public API endpoint for embeddings, vector store, chunking, retrieval, prompt-building, or
  decision-layer operations (all internal-only)
- An LLM-based (as opposed to rule-based) question router
- Any actual LLM call using a `BuiltRagPrompt` — `RagPromptBuilder` only shapes the prompt text
- Conversation memory / multi-turn context in prompt building
- Any pipeline wiring the decision layer, LLM generation, retrieval, and prompt building into a
  full RAG flow
- Auth, rate limiting, observability/logging pipeline

These land in later milestones once the infrastructure is confirmed stable.
