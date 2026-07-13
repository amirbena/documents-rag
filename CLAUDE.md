# CLAUDE.md

Working guide for Claude Code in this repository. Read this before making changes.

## Project context

`documents-rag` is a production-style **local RAG (Retrieval-Augmented Generation) learning and
portfolio project**. It is built incrementally, milestone by milestone, to demonstrate clean
architecture, correct local infrastructure (Docker Compose, Postgres, Redis, Qdrant, Ollama), and
disciplined engineering practice — not to ship a finished product quickly. Favor clarity and
correctness over speed or cleverness.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the system design and [README.md](README.md) for how
to run and test it.

## Working rules

1. **Documentation stays in sync with code.** Any meaningful code change (new module, changed
   behavior, new service, new config) must come with a matching documentation update in the same
   change — not deferred to "later."
2. **Docstrings are required, not optional.** Every module gets a concise module-level docstring
   explaining its responsibility. Every public class and function gets a short docstring. Skip
   docstrings only for trivial one-line helpers or `__init__.py` re-exports. Do not write
   multi-paragraph docstrings — one or two lines is enough.
3. **Architecture docs must reflect the real implementation.** If a change adds, removes, or
   rewires a service, endpoint, environment variable, or provider, update
   [ARCHITECTURE.md](ARCHITECTURE.md) to match. Never let it describe something that no longer
   exists or omit something that now does.
4. **Ship small, incremental milestones.** Prefer one clear, scoped, verifiable change over a
   large bundled change. Do not implement future milestones early "while you're in there" — stick
   to what was asked.
5. **Quality gates must pass before a change is considered done.** Run `make verify` before
   finishing any implementation task, and always before pushing/opening a PR — it runs, in
   order, stopping at the first failure:
   - `make test` (`pytest -q`)
   - `make lint` (`ruff check .`)
   - `make typecheck` (`mypy app`)
   - `make compose` (`docker compose config`)

   If `make` isn't available, run the four underlying commands individually in that order as a
   fallback. All four must pass cleanly either way. If one fails, fix the underlying issue rather
   than skipping or loosening the gate.

   Installing the pre-commit hook (`./scripts/install-git-hooks.sh`) makes `make verify` run
   automatically on every commit — this enforces the gate mechanically, but it does not replace
   the responsibility above: run `make verify` explicitly before finishing a task or opening a
   PR regardless of whether the hook is installed locally. Never bypass the hook (`--no-verify`)
   or any other quality gate unless the user explicitly asks for it.

## Function Documentation

- Every public function and public method gets a concise one-line docstring stating its intent —
  what it's for, not how it works. Don't restate the signature or implementation.
- Keep it to one line. If you need more than one line to explain intent, the function is probably
  doing too much.
- Trivial private helpers (`_helper`, single-line internal utilities) don't need a docstring unless
  their behavior is non-obvious from the name and signature alone.

Example:

```python
def get_settings() -> Settings:
    """Return the cached application settings."""
```

## Provider Stubs

- **Future provider stubs are allowed.** A placeholder class for a provider we intend to support
  later (e.g. `OpenAIProvider`, `GeminiProvider`, `AnthropicProvider`) may be added ahead of its
  real implementation, so the provider factory and config have a place for it to land.
- **Stubs must not silently call external APIs.** No HTTP calls, no SDK calls, no reading external
  API keys "just in case" — a stub does nothing except fail clearly.
- **Stubs must fail explicitly until implemented.** Every method on a stub raises a clear,
  named error (e.g. `ProviderNotImplementedError("<Provider> provider is not implemented yet.")`)
  rather than returning empty/default data or silently no-op'ing.
- **The backend must never silently fall back to Ollama when another provider is configured.**
  If `LLM_PROVIDER`/`EMBEDDING_PROVIDER`/`VECTOR_STORE_PROVIDER` names a provider other than
  Ollama, the factory must resolve to that provider (real or stub) or raise a clear configuration
  error — it must never quietly substitute the Ollama implementation instead.

## Provider vs. Model Configuration

- **Keep "which provider" and "which model" as separate settings.** `LLM_PROVIDER` selects the
  backend (e.g. `ollama`); `LLM_MODEL` selects which model that backend uses (e.g. `llama3.1`).
  Never conflate the two into a single setting — changing the model must not require touching
  provider selection, and vice versa.
- **Preserve backward compatibility when introducing a new setting that supersedes an old one.**
  `LLM_MODEL` falls back to the older `OLLAMA_CHAT_MODEL` when unset
  (`Settings.resolved_llm_model`), so existing `.env` files keep working. Apply this same
  fallback pattern for future renames instead of a breaking cutover.
- **Don't extend model selection to embeddings.** `OLLAMA_EMBEDDING_MODEL` stays fixed and is
  not user-selectable via `LLM_MODEL` or any similar mechanism — swapping the embedding model
  would silently invalidate previously computed vectors, so it requires a deliberate, separate
  migration, not a config flag.

## Provider Implementation Style

- **Prefer calling a provider's HTTP API directly over its official SDK**, unless asked
  otherwise. Both `OllamaEmbeddingProvider`/`OllamaLLMProvider` and `QdrantVectorStore` call raw
  REST endpoints via `httpx` rather than pulling in `ollama`'s client library or `qdrant-client`.
  This keeps dependencies minimal and behavior fully visible/testable via mocked `httpx`
  transports instead of SDK-specific mocking.

## Storage Abstraction Style

- **Ingestion/upload/extraction/re-index code depends only on `FileStorage`
  (`app/storage/contract.py`), never on `LocalFileStorage`/`MinioFileStorage` concretely, and
  never on a filesystem path or a MinIO SDK type.** `app/api/v1/routes/documents.py`,
  `app/services/document_upload_service.py`, `app/services/ingestion_worker.py`,
  `app/services/document_text_extractor.py`, and `app/services/reindex_service.py` all take a
  `FileStorage` as a constructor/function parameter (or resolve one via
  `app/storage/factory.py`'s `create_file_storage()`) — never instantiate a concrete storage
  class themselves, and never branch on `settings.file_storage_provider`.
- **Provider SDK types never leave a storage adapter.** MinIO SDK exceptions, `urllib3` response
  objects, and local filesystem exceptions (`OSError`, `FileNotFoundError`, etc.) are translated
  to the `app.storage.errors.StorageError` hierarchy at the `LocalFileStorage`/`MinioFileStorage`
  boundary, preserving the original exception as `__cause__` — no other module ever catches or
  re-raises an SDK-specific exception type.
- **Object keys are provider-neutral and application-generated.** `app/storage/keys.py`'s
  `generate_object_key()` is the only place a new key is created; a storage provider (especially
  `MinioFileStorage`) never invents its own key. `validate_object_key()` rejects absolute paths
  and `..` traversal before any provider call.
- **Presigned/download URLs are never persisted as a document's identity.** The persisted
  identity is always `Document.storage_provider`/`storage_bucket`/`storage_key` — never a
  `generate_download_url()` result, which is time-limited (MinIO) or an internal-only
  representation (local) and must be regenerated on demand, never stored or logged.
- **Storage and PostgreSQL are non-atomic — document this precisely, never claim otherwise.**
  `app/services/document_upload_service.py`'s `upload_document()` saves the object before
  persisting/committing the `Document`/`IngestionJob` rows; a commit failure after a successful
  save triggers a best-effort object delete (failure there is logged, never hidden, and never
  replaces the original DB exception, which always propagates unchanged).
- **A new storage provider must pass the same contract-level test suite** the existing
  implementations do (see `tests/test_local_file_storage.py`/`tests/test_minio_file_storage.py`
  and their Testcontainers integration counterparts) before being wired into
  `create_file_storage()` — never merged with only implementation-specific tests.
- **Local-path assumptions must never re-enter ingestion/extraction code.**
  `DocumentTextExtractor` reads bytes via `FileStorage.read()` and parses them in memory
  (`io.BytesIO`) — never `Path(...)`/`open(...)` against a document's storage location. If a
  future parser genuinely requires a real filesystem path, introduce an explicit, narrowly-scoped
  temporary-materialization boundary rather than leaking a path assumption into the extractor
  itself.
- **Documentation must stay consistent whenever the storage contract, providers, object-key
  strategy, or persisted storage identity change.** A change to `app/storage/`, `Document`'s
  storage-identity columns, or `FILE_STORAGE_PROVIDER`/`MINIO_*`/`LOCAL_STORAGE_ROOT` settings
  must come with a matching update to "Storage Abstraction (Phase 2.6/2.7)" in
  [ARCHITECTURE.md](ARCHITECTURE.md) and the "Storage abstraction" section of
  [README.md](README.md) — in the same change, not deferred.
- **Do not silently broaden storage scope beyond Phase 2.6/2.7.** No document-deletion endpoint,
  lifecycle/download/listing API, orphan-object cleanup worker, hash-based deduplication, AWS
  S3/Cloudflare R2 implementation, or presigned multipart upload support belongs in this layer
  until a future phase explicitly adds it — `MinioFileStorage`'s `delete()`/
  `generate_download_url()` exist to satisfy the `FileStorage` contract, not because a route
  calls them yet.

## RAG Engine Compatibility Style

- **`CustomRagEngine` (wrapping `RagOrchestrator`) remains the default and reference RAG engine.**
  `RAG_ENGINE` defaults to `custom`; every other `RagEngine` implementation is judged against its
  behavior, not the other way around. Never change this default without an explicit user request.
- **A LangChain-based (or any alternative) RAG engine must be adapter-based**, not a rewrite.
  It must reuse `RuleBasedRagDecider`/`RetrievalService`/`RagPromptBuilder` and the existing
  provider factory (`app/rag/providers/provider_factory.py`) — see `app/rag/engines/
  langchain_adapters.py` (`ProviderBackedLLM`/`ProviderBackedEmbeddings`/`ProviderBackedRetriever`)
  for the established pattern: wrap an already-resolved provider/service instance, never
  reimplement its logic.
- **An alternative RAG engine must never bypass the provider factory.** It must resolve
  `LLMProvider`/`EmbeddingProvider`/`VectorStore` via `get_llm_provider()`/
  `get_embedding_provider()`/`get_vector_store()`, exactly like `RagOrchestrator` does — never
  construct an Ollama/OpenAI/Gemini/Anthropic/Qdrant client directly inside an engine or its
  adapters.
- **An alternative RAG engine must never select a different embedding model, vector size, or
  Qdrant collection than the one already configured** (`OLLAMA_EMBEDDING_MODEL`, `VECTOR_SIZE`,
  `QDRANT_COLLECTION_NAME`) — switching `RAG_ENGINE` must never re-embed a document, create a new
  collection, or change a chunk/point ID.
- **Every `RagEngine` implementation must preserve the public API/SSE contract exactly.**
  `POST /api/v1/chat`'s event types (`metadata`/`token`/`done`/`error`), field shapes, ordering
  guarantees (metadata first, `done` exactly once on success, one `error` with no `done` on
  failure), and safe-error-message rules apply identically regardless of which engine is
  configured — the frontend must never need to know which one is selected.
- **An unsupported `RAG_ENGINE` value must fail explicitly** (`UnsupportedRagEngineError`) —
  mirroring the existing `LLM_PROVIDER`/`EMBEDDING_PROVIDER`/`VECTOR_STORE_PROVIDER` rule, never
  silently fall back to `custom` or silently switch which engine/provider is used.
- **API routes must remain engine-agnostic.** `app/api/v1/routes/chat.py` depends on the abstract
  `RagEngine` (via the engine factory), never on a concrete engine class, and must never branch on
  `RAG_ENGINE` or contain decision/retrieval/prompt/provider-selection logic of its own (see
  "Route Layer Style" below).
- **LangGraph must not be introduced until a real agentic workflow requires it** (multi-step tool
  use, conditional branching driven by intermediate LLM output, etc.). The current decide-then-
  generate flow has no agent loop for LangGraph's graph/state machinery to add value to.

## Multilingual RAG Style

- **Prompt ownership lives in the shared `PromptProvider`, never in an engine.** Fixed/governed
  response text (`clarification`/`no_results`/`out_of_scope`/`grounded_answer`/`direct_answer`)
  is resolved through `app/rag/prompts/provider.py`'s `PromptProvider`, which both
  `RagOrchestrator` and `LangChainRagEngine` call — neither engine may hardcode this text, import
  it from the other's implementation module, or reintroduce anything resembling the old
  `app/rag/responses.py` (removed).
- **No engine-specific prompt catalogs.** There is exactly one `PromptCatalog`
  (`app/rag/prompts/catalog.py`) for the whole platform. Do not add a LangChain-only or
  Custom-only prompt catalog, even a small one — both engines must resolve identical content for
  the same `(PromptType, SupportedLanguage)` pair.
- **Explicit embedding/index versioning is mandatory.** Any change to the embedding
  provider/model/dimension or the chunking approach must be reflected by bumping
  `EMBEDDING_VERSION`/`CHUNKING_VERSION` (see `app/rag/embedding_config.py`) — never change what a
  collection's vectors mean without changing which collection they live in.
  `get_active_embedding_config()` is the only function that should resolve these settings for
  indexing purposes; do not read `EMBEDDING_PROVIDER`/`EMBEDDING_MODEL`/`VECTOR_SIZE` directly
  from `Settings` in ingestion or retrieval code.
- **Collection migration must stay safe.** Never silently recreate or delete a Qdrant collection
  whose dimension doesn't match the active configuration (raise
  `IncompatibleIndexConfigurationError` instead, per `app/services/index_registry.py`), never
  delete a document's existing vectors before its replacement index write has already succeeded,
  and never auto-delete an old collection at startup — `retire_collection()` is a bookkeeping-only
  status flip; actually removing Qdrant data is a separate, deliberate operational action.
- **Documentation must stay consistent whenever prompts, languages, indexing metadata, or
  collection routing change.** A change to `app/rag/prompts/`, `app/rag/language.py`,
  `app/rag/embedding_config.py`, `app/models/index_collection.py`, or `Document`'s indexing
  columns must come with a matching update to "Multilingual RAG Foundation" in
  [ARCHITECTURE.md](ARCHITECTURE.md) and, where relevant, the "Multilingual RAG foundation"
  section of [README.md](README.md) — in the same change, not deferred.
- **Real Ollama/real embedding models stay out of the default automated suites.** Unit,
  integration, and backend E2E tests use `MultilingualFakeEmbeddingProvider`
  (`tests/multilingual_fixtures.py`) or an equivalent fake — never a real multilingual model
  download or call. A real-model evaluation belongs in `make smoke-multilingual-real`
  (`scripts/smoke_multilingual_real.py`), a separate, manual, non-blocking target never invoked
  by `make verify`/`make test*`/CI, same as the existing real-Ollama smoke-suite boundary.
- **Validate real embedding-vector output, not just configuration.** Every embedding batch
  (ingestion, re-index, and the query vector at retrieval) must pass through
  `app/rag/embedding_validation.py`'s `validate_embeddings()` before any Qdrant write/search or
  document-indexed marking — a wrong vector count or dimension must fail loudly
  (`EmbeddingResultCountMismatchError`/`EmbeddingDimensionMismatchError`), never write partial or
  mismatched data.
- **The decision layer stays single and shared, in both languages.** `RuleBasedRagDecider`
  (`app/rag/decision.py`) is the one decision service; do not add a
  `CustomHebrewDecider`/`LangChainHebrewDecider` or any other engine-specific routing logic.
  Hebrew patterns must match meaningful intent phrasing (a document/file reference, an extraction
  verb near a sensitive noun) — never bare Hebrew-script detection.
- **Generative system prompts share one English instruction, never duplicated per language.**
  `PromptCatalog.get_shared_instructions()` is English-only; the per-language piece is only the
  explicit response-language directive (`get_response_language_directive()`) — never instruct the
  model to "answer in English and translate," and never claim the model "thinks in English."
- **Re-index outcomes must be represented with a typed result, not a bare bool.**
  `reindex_document()` returns `ReindexResult`/`ReindexOutcome` — a bare `bool` cannot represent
  "zero-chunk document" or "the new collection/metadata committed but the old collection's
  cleanup failed" distinctly from a plain success. A zero-chunk document is marked indexed (with
  no vectors) — see `ARCHITECTURE.md`'s "Zero-chunk behavior" — never silently left unindexed.
- **A legacy-vector cleanup failure is tracked, never silently dropped or conflated with
  re-index failure.** Persist it as a `VectorCleanupJob` (`app/models/vector_cleanup_job.py`) and
  expose a retryable `retry_cleanup_job()` — retried regardless of whether the document itself is
  still stale, since cleanup success/failure is independent of `is_document_stale()`. Full
  document deletion (`delete_all_tracked_document_vectors()`) must clean every collection tracked
  by a pending/failed cleanup job for that document, not just its current one, and must attempt
  every resolved collection independently — one collection's delete failing must never stop,
  skip, or abort attempts against the others, or silently fall back to active-only semantics.
- **Any user-facing or lifecycle-level document deletion must call
  `delete_all_tracked_document_vectors(..., session)`.** `delete_current_document_vectors(...)`
  (no `session` parameter, active-collection only) is only valid for explicitly scoped
  active-collection operations such as rollback or current-index repair — never for a real
  document-deletion path, where leaving a historical collection's vectors behind would be a
  silent partial deletion.
- **Never claim the Qdrant/PostgreSQL boundary is atomic, or that a failed attempt is
  indistinguishable from one that never ran.** A Qdrant write can succeed before a Postgres
  commit fails; `reindex_document()` rolls back and expires the `Document` in that case, but the
  Qdrant points already exist (retry-safe via deterministic point IDs) — document this precisely,
  per `ARCHITECTURE.md`'s "Re-index" transaction-semantics description, rather than glossing over
  it.

## Database Testing Style

- **Do not add SQLite/`aiosqlite` for testing database-touching code.** The project targets
  Postgres, and code that depends on Postgres-specific semantics (e.g.
  `with_for_update(skip_locked=True)` row locking) is not correctly represented by SQLite even
  when SQLite accepts the same SQLAlchemy call — it silently behaves differently.
- **Use a fake session/repository double for unit tests.** Tests for code that reads/writes via
  `AsyncSession` (e.g. `tests/test_document_upload.py`, `tests/test_ingestion_worker.py`) use a
  small in-memory fake implementing only the methods the code under test actually calls
  (`add`, `execute` returning a fake scalar result, `get`, `commit`), faithfully simulating the
  real query's filter/ordering logic in plain Python rather than executing real SQL.
- **Use a real Postgres integration test — via Testcontainers — for behavior a fake session
  cannot faithfully represent** (row-level locking, `FOR UPDATE SKIP LOCKED`, real constraint
  enforcement, real Alembic migrations). See "Integration Testing Style" below.

## Integration Testing Style

- **Use Testcontainers for Python, not the main `docker-compose.yml`, for integration tests.**
  `tests/integration/` starts its own ephemeral Postgres/Qdrant containers via Testcontainers on
  dynamically assigned ports — never the repository's `docker-compose.yml` stack, and never a
  fixed host port or a persistent Compose volume. Local development continues to use
  `docker-compose.yml` exactly as before; that workflow and the integration suite must stay
  independent of each other.
- **Never use SQLite to simulate Postgres locking/transaction behavior** (this generalizes the
  "Database Testing Style" rule above to the integration tier too). Use a real Postgres container
  for any test asserting `FOR UPDATE SKIP LOCKED`, isolation-level, or constraint-enforcement
  behavior — a fake session or a different database engine cannot be trusted to match Postgres
  here, even when the same SQLAlchemy call superficially "works" against it.
- **Use a real Qdrant container for HTTP/data-contract integration tests** — verifying
  `QdrantVectorStore`'s actual request/response shape, ranking, and error behavior against a
  mocked `httpx` transport only proves the mock was self-consistent, not that it still matches
  real Qdrant.
- **Keep real Ollama outside the default integration suite.** No integration test in
  `tests/integration/` may start a real Ollama container or pull a real model — use a small,
  fixed-vector fake embedding provider (or an equivalent fake for LLM output) wherever the
  pipeline needs one. Real-Ollama verification belongs in a separate, future manual/nightly smoke
  suite, not the suite that runs on every `make test-integration`.
- **Do not add slow integration tests to the pre-commit hook without explicit approval.** The
  pre-commit hook runs `make verify`, which must stay fast and Docker-independent (beyond
  `docker compose config`, which starts nothing) — `make test-integration`/
  `make verify-integration` are separate, manually-invoked targets and stay that way unless the
  user explicitly asks to change that.
- **Integration tests must use dynamic ports and fully ephemeral state.** Never hardcode a host
  port for a Testcontainers-managed service, and never write state that outlives the test
  session — containers, their data, and any temp files must be gone once the run ends.
- **Never let a test connect to a production service.** Integration fixtures must fail loudly
  (before starting any container or test) if the ambient environment looks like production —
  see the guard fixture in `tests/integration/conftest.py` — rather than silently running against
  whatever `DATABASE_URL`/`QDRANT_URL` happens to be set.

## Backend E2E Testing Style

- **Backend E2E tests use Testcontainers and fully ephemeral state, exactly like the integration
  suite.** `tests/e2e/backend/` starts its own ephemeral Postgres/Qdrant containers via
  Testcontainers on dynamically assigned ports — never the repository's `docker-compose.yml`,
  never a fixed host port, never a persistent Compose volume — and every test gets its own
  isolated database rows and Qdrant collection (see `tests/e2e/backend/conftest.py`).
- **Backend E2E tests traverse the real public HTTP boundary where practical.** Drive the app
  through a real ASGI HTTP client (`httpx.AsyncClient` + `ASGITransport`) against the real
  FastAPI app — `POST /api/v1/documents`, `POST /api/v1/chat`, `GET /health*` — rather than
  calling service/route functions directly, so the actual request/response contract (status
  codes, SSE framing, validation errors) is what's exercised, not an internal shortcut.
- **AI providers remain deterministic fakes in the default backend E2E suite.** Real
  Ollama never runs and no model is ever pulled here — `FakeEmbeddingProvider` and
  `FakeStreamingLLMProvider`/`FakeFailingLLMProvider` (`tests/e2e/backend/fakes.py`) stand in,
  swapped via monkeypatching the provider-factory function each consuming module already
  imports, never a production-code branch on `APP_ENV`. The vector store is never faked —
  `QdrantVectorStore` keeps talking to the real ephemeral Qdrant container.
- **Real Ollama belongs in a separate, future manual/nightly smoke suite** — never added to
  `tests/e2e/backend/`, `make test-e2e-backend`, or `make verify-e2e-backend`.
- **Do not add backend E2E tests to the pre-commit hook without explicit approval.** The
  pre-commit hook runs `make verify`, which must stay fast and Docker-independent — `make
  test-e2e-backend`/`make verify-e2e-backend` are separate, manually-invoked targets and stay
  that way unless the user explicitly asks to change that.
- **Never use the main Compose environment (`docker-compose.yml`) as the E2E test environment.**
  Backend E2E fixtures must fail loudly (before starting any container or test) if the ambient
  environment looks like production — see the guard fixture in `tests/e2e/backend/conftest.py` —
  rather than silently running against whatever `DATABASE_URL`/`QDRANT_URL` happens to be set.

## Operational Endpoints

- **Operational endpoints remain unversioned.** `GET /health`, `/health/live`, `/health/ready`,
  and `/health/dependencies` (`app/api/routes/health.py`) are registered on `app` with no
  `/api/v1` (or any future `/api/vN`) prefix, and must stay that way. Business API versioning
  changes what request/response shapes a client of the RAG features sees; it must never affect
  where infrastructure (Kubernetes probes, load balancers, ArgoCD, monitoring) finds the
  operational health contract.
- **Liveness must not depend on external services.** `GET /health/live` (and `GET /health`) must
  never call Postgres, Redis, Qdrant, Ollama, or any other external service — if it did, a
  temporary dependency outage would make Kubernetes restart an otherwise-healthy process. Only
  `GET /health/ready` and `GET /health/dependencies` may perform dependency I/O.
- **Readiness must reflect required runtime dependencies.** `GET /health/ready` returns `503` if
  any dependency marked `required=True` in `app/services/platform_health.py` fails its check.
  Whether a given dependency is `required` must track whether the codebase actually depends on it
  today (e.g. `redis` is currently `required=False` because nothing reads or writes it yet) — flip
  it to `True` the same PR that starts actually using it, not before and not long after.
- **Health responses must never expose secrets or internal connection details.** No check result
  (`DependencyCheckResult.detail`, or any `/health/*` response) may include a raw exception
  message, a connection string, a credential, a stack trace, or a provider's raw response body —
  use a fixed, generic message per failure mode instead (see the existing checks in
  `app/services/platform_health.py` for the pattern).
- **Future API version changes must not move these endpoints under `/api/vN`.** If the business
  API is ever versioned to `/api/v2`, `/health*` stays exactly where it is — this is the whole
  point of keeping it unversioned in the first place.

## Route Layer Style

- **Keep FastAPI route handlers thin.** A route function should do only: request
  validation/parsing (via the Pydantic request schema), dependency injection (`Depends(...)`),
  one call into a service function, applying whatever HTTP status the service already decided,
  and returning the response body/model the service already built. No business logic, no
  aggregation (filtering, counting, computing an overall status, building an error summary), and
  no direct provider/database calls belong in a route module.
- **Business/aggregation logic lives in the service layer**, as small, well-named, independently
  unit-testable functions — prefer pure, synchronous functions for pure computation (e.g.
  `build_readiness_result(checks)`) separate from the async I/O-performing orchestration around
  them (e.g. `get_readiness_result(settings)`), so aggregation logic can be tested without
  mocking any I/O at all.
- **Prefer a typed result object over route-side status-code logic** when a service needs to
  communicate both a response body and an HTTP status back to a route — see
  `platform_health.ReadinessResult` (`response` + `status_code`) for the pattern. The route
  should only ever copy `result.status_code` onto the response, never re-derive it.
- **Established examples**: `POST /api/v1/chat` (`app/api/v1/routes/chat.py`) delegates entirely
  to `RagOrchestrator`; `GET /health/ready`/`GET /health/dependencies`
  (`app/api/routes/health.py`) delegate entirely to `app/services/platform_health.py`. When
  adding a new route with any nontrivial logic, follow the same split — and if a review finds
  aggregation/business logic creeping into a route module, that's a bug to fix, not a style
  nitpick to defer.

## Pull Request Workflow

- **Verify GitHub CLI before any GitHub operation.** Run `gh --version` and `gh auth status`
  first. If either fails, stop, report it, and do not push or open a PR.
- **Check the current branch before pushing.** Confirm `git branch --show-current` is the
  intended feature branch — never push from `main` on someone's behalf.
- **Verify working tree status before committing/pushing.** Run `git status` and review the
  diff; only stage the files that belong to the change.
- **Never push unrelated files.** Commit and push exactly what the task scoped — no drive-by
  cleanups bundled into an unrelated PR.
- **Prefer small, focused PRs.** One milestone or one concern per PR, matching the "small
  incremental milestones" rule above.
- **Use the repository PR template.** Fill in `.github/pull_request_template.md` — don't write a
  free-form description instead of it.

### Using the PR template with `gh`

When opening a pull request with GitHub CLI:

1. Read `.github/pull_request_template.md` first, before drafting any PR body.
2. Use its sections (Summary, Why, Changes, Verification, Explicit exclusions / intentionally
   not implemented, Next recommended milestone) as the PR body structure — do not invent a
   different structure or skip a section.
3. Write the filled-in body to a temporary file, then pass it with
   `gh pr create --body-file <file>` — do not pass an ad-hoc description inline with `--body`
   when the template exists.

### PR title style

Short, imperative, present tense, no trailing period — e.g. `Add Ollama provider health checks`.

### PR description format

Every PR description follows this structure, in this order:

1. **Summary** — one or two sentences on what the PR does.
2. **Why** — the motivating requirement or problem.
3. **Changes** — bullet list of what was added/modified.
4. **Verification** — the exact commands run and their output/result (e.g. `pytest -q`,
   `ruff check .`, `mypy app`, `docker compose config`). Include real output, not a claim that
   it passed.
5. **Explicit exclusions / intentionally not implemented** — what this PR deliberately does not
   do, so reviewers don't wonder if something was missed.
6. **Next recommended milestone** — one concrete, scoped suggestion for what comes after this PR.

If the PR is documentation-only or otherwise doesn't change application behavior, say so
explicitly in the Summary (e.g. "Documentation-only change; no application behavior changed.").

## Final report format

At the end of any non-trivial change, report back with these sections, in this order:

- **What changed** — a short summary of the actual change.
- **Why it changed** — the motivating requirement or problem.
- **Files changed** — list of files touched.
- **Verification** — the exact commands run and their results (pytest/ruff/mypy/docker compose
  config, plus any manual verification like curling an endpoint).
- **Next recommended milestone** — one concrete, scoped suggestion for what to build next.

## Boundaries

- Do not implement RAG business logic (ingestion, embeddings, retrieval, chat) unless explicitly
  asked — this project is intentionally staged milestone by milestone.
- Do not introduce new frameworks or heavy dependencies (e.g. LangChain) without being asked.
- Do not weaken or bypass quality gates (no `--no-verify`, no skipping failing tests/lint/type
  checks to "get it green").
