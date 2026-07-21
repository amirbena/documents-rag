# documents-rag

## Project overview

`documents-rag` is a local-first, framework-independent, multilingual (Hebrew + English) RAG
(Retrieval-Augmented Generation) platform. It runs entirely via Docker Compose — FastAPI,
PostgreSQL, Redis, Qdrant, and Ollama for local LLM/embedding inference — with no required
external API calls or cloud dependencies.

**Major capabilities:** document upload and asynchronous ingestion (extraction, chunking,
embedding, indexing) across `.txt`/`.md`/`.pdf`/`.docx`/`.xlsx`; streaming, source-attributed RAG
chat over Server-Sent Events; two interchangeable RAG execution engines (a custom orchestrator and
an optional LangChain-backed engine) with an identical public contract; hash-based upload
deduplication; full asynchronous document deletion; build-ahead, zero-downtime re-indexing with
explicit operator activation; and read-only reconciliation/audit reporting across Postgres, object
storage, and Qdrant.

**Current maturity and scope:** this is a portfolio/learning-project backend, feature-complete
through Phase 2.8 (document lifecycle: observe, recover, delete, deduplicate, upgrade,
reconcile — see [docs/document-lifecycle/](docs/document-lifecycle/README.md)). No frontend
exists yet. Backend lifecycle architecture is currently frozen pending UI/browser E2E validation —
see [analysis/phase-2.8-completion-and-backend-freeze-audit.md](analysis/phase-2.8-completion-and-backend-freeze-audit.md)
for that decision's full rationale.

## Architecture summary

Three storage systems with distinct roles — PostgreSQL (lifecycle authority), object storage
(original-content authority, local disk or MinIO), and Qdrant (rebuildable derived vector index) —
behind a provider-abstraction layer for embeddings/LLM/vector-store, and a `RagEngine` abstraction
letting the custom and LangChain execution paths share one public API/SSE contract.

Full detail, module ownership map, and dependency-direction rules:
**[docs/architecture/](docs/architecture/README.md)**.

## Initial setup

**Prerequisites:** Python 3.11+, Docker + Docker Compose (`docker compose version`), Git, and
optionally the [GitHub CLI](https://cli.github.com/) (`gh`) for the PR workflow.

First-time onboarding, in order:

1. **Create and activate a virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. **Install dependencies:**
   ```bash
   pip install -e ".[dev]"
   ```
3. **Copy the environment file:**
   ```bash
   cp .env.example .env
   ```
4. **Start Docker Compose** (brings up `app`, `postgres`, `redis`, `qdrant`, `ollama`):
   ```bash
   docker compose up --build
   ```
5. **Run Alembic migrations** — Docker Compose starts Postgres but does not apply migrations
   automatically:
   ```bash
   docker compose exec app alembic upgrade head
   ```
6. **Verify app health:**
   ```bash
   curl http://localhost:8000/health
   # {"status":"ok","service":"documents-rag","version":"0.1.0"}
   ```
7. **Pull the required Ollama models:**
   ```bash
   docker compose exec ollama ollama pull llama3.1
   docker compose exec ollama ollama pull bge-m3
   ```
8. **Verify Ollama health:**
   ```bash
   curl http://localhost:8000/api/v1/providers/ollama/health
   ```
9. **Install the git pre-commit hook** (runs `make verify` automatically before every commit):
   ```bash
   ./scripts/install-git-hooks.sh
   ```
10. **Run the full verification suite:**
    ```bash
    make verify
    ```

**Running the app without Docker** (once the steps above are done and Postgres/Redis/Qdrant/Ollama
are reachable some other way, e.g. `docker compose up postgres redis qdrant ollama` with `.env`
pointed at `localhost`):

```bash
uvicorn app.main:app --reload
```

**`python app/main.py` does not start the server** — it only defines the FastAPI `app` object; the
process imports the module and exits with no error, which is easy to mistake for success.

Container topology, migration sequencing detail, and the full health/readiness contract:
**[docs/deployment/](docs/deployment/README.md)**. Repository conventions, contribution workflow,
and PyCharm run configuration: **[docs/development/](docs/development/README.md)**.

## Documentation index

| Directory | Covers |
|---|---|
| [docs/architecture/](docs/architecture/README.md) | System overview, module ownership, dependency direction, invariants |
| [docs/development/](docs/development/README.md) | Local setup, conventions, contribution workflow |
| [docs/testing/](docs/testing/README.md) | Test taxonomy, suite/fixture ownership, where a new test belongs |
| [docs/operations/](docs/operations/README.md) | Worker execution, lifecycle recovery commands, reconciliation-to-repair mapping |
| [docs/providers/](docs/providers/README.md) | Embedding/LLM/vector-store provider abstraction and selection |
| [docs/storage/](docs/storage/README.md) | Relational, object, and vector storage ownership and consistency |
| [docs/rag/](docs/rag/README.md) | Retrieval/generation flow, RAG engine ownership |
| [docs/langchain/](docs/langchain/README.md) | LangChain-specific engine, parity expectations |
| [docs/multilingual/](docs/multilingual/README.md) | Hebrew/English language handling, prompt catalog |
| [docs/document-lifecycle/](docs/document-lifecycle/README.md) | **Canonical** lifecycle state machines and API contracts |
| [docs/configuration/](docs/configuration/README.md) | Full environment variable reference |
| [docs/deployment/](docs/deployment/README.md) | Container topology, migration sequencing, health contract |
| [docs/backend-e2e/](docs/backend-e2e/README.md) | Backend E2E scope, environment, execution, diagnosis |
| [docs/troubleshooting/](docs/troubleshooting/README.md) | Common failures and verified recovery steps |

## Verification

```bash
make test              # fast unit suite (no Docker)
make test-integration   # Testcontainers-based integration suite (needs Docker)
make test-e2e-backend   # Testcontainers-based backend E2E suite (needs Docker)
make lint               # ruff check .
make typecheck          # mypy app
make verify             # test + lint + typecheck + compose, stopping at the first failure
```

`make verify` is the canonical pre-commit/pre-PR gate. Install the git hook that runs it
automatically: `./scripts/install-git-hooks.sh`. Full command reference, including feature-slice
convenience commands: [docs/testing/](docs/testing/README.md).

## Current limitations

No stale-`PROCESSING` recovery for deletion or re-index jobs (only ingestion has this); only the
latest ingestion attempt is exposed via the API (full history is retained in Postgres but not
enumerable); download/upload buffer the full object in memory rather than streaming; no CI
workflow exists in this repository. Full, per-domain limitation lists live in each documentation
directory above — see in particular
[docs/document-lifecycle/](docs/document-lifecycle/README.md#current-limitations) and
[docs/operations/](docs/operations/README.md#current-limitations).

## Deferred behavior

Proposed or roadmap behavior that is **not** part of the current implementation is documented
separately, under its own "Deferred Behavior" heading, in each relevant `docs/` directory — never
described there or here as if it already exists. Notable examples: a generic reconciliation
"repair" API (will never be built — repair is bounded, domain-specific commands instead), a real
scheduler deployment for any operational script, stale-deletion/re-index-job recovery, an
LLM-based question router, and frontend/browser E2E (no frontend exists yet).

## GitHub CLI / PR workflow

Pull requests are created from the terminal with the [GitHub CLI](https://cli.github.com/)
(`gh`), not the web UI:

```bash
gh --version
gh auth status
```

PRs should be small and focused, with verification results included in the description. See
[.github/pull_request_template.md](.github/pull_request_template.md) and
[CLAUDE.md](CLAUDE.md)'s "Pull Request Workflow" for the full template and conventions.
