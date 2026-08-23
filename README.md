# documents-rag

## 1. Project overview

`documents-rag` is a local-first, multilingual (Hebrew + English) RAG (Retrieval-Augmented
Generation) backend. It lets you upload documents, ingests them into a searchable vector index,
and answers questions over that content with streaming, source-attributed chat — using local
infrastructure and [Ollama](https://ollama.com/) for LLM/embedding inference by default, with no
required external API calls or cloud dependencies.

**Status:** backend-only (no frontend yet), functionally complete for the document lifecycle
(upload, ingest, chat, delete, re-index, reconcile) and hardened for deployment readiness
(process lifecycle, config validation, timeouts/retries, structured logging). See
[Current limitations](#10-current-limitations) below and
[docs/architecture/](docs/architecture/README.md) for what that means in practice.

## 2. What you need before starting

**Required:**

- [Git](https://git-scm.com/)
- Python 3.11+
- [Docker](https://docs.docker.com/get-docker/) and Docker Compose (`docker compose version`)
- Enough free local RAM/disk to run Postgres, Redis, Qdrant, Ollama, and the application together
  — Ollama's pulled models (an LLM and an embedding model) are each multi-gigabyte downloads

**Optional:**

- [GitHub CLI](https://cli.github.com/) (`gh`) — only needed for the PR workflow, see
  [Contribution / PR workflow](#11-contribution--pr-workflow)
- MinIO (started automatically by `docker compose up`, but only actually used when
  `FILE_STORAGE_PROVIDER=minio` — the default `local` provider needs no object-storage service)

No other platform-specific tooling is required.

## 3. Installation / first-time setup

```bash
# 1. Clone and enter the repository
git clone <repo-url>
cd documents-rag

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install Python dependencies
pip install -e ".[dev]"

# 4. Copy the environment file
cp .env.example .env

# 5. Start the Docker Compose stack (app, postgres, redis, qdrant, ollama, minio)
docker compose up --build

# 6. In another terminal: apply Alembic migrations — Compose starts Postgres but does not
#    apply migrations automatically
docker compose exec app alembic upgrade head

# 7. Pull the required Ollama models
docker compose exec ollama ollama pull llama3.1
docker compose exec ollama ollama pull bge-m3

# 8. Install the git pre-commit hook (runs `make verify` automatically before every commit)
./scripts/install-git-hooks.sh
```

Full container topology and migration-sequencing detail: [docs/deployment/](docs/deployment/README.md).
Full environment variable reference: [docs/configuration/](docs/configuration/README.md).

## 4. Running the project

**Full Docker Compose (the normal local stack):**

```bash
docker compose up --build
```

**Application locally, dependencies in Docker** — once the app has been set up per section 3, you
can instead run only the infrastructure in Compose and the API directly with Uvicorn (useful for
faster iteration/debugging):

```bash
docker compose up postgres redis qdrant ollama   # and minio, if FILE_STORAGE_PROVIDER=minio
# point .env at localhost (e.g. DATABASE_URL=...@localhost:5432/..., REDIS_URL=redis://localhost:6379/0, ...)
uvicorn app.main:app --reload
```

**`python app/main.py` does not start the server** — it only defines the FastAPI `app` object; the
process imports the module and exits with no error, which is easy to mistake for success. Always
start the API through `docker compose up` or `uvicorn app.main:app`.

## 5. Verifying that it works

```bash
curl http://localhost:8000/health                            # basic liveness
curl http://localhost:8000/health/ready                       # readiness (dependency checks)
curl http://localhost:8000/api/v1/providers/ollama/health      # Ollama reachability + configured models
```

A healthy `/health` response looks like `{"status":"ok","service":"documents-rag","version":"0.1.0"}`.
Full health/readiness contract: [docs/deployment/](docs/deployment/README.md).

## 6. Development verification

```bash
make test              # fast unit suite (no Docker)
make test-integration   # Testcontainers-based integration suite (needs Docker)
make test-e2e-backend   # Testcontainers-based backend E2E suite (needs Docker)
make lint               # ruff check .
make typecheck          # mypy app
make verify              # test + lint + typecheck + compose, stopping at the first failure
```

`make verify` is the canonical pre-commit/pre-PR gate. The git hook installed in section 3
(`./scripts/install-git-hooks.sh`) runs it automatically before every commit. Full command
reference, including feature-slice convenience targets: [docs/testing/](docs/testing/README.md).

## 7. Architecture at a glance

FastAPI serves the API; **PostgreSQL** is the lifecycle authority (documents, jobs, state);
**object storage** (local disk or MinIO) holds original uploaded content; **Qdrant** is a
rebuildable derived vector index, never a source of truth. **Redis** and **Ollama** (local LLM +
embedding inference) back the async job/queue and generation paths respectively. A provider
abstraction layer decouples embedding/LLM/vector-store choice from the rest of the app, and a
`RagEngine` abstraction lets a custom orchestrator and an optional LangChain-backed engine share
one public API/SSE contract.

Full detail, module ownership map, and dependency-direction rules:
[docs/architecture/](docs/architecture/README.md). Provider selection and timeout/retry policy:
[docs/providers/](docs/providers/README.md). Storage ownership and consistency:
[docs/storage/](docs/storage/README.md). Retrieval/generation flow:
[docs/rag/](docs/rag/README.md).

## 8. Capabilities

- Document upload and asynchronous ingestion (extraction, chunking, embedding, indexing) across
  `.txt`, `.md`, `.pdf`, `.docx`, and `.xlsx`
- Streaming, source-attributed RAG chat over Server-Sent Events, in Hebrew and English
- Two interchangeable RAG execution engines — a custom orchestrator and an optional
  LangChain-backed engine — with an identical public contract
- Hash-based upload deduplication
- Full asynchronous document deletion
- Build-ahead, zero-downtime re-indexing with explicit operator activation
- Read-only reconciliation/audit reporting across Postgres, object storage, and Qdrant

Full lifecycle state machines and API contracts (canonical):
[docs/document-lifecycle/](docs/document-lifecycle/README.md).

## 9. Documentation

"I need to understand X — where do I go?"

| Directory | Covers |
|---|---|
| [docs/architecture/](docs/architecture/README.md) | System overview, module ownership, dependency direction, invariants |
| [docs/document-lifecycle/](docs/document-lifecycle/README.md) | **Canonical** lifecycle state machines and API contracts |
| [docs/development/](docs/development/README.md) | Local setup, conventions, contribution workflow |
| [docs/testing/](docs/testing/README.md) | Test taxonomy, suite/fixture ownership, where a new test belongs |
| [docs/operations/](docs/operations/README.md) | Worker execution, lifecycle recovery commands, reconciliation-to-repair mapping |
| [docs/providers/](docs/providers/README.md) | Embedding/LLM/vector-store provider abstraction and selection |
| [docs/storage/](docs/storage/README.md) | Relational, object, and vector storage ownership and consistency |
| [docs/rag/](docs/rag/README.md) | Retrieval/generation flow, RAG engine ownership |
| [docs/langchain/](docs/langchain/README.md) | LangChain-specific engine, parity expectations |
| [docs/multilingual/](docs/multilingual/README.md) | Hebrew/English language handling, prompt catalog |
| [docs/configuration/](docs/configuration/README.md) | Full environment variable reference |
| [docs/deployment/](docs/deployment/README.md) | Container topology, migration sequencing, health contract |
| [docs/backend-e2e/](docs/backend-e2e/README.md) | Backend E2E scope, environment, execution, diagnosis |
| [docs/troubleshooting/](docs/troubleshooting/README.md) | Common failures and verified recovery steps |
| [alembic/README.md](alembic/README.md) | Migration workflow, history reset constraints |

## 10. Current limitations

- No automated stale-`PROCESSING` recovery for deletion or re-index jobs (only ingestion has
  this, and it's script-only/manual) — see
  [docs/operations/](docs/operations/README.md#current-limitations)
- Only the latest ingestion attempt is exposed via the API (full history is retained in Postgres
  but not enumerable)
- Upload/download buffer the full object in memory rather than streaming
- No in-place Alembic upgrade path from pre-baseline-reset revisions — **existing local databases
  must be recreated, not upgraded**, see
  [alembic/README.md](alembic/README.md#migration-history-reset-phase-210)
- No CI workflow exists in this repository yet

Full, per-domain limitation lists live in each documentation directory above, in particular
[docs/document-lifecycle/](docs/document-lifecycle/README.md#current-limitations) and
[docs/operations/](docs/operations/README.md#current-limitations).

## 11. Contribution / PR workflow

Pull requests are created from the terminal with the [GitHub CLI](https://cli.github.com/)
(`gh`), not the web UI:

```bash
gh --version
gh auth status
```

Branch management, commit, and PR conventions are defined in [AGENTS.md](AGENTS.md) — read it
before making changes. PR body structure: [.github/pull_request_template.md](.github/pull_request_template.md).
