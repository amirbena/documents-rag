# Development

Local development setup, repository conventions, and the contribution workflow. See
[docs/testing/](../testing/README.md) for test placement rules and
[docs/architecture/](../architecture/README.md) for module-boundary rules in full.

## Prerequisites

- Python 3.11+
- Docker + Docker Compose (`docker compose version`)
- Git
- [GitHub CLI](https://cli.github.com/) (`gh`) — only needed for the PR workflow, not for running
  the app

## Local setup

The full first-time onboarding walkthrough (prerequisites, virtualenv, Docker Compose, Alembic,
Ollama models, pre-commit hook) lives in the root **[README.md](../../README.md#initial-setup)**
— that is the canonical copy; this page does not repeat it.

See [docs/deployment/](../deployment/README.md) for the full container topology and readiness
contract, and [docs/configuration/](../configuration/README.md) for every environment variable.

## Running the app

**`python app/main.py` does not start the server** — it only defines the FastAPI `app` object; the
process imports the module and exits, with no error, which is easy to mistake for success. See
the root [README.md](../../README.md#initial-setup) for the recommended `docker compose up --build`
flow and the app-only `uvicorn` alternative.

## Repository conventions

- **Package `__init__.py` files stay minimal** (a one-line docstring) — they never re-export
  package contents. Import from the canonical module directly:
  `from app.services.documents.query_service import get_document`, never
  `from app.services.documents import get_document`.
- **Route layer style**: routes parse/inject/call-one-service/copy-status — no business logic,
  aggregation, or direct provider/DB call in a route module.
- **Docstrings**: every module gets a one/two-line docstring; every public function/class gets a
  one-line docstring stating intent, not implementation.
- **Providers**: prefer calling a provider's HTTP API directly over its SDK — see
  [docs/providers/](../providers/README.md).

See [docs/architecture/](../architecture/README.md)'s Dependency Direction Rules for the full,
authoritative module-boundary list (which package may import which).

## Module-boundary expectations

Before adding code, identify which existing package owns the concern — see
[docs/architecture/](../architecture/README.md)'s ownership map. Do not create a new top-level
package or abstraction layer (`repositories/`, `domain/`, `application/`) for a concern an existing
package already owns.

## Contribution workflow

1. Create a feature branch from `main`.
2. Make focused changes — prefer small, single-concern commits and PRs (one milestone/concern per
   PR).
3. Run `make verify` before committing (or rely on the installed pre-commit hook — see below).
4. Open a PR via `gh pr create` (not the web UI) — see the PR template at
   [.github/pull_request_template.md](../../.github/pull_request_template.md).
5. Address review, then squash-merge (this repository's established convention) once approved.

```bash
gh --version
gh auth status
```

PR descriptions should include actual verification results (test/lint/type-check output), not just
a claim that checks passed.

## Common development commands

```bash
make test          # fast unit suite
make lint          # ruff check .
make typecheck      # mypy app
make compose        # docker compose config (validates only, starts nothing)
make verify         # all of the above, in order, stopping at the first failure
make help           # full command reference
```

Full verification/testing command reference: [docs/testing/](../testing/README.md).

## Pre-commit verification

```bash
./scripts/install-git-hooks.sh
```

Installs a git hook that runs `make verify` automatically before every commit — it only checks
(never auto-fixes or stages/commits anything). Skip only in a genuine emergency with
`git commit --no-verify`, and prefer fixing the underlying issue.

## Current Limitations

- No CI workflow exists (`.github/workflows/` absent) — `make verify` and the pre-commit hook are
  the only automated gates; nothing runs automatically on push or PR open.
- No linter/formatter auto-fix is applied by the pre-commit hook — it only checks.

## Deferred Behavior

- A CI pipeline (GitHub Actions or equivalent) running `make verify`/`make verify-integration`/
  `make verify-e2e-backend` automatically on push/PR.
