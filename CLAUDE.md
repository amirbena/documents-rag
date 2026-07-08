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
5. **Quality gates must pass before a change is considered done.** Run all of:
   - `pytest`
   - `ruff check .`
   - `mypy app`
   - `docker compose config`

   All four must pass cleanly. If one fails, fix the underlying issue rather than skipping or
   loosening the gate.

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
