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
   finishing any implementation task — it runs, in order, stopping at the first failure:
   - `make test` (`pytest -q`)
   - `make lint` (`ruff check .`)
   - `make typecheck` (`mypy app`)
   - `make compose` (`docker compose config`)

   If `make` isn't available, run the four underlying commands individually in that order as a
   fallback. All four must pass cleanly either way. If one fails, fix the underlying issue rather
   than skipping or loosening the gate.

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
