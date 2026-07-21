# Architecture

This document has moved. As of the Phase 2.9 documentation refactor, all of the content
previously on this page has been reorganized into the `docs/` directory, split by domain so each
area has one canonical owner instead of one growing file:

- **System overview, module ownership, dependency rules, invariants** →
  [docs/architecture/](docs/architecture/README.md)
- **Document/ingestion/deletion/re-index/activation/cleanup/reconciliation lifecycle, state
  machines, and API contracts** → [docs/document-lifecycle/](docs/document-lifecycle/README.md)
  (the canonical lifecycle reference)
- **Provider abstraction (Ollama/Qdrant/future LLM stubs)** →
  [docs/providers/](docs/providers/README.md)
- **Storage (relational/object/vector)** → [docs/storage/](docs/storage/README.md)
- **Retrieval/generation flow and RAG engine ownership** → [docs/rag/](docs/rag/README.md)
- **LangChain-specific engine** → [docs/langchain/](docs/langchain/README.md)
- **Hebrew/English multilingual behavior** → [docs/multilingual/](docs/multilingual/README.md)
- **Environment variables** → [docs/configuration/](docs/configuration/README.md)
- **Container topology, migrations, health contract** → [docs/deployment/](docs/deployment/README.md)
- **Test architecture** → [docs/testing/](docs/testing/README.md) and
  [docs/backend-e2e/](docs/backend-e2e/README.md)
- **Worker execution, reconciliation-to-repair mapping** → [docs/operations/](docs/operations/README.md)

Start at the repository root [README.md](README.md)'s Documentation Index for the full map.

This file is kept only so existing links to `ARCHITECTURE.md` keep resolving to the redirect
above — it carries no independent content and should not be edited further; update the relevant
`docs/` page instead.
