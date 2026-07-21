# Multilingual (Hebrew + English)

Multilingual retrieval and language-aware prompting are shared platform capabilities, reached
identically by both `CustomRagEngine` and `LangChainRagEngine` — neither engine implements its own
language detection, prompt catalog, or embedding-version selection. See
[docs/rag/](../rag/README.md) for the engines this sits underneath.

## Flow

```
Question → LanguageDetector → PromptProvider → PromptCatalog → ResolvedPrompt → RagEngine
```

## Language detection

`ScriptBasedLanguageDetector` (`app/rag/language.py`) — deterministic, **word-level** (not
character-level) Hebrew/Latin script-dominance counting: each whitespace/punctuation-split word is
classified Hebrew, Latin, or ignored (digits/punctuation), and whichever script has more *words*
wins. Word-level classification keeps a handful of Latin-script technical identifiers (Kafka,
Qdrant, Kubernetes) embedded in an otherwise-Hebrew sentence from outweighing the surrounding
Hebrew. An exact tie or no Hebrew/Latin words at all falls back to `DEFAULT_RESPONSE_LANGUAGE`.

## Prompt catalog

`PromptType` (`grounded_answer`/`direct_answer`/`clarification`/`no_results`/`out_of_scope`) ×
`SupportedLanguage` (`he`/`en`) → `PromptCatalog` (`app/rag/prompts/`). The two generation-backed
types share **one English-authored governance instruction each — never duplicated per language**:
answer only from context, never fabricate, preserve `[S1]`/`[S2]` labels and quoted source text
untranslated, never translate code/API names/filenames/commands/error messages. A per-language
response-language directive is appended separately (never "answer in English and translate"). The
three no-LLM-call types are naturally authored per language, since they bypass the LLM entirely.

`PromptProvider.resolve(prompt_type, question)` is the single seam both engines call — neither
imports this text from the other's implementation module.

## Embedding model selection

`OLLAMA_EMBEDDING_MODEL` defaults to **`bge-m3`** (1024-dim, supports 100+ languages including
Hebrew) — the actual runtime default, requiring `ollama pull bge-m3`. `EMBEDDING_VERSION` defaults
to `v2`. Changing the embedding model always produces a new versioned Qdrant collection (via
`EmbeddingIndexConfig.collection_name`) and requires re-indexing existing documents — the previous
collection's vectors are never deleted automatically. See
[docs/document-lifecycle/](../document-lifecycle/README.md) for the re-index build/activation
cycle, and [docs/configuration/](../configuration/README.md) for the full variable list.

The legacy English-oriented `nomic-embed-text` (768-dim) remains configurable via
`EMBEDDING_MODEL=nomic-embed-text` + `VECTOR_SIZE=768` + `EMBEDDING_VERSION=v1`, documented only as
an explicit opt-out for installations that don't need Hebrew retrieval.

## Supported languages

Hebrew and English only. `DEFAULT_RESPONSE_LANGUAGE` must be `he` or `en`.

## Citation behavior

Source titles/filenames, quoted text, and page/sheet metadata are language-independent — a Hebrew
answer can cite an English-titled source and vice versa; nothing translates a citation or document
title.

## Test ownership

Automated tests (unit/integration/E2E) **never** depend on a real embedding model or download —
they use `MultilingualFakeEmbeddingProvider` (`tests/multilingual_fixtures.py`), a deterministic
bag-of-concepts hashing embedding with a small Hebrew/English synonym table, proving the retrieval
*wiring* works cross-language, not real model quality.

```bash
make test-multilingual-rag       # unit + integration + E2E matrix (needs Docker)
make verify-multilingual-rag
make smoke-multilingual-real     # OPTIONAL, MANUAL: real bge-m3 model, 5 scenarios,
                                  # never run by make verify/test*/CI
```

## Current Limitations

- Only Hebrew and English are supported — no other language's script-dominance rules exist.
- Language detection is deterministic script-counting, not an ML model — a question with no
  Hebrew/Latin words at all (numbers/punctuation only) falls back to the configured default.
- Broader recall/ranking evaluation on a larger real-model corpus remains future work; the fake
  provider proves wiring correctness only.

## Deferred Behavior

- A database-backed, runtime-editable prompt system — the current catalog is a flat, hardcoded
  he/en dict. Deferred until the platform needs more languages or non-developer prompt edits; do
  not add this machinery ahead of an actual need.
- Additional languages beyond Hebrew/English.
- Production-scale multilingual retrieval-quality evaluation (the real-model smoke test is
  illustrative only, not a benchmark).
