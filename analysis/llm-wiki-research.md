# LLM Wiki Research

**Status:** Research only. No implementation code was touched or proposed as a plan of record. This
document distinguishes **[Implemented]** (verified against current code/docs), **[External]**
(researched approaches from outside this repository, cited), and **[Proposed]** (this document's own
recommendation, not yet built) throughout.

## Executive summary

**[Proposed]** The current RAG pipeline ([documents-rag](../README.md)) is a clean, single-purpose
flat-chunk vector retrieval system: extract → chunk → embed → Qdrant search → labeled/attributed
prompt → LLM. It has no notion of entities, topics, cross-document relationships, or corpus-level
overview — every answer is assembled from whatever chunks a single top-k similarity search returns.
That is sufficient for single-fact, single-document questions and is already well-grounded
(citations are chunk-level and mandatory). It measurably struggles on relationship questions ("how
is X related to Y"), "what depends on X" questions, and broad "summarize everything about Y across
the corpus" questions, because flat top-k retrieval has no mechanism to *aggregate* evidence spread
across many chunks/documents or to *know what exists* before a query arrives.

An LLM-generated Wiki layer — entity/topic pages, generated summaries, or a knowledge graph, always
citing back to source chunks — is a well-established pattern for exactly this gap (GraphRAG,
RAPTOR, hierarchical RAG; see [Existing approaches](#existing-approaches)). It adds real value for
the query categories above. It is not free: it adds a second generation-and-storage pipeline, a new
staleness/lifecycle surface (what happens to a Wiki fact when its source document is deleted or
re-indexed), and a hallucination-amplification risk that one-shot RAG doesn't have — a wrong
generated fact, once stored, is retrieved and repeated indefinitely rather than being a one-time
model slip.

**Recommendation: build a narrow slice, in two stages, not the full idea at once.** A
**Lightweight Wiki (Option 1)** — per-document LLM summaries, a small flat list of extracted
entities/topics, and a small set of grounded natural-language claims (each tied to source chunk
IDs — the minimum representation that can actually state a relationship like "X publishes Y"
without typed `subject/predicate/object` triples or graph traversal) — is justified. **Stage 1**
generates this and stores it in PostgreSQL only, for direct inspection against the evaluation set;
**Stage 2** — embedding it into a second Qdrant collection and wiring it into chat retrieval — is
deliberately deferred until Stage 1 proves the generated content is accurate enough to be worth
surfacing to a user. A full knowledge graph (Neo4j, typed relationship traversal) is **not**
justified for this repository's current scale and query patterns; it would be solving a problem the
corpus doesn't yet have. See [Recommended MVP](#recommended-mvp) for the concrete two-stage scope
and [Evaluation plan](#evaluation-plan) for how to prove it before investing further. **The immediate
next step is neither Stage 1 nor Stage 2** — it is a small local-LLM generation spike (Stage 0, see
[Proposed next steps](#proposed-next-steps)) that validates the currently-configured local model can
actually produce this structure reliably, before any persistence/lifecycle code is written.

---

## Current architecture fit

**[Implemented]** Verified against [`AGENTS.md`](../AGENTS.md), [`docs/architecture/README.md`](../docs/architecture/README.md),
[`docs/document-lifecycle/README.md`](../docs/document-lifecycle/README.md),
[`docs/rag/README.md`](../docs/rag/README.md), [`docs/storage/README.md`](../docs/storage/README.md),
[`docs/providers/README.md`](../docs/providers/README.md),
[`docs/multilingual/README.md`](../docs/multilingual/README.md),
[`docs/langchain/README.md`](../docs/langchain/README.md), and the corresponding `app/` modules.

### Ingestion → retrieval pipeline today

```
Upload -> object storage save + Document/IngestionJob rows (PENDING)
       -> IngestionWorker: extract -> chunk -> embed -> Qdrant upsert -> Document marked indexed

Chat -> RuleBasedRagDecider (NEEDS_RETRIEVAL / DIRECT_LLM / CLARIFICATION_NEEDED / OUT_OF_SCOPE)
     -> RetrievalService.retrieve(query) -> embed query -> Qdrant top-k search (score threshold)
     -> RagPromptBuilder.build() -> labeled [S1]/[S2] context, PromptSource per chunk
     -> LLMProvider.stream_generate() -> SSE: metadata -> token(s) -> done | error
```

- **Ownership**: `app/services/documents/{text_extractor,chunker}.py` (extraction/chunking),
  `app/services/ingestion/worker.py` (embed + Qdrant upsert), `app/rag/retrieval_service.py`
  (query-time search), `app/rag/prompt_builder.py` (attribution), `app/rag/orchestrator.py`
  (`RagOrchestrator`, the reference decide→retrieve→prompt→generate composition), `app/rag/engines/`
  (`CustomRagEngine` default, `LangChainRagEngine` optional/parity-tested adapter — both produce the
  identical SSE contract and reuse the same `RetrievalService`/`PromptProvider`/provider factory,
  never reimplementing them).
- **Retrieval is flat**: `RetrievalService.retrieve()` does one embed + one Qdrant `search_similar()`
  call against the single active versioned collection, returns up to `retrieval_top_k` chunks above
  `retrieval_score_threshold`. There is no reranking, no query expansion, no multi-hop retrieval, and
  no notion of "what documents/entities exist" prior to a query — every question is answered purely
  from whatever chunks are nearest in embedding space.
- **Chunking is content-agnostic**: `DocumentChunker` splits page/sheet text into fixed-size,
  word-boundary-aware, overlapping chunks (`chunk_size=1000`, `chunk_overlap=200` by default),
  preserving only `page_number`/`sheet_name` — no semantic/topic-aware splitting, no per-document
  title or subject field beyond `original_filename`.
- **Attribution is chunk-level and mandatory**: `RagPromptBuilder` labels every context block
  `[S1]`/`[S2]`… tied 1:1 to a `PromptSource {document_id, chunk_id, source, score, page_number,
  sheet_name}`; a system-prompt instruction forbids inventing information outside the supplied
  context, and prompts explicitly preserve `[S]` labels and quoted text untranslated (see
  [Multilingual](#multilingual-considerations)).
- **Postgres is the lifecycle authority** — `Document`/`IngestionJob`/`DocumentDeletionJob`/
  `ReindexJob`/`VectorCleanupJob`/`IndexCollection` rows are the source of truth for what exists and
  its state; **object storage** is the only place original bytes live; **Qdrant is always rebuildable
  derived state** — never authoritative for anything, and every vector can be regenerated from
  Postgres + object storage via re-index. Any Wiki layer must respect the same three-way split, not
  invent a fourth kind of "source of truth."
- **Provider abstraction**: `app/rag/providers/provider_factory.py` resolves `EmbeddingProvider`/
  `LLMProvider`/`VectorStore` from settings; only Ollama (embedding + LLM) and Qdrant are actually
  implemented — OpenAI/Gemini/Anthropic LLM providers are typed stubs that raise
  `ProviderNotImplementedError`. Any Wiki-generation LLM call must go through this same factory, never
  construct a client directly (`AGENTS.md` §3).
- **Multilingual**: one shared `PromptCatalog`/`PromptProvider` for Hebrew/English, `bge-m3`
  multilingual embeddings (1024-dim, 100+ languages) by default. No translation step exists anywhere
  in the pipeline today.
- **Re-indexing**: build-ahead, deterministic point IDs (overwrite not duplicate), activation is a
  single atomic Postgres cutover that also creates a `VectorCleanupJob` for the vacated collection —
  the *document's* embeddings are always fully rebuildable, but nothing today regenerates anything
  *derived from* those embeddings (there is no derived layer yet).
- **Deletion lifecycle**: vectors-before-storage, always uses the tracked (not partial) vector
  deletion path across current + historical + re-index-target collections; a `410 Gone` (never `404`)
  is returned for a deleted document's original content. **No existing mechanism removes anything
  derived from a document beyond its own vectors and object** — this is the exact gap a Wiki layer's
  deletion semantics would need to fill (see [Lifecycle implications](#lifecycle-implications)).
- **Reconciliation** (`app/services/reconciliation/`): strictly read-only, cross-domain
  (Postgres+storage+Qdrant) audit producing severity-graded findings (`INFO`/`WARNING`/`ERROR`);
  never mutates, never repairs. A dependency failure becomes a `WARNING` finding, never silent
  absence-proof. This is the natural home for a future "Wiki inspection" finding, not a new
  subsystem (see [Reconciliation](#reconciliation)).
- **RAG engine boundary**: `RAG_ENGINE=custom|langchain`, both adapter-based over the same shared
  `RetrievalService`/`PromptProvider`/provider factory; LangGraph is explicitly deferred until a real
  multi-step/conditional workflow needs it.

### Where a Wiki layer could fit without violating current ownership boundaries

**[Proposed]** A Wiki-aware retrieval path (parallel retrieval, routing) would itself be exactly the
kind of multi-step/conditional workflow `docs/langchain/README.md` says LangGraph is deferred until —
worth flagging as the first plausible future argument *for* LangGraph, though still not a reason to
adopt it preemptively or as part of this feature's MVP.

- **Generation**: a new step *after* `IngestionWorker` completes (`IngestionJob.status == COMPLETED`),
  never inside it — ingestion's job is indexing, not knowledge synthesis, and `AGENTS.md` explicitly
  forbids adding new lifecycle statuses or business logic without being asked. A Wiki-generation step
  should be its own out-of-band job type (mirroring `IngestionJob`/`ReindexJob`'s pattern: its own
  table, its own worker/script, its own append-only status machine), not a field bolted onto
  `IngestionJob`.
- **Storage**: a new `app/services/wiki/` (or similar) sibling package — analogous to
  `app/services/reconciliation/` in that it may need to read from both `documents/` and `indexing/`
  domains (it needs a document's chunks and its active collection) without being imported *by*
  either. It must never be imported by `app/services/indexing/*` (one-directional dependency rule).
- **Retrieval-time participation**: a new method on (or alongside) `RetrievalService`, called
  optionally by the orchestrator — never replacing `RetrievalService.retrieve()`'s existing chunk
  search, and never changing the `POST /chat` SSE contract (`metadata → token(s) → done | error`)
  that both engines currently guarantee identically.
- **Provider use**: any Wiki-generation LLM call must go through `provider_factory.get_llm_provider()`
  — it is deployment-configured (Ollama today), not a new provider path.

---

## What an LLM Wiki means

**[Proposed]** distinctions, evaluated against this repository's actual query patterns.

**Terminology note:** what Stage 1 actually builds is more precisely a **Generated Knowledge Layer**
— per-document `summary` + `entities/topics` + `grounded claims` (each with source-chunk
provenance) — not a "Wiki" in the traditional sense of browsable pages. "Wiki" throughout this
document names the *feature concept*: the possible future user-facing presentation and retrieval
surface over that generated data (Stage 2 and beyond), not the Stage 1 data model itself. The rest of
this document continues to use "Wiki" as shorthand for the whole feature, consistent with the
research prompt it answers — this note exists only so the term isn't read as implying Stage 1
produces browsable pages or a retrieval surface, which it deliberately does not.

| Interpretation | Description | Verdict |
|---|---|---|
| **A. Generated topic pages** | LLM-authored prose page per entity/topic/document, with a summary, related-concepts list, and source references. | **Useful, in a lightweight form.** This is the most directly answerable-from-existing-primitives interpretation: a per-document summary is a bounded, cheap generation task with a natural provenance model (the whole document). A per-*topic* page that spans documents needs entity/topic extraction first (see B) but the *page* itself is still just prose + citations. |
| **B. Entity/relationship extraction** | Structured `(subject, predicate, object)` triples with typed relationships. | **Not adopted as typed triples — but a bounded slice is required for relationship questions to be answerable at all.** A flat entity list (name + description) alone cannot state "X publishes Y" — no field in that shape holds relational information. The minimum viable slice is **grounded natural-language claims** (one sentence + source chunk IDs each, e.g. "Reconciliation Service consumes PaymentCompleted events emitted by Payment Service" cited to chunk-17/chunk-42) — deliberately short of typed triples, subject/predicate/object fields, or graph traversal, but the smallest representation that actually contains a relationship. Typed relation extraction (full B) remains the highest-risk piece and is not adopted; see [Recommended MVP](#recommended-mvp). |
| **C. Hierarchical summaries** | Corpus → Document → Section → Chunk, with generated summaries at each level. | **Useful for the "broad overview" / "summarize everything about Y" query class**, and it composes naturally with A (a document-level summary *is* one level of this hierarchy). A full multi-level tree (RAPTOR-style recursive clustering) is more machinery than a document-level summary needs for this corpus's current size. |
| **D. Full graph-style knowledge layer** | Entities + typed relationships + topics + evidence, queried via graph traversal. | **Not justified yet.** This is the union of B and a dedicated graph store/traversal engine (see [Storage options](#storage-options)) — real complexity (a new database, new query patterns, new consistency model) for a corpus and query volume that hasn't demonstrated it needs multi-hop traversal over flat retrieval + a lightweight entity list. |

**Recommendation for this repository:** **A (document/topic summaries) is the core of a useful MVP
Wiki**, extended with a flat entity/topic list *and* a small set of grounded natural-language claims
(the bounded slice of B described above — required specifically so relationship/ownership/dependency
questions have an actual answer to retrieve, not just a list of unconnected entity names). **C**'s
document-level tier is already covered by A; going further up (corpus-level, cross-document
synthesis) or down (section-level) is not justified until this is proven to help (see
[Evaluation plan](#evaluation-plan)). **D**, and typed relationship extraction/traversal in general,
should not be built now — see [Recommended MVP](#recommended-mvp) and
[Option 2/3](#options-considered) for what would need to be true first.

---

## Existing approaches

**[External]** Researched, not adopted wholesale. Each entry: what it solves, ingestion/indexing
cost, retrieval-time implications, update/deletion complexity, hallucination/provenance risk, and fit
for this repository.

### GraphRAG (Microsoft Research)

Builds an LLM-extracted entity knowledge graph from source documents, then runs community detection
(the Leiden algorithm) over that graph and pre-generates multi-tier community summaries; at query
time, "global" questions are answered by map-reducing over relevant community summaries, while
"local" questions use direct entity/relationship lookup [GraphRAG: Improving global search via
dynamic community selection — Microsoft Research](https://www.microsoft.com/en-us/research/blog/graphrag-improving-global-search-via-dynamic-community-selection/);
[From Local to Global: A Graph RAG Approach to Query-Focused Summarization — Microsoft
Research](https://www.microsoft.com/en-us/research/publication/from-local-to-global-a-graph-rag-approach-to-query-focused-summarization/).
It specifically targets questions flat vector RAG cannot answer at all — "abstract, global" questions
with no single relevant passage (their own example: "catch me up on the last two weeks of updates").

- **Ingestion cost**: full entity/relationship extraction over every document, plus a graph-clustering
  pass and hierarchical community summarization — a substantial, LLM-call-heavy indexing pipeline on
  top of normal chunk/embed.
- **Retrieval-time implications**: two distinct retrieval modes (local entity lookup vs. global
  community-summary map-reduce), meaning query routing logic the current `RuleBasedRagDecider`
  doesn't have.
- **Update/deletion complexity**: the graph and its community structure must be re-clustered as the
  corpus changes — deleting or adding a document can shift community boundaries, not just remove a
  few nodes.
- **Fit for this repository**: not justified now. This solves a "broad corpus overview with no clear
  single answer" problem class this repository doesn't yet have evidence of needing, at a cost
  (community detection, dual retrieval modes) far beyond a document-count/query-volume that hasn't
  demonstrated the need. Worth revisiting only if [Option 1](#option-1--lightweight-wiki-recommended)
  proves genuinely insufficient for broad-overview queries.

### RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)

Recursively embeds, clusters, and summarizes chunks bottom-up into a tree with multiple abstraction
levels, then retrieves from whichever level(s) best match a query — collapsing chunk-level detail and
document-level abstraction into one retrieval structure rather than two separate systems
[RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval (arXiv:2401.18059)](https://arxiv.org/pdf/2401.18059).
It solves multi-step, cross-passage reasoning questions within long documents (their own benchmark:
+20% absolute accuracy on QuALITY when RAPTOR retrieval is coupled with GPT-4).

- **Ingestion cost**: recursive clustering + summarization at each tree level — cheaper than GraphRAG
  (no relationship extraction/community detection) but still a nontrivial multi-pass indexing job.
- **Retrieval-time implications**: retrieval must choose which tree level(s) to search, or search
  multiple levels and merge — more than a single flat top-k call.
- **Update/deletion complexity**: a tree built bottom-up from chunk clusters isn't naturally
  incremental — adding/removing one document's chunks can change cluster membership at every level
  above it, unlike this repository's existing chunk/embed model where a document's vectors are fully
  independent of every other document's.
- **Fit for this repository**: the *document-level* summary tier is worth adopting now (it is exactly
  Option 1's "A. generated topic pages" scoped to one document); the *recursive multi-level tree* is
  not — it targets long single-document, multi-step reasoning depth this platform's current document
  sizes and query patterns don't obviously require, and its non-incremental update story conflicts
  with this platform's per-document, always-rebuildable-independently lifecycle model.

### Knowledge-graph-enhanced RAG / entity-centric retrieval / hybrid semantic + structured retrieval

**[External]** General surveys ([Knowledge Graph-Guided Retrieval Augmented Generation (arXiv:2502.06864)](https://arxiv.org/pdf/2502.06864);
[Retrieval-Augmented Generation for Natural Language Processing: A Survey (arXiv:2407.13193)](https://arxiv.org/pdf/2407.13193))
describe the broader family this repository's proposed Option 1/2 sit inside: augmenting vector
retrieval with structured facts that carry explicit provenance, consistency constraints, and
queryable relationships, generally to reduce hallucination and answer relationship-shaped questions
vector similarity alone can't reliably surface.

- **Problem solved**: the same class Option 1/B targets — "how is X related to Y", "who owns this" —
  where the *answer* is a fact a knowledge graph can state directly rather than a passage a similarity
  search must get lucky enough to retrieve.
- **Provenance-aware knowledge graphs specifically**: verified/curated data with source traceability
  per fact is the pattern this document's [Grounding and provenance](#grounding-and-provenance-requirements)
  section adopts directly — every generated fact must carry its supporting chunk/document IDs, not be
  presented as free-standing truth.
- **Fit**: informs the recommended MVP's provenance model; does not itself justify a graph database
  (see [Storage options](#storage-options) — the provenance requirement is served by a Postgres foreign
  key, not a graph engine).

### Hallucination amplification in generated/stored knowledge

**[External]** A one-time RAG hallucination is bounded to a single response; a hallucinated fact
written into a persistent knowledge store and later retrieved is not — subsequent queries repeat it
as if it were sourced. Attribution/provenance techniques (aligning generated spans to source passages
with a confidence score) measurably reduce but do not eliminate this: one cited approach reports
token-level provenance matching ground truth in "78% of cases, a 15-point gain over post-hoc citation
methods" [Attribution Techniques for Mitigating Hallucinated Information in RAG Systems: A Survey — UBOS](https://ubos.tech/attribution-techniques-for-mitigating-hallucinated-information-in-rag-systems-a-survey-4/),
and more generally "mechanisms that reduce incidental hallucination can amplify adversarial error when
the knowledge base is compromised" (same source) — i.e., provenance metadata reduces *undetected*
hallucination but a wrong-but-plausible generated fact with a real-looking citation is not
self-correcting. This directly motivates treating any Wiki content as **derived and
non-authoritative, never retrieved without also surfacing/verifying its cited source** (see
[Grounding and provenance](#grounding-and-provenance-requirements)).

### Corpus-wide topic discovery at index time

**[External]** Some hierarchical-RAG approaches perform unsupervised topic discovery across the whole
corpus at indexing time, building a persistent, query-independent knowledge base rather than a
per-query structure (general pattern noted across sources on hierarchical RAG). This is the
"cross-document summaries"/"topic pages" end of the spectrum in the research prompt (interpretation
A/C at the corpus level) — explicitly **out of MVP scope** here; see [Recommended MVP](#recommended-mvp).

### What was deliberately not adopted wholesale

No framework (LangChain graph tooling, a GraphRAG library, a dedicated agent framework) is assumed
necessary. `AGENTS.md` explicitly forbids introducing new heavy dependencies (LangChain itself was
only adopted as an *optional* adapter, and LangGraph remains deferred) without being asked, and this
research's job is to learn from these approaches' *ideas* (provenance modeling, hierarchical
summarization, entity extraction) — not to import their tooling.

---

## Problems it could solve — query-type comparison

**[Proposed]** assessment against the query categories in the research brief, using the current
architecture described above.

| Query type | Existing flat retrieval sufficient? | Retrieval tuning alone (reranking, top-k, threshold) helps? | Hierarchical summaries help? | LLM Wiki materially helps? | Graph traversal required? |
|---|---|---|---|---|---|
| Single-chunk factual ("what is X's timeout value") | **Yes** — this is exactly what top-k chunk search + attribution is built for. | N/A | No | No | No |
| "How is X related to Y?" | **No** — only works if one chunk happens to state the relationship explicitly. | Marginal — reranking doesn't create a fact that isn't in any single chunk. | Marginal | **Yes, if grounded claims are generated** — a flat entity list alone cannot answer this (no field holds relational information); a generated claim sentence with source citations can. | Not required for a 1-hop relation; would matter for multi-hop chains. |
| "What services depend on X?" | **No** — same shape as above, often worse (dependency facts are frequently scattered across multiple documents, none phrased as a list). | Marginal | Marginal | **Yes, if grounded claims are generated** — same requirement as above; an entity list of names/descriptions alone is insufficient. | Only if the answer requires transitive/multi-hop dependency chains, which this corpus hasn't demonstrated. |
| "Summarize everything the documents say about Y" | **Partial** — top-k retrieval caps at `retrieval_top_k` chunks; a topic discussed across 20 chunks in 5 documents will be under-represented. | Marginal — raising top-k trades recall for prompt bloat/noise, doesn't fix the ceiling. | **Yes** | **Yes** — this is squarely what a per-document/per-topic summary is for. | No |
| "Who owns this component?" | **Depends** — works if stated in one retrievable chunk; fails if ownership is implicit/scattered. | Marginal | No | **Yes**, if ownership is captured as an extracted fact with provenance. | No |
| "What changed across these documents?" | **No** — the current pipeline has no document-to-document diffing or versioning concept at all (re-indexing rebuilds vectors, it doesn't track content deltas). | No | No | **Partial** — a Wiki could note "document A supersedes/contradicts document B" if explicitly modeled, but this is a new capability, not a byproduct of summarization. | No — this is a provenance/versioning problem, not a graph-traversal problem. |
| "Give me an overview of this architecture" | **No** — no single chunk is an overview; flat retrieval returns fragments, not synthesis. | Marginal | **Yes** | **Yes** — corpus/topic-level summary is exactly this. | No |
| Questions needing evidence from many distant chunks | **No**, structurally — `retrieval_top_k` bounds how many chunks even reach the prompt. | Helps somewhat (better ranking of a fixed budget) but doesn't remove the ceiling. | **Yes** | **Yes** | Only if "distant" means graph-distant (multi-hop), not just corpus-distant. |

**Conclusion**: the query classes an LLM Wiki genuinely helps with are relationship/ownership
questions and broad/summary questions — not single-fact lookups, which the existing pipeline already
handles well. None of the evaluated query classes in this repository's actual scope require graph
*traversal* (multi-hop reasoning) — they require **aggregation** (many chunks → one fact/summary,
served by summaries), **discoverability** (what entities/topics exist at all, served by the flat
entity/topic list), and **existence of a relationship-shaped fact that no single chunk states as such**
(served specifically by grounded natural-language claims — an entity list alone cannot state a
relationship; see [Grounding and provenance](#grounding-and-provenance-requirements)). All three are
distinct roles within one lightweight **summary/entity/grounded-claim layer with provenance** — not
interchangeable, and not reducible to "a flat entity/summary layer" alone.

---

## Grounding and provenance requirements

**[Proposed]** A generated Wiki is synthesized data, never primary truth — this section is
non-negotiable for any MVP.

### Provenance model

**Every generated record belongs to exactly one document — this is a deliberate MVP simplification,
not an oversight.** If three documents each independently support the same real-world claim, the MVP
stores three independent records, one per document, each with its own single-document provenance.
There is no cross-document entity deduplication, no fact merging, and no shared/canonical record in
the MVP — a generated record's `document_id` is fixed at generation time and never re-attributed.
Cross-document synthesis (recognizing that two independently-generated records describe the same
real-world fact) is an explicitly deferred future capability, not part of this MVP — see
[Recommended MVP](#recommended-mvp).

Every generated summary, entity, or claim must carry:

```json
{
  "id": "wiki-record-uuid",
  "document_id": "source document id (Postgres Document.id) — exactly one, always",
  "kind": "document_summary | entity | claim",
  "text": "generated prose: the summary text, the entity's one-line description, or the claim sentence",
  "sources": [
    { "chunk_id": "..." }
  ],
  "source_content_hash": "Document.content_hash at generation time",
  "chunking_version": "Document.chunking_version at generation time",
  "generation_model": "provider/model used",
  "generation_prompt_version": "internal version of the generation prompt/instructions used",
  "generated_at": "timestamp"
}
```

A `claim` record (e.g. "Reconciliation Service consumes PaymentCompleted events emitted by Payment
Service", sourced to `chunk-17`/`chunk-42`) is a natural-language sentence, never a typed
`subject`/`predicate`/`object` triple — see [What an LLM Wiki means](#what-an-llm-wiki-means) for why
this bounded slice, rather than typed relationship extraction, is what the MVP adopts.

- **Chunk-level provenance is mandatory**, not optional — every record must cite the exact chunk(s) it
  was derived from, mirroring `PromptSource {document_id, chunk_id}` already used for retrieval
  attribution. Document-level-only provenance ("this came from document X somewhere") is insufficient
  to let a reader or a downstream retrieval step verify a claim. Since every record belongs to one
  document (above), `sources` need only carry `chunk_id` — the owning `document_id` is already the
  record's own field, not a per-source repetition.
- **No confidence/support-count model in the MVP**: because records are never merged across
  documents, there is no notion of "this claim is supported by N documents" to track — each record's
  confidence is simply "does its own cited chunk actually support its own text," independently
  verifiable per record. A future cross-document synthesis capability could reintroduce a
  support-count concept once merging itself is designed; the MVP does not need it.
- **Contradictory documents**: out of MVP scope, not merely "handled naively." Detecting that two
  documents disagree requires comparing records *across* documents — the same cross-document
  operation this MVP deliberately excludes. Two documents making conflicting claims simply produce two
  independent claim records, each accurately grounded in its own source; if both are retrieved
  together at answer time (once Stage 2 retrieval integration exists), the conflict is visible to the
  answering LLM and the reader from the two records' citations, without the platform itself detecting
  or flagging it as a "contradiction." Automatic contradiction detection is a future capability that
  depends on cross-document synthesis existing first.
- **Staleness tracks the source content, chunking, and generation itself — never the embedding model
  or Qdrant index.** `source_content_hash`/`chunking_version` detect whether the text a record was
  actually generated from could have changed; `generation_model`/`generation_prompt_version` detect
  whether a later, improved generation pass would produce different output from the *same* source.
  None of these are affected by re-embedding or re-indexing: a document's `Document.content_hash` and
  `Document.chunking_version` do not change merely because its vectors are rebuilt under a new
  embedding model (see [Document update / re-index](#document-update--re-index) — re-index reuses the
  same extracted/chunked text). A Wiki record's `embedding_version`/Qdrant collection is therefore
  **never** part of its staleness signal — see [Lifecycle implications](#lifecycle-implications) for
  the one case (a `chunking_version` bump) where it does matter.
- **Deletion semantics**: when a document reaches `DocumentDeletionJob.COMPLETED`, every Wiki record
  for that `document_id` is deleted outright — no recomputation, no partial removal, since every
  record already belongs to exactly one document by construction. This is materially simpler than a
  cross-document model would require, which is itself a reason to prefer the per-document-only MVP
  scope.
- **Regeneration semantics**: regeneration is always a full replace of a document's Wiki content, not
  an in-place patch — mirroring how re-index always rebuilds a document's vectors wholesale rather
  than diffing. This keeps the provenance model simple (no partial-update reconciliation logic) at
  the cost of some redundant LLM calls on unrelated re-chunking triggers — an acceptable MVP trade
  given Wiki generation is explicitly non-authoritative and best-effort.
- **Never used without retrievable evidence**: a Wiki fact/summary must never be presented to an end
  user (in a chat answer) without the underlying chunk citation being retrievable/verifiable — i.e.,
  if Wiki content participates in an answer, its `sources` chunk IDs are surfaced exactly like
  `PromptSource` chunk IDs are today, not replaced by "the Wiki says." This is the direct mitigation
  for hallucination amplification: a wrong generated fact, if it must always drag its (real,
  re-checkable) source citation along, is falsifiable by a reader in the same way a wrong RAG answer
  already is — it does not get to become an unverifiable "known fact" the system just repeats.

### Hallucination amplification — explicit treatment

**[Proposed]** This is the single biggest risk this feature introduces that the current architecture
does not have. Current RAG's worst failure mode is a wrong one-shot answer over correctly-retrieved
context, bounded to one response. A Wiki's worst failure mode is a wrong fact that gets *written down
once* and then *retrieved and repeated* across many future answers — the error compounds instead of
being independently re-rolled each time. Mitigations are split by stage below — "Stage 1" and "Stage
2" here mean exactly [Recommended MVP](#recommended-mvp)'s Stage 1 (generation + Postgres + offline
inspection, no retrieval integration) and Stage 2 (conditional Qdrant + chat retrieval integration);
"the MVP" alone is never used in this section to avoid ambiguity between the two.

**Stage 1 mitigations** (apply regardless of whether Stage 2 is ever built):

1. Generated knowledge (summary/entity/claim) is explicitly derived and non-authoritative — never
   treated as a fact the platform itself vouches for (see
   [Grounding and provenance](#grounding-and-provenance-requirements)).
2. Every summary/entity/claim carries chunk-level provenance, never document-level-only or
   freestanding text (above).
3. Generated records are inspected/evaluated (Stage 1's validation step, see
   [Recommended MVP](#recommended-mvp)) *before* any retrieval integration exists at all — a bad
   generation never reaches a user through chat, because chat retrieval simply doesn't read this data
   in Stage 1.
4. Wiki generation failure never blocks or degrades normal chat/retrieval — see
   [Failed generation](#failed-generation).
5. No automatic write-back: nothing feeds a generated record into another generated record's
   generation (no recursive summarization of summaries) — this specifically avoids the "generation
   reads previously-generated, possibly-wrong content as if it were source" compounding failure mode
   that makes agentic/iterative systems worse than one-shot ones.
6. No cross-document synthesis or merging (see
   [Grounding and provenance](#grounding-and-provenance-requirements)) — a wrong generated record can
   only ever be as wrong as its own single document's generation, never amplified by being folded
   into a merged, multi-document "consensus" fact.

**Stage 2 mitigations** (apply only if Stage 2 is actually built):

7. Wiki retrieval is always additive context alongside normal chunk retrieval, never a replacement
   for it (see [Retrieval architecture](#retrieval-architecture)) — the LLM generating the final
   answer still sees real chunks, not just the Wiki's claim.
8. Generated context remains distinctly labeled from source-chunk context in the prompt (e.g. `[W1]`
   vs. `[S1]`), never merged into the same label space — a reader/prompt can always tell generated
   content from source-grounded content apart.
9. Source chunks remain independently retrievable/verifiable — Wiki participation in an answer never
   substitutes for, or hides, the underlying `PromptSource`-style chunk citation (above).
10. The existing chat grounding/citation contract (`POST /chat`'s `sources`, the `[S]`-label
    attribution behavior) remains intact and unchanged in kind — Wiki content only ever adds more
    labeled, cited context blocks to it.

---

## Lifecycle implications

**[Proposed]**, following the same append-only, typed-result, explicit-not-automatic conventions
`AGENTS.md` establishes for every existing job type.

### Upload / ingestion

Generate **asynchronously, post-ingestion** — not during normal ingestion, and not synchronously
on-demand for the MVP. Rationale: `IngestionJob`'s job is indexing (the retrieval-critical path);
adding LLM summarization/extraction to it would make normal document availability depend on Wiki
generation succeeding, which is exactly backward given Wiki content is non-authoritative. A new job
type (e.g. `WikiGenerationJob`, own table, own append-only status machine, triggered only after
`IngestionJob.status == COMPLETED`) mirrors the existing `ReindexJob` pattern: scheduled separately,
processed by its own worker/script, never invoked from an HTTP-request path. **Scheduled corpus
rebuilding** (regenerating cross-document/corpus-level summaries) is explicitly out of MVP scope —
see [Recommended MVP](#recommended-mvp).

### Document update / re-index

Re-indexing rebuilds a document's vectors from the same stored, previously-extracted/chunked text —
`app/services/indexing/reindex_service.py`'s build step re-extracts and re-chunks under the pinned
target configuration, but for an embedding-model-only re-index, `chunking_version` is unchanged. A
document's own Wiki summary/entities/claims therefore do **not** need regeneration, and are **not**
marked stale, purely because of a re-index that only changes the embedding model/collection — the
source text a record cites is unaffected. **The only re-index-adjacent trigger that matters is a
`chunking_version` change**: if re-chunking logic itself changes (a new `chunking_version`, not just a
new embedding model), a Wiki record's `source_chunk_ids` may no longer correspond to the same text —
compare the record's stored `chunking_version` against the document's current one to detect this case
specifically, never against `embedding_version`/`indexed_at`. Nothing in the MVP auto-regenerates
Wiki content in either case; staleness is flagged for operator attention (see
[Reconciliation](#reconciliation)), and regeneration stays a bounded, explicit action, exactly like
vector cleanup and activation are today.

### Deletion

- On `DocumentDeletionJob` reaching `COMPLETED` (never earlier, mirroring how `content_hash` is only
  released on `COMPLETED`, never `PENDING`/`PROCESSING`/`PARTIALLY_FAILED`): delete every Wiki
  record for that `document_id` outright. Because every record belongs to exactly one document by
  construction (see [Grounding and provenance](#grounding-and-provenance-requirements)), this is a
  plain delete-by-`document_id` — no recomputation, no partial removal, no cross-document
  contribution tracking to update.
- No regeneration is triggered synchronously by deletion, and there is no corpus-level summary in the
  MVP to become stale as a side effect (corpus-level synthesis is explicitly out of scope — see
  [Recommended MVP](#recommended-mvp)).

### Failed generation

**Wiki state must be explicitly derived/non-authoritative — this is the load-bearing design decision
for the whole feature.** A document with `IngestionJob.status == COMPLETED` and
`WikiGenerationJob.status == FAILED` (or no Wiki job at all) must remain **fully usable in ordinary
RAG chat** — retrieval, prompt building, and generation over its chunks are completely unaffected by
Wiki generation failing, exactly as they are today (the current pipeline has zero dependency on
anything Wiki-shaped). This is enforced structurally by keeping Wiki generation a separate job type
with no dependency edge *into* the ingestion/retrieval path, only *out of* it (reads completed
ingestion state, never gates it).

### Reconciliation

**[Proposed]** Wiki state belongs in reconciliation as a new, clearly-scoped finding category — not a
new subsystem. `app/services/reconciliation/document_audit_service.py` already has the exact shape
needed: a `WIKI_GENERATION_FAILED` finding, and a `WIKI_CONTENT_STALE` finding specifically when a
record's stored `chunking_version` no longer matches the document's current one (never triggered by
an embedding-model/index change alone — see [Document update / re-index](#document-update--re-index)),
both at `INFO`/`WARNING` severity (never `ERROR` — a missing/stale Wiki is never a correctness problem
for ordinary RAG, unlike a missing vector, which is), following the same pattern as
`REINDEX_TARGET_BUILT_NOT_ACTIVATED` or `VECTOR_CLEANUP_INCOMPLETE`: read-only, informational, "an
operator may want to run the existing bounded regeneration command," never auto-triggering anything
from the audit itself.

---

## Storage options

**[Proposed]**

### PostgreSQL

New tables: `wiki_documents` (one row per document, holding the generated summary text +
generation metadata), `wiki_entities` (extracted entity/topic name + description), `wiki_claims`
(one grounded natural-language claim sentence per row), each with a `sources` JSONB column of chunk
IDs (given the low query complexity needed — a join table is not warranted at this scale).

- **Pros**: reuses the platform's existing lifecycle-authority system; transactional consistency with
  document deletion (delete Wiki rows in the same worker transaction that finalizes
  `DocumentDeletionJob`); no new infrastructure; trivial to add a reconciliation finding against it
  (already same database); foreign-key `ON DELETE` behavior can enforce the provenance-cleanup
  invariant structurally rather than relying on application code alone.
- **Cons**: no native semantic search over generated text — a "find the Wiki page about X" query
  needs either a Qdrant-side index (below) or a slow `ILIKE`/full-text search.

### Qdrant (additional collection) — **Stage 2, not part of the first MVP**

Embed generated Wiki pages/entity/claim text into a second, clearly-named Qdrant collection
(distinct from the chunk collection — never mixed into the same one, since a Wiki page's embedding
and a chunk's embedding are not interchangeable retrieval units) — **only after** Stage 1 (generation
+ Postgres storage + offline inspection, see [Recommended MVP](#recommended-mvp)) has shown the
generated content is accurate enough to be worth making retrievable at all. Building this
unconditionally as part of the first increment would mean committing retrieval infrastructure before
there is any evidence the underlying generated content is good — see
[Retrieval architecture](#retrieval-architecture) for the same staging applied to the retrieval side.

- **Pros** (once warranted): reuses the exact same `EmbeddingProvider`/`QdrantVectorStore`
  abstractions already in place; gets semantic Wiki lookup essentially for free, using the same
  multilingual embedding model already validated for Hebrew/English; still fully rebuildable from
  Postgres (same "Qdrant is never authoritative" invariant this platform already holds for chunk
  vectors).
- **Cons**: one more collection to version/rebuild if the embedding model/version changes; one more
  thing a reindex-style operation eventually needs; real infrastructure cost to pay *before* Stage 1
  has validated it's worth paying, if built prematurely.

### Graph database (Neo4j or similar)

- **Evaluation**: **premature for the MVP and likely premature beyond it**, given the query-type
  analysis above found no evaluated query class in this repository's actual scope that requires
  multi-hop graph traversal (as opposed to a flat entity list with a "related to" field, which
  Postgres already expresses fine). Adopting a graph database means a second database technology,
  a new consistency/backup/operational story, and a new query language (Cypher) — real infrastructure
  cost — to serve a traversal need that hasn't been demonstrated. `AGENTS.md` explicitly warns against
  introducing new heavy dependencies without being asked; a graph DB is exactly that.

### Recommended: Hybrid, Postgres-primary

```
Stage 1 (MVP): PostgreSQL only
  authoritative generated structure (summaries, entities, grounded claims) + provenance (chunk refs)

Stage 2 (only if Stage 1 proves the content is good — not built up front):
  Qdrant (new collection)
    semantic index over Wiki page/entity/claim text, for "search the Wiki" style lookup
```

This mirrors the platform's existing hybrid exactly (Postgres authoritative, Qdrant derived/
rebuildable) rather than inventing a new storage pattern — the only change from the platform's usual
pattern is that, for this specific feature, the Qdrant half is deliberately staged after validation
rather than built alongside Postgres from day one.

---

## Retrieval architecture

**[Proposed]** **Everything in this section is Stage 2 — not part of the first MVP.** The first MVP
(Stage 1, see [Recommended MVP](#recommended-mvp)) generates Wiki content and stores it in Postgres
for direct, offline inspection; it does not touch `RetrievalService`, `RagPromptBuilder`,
`RagOrchestrator`, or the `POST /chat` contract at all. This section describes how retrieval
integration *would* work, to be built only once Stage 1 has shown the generated content is worth
retrieving — building any of the following unconditionally as part of the first increment would mean
committing to production retrieval-contract changes before there is evidence the underlying
generated content is accurate.

### Parallel retrieval (recommended for Stage 2, if Wiki retrieval is added at all)

```
Query
 ├─ raw chunk retrieval (RetrievalService, unchanged)
 └─ Wiki retrieval (new, optional call — entity/summary lookup by embedding or exact-name match)
        ↓
     merge into RagPromptBuilder's context (Wiki content labeled distinctly from chunk content,
     e.g. [W1] vs [S1], never merged into the same label space — so a reader/prompt can tell
     generated context from source-grounded context apart)
        ↓
        LLM
```

- Keeps `RetrievalService.retrieve()` completely unchanged — Wiki retrieval is strictly additive,
  called from the orchestrator (or not at all, for a first cut — see MVP scope), never replacing the
  existing chunk search.
- Preserves the `POST /chat` SSE contract exactly (`AGENTS.md`'s explicit requirement for any
  RAG-adjacent change) — Wiki-sourced context is just more labeled context blocks in the same
  `BuiltRagPrompt`, with `PromptSource`-equivalent attribution so `sources` in the response still
  points at real, retrievable evidence (never a bare "from the Wiki").

### Wiki-first expansion

```
Query -> Wiki entity/topic lookup -> identify relevant source chunks -> grounded retrieval
```

- More sophisticated (uses the Wiki as an index *into* chunk retrieval, not just extra context), but
  requires the Wiki's entity→chunk mapping to be reliable and complete enough to trust as a filter —
  a bigger bet on Wiki quality than parallel retrieval, which only ever adds context and never removes
  a chunk the flat search would have found anyway. Worth revisiting once Option 1 has proven the
  Wiki's extraction quality is trustworthy.

### Query routing

Deciding *which* retrieval path a question needs (chunk-only vs. chunk+Wiki vs. Wiki-first) is itself
a new decision axis, on top of `RuleBasedRagDecider`'s existing 4-way classification. **Not necessary
for MVP**: parallel retrieval run unconditionally (Wiki lookup is cheap — one extra embedding + small
Postgres/Qdrant query) sidesteps needing a router at all, at the cost of always paying that small extra
latency. Routing only becomes worth its complexity once query volume/cost pressure justifies avoiding
the Wiki lookup for the (majority) single-fact queries that don't need it.

---

## Multilingual considerations

**[Proposed]**

- **Generation language**: generate Wiki summaries/entity descriptions **in the source document's
  language**, not translated — this matches the existing platform principle that source text (quoted
  chunks, titles) is never translated (`docs/multilingual/README.md`). A Hebrew document should get a
  Hebrew-language summary.
- **No single canonical internal language**: introducing one would require a translation step this
  platform has deliberately avoided everywhere else (citations, source titles) — adding it only for
  Wiki content would be a new, inconsistent behavior.
- **Multilingual aliases for entities**: worth a `canonical_name` + list of `aliases` (e.g. an entity
  extracted from both a Hebrew and an English document under different surface forms) — but this is
  an MVP-plus refinement, not required for a first cut where entities are scoped per-document.
- **Hebrew query over English content / vice versa**: the existing `bge-m3` embedding model already
  handles this for chunk retrieval (100+ languages, validated in
  `docs/multilingual/README.md`/`MultilingualFakeEmbeddingProvider` tests) — embedding Wiki
  summaries/entities with the same model should transfer the same cross-lingual matching capability
  without any extra translation machinery. This should be **verified empirically** as part of the
  evaluation plan (a Hebrew-query-over-English-Wiki-content case), not assumed.
- **Simplest viable approach**: reuse the platform's existing embedding provider and language
  detector unchanged; do not add a translation provider or a language-specific Wiki catalog.

---

## Options considered

**[Proposed]**

### Option 0 — No Wiki

Improve current RAG only: raise `retrieval_top_k` for broad questions, add reranking, tune
`retrieval_score_threshold`, or add a lightweight query-expansion step. These are real, cheap
improvements — but they cannot fix the structural gap (no chunk, however well-ranked, states a fact
that spans documents; no amount of reranking manufactures a corpus overview). Worth doing regardless
of the Wiki decision, since they're low-risk and independently valuable — but they don't substitute
for it on the relationship/overview query classes identified above.

### Option 1 — Lightweight Wiki (recommended)

Per-document LLM summary + flat extracted entity/topic list + grounded natural-language claims,
chunk-level provenance, Stage 1 stored in Postgres only (no Qdrant, no chat integration); Stage 2
(conditional on Stage 1 results) adds an optional Qdrant semantic index and parallel-retrieval chat
participation. No graph DB, no typed relationship traversal ever, no cross-document fact merging in
either stage, generated asynchronously post-ingestion, explicitly non-authoritative/best-effort.

| Dimension | Assessment |
|---|---|
| User value | Stage 1 alone (inspectable Postgres content) already validates whether the relationship/ownership/overview query classes are answerable; Stage 2 is what makes that value reachable through chat |
| Implementation complexity | Moderate — one new job type/worker, one new package, reuses every existing provider/storage abstraction; Stage 2's retrieval/Qdrant work is not paid unless Stage 1 justifies it |
| Operational complexity | Low — no new infrastructure component in Stage 1; Stage 2 adds one Qdrant collection, only if reached |
| Lifecycle impact | Contained — one new append-only job type; deletion is a plain delete-by-`document_id` (no cross-document recomputation, since records are never merged); reconciliation hooks as described above |
| Storage changes | Stage 1: three new Postgres tables (`wiki_documents`, `wiki_entities`, `wiki_claims`). Stage 2 (conditional): one additional Qdrant collection |
| Retrieval changes | Stage 1: none — `RetrievalService`/`POST /chat` untouched. Stage 2 (conditional): additive only (parallel retrieval); `POST /chat` contract still unchanged |
| Hallucination risk | Present but bounded and mitigated (mandatory single-document provenance, additive-not-replacing retrieval once Stage 2 exists, never authoritative) |
| Testability | High — deterministic provenance checks, fakeable LLM provider (same pattern as existing RAG tests), no real-model dependency needed in unit/integration tiers |
| Portfolio value | High — demonstrates provenance-aware generation design, a genuinely differentiating RAG-platform capability |
| Future extensibility | Clean path to Option 2 (typed relations) if entity extraction proves reliable |

### Option 2 — Structured knowledge graph

Typed entities + typed relationships + provenance edges + graph-style retrieval (still likely
Postgres-modeled — adjacency/edge tables — rather than a dedicated graph DB, unless traversal need is
proven).

| Dimension | Assessment |
|---|---|
| User value | Higher than Option 1 *only if* multi-hop relationship questions actually appear in real usage — unproven for this corpus |
| Implementation complexity | High — typed relation extraction is a harder, more hallucination-prone LLM task than summarization |
| Operational complexity | Moderate-high if a real graph DB is introduced; moderate if kept in Postgres |
| Lifecycle impact | Harder — a typed relationship's provenance/deletion semantics are more complex than a flat fact's (which relation "wins" under contradiction, cascading edge deletion) |
| Storage changes | New edge/relation tables at minimum; a new database if traversal need is proven |
| Retrieval changes | Needs actual graph queries or multi-hop expansion logic — more than parallel retrieval |
| Hallucination risk | Higher — typed relation extraction is more failure-prone than "summarize this document" |
| Testability | Harder — correctness of a typed relation is a harder thing to assert deterministically than "a citation exists" |
| Portfolio value | High, but only if genuinely warranted — an unjustified graph is a portfolio negative (looks like resume-driven development), not a positive |
| Future extensibility | This *is* the extension of Option 1 — build it only after Option 1's entity extraction is validated as reliable |

### Option 3 — Advanced GraphRAG / hierarchical architecture

Community detection, multi-tier summarization, dual local/global retrieval modes (per
[GraphRAG](#graphrag-microsoft-research)/[RAPTOR](#raptor-recursive-abstractive-processing-for-tree-organized-retrieval)
above).

| Dimension | Assessment |
|---|---|
| User value | Only relevant at a corpus scale/query diversity this repository hasn't demonstrated |
| Implementation complexity | Very high — clustering, community summarization, dual retrieval modes, query routing |
| Operational complexity | High — significant new indexing pipeline, likely new infra |
| Lifecycle impact | Severe — non-incremental updates (re-clustering on document add/remove) conflict directly with this platform's per-document-independent lifecycle model |
| Storage changes | Substantial |
| Retrieval changes | Substantial — new routing logic, likely the first real justification for LangGraph |
| Hallucination risk | Compounds across summarization layers |
| Testability | Hard — community structure is nondeterministic-ish and hard to assert against in CI without real models |
| Portfolio value | High risk of over-engineering signal for a portfolio/learning project at this stage — better demonstrated *after* proving the simpler layer's value |
| Future extensibility | N/A — this is the ceiling, not a stepping stone |

Not justified now. Revisit only if Option 1, once evaluated, shows the corpus/query pattern genuinely
needs global/abstractive synthesis beyond what per-document summaries provide.

---

## Recommended MVP

**[Proposed]**

**Should this be built at all?** Yes, as a narrow slice (Option 1), **staged in two increments** — the
query-type analysis shows a real, currently-unserved gap (relationship/ownership/overview questions),
and the provenance model above bounds the hallucination-amplification risk to something no worse than
the platform's existing "a Qdrant write can succeed while Postgres fails" documented-not-glossed-over
risk posture. The two-stage split exists specifically so retrieval-contract changes (Stage 2) are
never built before there is evidence the generated content (Stage 1) is worth surfacing — see the
reasoning in [Retrieval architecture](#retrieval-architecture) and [Storage options](#storage-options).

**Stage 1 itself is not the immediate next step.** Before committing to `WikiGenerationJob` or any
new database table, this repository should first run a small, throwaway generation spike against the
actually-configured local LLM — see [Proposed next steps](#proposed-next-steps)'s **Stage 0**. Stage 1
below describes the target shape *once* Stage 0 shows local generation is reliable enough to justify
building persistence and lifecycle machinery around it.

### Stage 1 — generate and validate (the target architecture, once Stage 0 justifies it)

1. A new `WikiGenerationJob` table/lifecycle (own append-only state machine: `PENDING → PROCESSING →
   COMPLETED | FAILED`, one active job per document, mirroring `IngestionJob`'s pattern exactly),
   triggered only after a document's `IngestionJob` reaches `COMPLETED` — never blocking or
   participating in ingestion itself.
2. Generation produces, per document: one LLM-written summary (plain prose, source language, citing
   which chunks it drew from); a small flat list of extracted entities/topics (name + one-line
   description + supporting chunk IDs); and a small set of grounded natural-language claims (one
   sentence + supporting chunk IDs each — e.g. "Reconciliation Service consumes PaymentCompleted
   events emitted by Payment Service" cited to specific chunks). No typed relationships, no
   cross-document synthesis, no fact merging across documents.
3. Storage: three Postgres tables (`wiki_documents`, `wiki_entities`, `wiki_claims`), each row scoped
   to exactly one `document_id`, with mandatory chunk-id provenance and the `source_content_hash`/
   `chunking_version`/`generation_model`/`generation_prompt_version` fields from
   [Grounding and provenance](#grounding-and-provenance-requirements) — **no Qdrant collection, no
   chat/retrieval integration in this stage.**
4. Validation: inspect the stored Postgres content directly against the [Evaluation plan](#evaluation-plan)'s
   query categories — does the generated summary/entities/claims for the relevant document(s) actually
   contain a correct, groundable answer, read manually (or scripted) off the rows themselves? This is
   the checkpoint that decides whether Stage 2 is worth building at all.
5. Deletion: on `DocumentDeletionJob.COMPLETED`, delete every Wiki row for that `document_id` — a
   plain delete-by-`document_id`, no cross-document recomputation (see
   [Grounding and provenance](#grounding-and-provenance-requirements)).
6. Reconciliation: one new `INFO`/`WARNING`-only finding category for missing/stale/failed Wiki
   generation, added to the existing `document_audit_service.py`.
7. Failure isolation: a document with failed/absent Wiki generation remains 100% usable in ordinary
   chat — enforced by construction (no dependency edge from ingestion/retrieval into Wiki state; Stage
   1 doesn't touch the chat path at all).

### Stage 2 — retrieval integration (only if Stage 1's validation succeeds)

8. Embed Wiki summary/entity/claim text into a new, dedicated Qdrant collection (see
   [Storage options](#storage-options)).
9. Wiki content participates in chat answers only as **additive, distinctly-labeled** context
   alongside normal chunk retrieval (parallel retrieval, no routing — see
   [Retrieval architecture](#retrieval-architecture)) — `POST /chat`'s SSE contract and `sources`
   attribution shape are unchanged in kind (still chunk-cited), only larger in what can populate them.

**What should explicitly NOT be included in either stage:**
- No typed relationship extraction/traversal (Option 2).
- No knowledge graph / graph database (Option 2/3).
- No cross-document fact merging, entity deduplication, or canonical cross-document facts — every
  record belongs to exactly one document, always (see
  [Grounding and provenance](#grounding-and-provenance-requirements)).
- No corpus-level synthesis, no community detection, no multi-tier hierarchy (Option 3) — only
  per-document summaries/entities/claims.
- No Wiki-first retrieval expansion or query routing, even in Stage 2 — parallel retrieval only.
- No automatic regeneration on re-index (staleness is flagged only when `chunking_version` actually
  changes, never on an embedding-model-only re-index — see
  [Document update / re-index](#document-update--re-index)).
- No scheduled/batch corpus rebuilding.
- No multilingual alias resolution across entities (defer until per-document scoping proves
  insufficient).
- No new frameworks/dependencies (LangGraph, a graph-DB client, a dedicated NER library) — reuse the
  existing `LLMProvider`/`EmbeddingProvider`/`VectorStore` abstractions.
- **Stage 2 is not automatically part of "the MVP"** — it is built only after Stage 1's validation
  step (item 4 above) shows the generated content is accurate; if Stage 1 shows poor quality (a real
  risk given local/Ollama generation — see [Risks and open questions](#risks-and-open-questions)),
  Stage 2 should not be built at all.

**Where it fits in the current architecture:** a new sibling service package (analogous to
`app/services/reconciliation/` — reads across `documents/`/`indexing/` domains without being imported
by either), a new job-worker pair (mirroring `ReindexWorker`'s script-driven, non-HTTP-triggered
pattern), and, in Stage 1 only, one addition to `document_audit_service.py`. Stage 2 (conditional)
additionally adds a hook into `RagOrchestrator`'s retrieval step. Stage 1 makes zero changes to
`RetrievalService`, `RagPromptBuilder`'s existing chunk-attribution behavior,
`CustomRagEngine`/`LangChainRagEngine` parity guarantees, or the `POST /chat` API contract; Stage 2
still makes no *breaking* changes to any of them.

**What data structures would be needed:** `wiki_documents(document_id, summary_text, source_chunk_ids,
source_content_hash, chunking_version, generation_model, generation_prompt_version, generated_at)`;
`wiki_entities(id, document_id, name, description, source_chunk_ids, ...same generation-metadata
columns...)`; `wiki_claims(id, document_id, claim_text, source_chunk_ids, ...same generation-metadata
columns...)`; a `WikiGenerationJob` table matching `IngestionJob`'s existing column shape (status,
timestamps, error info, one-active-per-document partial unique index).

**What lifecycle events affect it:** ingestion completion (triggers Stage 1 generation), document
deletion (deletes all Wiki rows for that `document_id`, no recomputation needed), re-index (marks
staleness only when `chunking_version` changed, never on an embedding-model-only rebuild; no
auto-regeneration either way), no effect from activation/vector-cleanup (those are purely
Qdrant-collection-cutover concerns for the *chunk* collection, orthogonal to Wiki content which is
keyed on `document_id`, not `collection_name`).

**How would retrieval use it:** not at all in Stage 1. In Stage 2 (conditional): parallel, additive
lookup alongside `RetrievalService.retrieve()`, merged into `RagPromptBuilder`'s context under a
distinct label space, always carrying its source chunk citations forward into the response's
`sources`.

**How would source grounding work:** exactly the provenance model in
[Grounding and provenance](#grounding-and-provenance-requirements) — every summary/entity/claim
carries chunk-id references within its own document; nothing is presented to a user (once Stage 2
exists) without its citation trail intact.

**How would we measure whether it improves RAG:** Stage 1's own validation step (item 4 above) is the
first measurement — direct inspection of generated Postgres content against the query categories,
requiring no retrieval integration. Only once that passes does the full [Evaluation plan](#evaluation-plan)'s
chat-level RAG-vs-RAG+Wiki comparison (which does require Stage 2) become the next gate, before
considering Option 2/3.

---

## Evaluation plan

**[Proposed]** This plan has two stages, matching [Recommended MVP](#recommended-mvp)'s Stage 1/Stage
2 split. **Stage 1 evaluation requires no retrieval integration**: it is a direct, offline read of the
generated Postgres rows (`wiki_documents`/`wiki_entities`/`wiki_claims`) against the same query
categories below — for each question, does the stored content for the relevant document(s) contain a
correct, groundable answer, judged by reading the rows and their cited chunks directly? Only once
Stage 1 passes does the **Stage 2 comparison** (RAG vs. RAG+Wiki, requiring Stage 2's retrieval
integration to exist) become relevant.

### Evaluation set (query categories)

Build a small, hand-curated set (10-20 questions total is enough to be directionally useful) spanning:

1. **Single-chunk factual** — expect no difference between RAG and RAG+Wiki (sanity check that Wiki
   doesn't regress the cases the current system already handles well).
2. **Multi-document synthesis** ("summarize everything about Y across documents").
3. **Relationship questions** ("how does X relate to Y", "what depends on X").
4. **Broad topic overview** ("give me an overview of this architecture").
5. **Contradictory-source questions** (two documents disagree — since the MVP stores independent
   per-document claims with no cross-document contradiction detection, this category checks whether
   both documents' independently-generated claims are each individually correct and well-grounded,
   and, once Stage 2 exists, whether retrieving both together lets the answering LLM/reader notice the
   conflict from the two citations — not whether the platform itself flags it).
6. **Hebrew query over English content.**
7. **English query over Hebrew content.**

### Comparison

**Stage 1** (no retrieval integration needed): for each question, read the relevant document(s)'
generated `wiki_documents`/`wiki_entities`/`wiki_claims` rows directly and judge whether they contain
a correct, groundable answer. **Stage 2** (only after Stage 1 passes): run each question through
`RAG` (current pipeline, Wiki disabled) and `RAG + Wiki` (parallel retrieval enabled), same underlying
LLM/embedding models, same corpus.

### Metrics

- **Answer correctness** (human-graded pass/fail against the source documents — the only ground truth
  that matters here, given no labeled dataset exists).
- **Source correctness** (do the cited chunk/document IDs in the response actually support the
  claim — directly testable given both pipelines already expose `sources`).
- **Retrieval relevance** (are the chunks/Wiki entries actually pertinent to the question).
- **Completeness** (for multi-document synthesis questions specifically — does the answer cover what
  the corpus actually says, or only what fit in `retrieval_top_k`).
- **Hallucination rate** (any claim in the answer not traceable to a cited source — this is the
  headline risk metric for this feature specifically).
- **Latency** (Wiki lookup adds a call; measure the actual added P50/P95 on real hardware, not an
  estimate).
- **Ingestion cost** (extra LLM calls per document for generation — token count and wall-clock time).
- **Token cost** (extra context tokens added to the chat prompt when Wiki content participates).
- **Number of retrieved source chunks** (does Wiki context let the LLM answer with fewer/more
  chunk citations than chunk-only retrieval).

### Recommended acceptance criteria for the MVP experiment

- **Must not regress**: single-chunk factual answer correctness and hallucination rate must not
  measurably worsen with Wiki enabled (guards against the "additive context confuses the model"
  failure mode).
- **Must improve**: answer completeness and correctness on the multi-document-synthesis and
  relationship-question categories, with source correctness held constant or better (a "more complete
  but less grounded" result should count as a failure, not a win, given this document's stance on
  hallucination amplification).
- **Must stay bounded**: latency overhead and ingestion cost are reported, not gated — this is a
  portfolio/learning project, not a production SLA, but a wildly disproportionate cost (e.g. 10x
  ingestion time for marginal gain) is a signal Option 1's scope needs narrowing further, not evidence
  to proceed to Option 2/3.
- Only once these hold should Option 2 (typed relationships) be considered — and only for the specific
  query subclass (multi-hop relationship chains) that Option 1's flat entity list demonstrably cannot
  answer.

---

## Risks and open questions

**[Proposed]**

- **Corpus scale is currently unknown/likely small** (portfolio project) — the evaluation plan's
  conclusions may not generalize if/when the corpus grows by orders of magnitude; re-run the
  evaluation at any meaningfully larger scale before trusting Option 1's verdict indefinitely.
- **Entity extraction quality with the current LLM provider (Ollama/local models)** is unverified —
  the GraphRAG/RAPTOR literature largely reports results with frontier hosted models (GPT-4-class);
  local-model extraction quality should be spot-checked early, since a low-quality entity list
  actively hurts (wrong "related concepts" are worse than none) more than a low-quality summary does.
- **No contradiction detection exists in the MVP at all** (not even a naive one) — see
  [Grounding and provenance](#grounding-and-provenance-requirements): since records are never merged
  or compared across documents, two documents making conflicting claims simply produce two
  independent, individually-accurate claim records with no platform-level flag connecting them. This
  is an explicit scope boundary, not a quality gap to be improved later within the same mechanism —
  any future contradiction detection would need cross-document synthesis to exist first.
- **No stale-`WikiGenerationJob` recovery is specified** — following this platform's own precedent
  (deletion/re-index jobs also have no stale-`PROCESSING` recovery), this should be an explicit,
  documented non-goal for the MVP rather than a silent gap, consistent with how the rest of this
  codebase treats the same limitation elsewhere.
- **Resolved (was previously an open question)**: `wiki_entities`/`wiki_claims` are **strictly
  per-document in both stages** (e.g. "Payment Service" extracted separately from three documents
  produces three independent rows, never merged) — see
  [Grounding and provenance](#grounding-and-provenance-requirements). This is settled as the MVP
  scope, not left open, specifically because merging requires a cross-document matching capability
  this MVP does not build; it remains the most likely *future* refinement once real usage data shows
  duplicate per-document entities are a practical annoyance.
- **Open question**: who/what triggers regeneration after a document's Wiki content is flagged stale
  (a `chunking_version` change — never an embedding-model-only re-index, see
  [Document update / re-index](#document-update--re-index)) — an operator-run script (mirroring
  `process_pending_reindex_jobs.py`), or should it be automatic? This document recommends
  **operator-run**, matching every other non-ingestion job type's current pattern, but it's a genuine
  judgment call rather than a forced conclusion.
- **Open question**: how should Stage 1's "does the generated content contain a correct answer"
  judgment (item 4 in [Recommended MVP](#recommended-mvp)) actually be scored — fully manual reading,
  or a scripted check (e.g. does the claim text contain expected keywords/entities)? This document
  does not resolve this; a first pass with manual judgment on a small (10-20 question) set is likely
  sufficient before investing in a scored harness.

---

## Proposed next steps

**[Proposed]** — not authorized by this research task; listed for the user's decision only. The
immediate next milestone is **Stage 0**, not Stage 1 — no `WikiGenerationJob`, no new database table,
and no persistence/lifecycle code should be written until Stage 0 has answered whether the currently
configured local LLM can actually produce the proposed structure reliably.

1. If the direction above is accepted, scope **Stage 0 — evaluation baseline + local-LLM generation
   spike** as its own dedicated task/branch (per `AGENTS.md`'s branch policy) — this research branch
   does not become that implementation branch. Stage 0 is deliberately a throwaway spike, not the
   start of Stage 1's persistence work.
2. Before running any generation spike, build the evaluation set from
   [Evaluation plan](#evaluation-plan) against the *current* RAG pipeline alone, to get a real
   correctness/completeness baseline to compare against later.
3. Run the Stage 0 spike itself: feed a handful of real documents' extracted chunks to the
   actually-configured local LLM provider and prompt it for the same structured output Stage 1 would
   persist — a summary, a flat entity/topic list, and grounded natural-language claims with source
   chunk references — without building any storage, job, or lifecycle machinery around it. This
   document does not design the spike's implementation (harness, prompt, or scoring mechanics) —
   that belongs to the Stage 0 task itself — but the spike should be able to answer:
   - Can the local model reliably follow the output contract (summary + entities/topics + grounded
     claims + chunk references), or does it need heavy prompt engineering / fail unpredictably?
   - Are generated claims actually supported by the chunks they cite, or does the model invent
     unsupported claims — and how often?
   - Are the extracted entities/topics useful signal, or mostly noise?
   - Is Hebrew output usable? Is English output usable? Can the model handle Hebrew and English
     source material consistently, including mixed-language documents?
   - Overall: is generation quality good enough to justify building Stage 1's persistence and
     lifecycle machinery at all, or does this change the recommendation (e.g. toward a stronger
     hosted model, a narrower claim scope, or not building the feature yet)?
4. Only after Stage 0 answers the questions above should Stage 1 (`WikiGenerationJob`, the three
   Postgres tables, the reconciliation hook) be scoped as a real implementation task.
