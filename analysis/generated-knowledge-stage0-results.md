# Generated Knowledge Stage 0 Results

**Status:** Real, disposable evaluation spike. This document reports observed results from actual
runs against the repository's currently-configured local LLM — it is evidence, not a claim about
implemented behavior. See [`analysis/llm-wiki-research.md`](llm-wiki-research.md) for the
architecture this spike validates; it defines Stage 0 exactly as "a small, throwaway generation
spike against the actually-configured local LLM" that answers whether local generation is reliable
enough to justify building Stage 1's persistence/lifecycle machinery.

**This document now has three parts, in chronological order, each preserved unchanged as historical
evidence once superseded.** [Stage 0](#goal) (below) is the original run — its systematic Hebrew
language-fidelity failure is why
[Stage 0.1 — Language-preservation follow-up](#stage-01--language-preservation-follow-up) exists.
Stage 0.1 fixed that, but its own real runs reproducibly surfaced a different problem — a
placeholder entity with an empty name/description — which is why
[Stage 0.2 — Empty-entity follow-up](#stage-02--empty-entity-follow-up) exists. **Stage 0.2, at the
end of this document, is this document's current final word** on whether Stage 1 is now justified
— not Stage 0's or Stage 0.1's own recommendation sections, which reflect only what was known at
the time each was written.

## Goal

Determine whether the currently configured local LLM (Ollama, `llama3.1`) can reliably generate the
Generated Knowledge Layer proposed in `analysis/llm-wiki-research.md` — a per-document summary,
entities/topics, and grounded natural-language claims, each citing real source chunk IDs — well
enough to justify building Stage 1 persistence/lifecycle infrastructure. No `WikiGenerationJob`,
database table, Qdrant collection, or chat integration was built as part of this spike.

## Experiment setup

- **Code (Stage 0-era location)**: `scripts/generated_knowledge_stage0.py` — reuses
  `app.rag.providers.provider_factory.get_llm_provider()` and the real `LLMProvider.generate()`
  contract exactly as production RAG code does; no second Ollama client, no bespoke HTTP call.
  Reuses `app.services.documents.chunker.DocumentChunk` for chunk representation. **This single-file
  path no longer exists** — Stage 0.1 refactored it into the package `scripts/generated_knowledge_stage0/`
  (see [Stage 0.1](#stage-01--language-preservation-follow-up) below for the current module
  layout); the description above reflects how Stage 0 was actually run at the time, not the current
  code location.
- **Output contract**: a JSON object `{"summary": {...}, "entities": [...], "claims": [...]}`,
  parsed and mechanically validated by `parse_and_validate()` (Stage 0: in the single script above;
  Stage 0.1 onward: in `scripts/generated_knowledge_stage0/validation.py`) — see
  [Output contract](#output-contract) and [Mechanical validation](#mechanical-validation) below.
- **Deterministic tests (Stage 0-era count)**: `tests/unit/scripts/test_generated_knowledge_stage0.py`
  (22 tests at the time) covered schema parsing, provenance validation, and `run_document()`'s
  failure-isolation contract against fake/mock LLM output — never real Ollama. **This file no longer
  exists** — Stage 0.1 split it into `tests/unit/scripts/generated_knowledge_stage0/` (currently 35
  tests across `test_prompt.py`/`test_validation.py`/`test_runner.py`; see
  [Stage 0.1](#stage-01--language-preservation-follow-up) below for what changed and why). The "22
  tests" figure describes Stage 0 as it ran, not the current suite.
- **Real-model run**: executed manually, twice, as an explicit experiment — **not** part of
  `make verify`/`make test`/CI, matching this repository's existing AI-provider testing policy (see
  `scripts/smoke_multilingual_real.py` for the established precedent this spike follows). Invoked as:

  ```bash
  OLLAMA_BASE_URL=http://localhost:11434 python3 -m scripts.generated_knowledge_stage0
  ```

## Model/provider

- **Provider**: `ollama` (`LLM_PROVIDER=ollama`, the repository default)
- **Model**: `llama3.1` (`resolved_llm_model` — the repository's default `OLLAMA_CHAT_MODEL`)
- **Endpoint**: locally reachable Ollama (`http://localhost:11434`) with `llama3.1` already pulled
- **Runs**: 2 full passes over all 5 sample documents (10 document-generations total), to check
  whether observed failure modes were one-off or reproducible
- **Latency observed**: 4.8s–10.6s per document (single `generate()` call, no retries, no streaming
  used at the application layer — `generate()` collects the full stream internally)

## Input documents

Five small, hand-written documents (2 chunks each except one single-chunk document), chosen to cover
the variation the task required — not a benchmark suite:

| Document | Language | Purpose |
|---|---|---|
| `en-payment-reconciliation` | English | Relationship-shaped content spread across two chunks of the same document (the target example shape from the research/task spec) |
| `he-vacation-policy` | Hebrew | Simple factual content, no relationships to extract |
| `en-unrelated-services` | English | **Adversarial**: two services (Service A/PostgreSQL, Service B/Kafka) with no stated relationship — the model must not invent one |
| `en-cooccurring-teams` | English | **Adversarial**: two teams co-occurring in one document, with one real stated fact (both report to the VP of Engineering) but no stated relationship between their actual work — tests whether entity co-occurrence alone triggers a fabricated relationship |
| `he-order-notification` | Hebrew | Relationship-shaped content spread across two chunks — the Hebrew counterpart to the English relationship case, to compare claim quality and language fidelity across languages on the same query shape |

## Output contract

```json
{
  "summary": {"text": "...", "source_chunk_ids": ["chunk-1"]},
  "entities": [
    {"name": "...", "description": "...", "source_chunk_ids": ["chunk-1"]}
  ],
  "claims": [
    {"text": "...", "source_chunk_ids": ["chunk-1", "chunk-2"]}
  ]
}
```

Matches `analysis/llm-wiki-research.md`'s target shape exactly: no typed `subject`/`predicate`/
`object`, no cross-document records (chunk IDs are validated against exactly one document's own
chunks per generation call), no graph structure. Claims are plain natural-language sentences that
may synthesize multiple chunks of the *same* document — matching the research document's own
worked example.

## Mechanical validation

`parse_and_validate()` enforces, deterministically, with no judge model:

- valid JSON (tolerating one common deviation: a `\`\`\`json ... \`\`\`` markdown fence around
  otherwise-valid JSON)
- every `source_chunk_ids` list is present and non-empty (missing provenance is rejected)
- every cited chunk ID exists in the caller-supplied allowed set — which is scoped to exactly one
  document's own chunks, so this simultaneously rejects unknown chunk IDs *and* any cross-document
  reference
- every `text`/`description` field is a non-empty string after stripping whitespace
- duplicate chunk IDs within one record are normalized (deduplicated, order-preserved), not rejected
- every problem found is collected and reported together (`Stage0ValidationError.issues`), not just
  the first

All 6 categories are covered by dedicated unit tests against fake/mock LLM output (22 tests at the
time, in the Stage 0-era single test file — see the current 35-test suite under
`tests/unit/scripts/generated_knowledge_stage0/` from Stage 0.1 onward),
independent of whether a real model run happens to exercise each path.

## English results

**4/5 English-language document-generations succeeded across both runs with no parse failures**
(`en-payment-reconciliation`, `en-unrelated-services`, `en-cooccurring-teams` — each run twice, 6
document-generations, 0 failures).

- **Summaries**: accurate, concise (1–2 sentences), no unsupported statements introduced in any run.
- **Entities**: consistently meaningful named concepts (`Payment Service`, `Reconciliation Service`,
  `Service A`, `Service B`, `Platform Team`, `Growth Team`), never every-noun noise. One run added
  `VP of Engineering` as an extra entity for the co-occurring-teams document — a defensible,
  non-noisy addition, not present in the second run (a minor inconsistency, not a quality problem).
- **Claims**: 12 English claims generated across both runs; all 12 manually classified
  **SUPPORTED** — see [Negative-case results](#negative-case-results) for the adversarial-case detail,
  which is the more important finding here.

## Hebrew results

**2/2 Hebrew documents produced fluent, well-formed output in both runs — but in English, not
Hebrew, in every single case (4/4 Hebrew-document generations).**

- `he-vacation-policy` (run 1): summary, entities (`company`, `vacation policy`, `full-time
  employees`), and claims were all written in English despite the source chunks being entirely
  Hebrew.
- `he-vacation-policy` (run 2): same — English output again, and this run additionally parsed
  successfully where run 1 failed (see [Failure modes observed](#failure-modes-observed)).
- `he-order-notification` (both runs): same — English output in both runs, despite the source
  chunks being entirely Hebrew.
- This directly violates the Stage 0 prompt's explicit instruction ("Write the summary, entity
  descriptions, and claims in the SAME language as the supplied chunks. Do not translate.") and the
  broader platform principle `analysis/llm-wiki-research.md` and `docs/multilingual/README.md`
  both establish — source content is never translated anywhere else in this platform.
- **Content accuracy despite the language violation**: the translated content itself was accurate
  and well-grounded — e.g. "21 days," "5 days carry-over," and the Order→OrderShipped→Notification
  relationship were all correctly captured. The failure is specifically language fidelity, not
  factual grounding.
- **Entity completeness gap specific to Hebrew**: `he-order-notification` extracted only one entity
  (`Notification`/`Notification Service`) in both runs, despite a claim directly naming "Order
  Service" as the event publisher — "Order Service" itself was never extracted as its own entity in
  either run. Not observed in the equivalent English relationship document
  (`en-payment-reconciliation`), where both services were consistently extracted as entities.

**This is the single most significant finding of this spike** — see
[Recommendation](#recommendation).

## Negative-case results

This was the most important question for the whole feature: does the model invent a relationship
between entities merely because they co-occur, or because two unrelated facts sit in the same
document?

- **`en-unrelated-services`** (Service A/PostgreSQL, Service B/Kafka, no stated relationship): run
  twice, **zero** instances of a fabricated relationship (e.g. "Service A communicates with Service
  B through Kafka" — the specific hallucination this document was designed to tempt). Both runs
  produced two independent, individually-accurate claims, one per service, with no connecting
  claim.
- **`en-cooccurring-teams`** (Platform Team, Growth Team, one real stated fact — both report to the
  VP of Engineering — but no stated relationship between their actual work): run twice, **zero**
  instances of a fabricated relationship between the teams' work (e.g. no invented "Platform Team's
  tooling supports Growth Team's experiments"). Both runs produced two independent, accurate claims
  about each team's own responsibility.
- Across 4 independent adversarial-document runs (2 documents × 2 runs), the model never once
  invented a relationship it was tempted toward. This is strong, directly-relevant evidence against
  the specific hallucination-amplification risk `analysis/llm-wiki-research.md` treats as the
  feature's central risk.

## Failure modes observed

1. **One parse failure out of 10 document-generations (10%)** — `he-vacation-policy`, run 1: the
   model cited `source_chunk_ids: ["he-vault-policy-0"]` in the summary — a near-miss, malformed
   chunk ID (`vault` instead of `vacation`) that does not match any chunk actually supplied
   (`he-vacation-policy-0`/`he-vacation-policy-1`). Mechanical validation correctly rejected this as
   an unknown-chunk-ID violation rather than silently accepting it — exactly the required behavior.
   The same document produced a valid, correctly-cited chunk ID in run 2, so this looks like an
   occasional generation slip rather than a systematic chunk-ID-handling defect, but it is real and
   should be expected to recur at a similar low rate.
2. **Systematic Hebrew→English translation** (see [Hebrew results](#hebrew-results)) — reproducible
   in 4/4 observed cases, not a one-off.
3. **No generation failures** — Ollama was reachable and responded successfully in all 10 attempts;
   no timeouts, no malformed streaming, no provider errors.
4. **Minor over-broad citation** (not a failure, but worth noting) — in 2 of the successful runs, a
   claim cited both of a document's chunks when only one was strictly necessary to support it (e.g.
   `he-order-notification` run 2's second claim cites both chunks for a fact stated only in chunk 0).
   The cited chunk never contradicted the claim in any observed case — this is imprecision, not
   hallucination, but worth tightening if Stage 1 is built.

## Quality assessment

| Dimension | Assessment |
|---|---|
| Output contract reliability | 9/10 (90%) valid on the first attempt; the one failure was a malformed chunk-ID reference, correctly caught by mechanical validation rather than silently accepted |
| Claim groundedness | 18/18 (100%) of manually-reviewed claims from successful parses classified **SUPPORTED** — 0 UNSUPPORTED, 0 PARTIALLY_SUPPORTED |
| Hallucinated relationships | 0 observed across 4 independent adversarial-document runs specifically designed to tempt one |
| Summary usefulness | High — accurate, concise, no unsupported statements in any of the 9 successful runs |
| Entity/topic usefulness | Mostly high (meaningful named concepts on English content); one generic/trivial extraction (`company`, `full-time employees` on the Hebrew vacation document) and one completeness gap (`Order Service` never extracted despite being named in a claim) |
| English quality | Acceptable — no issues beyond the general entity-completeness/over-citation nits above |
| Hebrew quality | **Not acceptable as-is** — content is accurate but is generated in English instead of Hebrew, in 100% of observed cases, directly violating both the experiment's own prompt instruction and this platform's broader "never translate source content" principle |
| Provenance reliability | High overall (90% clean first-attempt validity), with the one failure being a correctly-caught malformed chunk ID, not a silently-accepted one |

## Recommendation

### Stage 0 decision: **GO WITH CHANGES**

The core, highest-risk architectural question this spike existed to answer — *does the model invent
relationships between co-occurring entities, or hallucinate claims not actually stated in the
source* — came back strongly positive: **zero fabricated relationships across four independent
adversarial test runs**, and **100% of reviewed claims were source-supported**. This is exactly the
evidence `analysis/llm-wiki-research.md` said would need to hold before Stage 1 is worth building.
Output-contract reliability (90% first-attempt valid, with the one failure correctly caught rather
than silently accepted) and English-language quality are both good enough to proceed on.

**What clearly needs to change before Stage 1 is scoped:** Hebrew-language fidelity. The model
translated Hebrew source content into English in 100% of observed cases (4/4), which is disqualifying
on its own for a platform whose multilingual (Hebrew/English) support is a first-class, explicitly
documented capability — not an edge case. Building Stage 1's persistence layer today would mean
persisting English-language generated knowledge for Hebrew documents, silently defeating the "same
language as source" requirement this document and `docs/multilingual/README.md` both establish.

**Concrete adjustments to try before re-running this spike, in order of cheapest-first:**

1. **Strengthen the language-preservation instruction** — the current instruction ("Write the
   summary, entity descriptions, and claims in the SAME language as the supplied chunks. Do not
   translate.") is stated once, in a numbered list, alongside seven other rules. Try moving it
   immediately before the chunk text itself, repeating it once per language explicitly detected
   (mirroring `docs/multilingual/README.md`'s own pattern: a per-language response-language
   directive stated explicitly, in English, as its own final instruction — never buried in a general
   list), or adding a short Hebrew-language few-shot example.
2. **Detect the source language before prompting** (reusing `app.rag.language.
   ScriptBasedLanguageDetector`, already used elsewhere in this platform) and inject an explicit
   directive naming the detected language, the same way `PromptCatalog`'s response-language
   directives already work for chat answers.
3. **Re-run this same spike** against the adjusted prompt, specifically re-checking the two Hebrew
   documents, before concluding whether `llama3.1` can produce this structure in Hebrew at all, or
   whether a different configured model would be needed.
4. **Secondary, lower-priority fixes** (worth doing alongside the above, not blocking on their own):
   instruct the model to extract every named service/system it names in a claim as its own entity
   (addressing the `Order Service` completeness gap), and to cite only the chunk(s) strictly
   necessary for a given claim (addressing the minor over-citation pattern).

**Do not proceed directly to Stage 1** until a re-run confirms Hebrew output is generated in Hebrew.
The rest of this spike's results support proceeding once that specific issue is resolved — this is
not a NO-GO on the architecture itself, and is not evidence to reconsider the recommended MVP shape
in `analysis/llm-wiki-research.md`.

---

## Stage 0.1 — Language-preservation follow-up

**Status:** Real, disposable follow-up experiment, same policy as Stage 0 above — evidence, not a
claim about implemented behavior. Same model/provider, same sample set, so results are directly
comparable to Stage 0.

### Change tested

Two changes, both applied together (code refactor + prompt fix — see the spike's own module
structure for the refactor, unrelated to what's being measured here):

1. **Source-language detection**, reusing the existing `ScriptBasedLanguageDetector`
   (`app.rag.language`) directly — no new classification system, no LLM call used to detect
   language. The detector runs once per document, over the concatenation of all its chunk texts,
   before the prompt is built.
2. **An explicit, nearby language directive** in the generation prompt, replacing Stage 0's single
   top-of-prompt rule ("Write the summary, entity descriptions, and claims in the SAME language as
   the supplied chunks. Do not translate.") with a directive placed immediately before the
   output-format instruction — the text the model reads right before it starts generating JSON —
   naming the detected language explicitly and carving out proper nouns/API/event/code identifiers
   as legitimately exempt from translation. See `scripts/generated_knowledge_stage0/prompt.py`.

### Prompt/language strategy

```
SOURCE LANGUAGE: Hebrew

Generate ALL summary, entity descriptions, and claims in Hebrew.
Do not translate the source into English.
Entity names that are proper nouns, API names, product names, event names,
or code identifiers may remain in their original form.
```

placed directly before the JSON output-format instruction, itself placed directly before the
document's chunks — i.e. the last three things the model reads, in order, are: the language
directive, the exact output shape, then the source text. Mirrors the placement principle
`PromptCatalog.get_response_language_directive()` already uses for RAG chat answers (a short,
explicit, per-language directive, never "answer in English and translate"), adapted for
multi-field structured extraction rather than a single free-text answer.

### Hebrew results

**Complete reversal from Stage 0.** Both runs' `he-order-notification` document — the one Hebrew
document that parsed successfully in every Stage 0.1 run — produced its summary, both entities,
and every claim entirely in Hebrew:

- Run 1: `שירות ה-Notification אחראי על שליחת התראות, בעזרת שירות ה-Order.` (summary), entities
  `שירות ה-Notification` and `שירות ה-Order` (both Hebrew names and Hebrew descriptions), two
  Hebrew claims correctly grounded across both chunks.
- Run 2: `שירות ה-Notification אחראי על שליחת התראות למשתמשים.` (summary), entities
  `שירות ה-Notification` and `OrderShipped` (the event, correctly left as a Latin technical
  identifier per the directive's own carve-out), one Hebrew claim correctly grounded.

`he-vacation-policy` failed to parse in **both** Stage 0.1 runs — but not for a language reason:
its summary and claims were, in both runs, entirely and correctly in Hebrew (visible in the raw
output); the failure was a single entity record with an empty `"name"` (run 1) or empty
`"name"`+`"description"` (run 2), rejected by the same mandatory non-empty-text validation Stage 0
already enforced. See [Failure modes observed](#stage-01-failure-modes-observed) below — this is a
new, narrower failure mode, not the language failure Stage 0.1 exists to fix.

**Language-fidelity check result: 0 failures across every document that parsed in Stage 0.1** —
the deterministic `check_language_fidelity()` check (Part 3) never flagged a mismatch in either
run, for either language.

### English regression check

No regression observed. All three English documents (`en-payment-reconciliation`,
`en-unrelated-services`, `en-cooccurring-teams`) parsed successfully in both runs, with accurate,
concise summaries and correctly-English entities/claims — content quality is consistent with Stage
0's English results. The negative/adversarial documents specifically are covered in their own
section below.

### Provenance regression check

No regression. Every mandatory-provenance, unknown-chunk-ID, and empty-text check that Stage 0
enforced remains enforced identically in Stage 0.1 (same `validation.py` logic, extended, not
replaced) — the `he-vacation-policy` parse failures in both runs are this validation working
correctly, catching an empty entity name/description rather than silently accepting it.

### Negative-case regression check

No regression — **zero fabricated relationships** in either run of either adversarial document
(`en-unrelated-services`, `en-cooccurring-teams`), consistent with Stage 0's result. Notably, run 1
of `en-cooccurring-teams` this time extracted the reporting-line fact ("Both teams report into the
VP of Engineering, but their roadmaps are managed independently") as its own claim rather than only
inside an entity description — still accurately grounded, still no invented relationship between
the two teams' actual work.

### Comparison with Stage 0

| Metric | Stage 0 | Stage 0.1 | Change |
|---|---|---|---|
| Hebrew-document runs producing Hebrew output | 0/4 | 2/2 (of those that parsed) | **Fixed** |
| Schema/parse reliability (all documents, both runs) | 9/10 (90%) | 8/10 (80%) | **Regressed** — concentrated entirely in `he-vacation-policy` (0/2 in Stage 0.1 vs. 1/2 in Stage 0), driven by an empty entity name/description, not a language issue |
| Supported claims (of successfully-parsed documents) | 18/18 (100%) | 16/16 (100%) | No change |
| Unsupported/partially-supported claims | 0 | 0 | No change |
| Fabricated relationships in adversarial cases | 0/4 adversarial-document runs | 0/4 adversarial-document runs | No change |
| Entity quality | Mostly good; one generic/trivial pair (`company`, `full-time employees`); `Order Service` never extracted | Mostly good; `he-order-notification` now extracts the Order-side entity in run 1 (`שירות ה-Order`) — the earlier completeness gap improved, though inconsistently (run 2 extracted the event instead) | Mixed — no clear regression, modest improvement on the specific gap Stage 0 flagged |
| Provenance-validation failures | 1/10 (malformed chunk-ID reference) | 2/10 (both empty-name/description on the same document) | Nominally more failures, but a different, narrower, single-document root cause — not a provenance-*enforcement* regression, since validation caught both correctly |
| Generation latency | 4.8s–10.6s per document | 4.7s–10.7s per document | No material change |

### Final decision

**Stage 0.1 decision: GO WITH CHANGES**

The specific problem Stage 0.1 existed to fix — systematic Hebrew→English translation — is fixed:
**2/2 (100%)** of successfully-parsed Hebrew-document generations produced fully Hebrew summaries,
entities, and claims, a complete reversal from Stage 0's **0/4 (0%)**, achieved entirely by reusing
existing repository primitives (`ScriptBasedLanguageDetector`) and a more deliberately-placed
prompt directive — no new dependency, no new classification system, no LLM call spent on language
detection. Every other Stage 0 finding that mattered for the architecture holds: claim groundedness
stayed at 100% (16/16), zero fabricated relationships across every adversarial run, English quality
unchanged, and provenance validation continued to catch every malformed record rather than
silently accepting it.

**What still needs a bounded fix before Stage 1:** `he-vacation-policy` failed to parse in both
Stage 0.1 runs because the model produced an entity record with an empty `name`/`description`
rather than either a real name or, more sensibly, no entity at all — this document's actual content
(a simple factual policy with no clearly-nameable named service, only generic concepts like "the
company" or "full-time employees") appears to be a genuine weak spot independent of the language
fix. **Recommended adjustment:** add one explicit instruction — "if a document has no clearly
nameable entity/topic worth extracting, return an empty `entities` array; never invent a
placeholder or an empty-named record" — and re-run specifically against `he-vacation-policy` (and
ideally one more simple-factual document of each language) before concluding this is resolved.

**Do not proceed to Stage 1 until that specific adjustment is verified.** Stage 1 is justified in
every other respect this document has tested — the remaining gap is narrow, single-document, and
already caught (not silently accepted) by existing mechanical validation, not a systemic
correctness or hallucination problem.

### Stage 0.1 failure modes observed

1. **`he-vacation-policy` empty-entity failures (2/2 Stage 0.1 runs)** — see
   [Hebrew results](#hebrew-results-1) and [Final decision](#final-decision) above. Both the
   document's summary and claims were correctly, fully in Hebrew in both runs; only its single
   attempted entity record was malformed.
2. **No generation failures** — Ollama was reachable and responded successfully in all 10 attempts
   across both Stage 0.1 runs, matching Stage 0.
3. **No new hallucination pattern observed** — no fabricated relationship, no unsupported claim, in
   either run.

---

## Stage 0.2 — Empty-entity follow-up

**Status:** Real, disposable follow-up experiment, same policy as Stage 0/0.1 above. Same
model/provider, same sample set, so results are directly comparable across all three stages.

### Change tested

One prompt-only change (`scripts/generated_knowledge_stage0/prompt.py`), no parser change: an
explicit instruction that entities are optional — if a document has no meaningful named entity/
topic, the model must return `"entities": []`, and must never emit a placeholder entity with an
empty name or description. The instruction appears twice: once in the general rules list (rule 5)
and once, verbatim as its own directive (`ENTITIES ARE OPTIONAL: ...`), placed immediately before
the JSON output-format instruction — the same "close to what the model reads right before
generating" placement principle that fixed the Hebrew language issue in Stage 0.1.
`validation.py`'s `_validate_entity_dict()` is unchanged: an empty `name`/`description` is still
rejected exactly as strictly as before (verified by 3 new deterministic tests reproducing both
exact malformed shapes observed in real Stage 0.1 runs — empty name alone, and empty name +
description together — plus a whitespace-only name).

### he-vacation-policy result

**Fixed, reproducibly.** `he-vacation-policy` parsed successfully in **both** Stage 0.2 runs, each
time returning `"entities": []` — a genuine empty array, never a placeholder:

- Run 1: summary and 2 claims, all correctly grounded and in Hebrew; `entities (0)`.
- Run 2: summary and 2 claims, all correctly grounded and in Hebrew; `entities (0)`.

This is the exact target fix, confirmed reproducible across both runs (2/2, 0/2 empty-placeholder
failures — a complete reversal from Stage 0.1's 2/2 failures on this same document).

### Hebrew fidelity

Still fixed overall. `he-vacation-policy`'s summary and claims were fully Hebrew in both runs.
`he-order-notification` produced fully Hebrew summary and claims in both runs it reached that
stage (it parsed in run 1, failed to parse in run 2 for an unrelated reason — see
[Failure modes observed](#stage-02-failure-modes-observed) below). One narrow new observation in
run 1: one of three entity *descriptions* for `he-order-notification` — `"אירוע OrderShipped"`
("event OrderShipped") — was flagged by the deterministic language check. That two-word string is
exactly one Hebrew word and one Latin technical identifier, a genuine word-count tie;
`ScriptBasedLanguageDetector`'s documented tie-breaking rule falls back to the configured default
response language rather than guessing, and that fallback did not resolve to Hebrew here. This is
a detector edge case on an unusually short, technical-identifier-heavy description — not a
recurrence of Stage 0's systemic English-translation failure (the same run's summary and all three
claims for that same document were fully, correctly Hebrew).

### Schema/parse reliability

**Recovered to Stage 0's baseline.** 9/10 (90%) of Stage 0.2's 10 document-generations parsed
successfully — up from Stage 0.1's 8/10 (80%), and back at Stage 0's original 9/10 (90%). The one
Stage 0.2 parse failure was **not** `he-vacation-policy` (fully fixed, 2/2) — it was
`he-order-notification` in run 2, for a new and different reason: an unescaped `"` character inside
the Hebrew word `דוא"ל` ("email," a standard Hebrew abbreviation that includes a gershayim mark
rendered as a straight double-quote) broke JSON string escaping. Mechanical validation correctly
rejected this as invalid JSON rather than silently accepting a corrupted parse — see
[Failure modes observed](#stage-02-failure-modes-observed) below.

### Claim groundedness

**22/22 (100%) of manually-reviewed claims classified SUPPORTED** across both runs — 0 unsupported,
0 partially-supported — consistent with Stage 0 (18/18) and Stage 0.1 (16/16). One minor,
non-blocking provenance nit observed once (not in the claims themselves): `he-vacation-policy`'s
run-1 summary asserted both the 21-day entitlement (chunk 0) and the 5-day carryover cap (chunk 1)
but cited only chunk 0 in `source_chunk_ids` — an under-citation, not a fabrication (both facts are
genuinely present in the document, just not fully attributed). This did not recur in run 2, where
the same document's summary correctly cited only what it actually asserted.

### Provenance reliability

No regression. Every provenance/schema check Stage 0/0.1 enforced remains enforced identically and
was exercised again by real output in these runs (chunk-ID citation, non-empty text). The one
schema failure this stage (the JSON-escaping issue above) was caught, not silently accepted.

### English regression status

None. All three English documents parsed successfully in both runs with accurate, grounded content,
consistent with Stage 0 and Stage 0.1.

### Adversarial relationship result

**Zero fabricated relationships**, again, across all 4 adversarial-document runs
(`en-unrelated-services` × 2, `en-cooccurring-teams` × 2) — fully consistent with every prior stage.

### Comparison across all three stages

| Metric | Stage 0 | Stage 0.1 | Stage 0.2 |
|---|---|---|---|
| `he-vacation-policy` empty-placeholder-entity failures | n/a (chunk-ID typo instead) | 2/2 | **0/2** |
| Hebrew-document runs producing Hebrew output (of those that parsed) | 0/4 | 2/2 | 2/3 (3 parsed; 1 flagged by the language-fidelity check on the already-documented short entity-description edge case) |
| Schema/parse reliability (all documents, both runs) | 9/10 (90%) | 8/10 (80%) | 9/10 (90%) |
| Supported claims (of successfully-parsed documents) | 18/18 (100%) | 16/16 (100%) | 22/22 (100%) |
| Fabricated relationships in adversarial cases | 0/4 | 0/4 | 0/4 |
| New failure modes | chunk-ID typo (1/10) | empty-entity placeholder (2/10) | JSON-escaping of Hebrew gershayim punctuation (1/10); short-description language-detector tie-break (1/9 parsed) |

### Stage 0.2 failure modes observed

1. **`he-order-notification` JSON-escaping failure (run 2 only, 1/10 document-generations)** — an
   unescaped `"` inside the Hebrew abbreviation `דוא"ל` broke JSON parsing. New, not observed in
   Stage 0 or 0.1. Caught correctly by mechanical validation (rejected as invalid JSON, not
   silently accepted). Not reproduced in run 1 of the same document, so its rate cannot yet be
   distinguished from a one-off versus a recurring risk specific to Hebrew abbreviations containing
   a gershayim/geresh mark — worth watching in any future run, not fixed as part of this task (out
   of this task's explicit scope, which was the empty-entity issue only).
2. **`he-order-notification` short entity-description language-detector tie-break (run 1 only, 1/9
   parsed documents)** — see [Hebrew fidelity](#hebrew-fidelity) above. A narrow edge case on a
   two-word, exactly-tied description, not a recurrence of Stage 0's systemic translation failure.
3. **No regression in claim groundedness or adversarial safety** — 22/22 supported claims, 0
   fabricated relationships, both consistent with or better than prior stages.
4. **No generation failures** — Ollama responded successfully in all 10 attempts.

### Stage 0.2 decision

**GO**

The specific, narrowly-scoped problem this iteration existed to fix — `he-vacation-policy` emitting
a placeholder entity with an empty name/description instead of a genuinely empty `entities` array —
is fixed, reproducibly, with a prompt-only change that left the strict parser untouched (verified
by 3 new tests reproducing the exact malformed shapes real runs previously produced). None of the
core findings this document has built up across three stages regressed: claim groundedness held at
100% (22/22), zero fabricated relationships across every adversarial run to date, English quality
unaffected, and Hebrew fidelity remained fixed for the bulk of generated content (summaries and
claims) in every run.

Two new, narrow observations surfaced — a JSON-escaping edge case on Hebrew punctuation (1/10) and
a short-description language-detector tie-break (1/9 parsed) — both single-occurrence, both
correctly handled by existing mechanisms (the first caught and rejected by the parser; the second
caught and flagged by the language check, exactly as designed), and neither a return of a
previously-fixed systemic problem. Per this document's own decision-gate criteria, neither rises to
an "important regression": they are documented here as open, non-blocking follow-up items for
whoever picks up Stage 1's actual implementation — not conditions that must be re-verified before
Stage 1 can be scoped, the way the empty-entity issue was for Stage 0.1 → Stage 0.2.

**Stage 1 is now justified** by this document's own accumulated evidence: three independent rounds
of real-model evaluation (6 total runs) consistently show reliable schema output (~90%, with every
failure mechanically caught, never silently accepted), 100% claim groundedness (56/56 across all
three stages combined), zero hallucinated relationships across 12 adversarial-document runs, and — as
of this stage — correct Hebrew-language fidelity for the substantive generated content. Stage 1
implementation should carry forward the two open follow-up items above as known, low-severity risks
to watch, not as blockers.
