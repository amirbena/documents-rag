"""Stage 0 research spike: can the configured LLM reliably produce a grounded, per-document
Generated Knowledge Layer (summary + entities/topics + grounded natural-language claims, each
citing source chunk IDs), in the SAME language as its source, without inventing relationships
between merely co-occurring entities?

This is a disposable evaluation spike, not production code — see analysis/llm-wiki-research.md
for the architecture it validates and analysis/generated-knowledge-stage0-results.md for the
recorded outcome of real runs against this repository's configured LLM provider. It intentionally
builds nothing beyond a single generation call, deterministic output validation, and deterministic
language-fidelity validation: no WikiGenerationJob, no database table, no Qdrant collection, no
chat/retrieval wiring.

Reuses the existing LLMProvider abstraction (via provider_factory.get_llm_provider()) and the
existing ScriptBasedLanguageDetector (via app.rag.language) exactly as production code does — no
second Ollama client, no bespoke HTTP call, no second language-classification system, no LLM call
used merely to detect language.

Execution flow, one document at a time:

    input document/chunks
        -> determine source language (ScriptBasedLanguageDetector, over all chunk text)
        -> build prompt (prompt.build_stage0_prompt, with an explicit language directive)
        -> get configured LLMProvider (provider_factory.get_llm_provider)
        -> generate
        -> parse/validate (validation.parse_and_validate)
        -> check_language_fidelity (validation.check_language_fidelity, only if parsing succeeded)
        -> Stage0RunResult

Run manually via:

    OLLAMA_BASE_URL=http://localhost:11434 python -m scripts.generated_knowledge_stage0

Never invoked by make verify/test*/CI — same convention as scripts/smoke_multilingual_real.py.
"""

import asyncio
import sys
import time

from app.core.config import Settings, get_settings
from app.rag.language import LanguageDetector, ScriptBasedLanguageDetector
from app.rag.providers.llm_provider import LLMProvider
from app.rag.providers.provider_factory import get_llm_provider
from app.services.documents.chunker import DocumentChunk

from .models import Stage0Document, Stage0RunResult
from .prompt import build_stage0_prompt
from .validation import Stage0ValidationError, check_language_fidelity, parse_and_validate


def _doc(document_id: str, texts: list[str]) -> Stage0Document:
    chunks = tuple(
        DocumentChunk(document_id=document_id, chunk_id=f"{document_id}-{i}", text=text, chunk_index=i)
        for i, text in enumerate(texts)
    )
    return Stage0Document(document_id=document_id, chunks=chunks)


# Small, disposable sample set — not a benchmark suite. Unchanged from the original Stage 0 run
# so results remain directly comparable across the Stage 0 / Stage 0.1 boundary.
SAMPLE_DOCUMENTS: tuple[Stage0Document, ...] = (
    # English, relationship-shaped, spread across two chunks of the same document.
    _doc(
        "en-payment-reconciliation",
        [
            "Payment Service is a backend microservice responsible for processing customer "
            "payments. After a payment is successfully processed, Payment Service publishes a "
            "PaymentCompleted event to the message bus.",
            "Reconciliation Service subscribes to the message bus and consumes PaymentCompleted "
            "events. It uses these events to match incoming payments against expected invoices "
            "and flag any discrepancies.",
        ],
    ),
    # Hebrew, simple factual content, no relationships to extract.
    _doc(
        "he-vacation-policy",
        [
            "מדיניות החופשה של החברה קובעת כי כל עובד במשרה מלאה זכאי ל-21 ימי חופשה בשנה.",
            "ימי חופשה שלא נוצלו עד סוף השנה הקלנדרית ניתנים להעברה לשנה הבאה, עד למקסימום של 5 ימים.",
        ],
    ),
    # English negative/adversarial case: two unrelated services, no stated relationship.
    _doc(
        "en-unrelated-services",
        [
            "Service A is a Java-based service that persists all customer records in a "
            "PostgreSQL database.",
            "Service B is a Node.js service that publishes analytics events to a Kafka topic "
            "for downstream processing.",
        ],
    ),
    # English negative/adversarial case: two entities co-occur in one document with an actual
    # stated fact connecting them (reporting line), but no stated relationship between their
    # actual work — the tempting-but-unsupported inference is that one team's tooling affects
    # the other's roadmap.
    _doc(
        "en-cooccurring-teams",
        [
            "The Platform Team maintains the internal developer tooling used across the "
            "engineering organization. The Growth Team runs experiments on the marketing "
            "website to improve conversion rates. Both teams report into the VP of Engineering, "
            "but their roadmaps are managed independently.",
        ],
    ),
    # Hebrew, relationship-shaped, spread across two chunks — Hebrew counterpart to the English
    # relationship case above, to compare cross-language claim quality on the same query shape.
    _doc(
        "he-order-notification",
        [
            'שירות ה-Notification אחראי על שליחת התראות למשתמשים באמצעות דוא"ל ו-SMS.',
            "שירות ה-Order מפרסם אירוע OrderShipped בכל פעם שהזמנה נשלחת, ושירות ה-Notification "
            "מאזין לאירוע זה כדי לשלוח התראת משלוח למשתמש.",
        ],
    ),
)


async def run_document(
    llm: LLMProvider,
    detector: LanguageDetector,
    document: Stage0Document,
) -> Stage0RunResult:
    """Run the Stage 0 prompt for one document and validate the result. Never raises."""
    source_language = detector.detect(" ".join(chunk.text for chunk in document.chunks))
    prompt = build_stage0_prompt(document.chunks, source_language)
    allowed_chunk_ids = {chunk.chunk_id for chunk in document.chunks}

    start = time.monotonic()
    try:
        raw_output = await llm.generate(prompt)
    except Exception as exc:  # noqa: BLE001 - a spike must record any provider failure, not crash
        duration = time.monotonic() - start
        return Stage0RunResult(
            document_id=document.document_id,
            detected_source_language=source_language,
            generation_succeeded=False,
            parse_succeeded=False,
            language_fidelity_ok=None,
            generation_error=str(exc),
            parse_issues=[],
            language_issues=[],
            raw_output=None,
            knowledge=None,
            duration_seconds=duration,
        )
    duration = time.monotonic() - start

    try:
        knowledge = parse_and_validate(
            raw_output, document_id=document.document_id, allowed_chunk_ids=allowed_chunk_ids
        )
    except Stage0ValidationError as exc:
        return Stage0RunResult(
            document_id=document.document_id,
            detected_source_language=source_language,
            generation_succeeded=True,
            parse_succeeded=False,
            language_fidelity_ok=None,
            generation_error=None,
            parse_issues=exc.issues,
            language_issues=[],
            raw_output=raw_output,
            knowledge=None,
            duration_seconds=duration,
        )

    language_issues = check_language_fidelity(knowledge, source_language, detector)

    return Stage0RunResult(
        document_id=document.document_id,
        detected_source_language=source_language,
        generation_succeeded=True,
        parse_succeeded=True,
        language_fidelity_ok=not language_issues,
        generation_error=None,
        parse_issues=[],
        language_issues=language_issues,
        raw_output=raw_output,
        knowledge=knowledge,
        duration_seconds=duration,
    )


def _print_result(result: Stage0RunResult) -> None:
    print(
        f"=== {result.document_id} "
        f"(lang={result.detected_source_language.value}, {result.duration_seconds:.2f}s) ==="
    )
    if not result.generation_succeeded:
        print(f"GENERATION FAILED: {result.generation_error}")
        print()
        return
    if not result.parse_succeeded:
        print("PARSE FAILED:")
        for issue in result.parse_issues:
            print(f"  - {issue}")
        print("--- raw output ---")
        print(result.raw_output)
        print()
        return

    assert result.knowledge is not None
    knowledge = result.knowledge
    if not result.language_fidelity_ok:
        print("LANGUAGE FIDELITY FAILED:")
        for issue in result.language_issues:
            print(f"  - {issue}")
    print(f"summary: {knowledge.summary.text}")
    print(f"  sources: {knowledge.summary.source_chunk_ids}")
    print(f"entities ({len(knowledge.entities)}):")
    for entity in knowledge.entities:
        print(f"  - {entity.name}: {entity.description}  sources={entity.source_chunk_ids}")
    print(f"claims ({len(knowledge.claims)}):")
    for claim in knowledge.claims:
        print(f"  - {claim.text}  sources={claim.source_chunk_ids}")
    print()


async def _run_all(settings: Settings) -> list[Stage0RunResult]:
    llm = get_llm_provider(settings)
    detector = ScriptBasedLanguageDetector(settings)

    results: list[Stage0RunResult] = []
    for document in SAMPLE_DOCUMENTS:
        result = await run_document(llm, detector, document)
        _print_result(result)
        results.append(result)
    return results


async def main() -> int:
    settings = get_settings()

    print(f"LLM_PROVIDER={settings.llm_provider} model={settings.resolved_llm_model}")
    print(f"OLLAMA_BASE_URL={settings.ollama_base_url}")
    print()

    results = await _run_all(settings)

    generation_failures = sum(1 for r in results if not r.generation_succeeded)
    parse_failures = sum(1 for r in results if r.generation_succeeded and not r.parse_succeeded)
    language_failures = sum(1 for r in results if r.language_fidelity_ok is False)
    successes = sum(1 for r in results if r.parse_succeeded and r.language_fidelity_ok)
    total_summaries = sum(1 for r in results if r.knowledge is not None)
    total_entities = sum(len(r.knowledge.entities) for r in results if r.knowledge is not None)
    total_claims = sum(len(r.knowledge.claims) for r in results if r.knowledge is not None)

    print("=== summary ===")
    print(f"documents: {len(results)}")
    print(f"generation failures: {generation_failures}")
    print(f"parse failures: {parse_failures}")
    print(f"language-fidelity failures: {language_failures}")
    print(f"fully successful (parsed + language-faithful): {successes}")
    print(f"total summaries/entities/claims: {total_summaries}/{total_entities}/{total_claims}")

    return 1 if (generation_failures or parse_failures or language_failures) else 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
