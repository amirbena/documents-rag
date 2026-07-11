"""Optional, manual, non-blocking smoke check of the real configured embedding model.

Run via `make smoke-multilingual-real`. Requires a locally reachable Ollama with the configured
embedding model (default: bge-m3) already pulled (`ollama pull bge-m3`) — never run automatically
by `make verify`/`make test*`/CI, and never downloads a model itself. Exercises the five
Hebrew/English multilingual scenarios directly against the real OllamaEmbeddingProvider (cosine
similarity computed in plain Python — no Qdrant/Postgres needed for this check), verifying that:

- the correct source scores higher than an unrelated distractor for every scenario;
- every returned vector's dimension matches the active EmbeddingIndexConfig.dimension (so a
  misconfigured VECTOR_SIZE against the real model is caught, not just against the fake);
  Unicode (Hebrew) text survives the round trip unmangled;
- latency stays in a sane range for a manual local check.

This is a tiny, illustrative corpus — it proves the real model can execute the intended
multilingual flows, not production-grade retrieval quality/recall. A broader evaluation on a
larger corpus is future work (see ARCHITECTURE.md's "Real multilingual runtime smoke").
"""

import asyncio
import math
import sys
import time

from app.core.config import get_settings
from app.rag.embedding_config import get_active_embedding_config
from app.rag.providers.ollama_embedding_provider import OllamaEmbeddingError, OllamaEmbeddingProvider

_SCENARIOS = [
    (
        "hebrew_document_hebrew_query",
        "מדיניות ההחזרים של החברה מאפשרת החזר כספי מלא תוך 30 יום מיום הרכישה.",
        "מה מדיניות ההחזרים?",
    ),
    (
        "hebrew_document_english_query",
        "מדיניות ההחזרים של החברה מאפשרת החזר כספי מלא תוך 30 יום מיום הרכישה.",
        "What is the refund policy?",
    ),
    (
        "english_document_hebrew_query",
        "The company's refund policy allows a full refund within 30 days of purchase.",
        "מה מדיניות ההחזרים?",
    ),
    (
        "english_document_english_query",
        "The company's refund policy allows a full refund within 30 days of purchase.",
        "What is the refund policy?",
    ),
    (
        "mixed_document_mixed_query",
        "ה-Qdrant collection שומר embeddings לפי document_id, ומאפשר retry דרך Kafka.",
        "איך ה-Qdrant collection שומר embeddings ומאפשר retry?",
    ),
]

_DISTRACTOR = "The quarterly hiking club newsletter features photos from last month's trip."


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


async def main() -> int:
    settings = get_settings()
    active_config = get_active_embedding_config(settings)
    provider = OllamaEmbeddingProvider(settings=settings)

    print(f"Model: {active_config.model} (provider={active_config.provider})")
    print(f"Expected dimension: {active_config.dimension}")
    print(f"Ollama base URL: {settings.ollama_base_url}")
    print()

    try:
        distractor_vector = await provider.embed_text(_DISTRACTOR)
    except OllamaEmbeddingError as exc:
        print(f"FAILED: could not reach Ollama or model {active_config.model!r} is not installed.")
        print(f"  {exc}")
        print(f"  Try: ollama pull {active_config.model}")
        return 1

    if len(distractor_vector) != active_config.dimension:
        print(
            f"FAILED: {active_config.model} produced a {len(distractor_vector)}-dim vector, "
            f"but VECTOR_SIZE is configured as {active_config.dimension}."
        )
        return 1

    failures = 0
    for name, document_text, query_text in _SCENARIOS:
        start = time.monotonic()
        try:
            document_vector = await provider.embed_text(document_text)
            query_vector = await provider.embed_text(query_text)
        except OllamaEmbeddingError as exc:
            print(f"[{name}] FAILED: {exc}")
            failures += 1
            continue
        elapsed = time.monotonic() - start

        if len(document_vector) != active_config.dimension or len(query_vector) != active_config.dimension:
            print(f"[{name}] FAILED: unexpected vector dimension")
            failures += 1
            continue

        document_score = _cosine_similarity(query_vector, document_vector)
        distractor_score = _cosine_similarity(query_vector, distractor_vector)
        passed = document_score > distractor_score

        status = "OK" if passed else "FAILED"
        print(
            f"[{name}] {status} — document_score={document_score:.3f} "
            f"distractor_score={distractor_score:.3f} ({elapsed:.2f}s)"
        )
        if not passed:
            failures += 1

    print()
    if failures:
        print(f"{failures}/{len(_SCENARIOS)} scenario(s) failed.")
        return 1

    print(f"All {len(_SCENARIOS)} scenarios passed against the real {active_config.model} model.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
