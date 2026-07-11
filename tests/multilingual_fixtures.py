"""Shared deterministic multilingual test fixtures: a small Hebrew/English evaluation corpus and
a fake embedding provider that gives equivalent Hebrew/English concepts genuinely similar
vectors — not real multilingual model quality, just enough determinism to prove cross-language
retrieval wiring works. Used by both the integration suite and the backend E2E matrix so the
corpus/fake never drifts between tiers.

Real-model multilingual retrieval quality is NOT validated here — that requires a separate,
manual evaluation run against a real multilingual embedding model (see "Multilingual embedding
architecture" in ARCHITECTURE.md).
"""

import hashlib
import math
import re
from collections.abc import AsyncIterator

from app.rag.providers.embedding_provider import EmbeddingProvider
from app.rag.providers.llm_provider import LLMProvider

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)

# Parallel Hebrew/English vocabulary: every token on either side of one entry maps to the same
# canonical concept, so "vacation" and "חופשה" contribute to the same hashed dimension. Technical
# identifiers (kafka, kubernetes, qdrant, langchain) are the same spelling in both languages, so
# they naturally already do this without an entry here.
_CONCEPT_SYNONYMS: dict[str, str] = {
    # vacation policy concept
    "vacation": "concept_vacation",
    "חופשה": "concept_vacation",
    "policy": "concept_policy",
    "מדיניות": "concept_policy",
    "days": "concept_days",
    "יום": "concept_days",
    "ימים": "concept_days",
    "employee": "concept_employee",
    "employees": "concept_employee",
    "עובד": "concept_employee",
    "עובדים": "concept_employee",
    "entitled": "concept_entitlement",
    "entitlement": "concept_entitlement",
    "זכאים": "concept_entitlement",
    "זכאי": "concept_entitlement",
    # refund policy concept
    "refund": "concept_refund",
    "refunds": "concept_refund",
    "החזר": "concept_refund",
    "החזרים": "concept_refund",
    "return": "concept_return",
    "returned": "concept_return",
    "להחזיר": "concept_return",
    # distractor concept (pizza recipe) — must never match the above
    "pizza": "concept_pizza",
    "פיצה": "concept_pizza",
    "recipe": "concept_recipe",
    "מתכון": "concept_recipe",
    "dough": "concept_dough",
    "בצק": "concept_dough",
}


def _canonicalize(token: str) -> str:
    """Map a token to its shared cross-language concept ID, or itself if not a known synonym."""
    return _CONCEPT_SYNONYMS.get(token.lower(), token.lower())


def _hash_token(token: str, dimensions: int) -> int:
    """Map a token to a stable index in [0, dimensions) via SHA-256 — deterministic, no randomness."""
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % dimensions


class MultilingualFakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic bag-of-concepts hashing embedding with a small Hebrew/English synonym table.

    Behaves like a real multilingual embedding model only in one narrow, deliberate sense: a
    Hebrew word and its English counterpart in `_CONCEPT_SYNONYMS` hash into the same vector
    dimension, so equivalent Hebrew/English concepts produce genuinely similar vectors under
    cosine search — not because of real semantic understanding, but because both are canonicalized
    to the same token before hashing. Unrelated (distractor) content still hashes to unrelated
    dimensions, so it never becomes a false match.
    """

    def __init__(self, vector_size: int) -> None:
        self._vector_size = vector_size

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one deterministic embedding vector per input text, in the same order."""
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._vector_size
        for token in _TOKEN_PATTERN.findall(text.lower()):
            concept = _canonicalize(token)
            vector[_hash_token(concept, self._vector_size)] += 1.0

        norm = math.sqrt(sum(component * component for component in vector))
        if norm == 0.0:
            return vector
        return [component / norm for component in vector]


class MultilingualFakeLLMProvider(LLMProvider):
    """Yields a fixed, ordered sequence of text chunks — no model, no network call."""

    def __init__(self, chunks: tuple[str, ...] = ("Based ", "on ", "the ", "context, ", "yes.")) -> None:
        self.chunks = chunks

    async def generate(self, prompt: str) -> str:
        """Return the full completion by joining every fixed chunk."""
        return "".join(self.chunks)

    async def stream_generate(self, prompt: str) -> AsyncIterator[str]:
        """Yield each fixed chunk in order."""
        for chunk in self.chunks:
            yield chunk


# --- Deterministic multilingual evaluation corpus ----------------------------------------------

HEBREW_VACATION_DOCUMENT = (
    "מדיניות חופשה שנתית\n"
    "כל העובדים זכאים ל-20 ימי חופשה בשנה, בהתאם לוותק במשרה.\n"
).encode()

ENGLISH_VACATION_DOCUMENT = (
    b"Annual Vacation Policy\n"
    b"All employees are entitled to 20 vacation days per year, based on seniority.\n"
)

MIXED_TECHNICAL_DOCUMENT = (
    "Refund Processing Architecture / ארכיטקטורת עיבוד החזרים\n"
    "The refund service (שירות ההחזרים) uses Kafka for event streaming and Qdrant for "
    "similarity search, orchestrated through LangChain and deployed on Kubernetes.\n"
    "כאשר עובד מבצע בקשת החזר, המערכת שולחת אירוע ל-Kafka ומעדכנת את מסד הנתונים.\n"
).encode()

DISTRACTOR_DOCUMENT = (
    b"Pizza dough recipe: flour, water, yeast, and salt, kneaded for ten minutes and left to "
    b"rise for one hour before baking at high heat.\n"
)

# The rule-based decision layer (app/rag/decision.py) only recognizes English retrieval-trigger
# words ("document", "uploaded", ...) — deliberately unchanged by this milestone (see CLAUDE.md).
# So every retrieval-triggering question below embeds one such English word even when the
# question is otherwise Hebrew — this doubles as the required "Hebrew question containing English
# technical terminology" scenario, and still detects as Hebrew overall (Hebrew words dominate).
HEBREW_RETRIEVAL_QUESTION = "לפי ה-document שהועלה למערכת, כמה ימי חופשה מגיעים לעובד בשנה?"
ENGLISH_RETRIEVAL_QUESTION = (
    "According to the uploaded document, how many vacation days is an employee entitled to per year?"
)
MIXED_RETRIEVAL_QUESTION = (
    "According to the uploaded document, how does the refund service use Kafka ו-Kubernetes "
    "עבור עיבוד החזרים?"
)
