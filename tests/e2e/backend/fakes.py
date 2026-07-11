"""Deterministic fake AI providers for the backend E2E suite — no network calls, ever.

FakeEmbeddingProvider and FakeStreamingLLMProvider implement the same EmbeddingProvider/
LLMProvider interfaces the real Ollama-backed providers implement, so they are swapped in via
provider-factory monkeypatching (see conftest.py) rather than any branch in production code.
"""

import hashlib
import math
import re
from collections.abc import AsyncIterator

from app.rag.providers.embedding_provider import EmbeddingProvider
from app.rag.providers.llm_provider import LLMProvider

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def _hash_token(token: str, dimensions: int) -> int:
    """Map a token to a stable index in [0, dimensions) via SHA-256 — deterministic, no randomness."""
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % dimensions


class FakeEmbeddingProvider(EmbeddingProvider):
    """Deterministic bag-of-words hashing embedding — no model, no network call.

    Each text is embedded as an L2-normalized histogram of its tokens hashed into a fixed-size
    vector. Texts that share words produce vectors with high cosine similarity, so a query and
    its relevant indexed chunks land close together under Qdrant's real cosine search — without
    any real embedding model.
    """

    def __init__(self, vector_size: int) -> None:
        self._vector_size = vector_size

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one deterministic embedding vector per input text, in the same order."""
        return [self._embed_one(text) for text in texts]

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self._vector_size
        for token in _TOKEN_PATTERN.findall(text.lower()):
            vector[_hash_token(token, self._vector_size)] += 1.0

        norm = math.sqrt(sum(component * component for component in vector))
        if norm == 0.0:
            return vector
        return [component / norm for component in vector]


class FakeStreamingLLMProvider(LLMProvider):
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


class FakeFailingLLMProvider(LLMProvider):
    """Streams one chunk, then raises — exercises the chat endpoint's mid-stream error path."""

    def __init__(self, message: str = "simulated LLM failure") -> None:
        self._message = message

    async def generate(self, prompt: str) -> str:
        """Always raise; this fake simulates a failing LLM call."""
        raise RuntimeError(self._message)

    async def stream_generate(self, prompt: str) -> AsyncIterator[str]:
        """Yield one chunk, then raise mid-stream."""
        yield "Partial answer"
        raise RuntimeError(self._message)
