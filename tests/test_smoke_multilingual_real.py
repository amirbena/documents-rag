"""Tests for scripts/smoke_multilingual_real.py's failure-mode contract.

The optional real-model smoke check must fail clearly (non-zero exit, explicit message) when
Ollama or the configured model isn't reachable — never raise an unhandled exception, and never
be mistaken for a passing run. It must never be invoked by make verify/test*/CI itself; these
tests exercise `main()` directly against a fake failing provider instead.
"""

import pytest

from app.rag.providers.ollama_embedding_provider import OllamaEmbeddingError
from scripts.smoke_multilingual_real import main


class _UnreachableEmbeddingProvider:
    def __init__(self, settings=None, transport=None) -> None:
        pass

    async def embed_text(self, text: str) -> list[float]:
        raise OllamaEmbeddingError("Ollama unreachable at /api/embeddings: connection refused")


async def test_main_returns_nonzero_and_prints_clear_message_when_ollama_unreachable(
    monkeypatch, capsys
) -> None:
    """A missing/unreachable Ollama must fail clearly, not raise or hang."""
    import scripts.smoke_multilingual_real as smoke_module

    monkeypatch.setattr(smoke_module, "OllamaEmbeddingProvider", _UnreachableEmbeddingProvider)

    exit_code = await main()

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "FAILED" in output
    assert "ollama pull" in output.lower()


async def test_main_does_not_raise_on_failure(monkeypatch) -> None:
    """main() must return an error code, never propagate an unhandled exception, when unreachable."""
    import scripts.smoke_multilingual_real as smoke_module

    monkeypatch.setattr(smoke_module, "OllamaEmbeddingProvider", _UnreachableEmbeddingProvider)

    try:
        exit_code = await main()
    except OllamaEmbeddingError:
        pytest.fail("main() must catch OllamaEmbeddingError itself, not let it propagate")

    assert exit_code == 1
