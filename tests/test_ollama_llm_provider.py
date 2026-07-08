"""Tests for streaming OllamaLLMProvider with a mocked Ollama HTTP transport."""

import httpx
import pytest

from app.core.config import get_settings
from app.rag.providers.ollama_llm_provider import OllamaLLMError, OllamaLLMProvider


def _provider(transport: httpx.MockTransport) -> OllamaLLMProvider:
    return OllamaLLMProvider(settings=get_settings(), transport=transport)


def _ndjson_transport(lines: list[str]) -> httpx.MockTransport:
    """Build a mock transport whose /api/generate response is the given NDJSON lines."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = "\n".join(lines) + "\n"
        return httpx.Response(200, content=body.encode())

    return httpx.MockTransport(handler)


async def _collect(provider: OllamaLLMProvider, prompt: str) -> list[str]:
    return [chunk async for chunk in provider.stream_generate(prompt)]


async def test_stream_generate_yields_chunks_in_order() -> None:
    """A successful streamed response should yield chunks in arrival order, stopping at done."""
    transport = _ndjson_transport(
        [
            '{"response": "Hel", "done": false}',
            '{"response": "lo", "done": false}',
            '{"response": "", "done": true}',
        ]
    )
    provider = _provider(transport)

    chunks = await _collect(provider, "hi")

    assert chunks == ["Hel", "lo"]


async def test_generate_joins_streamed_chunks() -> None:
    """generate() should collect all streamed chunks into a single string."""
    transport = _ndjson_transport(
        [
            '{"response": "Hel", "done": false}',
            '{"response": "lo", "done": true}',
        ]
    )
    provider = _provider(transport)

    result = await provider.generate("hi")

    assert result == "Hello"


async def test_stream_generate_empty_prompt_raises_value_error() -> None:
    """Empty or whitespace-only prompt should be rejected before any HTTP call is made."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not make an HTTP request for an empty prompt")

    provider = _provider(httpx.MockTransport(handler))

    with pytest.raises(ValueError):
        await _collect(provider, "")

    with pytest.raises(ValueError):
        await _collect(provider, "   ")


async def test_stream_generate_unreachable_raises_ollama_llm_error() -> None:
    """A connection failure should raise OllamaLLMError."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    provider = _provider(httpx.MockTransport(handler))

    with pytest.raises(OllamaLLMError):
        await _collect(provider, "hi")


async def test_stream_generate_non_200_response_raises_ollama_llm_error() -> None:
    """A non-200 response should raise OllamaLLMError."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "internal error"})

    provider = _provider(httpx.MockTransport(handler))

    with pytest.raises(OllamaLLMError):
        await _collect(provider, "hi")


async def test_stream_generate_malformed_json_line_raises_ollama_llm_error() -> None:
    """A non-JSON line in the stream should raise OllamaLLMError."""
    transport = _ndjson_transport(["not json at all"])
    provider = _provider(transport)

    with pytest.raises(OllamaLLMError):
        await _collect(provider, "hi")


async def test_stream_generate_ending_without_done_raises_ollama_llm_error() -> None:
    """A stream that ends without a done=true line should raise OllamaLLMError."""
    transport = _ndjson_transport(
        [
            '{"response": "a", "done": false}',
            '{"response": "b", "done": false}',
        ]
    )
    provider = _provider(transport)

    with pytest.raises(OllamaLLMError):
        await _collect(provider, "hi")
