"""Incremental Server-Sent-Events parsing for the backend E2E suite.

Parses an httpx streaming response line by line as it arrives, instead of inspecting one fully
buffered response string — this is what actually exercises the endpoint's streaming semantics
(event order, incremental delivery) rather than just its final content.
"""

import json
from collections.abc import AsyncIterator

import httpx

SseEvent = tuple[str, dict]


async def iter_sse_events(response: httpx.Response) -> AsyncIterator[SseEvent]:
    """Yield (event_name, data) pairs as each SSE block completes, in arrival order."""
    event_name: str | None = None
    data_line: str | None = None

    async for line in response.aiter_lines():
        if line.startswith("event: "):
            event_name = line.removeprefix("event: ")
        elif line.startswith("data: "):
            data_line = line.removeprefix("data: ")
        elif line == "":
            if event_name is not None and data_line is not None:
                yield event_name, json.loads(data_line)
            event_name = None
            data_line = None
