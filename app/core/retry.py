"""Bounded, classification-aware retry with exponential backoff and jitter (Phase 2.10).

Hand-rolled (`asyncio.sleep`-based), not a new dependency — the logic needed here (bounded
attempts, transient-vs-permanent classification, capped backoff) is small enough that adding
`tenacity` or a similar library would be more machinery than this phase calls for, and it would
sit oddly alongside this phase's parallel choice not to introduce a new logging platform either.

Callers supply their own `is_transient(exc) -> bool` classifier — this module never decides what
counts as transient for a given provider; that judgment stays with the provider adapter that
actually knows its own error types (see `app.rag.providers.*`, `app.storage.minio_storage`).
"""

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    max_attempts: int,
    base_delay: float,
    max_delay: float,
    is_transient: Callable[[Exception], bool],
) -> T:
    """Call `fn()` up to `max_attempts` times, retrying only exceptions `is_transient` accepts.

    A permanent exception (per `is_transient`) is re-raised immediately, on the first attempt —
    it is never retried. A transient exception is retried with full-jitter exponential backoff
    (`random.uniform(0, min(max_delay, base_delay * 2 ** attempt))`) until attempts are exhausted,
    at which point the last transient exception is re-raised unchanged (never swallowed, never
    replaced with a generic "retry exhausted" wrapper — the caller's own error type/message is
    what a caller expects to catch).
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    last_exc: Exception | None = None
    for attempt in range(max_attempts):
        try:
            return await fn()
        except Exception as exc:  # noqa: BLE001 - classification is the caller's job, not ours
            if not is_transient(exc):
                raise
            last_exc = exc
            if attempt == max_attempts - 1:
                break
            delay = random.uniform(0, min(max_delay, base_delay * (2**attempt)))
            await asyncio.sleep(delay)

    assert last_exc is not None  # unreachable otherwise: either returned or raised above
    raise last_exc
