"""Tests for the bounded, classification-aware retry helper (Phase 2.10, app/core/retry.py)."""

import pytest

from app.core.retry import retry_async


class _Transient(Exception):
    pass


class _Permanent(Exception):
    pass


def _is_transient(exc: Exception) -> bool:
    return isinstance(exc, _Transient)


async def test_succeeds_on_first_attempt_without_sleeping(monkeypatch: pytest.MonkeyPatch) -> None:
    sleep_calls: list[float] = []
    monkeypatch.setattr("asyncio.sleep", lambda d: sleep_calls.append(d) or _noop())

    calls = {"count": 0}

    async def _fn() -> str:
        calls["count"] += 1
        return "ok"

    result = await retry_async(
        _fn, max_attempts=3, base_delay=0.01, max_delay=0.05, is_transient=_is_transient
    )

    assert result == "ok"
    assert calls["count"] == 1
    assert sleep_calls == []


async def _noop():
    return None


async def test_permanent_error_is_never_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("asyncio.sleep", lambda d: _noop())
    calls = {"count": 0}

    async def _fn() -> str:
        calls["count"] += 1
        raise _Permanent("permanent failure")

    with pytest.raises(_Permanent):
        await retry_async(
            _fn, max_attempts=5, base_delay=0.01, max_delay=0.05, is_transient=_is_transient
        )

    assert calls["count"] == 1


async def test_transient_error_is_retried_until_success(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("asyncio.sleep", lambda d: _noop())
    calls = {"count": 0}

    async def _fn() -> str:
        calls["count"] += 1
        if calls["count"] < 3:
            raise _Transient("transient failure")
        return "recovered"

    result = await retry_async(
        _fn, max_attempts=5, base_delay=0.01, max_delay=0.05, is_transient=_is_transient
    )

    assert result == "recovered"
    assert calls["count"] == 3


async def test_retry_exhaustion_reraises_the_last_transient_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("asyncio.sleep", lambda d: _noop())
    calls = {"count": 0}

    async def _fn() -> str:
        calls["count"] += 1
        raise _Transient(f"attempt {calls['count']}")

    with pytest.raises(_Transient, match="attempt 3"):
        await retry_async(
            _fn, max_attempts=3, base_delay=0.01, max_delay=0.05, is_transient=_is_transient
        )

    assert calls["count"] == 3


async def test_backoff_is_bounded_by_max_delay(monkeypatch: pytest.MonkeyPatch) -> None:
    sleep_calls: list[float] = []

    async def _fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    monkeypatch.setattr("asyncio.sleep", _fake_sleep)

    async def _fn() -> str:
        raise _Transient("always fails")

    with pytest.raises(_Transient):
        await retry_async(
            _fn, max_attempts=4, base_delay=10.0, max_delay=0.02, is_transient=_is_transient
        )

    assert len(sleep_calls) == 3  # one sleep between each of the 4 attempts, none after the last
    assert all(delay <= 0.02 for delay in sleep_calls)


def test_max_attempts_below_one_is_rejected() -> None:
    async def _fn() -> str:
        return "unreachable"

    with pytest.raises(ValueError, match="max_attempts"):
        import asyncio

        asyncio.run(
            retry_async(_fn, max_attempts=0, base_delay=0.01, max_delay=0.05, is_transient=_is_transient)
        )
