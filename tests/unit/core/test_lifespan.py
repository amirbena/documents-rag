"""Tests for the application lifespan (Phase 2.10, app/core/lifespan.py).

Driven directly against `build_lifespan()` with a stand-in engine — no real FastAPI app, no real
database — so these stay fast, fake-only unit tests per tests/unit/conftest.py's tier contract.
"""

import logging
from unittest.mock import AsyncMock

import pytest

from app.core.lifespan import build_lifespan


async def test_shutdown_disposes_the_engine() -> None:
    engine = AsyncMock()
    lifespan = build_lifespan(engine)

    async with lifespan(app=None):
        engine.dispose.assert_not_called()

    engine.dispose.assert_awaited_once()


async def test_startup_exception_still_disposes_already_registered_resources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A step between resource-acquisition and `yield` raising must not leak the engine.

    Forces the second `logger.info` call inside the lifespan (the "startup complete" log, which
    runs after the engine's dispose callback is registered but before `yield`) to raise — proving
    the AsyncExitStack unwinds and disposes the engine even though the lifespan never reached the
    "app is serving" point, exactly as a genuine startup failure would behave.
    """
    engine = AsyncMock()
    lifespan = build_lifespan(engine)

    call_count = {"n": 0}
    original_info = logging.Logger.info

    def _fail_on_second_call(self: logging.Logger, msg: str, *args: object, **kwargs: object) -> None:
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated startup failure")
        original_info(self, msg, *args, **kwargs)

    monkeypatch.setattr(logging.Logger, "info", _fail_on_second_call)

    with pytest.raises(RuntimeError, match="simulated startup failure"):
        async with lifespan(app=None):
            pytest.fail("lifespan body must not be reached after a startup failure")

    engine.dispose.assert_awaited_once()


async def test_lifespan_never_dispatches_shutdown_logs_without_reaching_yield(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No shutdown log line is emitted when startup itself never completes."""
    engine = AsyncMock()
    lifespan = build_lifespan(engine)

    events: list[str] = []
    original_info = logging.Logger.info

    def _capture(self: logging.Logger, msg: str, *args: object, **kwargs: object) -> None:
        extra = kwargs.get("extra") or {}
        event = extra.get("event")
        if event:
            events.append(event)
        if event == "app_startup_complete":
            raise RuntimeError("simulated startup failure")
        original_info(self, msg, *args, **kwargs)

    monkeypatch.setattr(logging.Logger, "info", _capture)

    with pytest.raises(RuntimeError):
        async with lifespan(app=None):
            pass  # pragma: no cover - unreachable after the simulated failure

    assert events == ["app_startup_begin", "app_startup_complete"]
    assert "app_shutdown_begin" not in events
    assert "app_shutdown_complete" not in events
