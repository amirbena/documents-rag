"""Shared fixtures for the entire unit tier — fakes/mocks only, no Docker, no real timing.

This conftest applies to every test under tests/unit/.
"""

import asyncio

import pytest


@pytest.fixture(autouse=True)
def _no_real_sleep_in_unit_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make asyncio.sleep instant everywhere in the unit tier.

    Phase 2.10's provider retry/backoff (app/core/retry.py) genuinely calls asyncio.sleep between
    attempts — real backoff delays have no place in a "well under a second" fake/mock-only unit
    suite. Any test that wants to assert the actual requested delay value should still capture the
    argument passed to this patched sleep, exactly as tests/unit/core/test_retry.py already does.
    """

    async def _instant_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr(asyncio, "sleep", _instant_sleep)
