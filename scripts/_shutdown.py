"""Cooperative SIGINT/SIGTERM handling shared by the standalone `process_pending_*.py` batch
scripts (Phase 2.10).

Each script installs these handlers for the duration of its own `asyncio.run(main())` invocation
only — this is never wired into FastAPI's lifespan (`app/core/lifespan.py`). The API process and
these standalone worker-script processes are separate, independently-managed lifecycles; nothing
here starts, stops, or is stopped by the other.

A handler installed here never raises inside the running work unit — it only flips a process-local
stop flag that each script's own loop checks before claiming its next job. A job already claimed
(and its worker's own claim/process/commit sequence) is never interrupted mid-flight by this
module; it always reaches its own next commit before the script's loop re-checks the flag. What
happens if the process is *force-killed* (not merely signaled) between a claim and that job's
terminal commit is a separate, pre-existing question answered per-job-type by each script's own
module docstring and by the worker/service module it calls — this module does not change that.
"""

import logging
import signal
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType

logger = logging.getLogger(__name__)

_STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM)


class StopRequested:
    """A process-local flag, set once by the first SIGINT/SIGTERM this process receives."""

    def __init__(self) -> None:
        self._requested = False
        self._signal_name: str | None = None

    def __bool__(self) -> bool:
        return self._requested

    def set(self, signal_name: str) -> None:
        """Record that a stop was requested; the first signal received wins, later ones no-op."""
        if not self._requested:
            self._requested = True
            self._signal_name = signal_name

    @property
    def signal_name(self) -> str | None:
        """The name of the signal that triggered the stop, or None if none has been received."""
        return self._signal_name


@contextmanager
def install_stop_signal_handlers() -> Iterator[StopRequested]:
    """Install SIGINT/SIGTERM handlers for this context's duration; restore prior handlers on exit.

    Yields a `StopRequested` flag the caller checks before claiming its next work unit. Previous
    handlers are always restored on exit (including on an exception) so this never leaks
    process-global signal state — required for tests, and for correctness if a future caller ever
    invokes this more than once in the same interpreter.
    """
    stop = StopRequested()
    previous_handlers: dict[int, object] = {}

    def _handle(received_signal: int, _frame: FrameType | None) -> None:
        signal_name = signal.Signals(received_signal).name
        logger.info(
            "Received %s; will stop before claiming the next job.",
            signal_name,
            extra={"event": "worker_stop_signal_received", "signal": signal_name},
        )
        stop.set(signal_name)

    try:
        for sig in _STOP_SIGNALS:
            previous_handlers[sig] = signal.signal(sig, _handle)
        yield stop
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)  # type: ignore[arg-type]
