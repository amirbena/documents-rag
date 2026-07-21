"""Tests for scripts/_shutdown.py's cooperative SIGINT/SIGTERM handling (Phase 2.10).

Uses real signal delivery (signal.raise_signal) rather than calling the internal handler function
directly — the point being tested is that a real OS signal, delivered to this process while the
context manager is active, sets the flag without raising, and that prior handlers are restored
on exit so no test leaks process-global signal state to any other test in the suite.
"""

import signal

from scripts._shutdown import StopRequested, install_stop_signal_handlers


def test_sigint_sets_the_stop_flag_without_raising() -> None:
    with install_stop_signal_handlers() as stop:
        assert not stop
        signal.raise_signal(signal.SIGINT)
        assert stop
        assert stop.signal_name == "SIGINT"


def test_sigterm_sets_the_stop_flag_without_raising() -> None:
    with install_stop_signal_handlers() as stop:
        assert not stop
        signal.raise_signal(signal.SIGTERM)
        assert stop
        assert stop.signal_name == "SIGTERM"


def test_first_signal_wins_a_second_signal_does_not_overwrite_it() -> None:
    with install_stop_signal_handlers() as stop:
        signal.raise_signal(signal.SIGINT)
        signal.raise_signal(signal.SIGTERM)
        assert stop.signal_name == "SIGINT"


def test_handlers_are_restored_after_the_context_exits() -> None:
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)

    with install_stop_signal_handlers():
        assert signal.getsignal(signal.SIGINT) is not original_sigint
        assert signal.getsignal(signal.SIGTERM) is not original_sigterm

    assert signal.getsignal(signal.SIGINT) is original_sigint
    assert signal.getsignal(signal.SIGTERM) is original_sigterm


def test_handlers_are_restored_even_if_the_body_raises() -> None:
    original_sigint = signal.getsignal(signal.SIGINT)

    class _Boom(Exception):
        pass

    try:
        with install_stop_signal_handlers():
            raise _Boom("body failed")
    except _Boom:
        pass

    assert signal.getsignal(signal.SIGINT) is original_sigint


def test_a_fresh_stop_requested_is_falsy_with_no_signal_name() -> None:
    stop = StopRequested()
    assert not stop
    assert stop.signal_name is None
