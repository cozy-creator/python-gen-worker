"""#462-T4: a worker whose capability token is permanently rejected must EXIT
instead of zombie-spinning in the reconnect backoff loop (#338) forever.

The contract:
  * UNAUTHENTICATED / PERMISSION_DENIED on connect/register or the control
    stream counts ONE consecutive auth failure.
  * After GEN_WORKER_MAX_AUTH_FAILURES (default 10) consecutive failures the
    worker logs a fatal line and exits (injectable exit func for tests).
  * Any inbound scheduler message proves auth succeeded and resets the counter.
  * Transient network errors (UNAVAILABLE etc.) neither count nor reset.

Bare-Worker pattern, same as test_reconnect_robustness.
"""

from __future__ import annotations

import threading
import time
from typing import Any, List, Tuple

import grpc
import pytest

from gen_worker.worker import Worker


class _FakeRpcError(grpc.RpcError):
    def __init__(self, code: grpc.StatusCode, details: str = "boom") -> None:
        self._code = code
        self._details = details

    def code(self) -> grpc.StatusCode:
        return self._code

    def details(self) -> str:
        return self._details


def _auth_worker(limit: int = 10) -> Tuple[Worker, List[int]]:
    w = Worker.__new__(Worker)
    w._max_auth_failures = limit
    w._auth_failure_count = 0
    w._running = True
    w._stop_event = threading.Event()
    exits: List[int] = []
    w._fatal_exit = exits.append  # type: ignore[assignment]
    return w, exits


class _UnknownMsg:
    """Minimal inbound message; falls through every _process_message branch."""

    def WhichOneof(self, _field: str) -> str:
        return "unknown_type"


# --------------------------------------------------------------------------- #
# Classification: only true auth codes count.                                  #
# --------------------------------------------------------------------------- #


def test_is_auth_rejection_classification() -> None:
    assert Worker._is_auth_rejection(grpc.StatusCode.UNAUTHENTICATED) is True
    assert Worker._is_auth_rejection(grpc.StatusCode.PERMISSION_DENIED) is True
    for code in (
        grpc.StatusCode.UNAVAILABLE,
        grpc.StatusCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.INTERNAL,
        grpc.StatusCode.CANCELLED,
        grpc.StatusCode.FAILED_PRECONDITION,
        grpc.StatusCode.UNKNOWN,
    ):
        assert Worker._is_auth_rejection(code) is False, code


# --------------------------------------------------------------------------- #
# Consecutive counting + exit trigger.                                          #
# --------------------------------------------------------------------------- #


def test_consecutive_auth_failures_trigger_exit_at_limit() -> None:
    w, exits = _auth_worker(limit=10)
    for _ in range(9):
        w._note_auth_failure(grpc.StatusCode.UNAUTHENTICATED, "token revoked")
    assert exits == [] and w._running is True, "must not exit below the budget"
    assert w._auth_failure_count == 9

    w._note_auth_failure(grpc.StatusCode.UNAUTHENTICATED, "token revoked")
    assert exits == [1], "10th consecutive auth failure must exit"
    assert w._running is False and w._stop_event.is_set()


def test_zero_limit_disables_the_cap() -> None:
    w, exits = _auth_worker(limit=0)
    for _ in range(50):
        w._note_auth_failure(grpc.StatusCode.PERMISSION_DENIED, "denied")
    assert exits == []


def test_inbound_message_resets_counter_consecutive_semantics() -> None:
    w, exits = _auth_worker(limit=3)
    w._last_server_msg_at = time.monotonic()

    w._note_auth_failure(grpc.StatusCode.UNAUTHENTICATED, "x")
    w._note_auth_failure(grpc.StatusCode.UNAUTHENTICATED, "x")
    assert w._auth_failure_count == 2 and exits == []

    # A successful auth (any inbound scheduler message) resets the counter.
    w._process_message(_UnknownMsg())  # type: ignore[arg-type]
    assert w._auth_failure_count == 0

    # The budget must be spent again from zero — consecutive, not cumulative.
    w._note_auth_failure(grpc.StatusCode.UNAUTHENTICATED, "x")
    w._note_auth_failure(grpc.StatusCode.UNAUTHENTICATED, "x")
    assert exits == []
    w._note_auth_failure(grpc.StatusCode.UNAUTHENTICATED, "x")
    assert exits == [1]


# --------------------------------------------------------------------------- #
# Receive-loop wiring: auth errors count, transient errors don't.              #
# --------------------------------------------------------------------------- #


class _RaisingStream:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    def __iter__(self) -> Any:
        raise self._exc


def _receive_loop_worker(limit: int = 10) -> Tuple[Worker, List[int], List[bool]]:
    w, exits = _auth_worker(limit=limit)
    torn_down: List[bool] = []
    w._handle_connection_error = lambda: torn_down.append(True)  # type: ignore[method-assign]
    return w, exits, torn_down


@pytest.mark.parametrize(
    "code", [grpc.StatusCode.UNAUTHENTICATED, grpc.StatusCode.PERMISSION_DENIED]
)
def test_receive_loop_counts_auth_rejection_and_reconnects(code: grpc.StatusCode) -> None:
    w, exits, torn_down = _receive_loop_worker()
    w._stream = _RaisingStream(_FakeRpcError(code, "credentials rejected"))
    w._receive_loop()
    assert w._auth_failure_count == 1
    assert torn_down == [True], "below the budget the worker still reconnects"
    assert exits == []


@pytest.mark.parametrize(
    "code",
    [
        grpc.StatusCode.UNAVAILABLE,
        grpc.StatusCode.DEADLINE_EXCEEDED,
        grpc.StatusCode.INTERNAL,
        grpc.StatusCode.CANCELLED,
    ],
)
def test_receive_loop_transient_errors_do_not_count(code: grpc.StatusCode) -> None:
    w, exits, torn_down = _receive_loop_worker()
    w._stream = _RaisingStream(_FakeRpcError(code, "network blip"))
    w._receive_loop()
    assert w._auth_failure_count == 0, f"{code} must not count as an auth failure"
    assert torn_down == [True]
    assert exits == []


def test_receive_loop_exits_on_final_auth_failure() -> None:
    w, exits, torn_down = _receive_loop_worker(limit=2)
    w._stream = _RaisingStream(_FakeRpcError(grpc.StatusCode.UNAUTHENTICATED, "revoked"))
    w._receive_loop()
    assert exits == []
    w._stream = _RaisingStream(_FakeRpcError(grpc.StatusCode.UNAUTHENTICATED, "revoked"))
    w._receive_loop()
    assert exits == [1]
    assert w._running is False and w._stop_event.is_set()


# --------------------------------------------------------------------------- #
# Init defaults guard (bare workers skip __init__, so pin the source).          #
# --------------------------------------------------------------------------- #


def test_init_reads_env_with_default_10() -> None:
    import inspect

    src = inspect.getsource(Worker.__init__)
    assert 'GEN_WORKER_MAX_AUTH_FAILURES", 10' in src
    assert "_auth_failure_count = 0" in src
    assert "_fatal_exit" in src
