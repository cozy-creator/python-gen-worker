"""Worker P0 reliability regressions (gen-worker #357).

Two live defects, both driven through the real Worker helpers via a bare
``Worker.__new__`` instance (same pattern as test_reconnect_robustness /
test_worker_dispatch), so we exercise the actual code paths:

1. Drain crash: ``_emit_worker_drain_result`` built ``pb.WorkerDrainResult`` with
   a ``worker_id`` field that #321 removed from the proto -> ValueError, swallowed
   by the receive loop, so drain never ran and pods only died by external kill.
2. Result loss on reconnect: ``_send_message`` refused to enqueue while
   ``_stop_event`` was set. That event is set on every transient reconnect (with
   ``_running`` still True), so a JobExecutionResult finished in the reconnect
   window was dropped and the request stranded orchestrator-side.
"""

from __future__ import annotations

import queue
import threading

from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.worker import OUTGOING_QUEUE_MAXSIZE, Worker


def _sendable_worker() -> Worker:
    """Bare Worker wired with just enough state for _send_message."""
    w = Worker.__new__(Worker)
    w._running = True
    w._stop_event = threading.Event()
    w._outgoing_queue = queue.Queue(maxsize=OUTGOING_QUEUE_MAXSIZE)
    return w


# --------------------------------------------------------------------------- #
# Fix 1: drain builds a valid proto (no removed worker_id field).
# --------------------------------------------------------------------------- #


def test_emit_worker_drain_result_builds_valid_proto() -> None:
    w = _sendable_worker()
    w._active_request_count = lambda: 3  # type: ignore[method-assign]
    w._emit_worker_event_bytes = lambda *a, **k: None  # type: ignore[method-assign]

    # Previously raised ValueError("WorkerDrainResult has no worker_id field").
    w._emit_worker_drain_result("scheduler_drain", "draining")

    msg = w._outgoing_queue.get_nowait()
    assert msg.WhichOneof("msg") == "worker_drain_result"
    assert msg.worker_drain_result.status == "draining"
    assert msg.worker_drain_result.reason == "scheduler_drain"
    assert msg.worker_drain_result.active_requests == 3


# --------------------------------------------------------------------------- #
# Fix 2: results enqueue during the reconnect window; refused only on shutdown.
# --------------------------------------------------------------------------- #


def test_result_enqueued_during_reconnect_window() -> None:
    w = _sendable_worker()
    # Reconnect in progress: _stop_event is set but _running stays True.
    w._stop_event.set()

    result = pb.WorkerSchedulerMessage(job_result=pb.JobExecutionResult(request_id="r1"))
    w._send_message(result)

    assert w._outgoing_queue.qsize() == 1, "result must survive the reconnect window"
    got = w._outgoing_queue.get_nowait()
    assert got.job_result.request_id == "r1"


def test_message_refused_only_on_genuine_shutdown() -> None:
    w = _sendable_worker()
    # stop() clears _running before setting _stop_event -> genuine shutdown.
    w._running = False
    w._stop_event.set()

    result = pb.WorkerSchedulerMessage(job_result=pb.JobExecutionResult(request_id="r2"))
    w._send_message(result)

    assert w._outgoing_queue.qsize() == 0, "no enqueue once the worker is stopping"
