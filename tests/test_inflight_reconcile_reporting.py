"""Worker-side tests for orchestrator-restart recovery (gen-orchestrator #346).

These exercise the in-flight reporting helpers and the bounded persistent
outgoing queue WITHOUT standing up a real gRPC stream or a discovered
endpoint. The Worker instance is built via object.__new__ and only the
attributes the helpers touch are populated.
"""

import queue
import threading

from gen_worker.worker import Worker, OUTGOING_QUEUE_MAXSIZE
from gen_worker.pb import worker_scheduler_pb2 as pb


def _bare_worker():
    w = object.__new__(Worker)
    w.worker_id = "w1"
    w._running = True
    w._stop_event = threading.Event()
    w._active_requests = {}
    w._active_requests_lock = threading.Lock()
    w._outgoing_queue = queue.Queue(maxsize=OUTGOING_QUEUE_MAXSIZE)
    return w


def _job_result_msg(rid):
    return pb.WorkerSchedulerMessage(job_result=pb.JobExecutionResult(request_id=rid))


def _event_msg():
    return pb.WorkerSchedulerMessage(worker_event=pb.WorkerEvent(request_id="", event_type="noise"))


def test_buffered_result_request_ids_scans_only_job_results():
    w = _bare_worker()
    w._outgoing_queue.put_nowait(_event_msg())
    w._outgoing_queue.put_nowait(_job_result_msg("r-buffered"))
    w._outgoing_queue.put_nowait(_event_msg())

    ids = w._buffered_result_request_ids()
    assert ids == ["r-buffered"]
    # Non-destructive: the queue is untouched.
    assert w._outgoing_queue.qsize() == 3


def test_buffered_result_empty_when_no_results():
    w = _bare_worker()
    w._outgoing_queue.put_nowait(_event_msg())
    assert w._buffered_result_request_ids() == []


def test_outgoing_queue_drops_oldest_on_overflow():
    w = _bare_worker()
    # Fill to capacity with distinguishable results.
    for i in range(OUTGOING_QUEUE_MAXSIZE):
        w._outgoing_queue.put_nowait(_job_result_msg(f"r{i}"))
    assert w._outgoing_queue.full()

    # One more send must drop the oldest (r0) and admit the newest.
    w._send_message(_job_result_msg("r-new"))
    assert w._outgoing_queue.qsize() == OUTGOING_QUEUE_MAXSIZE

    drained = []
    while not w._outgoing_queue.empty():
        drained.append(w._outgoing_queue.get_nowait().job_result.request_id)
    assert "r0" not in drained          # oldest dropped
    assert "r-new" in drained           # newest admitted
    assert drained[-1] == "r-new"       # appended at the tail


def test_inflight_report_unions_active_and_buffered():
    w = _bare_worker()
    # Two handlers running; one result buffered (completed, not yet shipped).
    w._active_requests = {"r-active-1": object(), "r-active-2": object()}
    w._outgoing_queue.put_nowait(_job_result_msg("r-buffered"))
    # A result for an already-active request must not be double-counted.
    w._outgoing_queue.put_nowait(_job_result_msg("r-active-1"))

    with w._active_requests_lock:
        in_flight = list(w._active_requests.keys())
    seen = set(in_flight)
    for rid in w._buffered_result_request_ids():
        if rid not in seen:
            seen.add(rid)
            in_flight.append(rid)

    assert set(in_flight) == {"r-active-1", "r-active-2", "r-buffered"}
    # No duplicate of r-active-1.
    assert in_flight.count("r-active-1") == 1
