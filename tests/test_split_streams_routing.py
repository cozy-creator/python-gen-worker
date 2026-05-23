"""#345 Improvement A (worker side): split outgoing stream routing.

Verifies _select_outgoing_queue routes JobExecutionResult to the results
queue and lifecycle/incremental events to the events queue when split-streams
are active, and falls back to the primary queue otherwise. Also checks the
opening-handshake registration carries the split-stream capability + role.
"""

from __future__ import annotations

import queue
import threading

from gen_worker.worker import OUTGOING_QUEUE_MAXSIZE, Worker
from gen_worker.pb import worker_scheduler_pb2 as pb


def _bare() -> Worker:
    w = object.__new__(Worker)
    w._running = True
    w._stop_event = threading.Event()
    w._outgoing_queue = queue.Queue(maxsize=OUTGOING_QUEUE_MAXSIZE)
    w._results_outgoing_queue = queue.Queue(maxsize=OUTGOING_QUEUE_MAXSIZE)
    w._events_outgoing_queue = queue.Queue(maxsize=OUTGOING_QUEUE_MAXSIZE)
    w._split_streams_enabled = True
    w._aux_streams_active = True
    w.worker_id = "w1"
    w.release_id = "r1"
    w.runpod_pod_id = ""
    w.worker_jwt = ""
    w._heartbeat_outgoing_queue = queue.Queue()
    return w


def _result_msg(rid: str = "req-1") -> pb.WorkerSchedulerMessage:
    return pb.WorkerSchedulerMessage(job_result=pb.JobExecutionResult(request_id=rid, success=True))


def _event_msg() -> pb.WorkerSchedulerMessage:
    return pb.WorkerSchedulerMessage(worker_event=pb.WorkerEvent(request_id="req-1", event_type="custom.x"))


def _delta_msg() -> pb.WorkerSchedulerMessage:
    return pb.WorkerSchedulerMessage(incremental_token_delta=pb.IncrementalTokenDelta(request_id="req-1"))


def _registration_msg() -> pb.WorkerSchedulerMessage:
    return pb.WorkerSchedulerMessage(worker_registration=pb.WorkerRegistration(is_heartbeat=True))


def test_result_routes_to_results_queue() -> None:
    w = _bare()
    assert w._select_outgoing_queue(_result_msg()) is w._results_outgoing_queue


def test_events_route_to_events_queue() -> None:
    w = _bare()
    assert w._select_outgoing_queue(_event_msg()) is w._events_outgoing_queue
    assert w._select_outgoing_queue(_delta_msg()) is w._events_outgoing_queue


def test_registration_stays_on_control_queue() -> None:
    w = _bare()
    assert w._select_outgoing_queue(_registration_msg()) is w._outgoing_queue


def test_fallback_to_primary_when_aux_inactive() -> None:
    w = _bare()
    w._aux_streams_active = False
    assert w._select_outgoing_queue(_result_msg()) is w._outgoing_queue
    assert w._select_outgoing_queue(_event_msg()) is w._outgoing_queue


def test_fallback_to_primary_when_split_disabled() -> None:
    w = _bare()
    w._split_streams_enabled = False
    assert w._select_outgoing_queue(_result_msg()) is w._outgoing_queue


def test_send_message_routes_into_results_queue() -> None:
    w = _bare()
    w._send_message(_result_msg("r-abc"))
    assert w._results_outgoing_queue.qsize() == 1
    assert w._outgoing_queue.qsize() == 0
    assert w._events_outgoing_queue.qsize() == 0


def test_heartbeat_uses_primary_queue_when_worker_jwt_enabled() -> None:
    w = _bare()
    w.worker_jwt = "worker.jwt"
    w._send_heartbeat_message(_registration_msg())

    assert w._outgoing_queue.qsize() == 1
    assert w._heartbeat_outgoing_queue.qsize() == 0


def test_heartbeat_uses_dedicated_queue_without_worker_jwt() -> None:
    w = _bare()
    w._send_heartbeat_message(_registration_msg())

    assert w._outgoing_queue.qsize() == 0
    assert w._heartbeat_outgoing_queue.qsize() == 1


def test_aux_handshake_registration_advertises_capability() -> None:
    w = _bare()
    msg = w._build_registration_message(is_heartbeat=True, stream_role="results")
    reg = msg.worker_registration
    assert reg.supports_split_streams is True
    assert reg.stream_role == "results"
    assert reg.is_heartbeat is True
    assert reg.resources.worker_id == "w1"


def test_buffered_result_ids_scans_both_queues() -> None:
    w = _bare()
    w._outgoing_queue.put_nowait(_event_msg())
    w._results_outgoing_queue.put_nowait(_result_msg("r-buffered"))
    ids = w._buffered_result_request_ids()
    assert ids == ["r-buffered"]
