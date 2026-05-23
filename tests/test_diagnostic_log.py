from __future__ import annotations

import json
import queue
import threading

from gen_worker.diagnostics import diagnostic_emitter_context, emit_diagnostic_log
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.worker import DIAGNOSTIC_LOG_MAX_PAYLOAD_BYTES, OUTGOING_QUEUE_MAXSIZE, Worker


def _bare_worker() -> Worker:
    w = object.__new__(Worker)
    w._running = True
    w._stop_event = threading.Event()
    w._outgoing_queue = queue.Queue(maxsize=OUTGOING_QUEUE_MAXSIZE)
    w._results_outgoing_queue = queue.Queue(maxsize=OUTGOING_QUEUE_MAXSIZE)
    w._events_outgoing_queue = queue.Queue(maxsize=OUTGOING_QUEUE_MAXSIZE)
    w._split_streams_enabled = True
    w._aux_streams_active = True
    w.worker_id = "worker-1"
    w.release_id = "release-1"
    w.runpod_pod_id = "pod-1"
    w.image_digest = "sha256:abc"
    return w


def test_emit_diagnostic_log_sends_typed_message_without_request_context() -> None:
    w = _bare_worker()

    ok = w._emit_diagnostic_log(
        category="setup",
        message="loaded pipeline",
        payload={"endpoint_class": "ImageGen", "status": "ok"},
        function_name="generate",
    )

    assert ok is True
    assert w._events_outgoing_queue.qsize() == 1
    msg = w._events_outgoing_queue.get_nowait()
    assert msg.WhichOneof("msg") == "worker_diagnostic_log"
    diagnostic = msg.worker_diagnostic_log
    assert diagnostic.worker_id == "worker-1"
    assert diagnostic.release_id == "release-1"
    assert diagnostic.runpod_pod_id == "pod-1"
    assert diagnostic.category == "setup"
    assert diagnostic.severity == "info"
    payload = json.loads(diagnostic.payload_json)
    assert payload["endpoint_class"] == "ImageGen"
    assert payload["function_name"] == "generate"
    assert payload["image_digest"] == "sha256:abc"


def test_emit_diagnostic_log_is_safe_when_stream_disconnected() -> None:
    w = _bare_worker()
    w._running = False

    assert w._emit_diagnostic_log(category="setup", message="not connected", payload={}) is True
    assert w._outgoing_queue.qsize() == 0
    assert w._events_outgoing_queue.qsize() == 0


def test_diagnostic_payload_truncates_and_redacts_secret_keys() -> None:
    w = _bare_worker()

    ok = w._emit_diagnostic_log(
        category="setup",
        message="large payload",
        payload={
            "hf_token": "secret-token",
            "items": ["x" * 4096 for _ in range(80)],
        },
    )

    assert ok is True
    msg = w._events_outgoing_queue.get_nowait()
    payload_json = bytes(msg.worker_diagnostic_log.payload_json)
    assert len(payload_json) <= DIAGNOSTIC_LOG_MAX_PAYLOAD_BYTES
    assert b"secret-token" not in payload_json
    payload = json.loads(payload_json)
    assert payload.get("truncated") is True
    assert payload["original_size_bytes"] > DIAGNOSTIC_LOG_MAX_PAYLOAD_BYTES


def test_public_emit_diagnostic_log_uses_worker_context() -> None:
    w = _bare_worker()

    with diagnostic_emitter_context(w._emit_diagnostic_log):
        assert emit_diagnostic_log("memory_sharing", "probe", {"same_pipeline": False}) is True

    msg = w._events_outgoing_queue.get_nowait()
    assert msg.WhichOneof("msg") == "worker_diagnostic_log"
    assert msg.worker_diagnostic_log.category == "memory_sharing"


def test_serial_setup_can_emit_diagnostic_without_request_context() -> None:
    w = _bare_worker()
    w._configure_torchinductor_cache_dir = lambda: None
    w._serial_class_specs = {}
    w._micro_batch_aggregators = {}

    class Endpoint:
        def setup(self) -> None:
            assert emit_diagnostic_log("setup", "from setup", {"stage": "setup"}) is True

    rec = {
        "started": False,
        "started_lock": threading.Lock(),
        "instance": Endpoint(),
        "endpoint_spec": None,
        "cls_name": "Endpoint",
    }

    w._ensure_serial_class_started(rec)

    assert rec["started"] is True
    msg = w._events_outgoing_queue.get_nowait()
    assert msg.WhichOneof("msg") == "worker_diagnostic_log"
    assert msg.worker_diagnostic_log.category == "setup"
    payload = json.loads(msg.worker_diagnostic_log.payload_json)
    assert payload["endpoint_class"] == "Endpoint"


def test_diagnostic_log_routes_to_primary_when_split_streams_inactive() -> None:
    w = _bare_worker()
    w._aux_streams_active = False

    w._emit_diagnostic_log(category="setup", message="fallback", payload={})

    assert w._outgoing_queue.qsize() == 1
    assert w._events_outgoing_queue.qsize() == 0
    assert w._outgoing_queue.get_nowait().WhichOneof("msg") == "worker_diagnostic_log"
