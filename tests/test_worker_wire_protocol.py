from __future__ import annotations

from gen_worker.wire_protocol import WIRE_PROTOCOL_MAJOR, WIRE_PROTOCOL_MINOR
from gen_worker.worker import Worker


def test_registration_advertises_wire_protocol(monkeypatch) -> None:
    sent = []

    w = Worker(
        scheduler_addr="127.0.0.1:65535",
        user_module_names=[],
        worker_jwt="test-jwt",
        reconnect_delay=0,
    )
    monkeypatch.setattr(w, "_send_message", lambda message: sent.append(message))

    w._register_worker(is_heartbeat=False)

    assert sent, "expected at least one outgoing registration message"
    reg = sent[0].worker_registration
    assert reg.protocol_major == WIRE_PROTOCOL_MAJOR
    assert reg.protocol_minor == WIRE_PROTOCOL_MINOR


def test_detects_protocol_incompatibility_marker() -> None:
    assert Worker._is_protocol_incompatibility("unsupported_worker_protocol:1.0 supported=1:1-9999")
    assert not Worker._is_protocol_incompatibility("not_leader:127.0.0.1:50051")
