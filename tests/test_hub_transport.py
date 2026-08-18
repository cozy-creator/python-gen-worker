"""Hub transport: connect, reconnect, the outbox, and typed refusals.

Sections keep their incident id; the full narratives live in the tracker.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import threading
import time
from concurrent import futures
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any, Iterator, List, Tuple

import grpc
import msgspec
import pytest
import requests
from harness import hub_env
from harness.blob_host import BlobHost
from harness.hardware_report_hub import closed_port_addr, old_hub, recording_hub
from harness.hub_double import FakeScheduler, hub_double, is_ready
from harness.toy_endpoints import EchoIn

from gen_worker import activity as activity_mod
from gen_worker import config as config_pkg
from gen_worker import cuda_probe, hardware_report, worker_credential
from gen_worker.config import Settings, load_settings
from gen_worker.cuda_probe import CudaProbeResult, classify_probe_failure
from gen_worker.executor import _map_exception
from gen_worker.hub_error import (
    HubApiError,
    parse_hub_error,
    raise_for_hub_error,
)
from gen_worker.lifecycle_intents import IntentRegistry
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.pb import worker_scheduler_pb2_grpc as pb_grpc
from gen_worker.transport import HandlerError, SendQueue, Transport

# ============================================================================
# pgw#869 — the worker outbox — losing the hub must not lose the
#   evidence.
# ============================================================================

_RUNNING = pb.ActivityState.ACTIVITY_STATE_RUNNING


_COMPLETED = pb.ActivityState.ACTIVITY_STATE_COMPLETED


def _payload_digest(error: str, detail: str) -> str:
    h = hashlib.sha256()
    h.update(error.encode())
    h.update(b"\x00")
    h.update(detail.encode())
    return h.hexdigest()


def _hub_row_key(msg: pb.WorkerMessage, worker_id: str = "w") -> Tuple:
    u = msg.activity_update
    return (
        worker_id, u.kind, u.phase, int(u.state), bool(u.self_stalled),
        _payload_digest(u.error, u.detail),
    )


def _fact(kind: str, phase: str, detail: str) -> pb.WorkerMessage:
    """A measurement: RUNNING, carrying a payload. Never coalescible."""
    return pb.WorkerMessage(activity_update=pb.ActivityUpdate(
        kind=kind, phase=phase, state=_RUNNING, detail=detail))


def _beat(kind: str, phase: str, seq: int) -> pb.WorkerMessage:
    """A liveness re-report: RUNNING, no payload. Coalescible by contract."""
    return pb.WorkerMessage(activity_update=pb.ActivityUpdate(
        kind=kind, phase=phase, state=_RUNNING, seq=seq))


def _activities(conn: Any, kind: str) -> List[pb.ActivityUpdate]:
    return [
        m.activity_update for m in list(conn.received)
        if m.WhichOneof("msg") == "activity_update"
        and m.activity_update.kind == kind
    ]


def _is_activity(kind: str, phase: str = "") -> Any:
    def pred(m: pb.WorkerMessage) -> bool:
        if m.WhichOneof("msg") != "activity_update":
            return False
        u = m.activity_update
        return u.kind == kind and (not phase or u.phase == phase)
    return pred


def _await_live_reporting(harness: Any) -> None:
    """pgw#869: Gate on the worker having bound its activity sink."""
    while activity_mod._sink is None:
        assert harness.alive, "worker exited before binding its activity sink"
        time.sleep(0.05)


def test_hub_killed_midactivity_replays_every_fact_exactly_once() -> None:
    kind = "self_mint_compile_pgw869"
    with hub_double(worker_id="outbox-acceptance") as (scheduler, harness):
        conn0 = scheduler.wait_connection(0)
        conn0.wait_for(is_ready)
        _await_live_reporting(harness)

        # A fact produced while the hub is up, and confirmed shipped.
        activity_mod.emit_event(kind, "entries=36 ceiling=8", phase="pool")
        conn0.wait_for(_is_activity(kind, "pool"))

        # The hub dies. The worker process does not — exactly the incident.
        conn0.kill()

        # The "mint" keeps producing measurements into a dead connection.
        during = [
            ("trace_graph", "26 evidence @ 0.99/s"),
            ("autotune", "per_entry_device_bytes=11881591040 basis=measured"),
            ("publish", "cell ck1-aeed10f6 published"),
        ]
        for phase, detail in during:
            activity_mod.emit_event(kind, detail, phase=phase)

        # The hub comes back and the worker redials on its own.
        conn1 = scheduler.wait_connection(1)
        for phase, _detail in during:
            conn1.wait_for(_is_activity(kind, phase))

        replayed = [u for u in _activities(conn1, kind)]

        # (a) NOTHING WAS LOST. Every fact produced during the outage arrived.
        assert {u.phase for u in replayed} >= {p for p, _d in during}
        for phase, detail in during:
            assert any(
                u.phase == phase and u.detail == detail for u in replayed
            ), f"{phase!r} arrived without its payload: {replayed}"

        # (b) IN ORDER. Replay is FIFO in production order, not reshuffled.
        order = [u.phase for u in replayed if u.phase in {p for p, _ in during}]
        first_seen = [p for i, p in enumerate(order) if p not in order[:i]]
        assert first_seen == [p for p, _d in during]

        # (c) EXACTLY ONCE, as the hub counts it: one row per fact.
        all_msgs = [
            m for c in (conn0, conn1) for m in list(c.received)
            if m.WhichOneof("msg") == "activity_update"
            and m.activity_update.kind == kind
        ]
        rows = {_hub_row_key(m) for m in all_msgs}
        assert len(rows) == 1 + len(during), (
            "each fact must collapse to one hub row; got "
            f"{sorted(r[2] for r in rows)}"
        )


def test_terminal_produced_during_the_outage_still_lands() -> None:
    """pgw#869: The mint FINISHES while the hub is away."""
    kind = "forge_terminal_pgw869"
    with hub_double(worker_id="outbox-terminal") as (scheduler, harness):
        conn0 = scheduler.wait_connection(0)
        conn0.wait_for(is_ready)
        _await_live_reporting(harness)
        conn0.kill()

        act = activity_mod.begin(kind)
        act.phase("publishing")
        act.completed()

        conn1 = scheduler.wait_connection(1)
        terminal = conn1.wait_for(
            lambda m: (
                m.WhichOneof("msg") == "activity_update"
                and m.activity_update.kind == kind
                and m.activity_update.state == _COMPLETED
            ),
        )
        assert terminal.activity_update.phase == "publishing"


def test_a_long_outage_coalesces_beats_and_keeps_every_measurement() -> None:
    """pgw#869: Getting this backwards is the failure mode the issue names."""
    kind = "outbox_longoutage_pgw869"
    with hub_double(worker_id="outbox-longoutage") as (scheduler, harness):
        conn0 = scheduler.wait_connection(0)
        conn0.wait_for(is_ready)
        _await_live_reporting(harness)
        conn0.kill()

        act = activity_mod.begin(kind)
        act.phase("compiling")
        for i in range(64):
            act.heartbeat()
            if i % 16 == 0:
                activity_mod.emit_event(
                    kind, f"measurement {i}", phase=f"m{i}")

        conn1 = scheduler.wait_connection(1)
        for i in range(0, 64, 16):
            conn1.wait_for(_is_activity(kind, f"m{i}"))

        got = _activities(conn1, kind)
        # Every measurement survived, with its payload.
        for i in range(0, 64, 16):
            assert any(
                u.phase == f"m{i}" and u.detail == f"measurement {i}"
                for u in got
            ), f"measurement {i} was dropped: {[u.phase for u in got]}"
        # 64 beats of one phase collapsed. They were always destined to be one
        # hub row (same conflict key, empty payload digest); the outbox simply
        # stops carrying 64 copies of it across the outage.
        beats = [u for u in got if u.phase == "compiling" and not u.detail]
        assert len(beats) <= 1, f"beats did not coalesce: {len(beats)}"


def test_reconnect_requeues_evidence_instead_of_clearing_it() -> None:
    """Loss site 1: `reset_for_reconnect` used to clear `_items` wholesale."""
    async def _run() -> None:
        q = SendQueue(maxsize=4)
        facts = [_fact("k", f"p{i}", f"d{i}") for i in range(3)]
        for f in facts:
            await q.put(f)
        await q.reset_for_reconnect()
        replayed = []
        while len(q):
            kind, msg = await q.get()
            assert kind == "evidence"
            replayed.append(msg)
        assert replayed == facts          # all three, in production order

    asyncio.run(_run())


def test_a_fact_the_sender_took_but_never_shipped_comes_back() -> None:
    """pgw#869: Loss site 2: the sender POPS before `write()`; a failed write lost it."""
    async def _run() -> None:
        q = SendQueue()
        msg = _fact("self_mint_compile", "trace_graph", "entries=36")
        await q.put(msg)
        kind, taken = await q.get()       # sender has it; the write then FAILS
        assert kind == "evidence"
        await q.reset_for_reconnect()
        _kind, replayed = await q.get()
        assert replayed == msg
        assert _hub_row_key(taken) == _hub_row_key(replayed)

        # A fact whose write RETURNED is re-offered once more after a dirty
        # reconnect, because a returned write is only a BUFFERED write (the
        # RESHIP_WINDOW argument). It does not accumulate: one slot, one copy.
        await q.mark_event_shipped(replayed)
        await q.reset_for_reconnect()
        assert q.pending_evidence_count == 1
        _kind, again = await q.get()
        assert _hub_row_key(again) == _hub_row_key(msg)
        await q.mark_event_shipped(again)
        assert q.pending_evidence_count == 0
        assert len(q) == 0            # nothing outstanding while the stream lives

    asyncio.run(_run())


def test_beats_coalesce_in_place_and_facts_never_do() -> None:
    async def _run() -> None:
        q = SendQueue(maxsize=2)          # deliberately smaller than the input
        await q.put(_beat("k", "compiling", 1))
        await q.put(_fact("k", "pool", "K=2 binding=vram"))
        for seq in range(2, 40):
            await q.put(_beat("k", "compiling", seq))
        await q.put(_beat("k", "publishing", 99))

        await q.reset_for_reconnect()
        out = []
        while len(q):
            _kind, msg = await q.get()
            out.append(msg.activity_update)
        # one slot per (kind, phase) beat, and the LATEST beat won
        compiling = [u for u in out if u.phase == "compiling"]
        assert len(compiling) == 1 and compiling[0].seq == 39
        assert len([u for u in out if u.phase == "publishing"]) == 1
        # the measurement kept its position ahead of the later beats
        assert [u.phase for u in out] == ["compiling", "pool", "publishing"]

    asyncio.run(_run())


def test_a_result_never_enters_the_evidence_execution_lane() -> None:
    """pgw#869: The precondition that makes `_evidence_bytes` total."""
    async def _run() -> None:
        q = SendQueue(maxsize=4)
        result = pb.WorkerMessage(job_result=pb.JobResult(
            request_id="r1", attempt=1))
        assert SendQueue._message_key(result) is None
        await q.put(result)
        await q.put(_fact("k", "pool", "K=2"))
        await q.put(_beat("k", "compiling", 1))
        # only the two facts are booked as evidence; the result rides its own
        # durable lane and is untouched by coalescing or shedding.
        assert q.pending_evidence_count == 2
        assert q.pending_result_keys == [("r1", 1)]

        kinds = []
        while len(q):
            kind, _msg = await q.get()
            kinds.append(kind)
        assert kinds.count("result") == 1
        assert kinds.count("evidence") == 2

    asyncio.run(_run())


def test_a_producer_never_blocks_on_a_dead_connection() -> None:
    """pgw#869: `maxsize` backpressure must not reach a path that just MEASURED something."""
    async def _run() -> None:
        q = SendQueue(maxsize=2)
        for i in range(200):
            await asyncio.wait_for(q.put(_fact("k", f"p{i}", f"d{i}")), 1.0)
        assert q.pending_evidence_count == 200

    asyncio.run(_run())


def _jwt(exp_unix: float, jti: str = "jti-1") -> str:
    """pgw#869: An UNSIGNED JWT."""
    def seg(obj: Any) -> str:
        raw = json.dumps(obj).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")
    return f"{seg({'alg': 'none'})}.{seg({'exp': exp_unix, 'jti': jti})}.sig"


def _transport(pod: str = "") -> Transport:
    """pgw#873: `runpod_pod_id` decides who the actuator is, so it is the axis every auth-ladder row below varie..."""
    return Transport(
        load_settings(worker_id="auth-probe", runpod_pod_id=pod), object())


def test_an_expired_credential_refused_by_the_hub_is_never_fatal() -> None:
    """pgw#869: Wall #2 through a different door."""
    worker_credential.reset()
    try:
        worker_credential.install(_jwt(time.time() - 600.0), 0.0)
        t = _transport()
        for _ in range(50):
            assert t._auth_rejection_is_fatal("invalid worker token state") is False
        # and the ladder is not merely out-run — it is UNREACHABLE: an
        # expired credential never even records a verdict.
        assert t._auth_verdicts == set()
    finally:
        worker_credential.reset()


def test_an_expired_rejection_confesses_once_per_credential() -> None:
    """pgw#869: Patient is not the same as silent — and the confession is a hub ROW, which the evidence lane now..."""
    worker_credential.reset()
    captured: List[pb.ActivityUpdate] = []
    prior = activity_mod._sink
    activity_mod._sink = captured.append
    try:
        worker_credential.install(_jwt(time.time() - 60.0, "jti-A"), 0.0)
        t = _transport()
        for _ in range(5):
            t._auth_rejection_is_fatal("invalid worker token state")
        worker_credential.install(_jwt(time.time() - 60.0, "jti-B"), 0.0)
        for _ in range(5):
            t._auth_rejection_is_fatal("invalid worker token state")
    finally:
        activity_mod._sink = prior
        worker_credential.reset()

    rows = [u for u in captured if u.kind == "worker_credential"]
    assert len(rows) == 2, f"one confession per credential, got {len(rows)}"
    assert all(u.phase == "expired_rejected_retrying" for u in rows)
    assert "jti-A" in rows[0].detail and "jti-B" in rows[1].detail


def test_a_LIVE_credential_refused_TWICE_escalates_on_a_POD_LESS_worker() -> None:
    """pgw#873 RED: the other half must not regress — revoked/superseded/ misconfigured is a verdict about this ..."""
    worker_credential.reset()
    try:
        worker_credential.install(_jwt(time.time() + 3600.0), 0.0)
        t = _transport()
        assert t._auth_rejection_is_fatal("invalid worker token state") is False
        # a DIFFERENT verdict on the same credential is new evidence, not a
        # repeat — it is not what this rule escalates on.
        assert t._auth_rejection_is_fatal("worker not expected") is False
        assert t._auth_rejection_is_fatal("invalid worker token state") is True
    finally:
        worker_credential.reset()


def test_a_POD_never_self_terminates_on_auth_and_says_why() -> None:
    """pgw#873 RED: `worker_wedge.go` is the actuator for a pod — it revokes the token and closes the pod-hour l..."""
    worker_credential.reset()
    captured: List[pb.ActivityUpdate] = []
    prior = activity_mod._sink
    activity_mod._sink = captured.append
    try:
        worker_credential.install(_jwt(time.time() + 3600.0, "jti-pod"), 0.0)
        t = _transport(pod="pod-abc123")
        for _ in range(20):
            assert t._auth_rejection_is_fatal("invalid worker token state") is False
    finally:
        activity_mod._sink = prior
        worker_credential.reset()

    rows = [u for u in captured
            if u.phase == "auth_verdict_deferred_to_hub"]
    assert len(rows) == 1, f"once per verdict, got {len(rows)}"
    assert "jti-pod" in rows[0].detail and "pod-abc123" in rows[0].detail


def test_a_rotation_retires_the_evidence() -> None:
    """pgw#869: The evidence is about a credential."""
    worker_credential.reset()
    try:
        worker_credential.install(_jwt(time.time() + 3600.0, "jti-old"), 0.0)
        t = _transport()
        assert t._auth_rejection_is_fatal("invalid worker token state") is False
        worker_credential.install(_jwt(time.time() + 3600.0, "jti-new"), 0.0)
        # would have been the fatal repeat under the old credential
        assert t._auth_rejection_is_fatal("invalid worker token state") is False
        assert t._auth_rejection_is_fatal("invalid worker token state") is True
    finally:
        worker_credential.reset()


def test_the_last_magic_number_in_the_reconnect_path_is_gone() -> None:
    """pgw#873: nothing measured said 3, and nothing said 60 s."""
    from gen_worker import transport as transport_mod

    assert not hasattr(transport_mod, "_AUTH_FAILURE_EXIT_THRESHOLD")
    assert not hasattr(transport_mod, "_AUTH_FAILURE_EXIT_WINDOW_S")
    source = Path(transport_mod.__file__).read_text()
    assert "_consecutive_auth_failures" not in source


def test_shedding_is_bounded_and_reports_itself() -> None:
    """Growth is bounded, coalescible sheds first, and a shed emits a fact."""
    async def _run() -> None:
        q = SendQueue(maxsize=4, evidence_max=8)
        for seq in range(6):
            await q.put(_beat("k", f"phase{seq}", seq))
        for i in range(20):
            await q.put(_fact("k", f"m{i}", f"measurement {i}"))

        assert q.pending_evidence_count <= 8 + 1   # +1 = the shed report itself
        assert q.shed_total >= 18

        await q.reset_for_reconnect()
        out = []
        while len(q):
            _kind, msg = await q.get()
            out.append(msg.activity_update)
        shed = [u for u in out if u.kind == "outbox_shed"]
        assert len(shed) == 1, "a shed must never be silent"
        assert str(q.shed_total) in shed[0].detail
        # coalescible beats went first: no beat survived while facts did
        assert not [u for u in out if u.phase.startswith("phase")]
        assert [u for u in out if u.phase.startswith("m")]

    asyncio.run(_run())


# ============================================================================
# pgw#1229 — the hub's typed refusal survives all the way onto the
#   wire.
# ============================================================================

HUB_CODE = "forbidden"


HUB_MESSAGE = "worker capabilities must use the exact input-asset resolver"


HUB_REQUEST_ID = "req_01J8ZQ"


class _Hub(BaseHTTPRequestHandler):
    """The routes this rig serves, each answering exactly like tensorhub."""

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/api/v1/media/urls":
            body = json.dumps({
                "error": {
                    "code": HUB_CODE,
                    "message": HUB_MESSAGE,
                    "request_id": HUB_REQUEST_ID,
                }
            }).encode()
            self._send(403, body, "application/json")
        elif self.path == "/no-body":
            self._send(403, b"", "application/json")
        elif self.path == "/proxy-outage":
            # ngrok with no healthy backend: HTML, not our envelope.
            self._send(503, b"<!DOCTYPE html><html>tunnel offline</html>",
                       "text/html")
        elif self.path == "/ok":
            self._send(200, b'{"ok":true}', "application/json")
        else:
            self._send(404, b'{"error":{"code":"not_found"}}', "application/json")

    def _send(self, status: int, body: bytes, ctype: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:  # noqa: D102
        pass


@pytest.fixture(scope="module")
def hub() -> Iterator[str]:
    srv = HTTPServer(("127.0.0.1", 0), _Hub)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_port}"
    finally:
        srv.shutdown()
        srv.server_close()


def test_raise_for_status_is_the_defect(hub: str) -> None:
    """The control: what the shipped call site produced, verbatim."""
    resp = requests.post(f"{hub}/api/v1/media/urls", json={"refs": ["x"]}, timeout=10)
    with pytest.raises(requests.HTTPError) as caught:
        resp.raise_for_status()
    _status, safe = _map_exception(caught.value)
    assert HUB_CODE not in safe
    assert HUB_MESSAGE not in safe
    assert "403 Client Error" in safe


def test_typed_body_reaches_the_wire(hub: str) -> None:
    """The fix: code AND message in the exception, and in `safe_message`."""
    resp = requests.post(f"{hub}/api/v1/media/urls", json={"refs": ["x"]}, timeout=10)
    with pytest.raises(HubApiError) as caught:
        raise_for_hub_error(resp, what="presign input assets")
    exc = caught.value

    assert exc.status_code == 403
    assert exc.code == HUB_CODE
    assert exc.message == HUB_MESSAGE
    assert exc.request_id == HUB_REQUEST_ID

    status, safe = _map_exception(exc)
    # A hub-authored 403 is a refusal, not a blip: never retried.
    assert status == pb.JOB_STATUS_FATAL
    assert HUB_CODE in safe
    assert HUB_MESSAGE in safe
    assert "presign input assets" in safe
    # One line — `_map_exception`'s generic arm keeps only splitlines()[0], and
    # a remedy split across lines is a remedy half-delivered.
    assert "\n" not in safe


def test_absent_body_degrades_to_the_status_line(hub: str) -> None:
    """pgw#1229: The failure mode of an error path is silence, never a second error."""
    resp = requests.post(f"{hub}/no-body", timeout=10)
    with pytest.raises(HubApiError) as caught:
        raise_for_hub_error(resp, what="presign input assets")
    status, safe = _map_exception(caught.value)
    assert status == pb.JOB_STATUS_RETRYABLE
    assert "403" in safe
    assert "presign input assets" in safe


def test_proxy_outage_is_retryable(hub: str) -> None:
    """An answer that is not the hub's envelope came from in front of it."""
    resp = requests.post(f"{hub}/proxy-outage", timeout=10)
    with pytest.raises(HubApiError) as caught:
        raise_for_hub_error(resp, what="presign input assets")
    assert caught.value.retryable is True
    status, _safe = _map_exception(caught.value)
    assert status == pb.JOB_STATUS_RETRYABLE


def test_success_passes_through(hub: str) -> None:
    resp = requests.post(f"{hub}/ok", timeout=10)
    assert raise_for_hub_error(resp) is resp


@pytest.mark.parametrize(
    "body,code,message",
    [
        ('{"error":{"code":"not_found","message":"no such repo"}}',
         "not_found", "no such repo"),
        # pgw#987 publish/gin shape: the code is a bare string beside `message`.
        ('{"error":"publish_repudiated","message":"audit findings"}',
         "publish_repudiated", "audit findings"),
        # A bare token with no prose is still the code...
        ('{"error":"insufficient_scope"}', "insufficient_scope", ""),
        # ...but prose is not a code.
        ('{"error":"the body could not be parsed"}',
         "", "the body could not be parsed"),
        ("<!DOCTYPE html>", "", "<!DOCTYPE html>"),
        ("", "", ""),
    ],
)
def test_envelope_shapes(body: str, code: str, message: str) -> None:
    err = parse_hub_error(body)
    assert err.code == code
    assert err.message == message


# ============================================================================
# gw#640 — a message-handler exception must never wear a dropped
#   socket's clothes.
# ============================================================================

class _Handlers:
    """Handlers whose on_message raises the way the live worker's did."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc
        self.disconnects = 0

    def build_hello(self):  # pragma: no cover - not reached in these tests
        return pb.Hello()

    async def on_hello_ack(self, ack):
        return None

    async def on_message(self, msg):
        raise self.exc

    async def on_disconnect(self):
        self.disconnects += 1


def _transport_gw640(handlers: Any, settings: Any) -> Transport:
    return Transport(settings, handlers)


def _settings():
    from gen_worker import config as gw_config

    return gw_config.current()


def test_handler_exception_is_its_own_class_not_a_transport_error():
    """The wrapper carries WHICH message and the original cause."""
    err = HandlerError("run_job", ValueError("payload decode blew up"))
    assert err.kind == "run_job"
    assert isinstance(err.cause, ValueError)
    assert "run_job" in str(err)
    assert "ValueError" in str(err)
    assert "payload decode blew up" in str(err)


def test_recv_loop_wraps_handler_raise_and_names_the_message(monkeypatch):
    """A RunJob handler raise surfaces as HandlerError(kind='run_job')."""
    boom = RuntimeError("msgpack decode failed")
    handlers = _Handlers(boom)
    t = _transport_gw640(handlers, _settings())

    run_job = pb.SchedulerMessage(run_job=pb.RunJob(request_id="r1", attempt=1))

    class _Stream:
        def __init__(self):
            self.reads = [run_job]

        async def read(self):
            return self.reads.pop(0)

    with pytest.raises(HandlerError) as caught:
        asyncio.run(t._recv_loop(_Stream()))
    assert caught.value.kind == "run_job"
    assert caught.value.cause is boom


def test_handler_failure_is_reported_through_the_worker_fatal_carrier(monkeypatch):
    """It dials the hub (bounded), naming the phase after the message kind."""
    t = _transport_gw640(_Handlers(RuntimeError("x")), _settings())
    seen = {}

    def _fake_report(settings, phase, exc, *, exit_code):
        seen["phase"] = phase
        seen["exc"] = exc
        seen["exit_code"] = exit_code
        return True

    import gen_worker.worker_fatal as wf

    monkeypatch.setattr(wf, "report_worker_fatal", _fake_report)

    cause = RuntimeError("msgpack decode failed")
    t._report_handler_failure(HandlerError("run_job", cause))
    assert seen["phase"] == "message_handler:run_job"
    assert seen["exc"] is cause

    # deduped: the reconnect loop must not re-dial the same fault every cycle
    seen.clear()
    t._report_handler_failure(HandlerError("run_job", RuntimeError("msgpack decode failed")))
    assert seen == {}

    # a DIFFERENT fault still reports
    t._report_handler_failure(HandlerError("run_job", ValueError("other")))
    assert seen["phase"] == "message_handler:run_job"


def test_transport_failures_are_still_plain_transport_failures():
    """EOF from the scheduler stays a ConnectionError, not a HandlerError."""
    import grpc

    t = _transport_gw640(_Handlers(RuntimeError("unused")), _settings())

    class _EofStream:
        async def read(self):
            return grpc.aio.EOF

    with pytest.raises(ConnectionError):
        asyncio.run(t._recv_loop(_EofStream()))


def test_waiting_intent_always_carries_blocker_retry_or_deadline():
    """gw#640: The hub's shadow validator requires one of the three on a WAITING state."""
    from gen_worker.lifecycle_intents import IntentRegistry
    from gen_worker.pb import worker_scheduler_pb2 as p

    reg = IntentRegistry("release-1", ["artifact-stat"])
    intent_id = reg.ensure_local_intent("materialize", "tensorhub/tiny@prod")
    assert intent_id.startswith("compat-materialize-"), intent_id

    reg.transition(
        intent_id,
        p.LIFECYCLE_INTENT_STATUS_WAITING,
        p.LIFECYCLE_INTENT_STAGE_FETCHING,
        reason=p.LIFECYCLE_WAIT_REASON_NETWORK_RETRY,
    )
    state = next(s for s in reg.snapshot().intents if s.intent_id == intent_id)
    blocked = bool(state.blocker_intent_id) or bool(state.blocker_request.request_id)
    assert blocked or state.next_retry_at_unix_ms > 0 or state.deadline_at_unix_ms > 0, (
        "a WAITING intent with no blocker and no retry must get a fallback "
        "deadline, or the hub rejects the whole snapshot"
    )


def test_explicit_waiting_fields_are_not_overwritten():
    """The fallback must only fill a genuine gap."""
    from gen_worker.lifecycle_intents import IntentRegistry
    from gen_worker.pb import worker_scheduler_pb2 as p

    reg = IntentRegistry("release-1", ["artifact-stat"])
    intent_id = reg.ensure_local_intent("materialize", "tensorhub/tiny@prod")
    reg.transition(
        intent_id,
        p.LIFECYCLE_INTENT_STATUS_WAITING,
        p.LIFECYCLE_INTENT_STAGE_FETCHING,
        reason=p.LIFECYCLE_WAIT_REASON_NETWORK_RETRY,
        next_retry_at_unix_ms=1234,
    )
    state = next(s for s in reg.snapshot().intents if s.intent_id == intent_id)
    assert state.next_retry_at_unix_ms == 1234
    assert state.deadline_at_unix_ms == 0, "an explicit retry time needs no deadline"


# ============================================================================
# pgw#654/#658 — id
#   race): the 0.58.0/0.60.0 fleet-wide false "model load failure".
# ============================================================================

_REF_A = "harness/slow-pipeline"


_REF_B = "harness/second-ref"


_FN = "slow-slot-echo"


_RELEASE = "rel-pgw654"


def _shadow_command(
    session_id: str, desired: pb.DesiredResidency,
) -> pb.DesiredStateCommand:
    """pgw#654: Mirror of tensorhub buildShadowDesiredStateCommand (a838b34a): MATERIALIZE per disk ref + FUNCTI..."""
    now_ms = int(time.time() * 1000)
    cmd = pb.DesiredStateCommand(
        worker_session_id=session_id,
        command_seq=desired.generation,
        goal_id=f"goal-gen-{desired.generation}",
        release_id=_RELEASE,
        config_generation=desired.config_generation,
        issued_at_unix_ms=now_ms,
        accept_by_unix_ms=now_ms + 2_000,
        first_action_by_unix_ms=now_ms + 60_000,
        mandatory=True,
    )
    for i, ref in enumerate(desired.disk_refs):
        digest = desired.snapshots[ref].digest if ref in desired.snapshots else ""
        ident = hashlib.sha256(f"materialize|{ref}|{digest}".encode()).hexdigest()[:16]
        cmd.intents.append(pb.DesiredIntent(
            intent_id=f"intent-mat-{ident}",
            kind=pb.DESIRED_INTENT_KIND_MATERIALIZE,
            cause=pb.DESIRED_INTENT_CAUSE_PREPOSITION,
            ref=ref,
            snapshot_digest=digest.encode(),
            desired_tier=pb.RESIDENCY_TIER_DISK,
            priority=len(desired.disk_refs) - i,
            mandatory=True,
        ))
    for i, instance in enumerate(desired.hot):
        raw = instance.SerializeToString(deterministic=True)
        bd = hashlib.sha256(raw)
        cmd.intents.append(pb.DesiredIntent(
            intent_id=f"intent-fn-{bd.hexdigest()[:16]}",
            kind=pb.DESIRED_INTENT_KIND_FUNCTION_READY,
            cause=pb.DESIRED_INTENT_CAUSE_COLD_BOOT,
            function_name=instance.function_name,
            desired_tier=pb.RESIDENCY_TIER_VRAM,
            binding_digest=bd.digest(),
            priority=len(desired.hot) - i,
            mandatory=True,
        ))
    return cmd


def test_first_job_on_converged_v5_worker_does_not_error(tmp_path: Path) -> None:
    """The live killer: converged command + in-flight >2s job + re-pass."""
    blobs = BlobHost(tmp_path)
    try:
        snap_a = blobs.one_file_snapshot(
            "snap-a", "blob-a", b"weights-a",
            path_in_snapshot="transformer/weights.txt",
        )
        snap_b = blobs.one_file_snapshot(
            "snap-b", "blob-b", b"weights-b",
            path_in_snapshot="transformer/weights.txt",
        )
        with hub_double(
            modules=("harness.toy_endpoints", "harness.slow_endpoints_pgw654"),
        ) as (scheduler, _harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            session_id = conn.hello.worker_session_id

            hot = [pb.DesiredInstance(
                function_name=_FN,
                models=[pb.ModelBinding(slot="pipeline", ref=_REF_A)],
            )]
            desired1 = pb.DesiredResidency(
                generation=1, release_id=_RELEASE,
                disk_refs=[_REF_A], hot=hot,
                snapshots={_REF_A: snap_a},
            )
            conn.send(hello_ack=pb.HelloAck(
                protocol_version=pb.PROTOCOL_VERSION_CURRENT,
                desired_residency=desired1,
                desired_state_command=_shadow_command(session_id, desired1),
            ))
            conn.wait_for(
                lambda m: m.WhichOneof("msg") == "state_delta"
                and _FN in m.state_delta.available_functions,
            )

            # FULL v5 convergence: the command's materialize intent must be
            # terminal SUCCEEDED — the live pods' state at first dispatch.
            def _mat_succeeded(m: pb.WorkerMessage) -> bool:
                if m.WhichOneof("msg") != "lifecycle_snapshot":
                    return False
                return any(
                    i.intent_id.startswith("intent-mat-") and i.status == 4
                    for i in m.lifecycle_snapshot.intents
                )

            conn.wait_for(_mat_succeeded)

            # First tenant job, runs well past the 2.0s grace.
            conn.send(run_job=pb.RunJob(
                request_id="r-long", attempt=1, function_name="slow-stream",
                input_payload=msgspec.msgpack.encode(EchoIn(text="x")),
            ))
            conn.wait_for(
                lambda m: m.WhichOneof("msg") == "job_accepted"
                and m.job_accepted.request_id == "r-long"
            )

            # Generation bump mid-job (live: planner adds the sibling ref).
            desired2 = pb.DesiredResidency(
                generation=2, release_id=_RELEASE,
                disk_refs=[_REF_A, _REF_B], hot=hot,
                snapshots={_REF_A: snap_a, _REF_B: snap_b},
            )
            conn.send(hello_ack=pb.HelloAck(
                protocol_version=pb.PROTOCOL_VERSION_CURRENT,
                desired_residency=desired2,
                desired_state_command=_shadow_command(session_id, desired2),
            ))

            # The job must complete and the worker must NEVER report ERROR.
            result = conn.wait_for(
                lambda m: m.WhichOneof("msg") == "job_result"
                and m.job_result.request_id == "r-long",
            ).job_result
            assert result.status == pb.JOB_STATUS_OK

            # And the bumped desired state still converges after the job. A
            # run_job cancels the reconcile mid-pass; pgw#845 was the cancel
            # landing on a granted ref lock, which leaked it and stopped this
            # worker converging anything, ever.
            conn.wait_for(
                lambda m: m.WhichOneof("msg") == "model_event"
                and m.model_event.ref == _REF_B
                and m.model_event.state == pb.MODEL_STATE_ON_DISK,
            )

            # Checked LAST, so it sees the whole run: an ERROR raised after the
            # job result used to be invisible to this assertion.
            errors = conn.count(
                lambda m: m.WhichOneof("msg") == "state_delta"
                and m.state_delta.phase == pb.WORKER_PHASE_ERROR
            )
            assert errors == 0, (
                "worker reported WORKER_PHASE_ERROR — the pgw#654 false "
                "model_load_failure regression is back"
            )
    finally:
        blobs.shutdown()


def test_ensure_intent_mints_carrier_for_terminal_command_work() -> None:
    """ensure_intent never returns "" for re-verified command work."""
    reg = IntentRegistry("rel-x", ["fn-a"])
    cmd = pb.DesiredStateCommand(
        worker_session_id=reg.worker_session_id, command_seq=1,
        goal_id="goal-1", release_id="rel-x", mandatory=True,
        issued_at_unix_ms=1, accept_by_unix_ms=2, first_action_by_unix_ms=3,
        intents=[pb.DesiredIntent(
            intent_id="mat-a", kind=pb.DESIRED_INTENT_KIND_MATERIALIZE,
            cause=pb.DESIRED_INTENT_CAUSE_PREPOSITION, ref="refA",
            snapshot_digest=b"d", desired_tier=pb.RESIDENCY_TIER_DISK,
            mandatory=True,
        )],
    )
    receipt = reg.apply_command(cmd)
    assert receipt.status == pb.GOAL_RECEIPT_STATUS_ACCEPTED

    # Command-owned while active.
    assert reg.ensure_intent(
        pb.DESIRED_INTENT_KIND_MATERIALIZE, ref="refA") == "mat-a"

    reg.transition("mat-a", pb.LIFECYCLE_INTENT_STATUS_RUNNING,
                   pb.LIFECYCLE_INTENT_STAGE_FETCHING)
    reg.transition("mat-a", pb.LIFECYCLE_INTENT_STATUS_SUCCEEDED,
                   pb.LIFECYCLE_INTENT_STAGE_ON_DISK)

    # Terminal command work: a re-pass gets a live compat carrier, not "".
    carrier = reg.ensure_intent(pb.DESIRED_INTENT_KIND_MATERIALIZE, ref="refA")
    assert carrier and carrier != "mat-a"
    assert reg.is_active(carrier)
    # The carrier is transitionable (reported_await can report on it).
    reg.transition(carrier, pb.LIFECYCLE_INTENT_STATUS_WAITING,
                   pb.LIFECYCLE_INTENT_STAGE_WAIT_TENANT_IDLE,
                   reason=pb.LIFECYCLE_WAIT_REASON_TENANT_WORK)


def test_generation_bump_does_not_supersede_worker_local_intents() -> None:
    """pgw#654: A command replaces command-owned work only; a live job/setup carrier must survive the bump."""
    reg = IntentRegistry("rel-x", ["fn-a"])

    def cmd(seq: int, refs: list) -> pb.DesiredStateCommand:
        c = pb.DesiredStateCommand(
            worker_session_id=reg.worker_session_id, command_seq=seq,
            goal_id=f"goal-{seq}", release_id="rel-x", mandatory=True,
            issued_at_unix_ms=1, accept_by_unix_ms=2, first_action_by_unix_ms=3,
        )
        for ref in refs:
            c.intents.append(pb.DesiredIntent(
                intent_id=f"mat-{ref}", kind=pb.DESIRED_INTENT_KIND_MATERIALIZE,
                cause=pb.DESIRED_INTENT_CAUSE_PREPOSITION, ref=ref,
                snapshot_digest=b"d", desired_tier=pb.RESIDENCY_TIER_DISK,
                mandatory=True,
            ))
        return c

    assert reg.apply_command(cmd(1, ["refA"])).status == pb.GOAL_RECEIPT_STATUS_ACCEPTED
    job_intent = reg.ensure_local_intent("job", "r1\x001", function_name="fn-a")
    reg.transition(job_intent, pb.LIFECYCLE_INTENT_STATUS_RUNNING,
                   pb.LIFECYCLE_INTENT_STAGE_READY, detail="executing")

    assert reg.apply_command(cmd(2, ["refB"])).status == pb.GOAL_RECEIPT_STATUS_ACCEPTED

    # Worker-local job carrier survived; the dropped command intent did not.
    assert reg.is_active(job_intent), "generation bump killed a live job intent"
    assert not reg.is_active("mat-refA"), "stale command intent not superseded"


def test_enter_error_phase_dials_cause(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every WORKER_PHASE_ERROR ships its cause on the gw#640 carrier."""
    from gen_worker import worker_fatal
    from gen_worker.lifecycle import Lifecycle

    dialed: list = []

    async def _fake_report(settings, detail):
        dialed.append(detail)
        return True

    monkeypatch.setattr(worker_fatal, "report_worker_error_async", _fake_report)

    lc = object.__new__(Lifecycle)  # stubbed Lifecycle (repo convention)
    lc.phase = pb.WORKER_PHASE_READY
    lc._settings = object()  # type: ignore[assignment]

    async def _run() -> None:
        lc._enter_error_phase("residency reconcile",
                              RuntimeError("unreported protocol await: x"))
        await asyncio.sleep(0)  # let the dial task run

    asyncio.run(_run())
    assert lc.phase == pb.WORKER_PHASE_ERROR
    assert len(dialed) == 1
    assert "unreported protocol await: x" in dialed[0]
    assert "residency reconcile" in dialed[0]


# ============================================================================
# test_hardware_report — gw#619/th#988: the worker's boot-time CUDA probe
#   failure must dial the hub with a typed HardwareUnsuitable report BEFORE
#   the pre-existing silent exit (cuda_probe.py, ...
# ============================================================================

@pytest.mark.parametrize(
    "reason, expected",
    [
        ("", "unknown"),
        ("torch unavailable: no module named torch", "torch_unavailable"),
        (cuda_probe.NO_DEVICE_REASON, "cuda_unavailable"),
        # th#591/th#979's exact real-world signature, reproduced verbatim.
        ("RuntimeError: CUDA initialization: driver too old (found version 12080)", "driver_too_old"),
        ("RuntimeError: CUDA-capable device(s) is/are busy or unavailable", "cuda_error"),
    ],
)
def test_classify_probe_failure_vocabulary(reason: str, expected: str) -> None:
    assert classify_probe_failure(reason) == expected


def _settings_hardware_report(**overrides: object) -> Settings:
    base = dict(
        orchestrator_public_addr="127.0.0.1:1",
        worker_id="worker-1",
        bootstrap_worker_jwt="",
        worker_image_digest="sha256:deadbeef",
        runpod_pod_id="pod-1",
    )
    base.update(overrides)  # type: ignore[arg-type]
    return Settings(**base)  # type: ignore[arg-type]


def test_build_hardware_report_degrades_safely_without_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*_a: object, **_kw: object) -> None:
        raise FileNotFoundError("no nvidia-smi on this box")

    monkeypatch.setattr(hardware_report, "_nvidia_smi_driver_and_gpu", lambda: ("", ""))
    probe = CudaProbeResult(ok=False, reason=cuda_probe.NO_DEVICE_REASON)
    report = hardware_report.build_hardware_report(probe, _settings_hardware_report())
    assert report.reason_class == "cuda_unavailable"
    assert report.detail == probe.reason
    assert report.image_digest == "sha256:deadbeef"
    assert report.instance_id == "pod-1"
    # torch IS importable in this env (it's a gen-worker dependency); driver
    # detection degrading to "" must not take torch_version down with it.
    assert report.torch_version


def test_build_hardware_report_uses_nvidia_smi_when_torch_cuda_is_down(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        hardware_report, "_nvidia_smi_driver_and_gpu", lambda: ("570.211.01", "NVIDIA GeForce RTX 4070")
    )
    probe = CudaProbeResult(
        ok=False, reason="RuntimeError: CUDA initialization: driver too old (found version 12080)"
    )
    report = hardware_report.build_hardware_report(probe, _settings_hardware_report())
    assert report.reason_class == "driver_too_old"
    assert report.driver_version == "570.211.01"
    assert report.gpu_name == "NVIDIA GeForce RTX 4070"


def test_report_hardware_unsuitable_delivers_to_a_new_hub() -> None:
    with recording_hub() as (servicer, addr):
        settings = _settings_hardware_report(orchestrator_public_addr=addr, bootstrap_worker_jwt="")
        probe = CudaProbeResult(ok=False, reason=cuda_probe.NO_DEVICE_REASON)
        delivered = hardware_report.report_hardware_unsuitable(settings, probe)
        assert delivered is True

        msg = servicer.wait_for_message(timeout=5.0)
        assert msg.WhichOneof("msg") == "hardware_unsuitable"
        hw = msg.hardware_unsuitable
        assert hw.worker_id == "worker-1"
        assert hw.reason_class == "cuda_unavailable"
        assert hw.detail == probe.reason
        assert hw.image_digest == "sha256:deadbeef"
        assert hw.instance_id == "pod-1"
        assert hw.reported_at_unix_ms > 0


def test_general_hub_double_preserves_a_diagnostic_first_stream() -> None:
    """The ordinary hub double mirrors both valid Connect first-message shapes."""
    scheduler = FakeScheduler()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    pb_grpc.add_WorkerSchedulerServicer_to_server(scheduler, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        settings = _settings_hardware_report(orchestrator_public_addr=f"127.0.0.1:{port}")
        probe = CudaProbeResult(ok=False, reason="the root failure survives")
        assert hardware_report.report_hardware_unsuitable(settings, probe) is True
        assert len(scheduler.diagnostic_reports) == 1
        report = scheduler.diagnostic_reports[0]
        assert report.reason_class == "cuda_error"
        assert report.detail == probe.reason
    finally:
        server.stop(grace=0)


def test_report_hardware_unsuitable_delivers_with_worker_jwt_identity() -> None:
    """worker_id/release_id fall back to the JWT claims when Settings.worker_id is unset — exactly Lifecycle's o..."""
    import base64
    import json

    from gen_worker import worker_credential

    payload = base64.urlsafe_b64encode(
        json.dumps({"sub": "jwt-worker", "release_id": "release-77"}).encode()
    ).rstrip(b"=")
    fake_jwt = b"header." + payload + b".sig"

    worker_credential.reset()
    try:
        worker_credential.install(fake_jwt.decode())
        with recording_hub() as (servicer, addr):
            settings = _settings_hardware_report(orchestrator_public_addr=addr, worker_id="")
            probe = CudaProbeResult(ok=False, reason="")
            assert hardware_report.report_hardware_unsuitable(settings, probe) is True
            hw = servicer.wait_for_message(timeout=5.0).hardware_unsuitable
            assert hw.worker_id == "jwt-worker"
            assert hw.release_id == "release-77"
    finally:
        worker_credential.reset()


def test_report_hardware_unsuitable_old_hub_rejects_gracefully() -> None:
    """A pre-gw#619 hub rejects the missing-Hello stream with FAILED_PRECONDITION — the worker must treat that a..."""
    with old_hub() as addr:
        settings = _settings_hardware_report(orchestrator_public_addr=addr)
        probe = CudaProbeResult(ok=False, reason="cuda oops")
        start = time.monotonic()
        delivered = hardware_report.report_hardware_unsuitable(settings, probe)
        elapsed = time.monotonic() - start
        assert delivered is False
        assert elapsed < 10.0


def test_report_hardware_unsuitable_unreachable_hub_is_bounded() -> None:
    """Connection-refused (nothing listening) must fall through to not-delivered within the bounded retry budget..."""
    settings = _settings_hardware_report(orchestrator_public_addr=closed_port_addr())
    probe = CudaProbeResult(ok=False, reason="cuda oops")
    start = time.monotonic()
    delivered = hardware_report.report_hardware_unsuitable(settings, probe)
    elapsed = time.monotonic() - start
    assert delivered is False
    assert elapsed < 10.0


def test_report_hardware_unsuitable_no_orchestrator_addr_is_a_noop() -> None:
    settings = _settings_hardware_report(orchestrator_public_addr="")
    probe = CudaProbeResult(ok=False, reason="cuda oops")
    assert hardware_report.report_hardware_unsuitable(settings, probe) is False


# ============================================================================
# pgw#995 — Hub-shaped env delivery reaches `Settings`.
# ============================================================================

def _boot(monkeypatch: pytest.MonkeyPatch, env: dict) -> None:
    """Replace the process environment with a pod's, exactly."""
    for name in list(os_environ_names()):
        monkeypatch.delenv(name, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)


def os_environ_names() -> list:
    import os

    return [n for n in os.environ if n.startswith(
        ("GEN_WORKER_", "TENSORHUB_", "WORKER_", "COZY_", "HF_"))]


def test_a_hub_delivered_env_value_reaches_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#995: The whole chain, end to end, with nothing hand-placed in the middle."""
    declarations = hub_env.declared_by(["HF_TOKEN"])
    entries = hub_env.EndpointEnvEntries(
        {"HF_TOKEN": "hf_operator_set_token"})

    delivery = hub_env.resolve(declarations, entries)
    assert delivery.env == {"HF_TOKEN": "hf_operator_set_token"}
    assert delivery.withheld == ()

    # The pod boots with image env + delivered entries, and NOTHING the test
    # process happened to be carrying.
    _boot(monkeypatch, hub_env.pod_environ({}, delivery))

    settings = load_settings()
    assert settings.hf_token == "hf_operator_set_token", (
        "a declared, delivered env did not reach Settings — the delivery chain "
        "is broken between the hub's resolve and config.loader")


def test_a_rebuild_that_stops_declaring_a_name_withholds_it_and_says_so(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#995: `GEN_WORKER_PREFER_AOT`, reproduced in milliseconds instead of three pods."""
    entries = hub_env.EndpointEnvEntries(
        {"HF_TOKEN": "hf_operator_set_token"})

    before = hub_env.resolve(
        hub_env.declared_by(["HF_TOKEN"]), entries)
    _boot(monkeypatch, hub_env.pod_environ({}, before))
    assert load_settings().hf_token == "hf_operator_set_token"

    # The rebuild. Nobody edited the entry; the worker function's env list
    # changed, so the release declares nothing.
    after = hub_env.resolve(hub_env.declared_by([]), entries)

    assert after.env == {}, "an undeclared name must not be injected"
    assert after.withheld_names() == ["HF_TOKEN"], (
        "the rebuild dropped a configured entry and reported NOTHING — this is "
        "precisely the silence that cost three pod attempts (th#1650)")
    assert after.withheld[0].reason == hub_env.WITHHELD_UNDECLARED
    assert "0 env name(s)" in after.withheld[0].detail, (
        "a withholding must distinguish 'this release declares nothing at all' "
        "(the rebuild case) from 'this one name was removed' (the intended one)")

    _boot(monkeypatch, hub_env.pod_environ({}, after))
    assert load_settings().hf_token == "", (
        "the withheld value still reached Settings — something other than the "
        "hub is supplying it, which is the substitution this harness exists to "
        "forbid")


def test_an_ambient_export_cannot_stand_in_for_a_hub_delivered_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pgw#995: A rig that lets the developer's own shell satisfy the assertion is decoration."""
    monkeypatch.setenv("HF_TOKEN", "ambient_shell_token")
    ambient = {"HF_TOKEN": "ambient_shell_token"}

    delivery = hub_env.resolve(
        hub_env.declared_by([]),
        hub_env.EndpointEnvEntries({"HF_TOKEN": "hf_operator_set_token"}))

    env = hub_env.pod_environ(ambient, delivery, strip=["HF_TOKEN"])
    _boot(monkeypatch, env)
    assert load_settings().hf_token == ""


def test_a_release_cannot_declare_its_way_into_the_platform_namespace(
) -> None:
    """pgw#763 delta 0: the process-split switch is platform-only."""
    delivery = hub_env.resolve(
        hub_env.declared_by(["GEN_WORKER_COMPUTE_CHILD"]),
        hub_env.EndpointEnvEntries({"GEN_WORKER_COMPUTE_CHILD": "1"}))
    assert delivery.env == {}
    assert delivery.withheld[0].reason == hub_env.WITHHELD_RESERVED


def test_the_loader_is_the_only_component_this_harness_talks_to() -> None:
    """pgw#995: Guard against the harness growing into a second config implementation."""
    src = (hub_env.__file__ or "")
    assert src
    text = open(src).read()
    for forbidden in ("load_settings", "Settings(", "msgspec"):
        assert forbidden not in text, (
            f"hub_env references {forbidden!r}: it must produce an ENVIRONMENT "
            f"and let the real loader turn it into config")
    assert config_pkg.load_settings is load_settings


def _rig():
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import micro_mint_rig  # noqa: PLC0415 - the script under test

    return micro_mint_rig


def test_rig_hub_env_mode_delivers_declared_and_strips_ambient() -> None:
    """pgw#995: `--hub-env` boots the mint child the way a pod is booted."""
    rig = _rig()
    env, withheld = rig.hub_delivered_env(
        {"PATH": "/usr/bin", "HF_TOKEN": "ambient-shell-value"},
        {"HF_TOKEN": "delivered-by-hub"})

    assert env["HF_TOKEN"] == "delivered-by-hub"
    assert env["PATH"] == "/usr/bin", "image env must survive"
    assert withheld == []


def test_rig_hub_env_mode_reports_an_undeclared_entry_instead_of_dropping_it(
) -> None:
    """pgw#995: The rig's whole reason to exist is turning a pod-only failure into a local one."""
    rig = _rig()
    env, withheld = rig.hub_delivered_env(
        {"PATH": "/usr/bin"}, {"COZY_SOMETHING_UNDECLARED": "x"})

    assert "COZY_SOMETHING_UNDECLARED" not in env
    assert len(withheld) == 1
    assert withheld[0]["name"] == "COZY_SOMETHING_UNDECLARED"
    assert withheld[0]["reason"] == hub_env.WITHHELD_UNDECLARED
    assert withheld[0]["detail"], "a withholding with no detail is a shrug"


def test_rig_strips_every_name_it_claims_to_deliver() -> None:
    """pgw#995: A name the rig DECLARES but does not STRIP is a hole: ambient value in, hub value never exercise..."""
    rig = _rig()
    assert set(rig.RIG_DECLARED_ENV) <= set(rig.RIG_STRIPPED_ENV), (
        "every declared name must also be stripped, or the ambient environment "
        "can satisfy the assertion the mode exists to make")
