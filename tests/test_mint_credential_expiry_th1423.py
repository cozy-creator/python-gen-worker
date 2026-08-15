"""A mint that outlives its credential must SAY SO.

A 28-minute inductor compile finishes past a 30m worker JWT TTL and publish
fails `(401): invalid worker token`. Two worker-side defects make that
unreadable:

* `CellPublisher._post` raised a BARE `RuntimeError`, so `_publish_failure_phase`
  had no `status`/`code` to group by and all three landed under the phase
  `RuntimeError` — indistinguishable from any other exception type on the wire;
* `aot_mint._publisher_from_settings` built the publisher with
  `worker_jwt=lambda: token`, a token CAPTURED at construction, so a rotation
  arriving during the mint could never be picked up.

These drive the REAL publisher against a localhost server speaking tensorhub's
real refusal envelope — real sockets, real threads, a real packed cell — because
what is under test is what the worker does with a live 401, not a mock's idea of
one. Every wait here is a `Thread.join()` on the publish thread itself: no
sleeps, no durations.

Run: pytest tests/test_mint_credential_expiry_th1423.py -q
"""

from __future__ import annotations

import base64
import http.server
import json
import threading
import time
import urllib.parse
from pathlib import Path

import pytest

from gen_worker import fleet_cells as fc
from gen_worker.hubio.client import HubPublishError
from harness.cell_meta import exported_cell_meta

LAPSE_S = 150  # how far past `exp` the presented credential is, in the JWT
FAMILY = "sdxl"

# a real exported-cell envelope. The publish path recomputes the key
# from the recorded blocks and refuses a cell that cannot state one, so the
# credential-lapse legs below have to ride a cell that could genuinely publish.
META = exported_cell_meta(family=FAMILY, gen_worker="0.76.6",
                          weight_lane="w8a8", lora_bucket=64)
COMPILED_GRAPH_KEY = META["compiled_graph_key"]


def _jwt(*, lifetime_s: float) -> str:
    """A real three-segment JWT carrying a real `exp`. Unsigned on purpose —
    the worker never verifies its own credential, it reads what it holds.

    `lifetime_s` is the credential's OWN remaining life, a claim the token
    carries. Nothing here waits on it: no row in this file is bounded by a
    clock.
    """
    issued = time.time()
    def seg(obj: dict) -> str:
        raw = json.dumps(obj).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")
    exp = int(issued + lifetime_s)
    return f"{seg({'alg': 'none'})}.{seg({'exp': exp, 'sub': 'pod-1'})}.sig"


class _Hub(http.server.BaseHTTPRequestHandler):
    """tensorhub's worker-JWT refusal, verbatim: `authenticateWorkerJWT` writes
    `{"error": {"code": "unauthorized", ...}}` via `httperrors.WriteOrchestrator`
    and books its th#1423 pod_event; the wire shape is all the worker sees."""

    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # noqa: D102
        pass

    def _json(self, code: int, body: dict) -> None:
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):  # noqa: N802
        srv = self.server
        path = urllib.parse.urlparse(self.path).path
        n = int(self.headers.get("Content-Length") or 0)
        _ = self.rfile.read(n) if n else b""
        bearer = (self.headers.get("Authorization") or "").split(" ")[-1]
        with srv.lock:
            srv.seen.append((path, bearer))
        self._json(401, {"error": {
            "code": "unauthorized",
            "message": "invalid worker token",
            "request_id": "req-1",
        }})


class _Server:
    def __init__(self):
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Hub)
        self.httpd.lock = threading.Lock()
        self.httpd.seen = []
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    @property
    def base(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def close(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def hub():
    s = _Server()
    try:
        yield s
    finally:
        s.close()


@pytest.fixture()
def artifact(tmp_path: Path) -> Path:
    # Every row is refused at publish-intent, before artifact transport opens
    # this path. Keep only the opaque carrier needed by the publisher API: the
    # behavior under test is the credential on the real HTTP leg.
    out = tmp_path / "mintdir" / "cell.tar.gz"
    out.parent.mkdir()
    out.write_bytes(b"\x11" * 4096)
    return out


def _publisher(hub, token: str) -> fc.CellPublisher:
    return fc.CellPublisher(base_url=hub.base, worker_jwt=lambda: token,
                            image_digest="sha256:" + "1" * 64)


# --- the wire: a 401 must arrive as a TYPED, groupable failure ---------------


def test_intent_401_carries_the_hubs_status_and_code(hub, artifact):
    """RED before the fix: `_post` raised a bare `RuntimeError`, so this
    `HubPublishError` never existed and `.status`/`.code` did not either."""
    live = _jwt(lifetime_s=900)
    with pytest.raises(HubPublishError) as caught:
        _publisher(hub, live).publish(FAMILY, artifact, dict(META))

    exc = caught.value
    assert exc.status == 401
    assert exc.code == "unauthorized"
    assert fc._publish_failure_phase(exc) == "unauthorized"
    # The refusal was really spoken over the wire, by the credential we hold.
    assert hub.httpd.seen == [("/v1/worker/compiled-graphs/publish-intent", live)]


def test_an_expired_credential_is_named_as_such_not_as_a_generic_401(
        hub, artifact):
    """The th#1423 shape exactly: a 28-minute compile hands publish-intent a
    credential already past its own `exp`. The hub cannot tell "expired" from
    "revoked" — the worker can, and only from the token it presented.

    RED before the fix: `_publish_failure_phase` returned `RuntimeError`."""
    dead = _jwt(lifetime_s=-LAPSE_S)
    with pytest.raises(HubPublishError) as caught:
        _publisher(hub, dead).publish(FAMILY, artifact, dict(META))

    exc = caught.value
    assert exc.status == 401
    assert exc.code == fc.CREDENTIAL_EXPIRED_CODE
    assert fc._publish_failure_phase(exc) == "worker_credential_expired"
    assert "past its own exp" in str(exc)


def test_the_lapse_is_on_the_wire_before_the_intent_is_spent(
        hub, artifact, monkeypatch):
    """The credential states its own `exp`, so "this mint outlived its
    credential" is knowable at the one moment it decides the outcome — not
    only afterwards from a 401 whose cause the hub cannot see."""
    seen: list = []
    monkeypatch.setattr(
        fc.activity_mod, "emit_event",
        lambda kind, detail, phase="", duration_ms=0, **_kw: seen.append((kind, phase, detail)))

    with pytest.raises(HubPublishError):
        _publisher(hub, _jwt(lifetime_s=-LAPSE_S)).publish(
            FAMILY, artifact, dict(META))

    legs = [(k, p, d) for k, p, d in seen if p == "credential_expired"]
    assert len(legs) == 1, seen
    assert "past_exp_s=1" in legs[0][2]  # 150s, measured — never a constant


def test_a_live_credential_publishes_no_lapse_leg(hub, artifact, monkeypatch):
    """The negative half: an in-date credential must not be accused."""
    seen: list = []
    monkeypatch.setattr(
        fc.activity_mod, "emit_event",
        lambda kind, detail, phase="", duration_ms=0, **_kw: seen.append((kind, phase)))

    with pytest.raises(HubPublishError):
        _publisher(hub, _jwt(lifetime_s=900)).publish(
            FAMILY, artifact, dict(META))

    assert not [p for _, p in seen if p == "credential_expired"]


# --- the production entry: the event the three failures were reported as ----


def test_the_background_publish_reports_the_grouped_phase(hub, artifact,
                                                          monkeypatch):
    """`_publish_async` is the path all three production failures took. Its
    `self_mint_publish_failed` phase is the group key the hub reads back.

    RED before the fix: `phase="RuntimeError"`."""
    seen: list = []
    monkeypatch.setattr(
        fc.activity_mod, "emit_event",
        lambda kind, detail, phase="", duration_ms=0, **_kw: seen.append((kind, phase)))

    thread = fc._publish_async(
        _publisher(hub, _jwt(lifetime_s=-LAPSE_S)),
        FAMILY, artifact, dict(META), compiled_graph_key_digest=COMPILED_GRAPH_KEY)
    thread.join()  # the publish thread itself is the bound — no clock

    assert ("self_mint_publish_failed", "worker_credential_expired") in seen


# --- the mint publisher must read the credential at USE time ----------------


@pytest.fixture()
def clean_process_credential():
    """Restore the process-wide `Settings` and credential this test replaces —
    both are module globals, and a leak would silently arm another test."""
    from gen_worker import worker_credential
    from gen_worker.config import process as config_process

    prior = config_process._SETTINGS
    prior_boot = worker_credential._BOOTSTRAP
    try:
        yield
    finally:
        config_process._SETTINGS = prior
        worker_credential.reset()
        worker_credential._BOOTSTRAP = prior_boot


def test_the_self_mint_publisher_reads_the_credential_at_use_time(
        monkeypatch, clean_process_credential):
    """RED before the fix: the mint publisher closed over
    `worker_jwt=lambda: token`, a token CAPTURED at construction, so a rotation
    landing during the mint — the only thing that can save a compile longer
    than the TTL — was invisible.

    pgw#1270 deleted the `aot_mint` CLI that built a second publisher of its
    own. `Executor._cell_publisher` is the ONE surviving construction site, so
    it is the one asserted; a captured token here would be the same defect in
    the only place left to have it.
    """
    from gen_worker import config, worker_credential
    from gen_worker.executor import Executor

    boot = _jwt(lifetime_s=60)
    rotated = _jwt(lifetime_s=3600)

    monkeypatch.setenv("WORKER_JWT", boot)
    monkeypatch.setenv("TENSORHUB_URL", "http://127.0.0.1:1")
    monkeypatch.delenv("TENSORHUB_PUBLIC_URL", raising=False)
    monkeypatch.delenv("TENSORHUB_TOKEN", raising=False)
    config.reload_for_test()
    # The pod's own starting state: nothing has handed the process-wide
    # credential source anything yet.
    worker_credential.reset()
    monkeypatch.setattr(worker_credential, "_BOOTSTRAP", "", raising=False)

    async def _send(_msg):
        return None

    ex = Executor([], _send)
    # th#1423: `worker_credential.current()` only answers once the PROCESS
    # ENTRY hands it the boot token (`entrypoint` / the procsplit parent).
    worker_credential.install_bootstrap(config.current())

    publisher = ex._cell_publisher()
    assert publisher.worker_jwt() == boot

    # 0.0 = expiry unknown to the installer; the token states its own.
    worker_credential.install(rotated, 0.0)
    assert publisher.worker_jwt() == rotated
