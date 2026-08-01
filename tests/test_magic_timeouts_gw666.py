"""gw#666 (th#1166 findings B-F): the last five gen-worker wall clocks.

Every tape here runs against REAL children and REAL loopback HTTP servers —
the point is precisely that a mock cannot tell you whether a slow-but-alive
peer survives, and that is the only property under test.

Each pair is deliberately discriminating: the "alive" case runs FAR longer
than the give-up window and must still succeed, which is exactly what the
deleted wall clocks made impossible.
"""

from __future__ import annotations

import asyncio
import http.server
import json
import os
import re
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional

import pytest
import requests
import hashlib

from gen_worker.stall import ProgressFloor, SilenceWindow

SRC = Path(__file__).resolve().parents[1] / "src" / "gen_worker"


# ---------------------------------------------------------------------------
# Helpers: a real loopback HTTP server driven by a per-request callback.
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _Server:
    """One loopback HTTP server whose every request is answered by ``handler``
    -> (status, body_bytes, extra_headers)."""

    def __init__(self, handler: Callable[[Any], tuple]) -> None:
        self.calls = 0
        outer = self

        class _H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _answer(self) -> None:
                outer.calls += 1
                # Drain the request body or the next keep-alive request on this
                # connection reads it as a request line.
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                status, body, headers = handler(self)
                self.send_response(status)
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            do_GET = _answer
            do_POST = _answer

            def log_message(self, *a: Any) -> None:
                pass

        self.port = _free_port()
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", self.port), _H)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def __enter__(self) -> "_Server":
        self.thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def _json_body(payload: dict) -> bytes:
    return json.dumps(payload).encode()


_IN_PROGRESS = _json_body({"error": {"code": "upload_complete_in_progress"}})


def _n_then_ok(n: int, hot: tuple, cold: tuple) -> Callable[[Any], tuple]:
    """Answer ``hot`` for the first ``n`` requests, then ``cold`` forever."""
    seen = {"n": 0}

    def _h(_req: Any) -> tuple:
        seen["n"] += 1
        return hot if seen["n"] <= n else cold

    return _h


# ---------------------------------------------------------------------------
# The primitive
# ---------------------------------------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def test_silence_window_measures_advances_not_runtime() -> None:
    clock = _Clock()
    w = SilenceWindow(10.0, now=clock)

    for _ in range(100):  # runs 900s — ten times the window — while advancing
        clock.t += 9.0
        assert not w.stalled()
        w.touch()

    clock.t += 10.0
    assert w.stalled()
    assert w.silent_for() == pytest.approx(10.0)


def test_silence_window_marker_advances_only_on_change() -> None:
    clock = _Clock()
    w = SilenceWindow(10.0, now=clock)

    assert w.touch_if_changed("building@7") is True  # first sighting counts
    clock.t += 6.0
    assert w.touch_if_changed("building@7") is False  # same token: no advance
    clock.t += 6.0
    assert w.stalled()
    assert w.touch_if_changed("building@8") is True
    assert not w.stalled()


def test_progress_floor_treats_a_trickle_as_no_progress() -> None:
    floor = ProgressFloor(1000)
    total = 0
    for _ in range(500):  # 500 bytes of "progress" — half the floor
        total += 1
        assert floor.cleared(total) is False
    assert floor.moved(total) == 500
    total += 500
    assert floor.cleared(total) is True
    assert floor.moved(total) == 0  # rebased


# ---------------------------------------------------------------------------
# B — runtimes/server.py: an engine boot is bounded by SILENCE
# ---------------------------------------------------------------------------

# A slow-booting engine: prints load progress for a while, THEN serves /health.
# Under the deleted `_DEFAULT_BOOT_TIMEOUT_S` this shape died for the sole
# crime of taking longer than the constant.
_CHATTY_ENGINE = """
import sys, time
from http.server import BaseHTTPRequestHandler, HTTPServer
port, talk_s = int(sys.argv[1]), float(sys.argv[2])
i = 0
end = time.time() + talk_s
while time.time() < end:
    print("loading weights shard %d" % i, flush=True)
    i += 1
    time.sleep(0.05)
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")
    def log_message(self, *a):
        pass
HTTPServer(("127.0.0.1", port), H).serve_forever()
"""

# An engine that says one thing and then wedges, never serving.
_WEDGED_ENGINE = """
import sys, time
open(sys.argv[1], "w").write(str(__import__("os").getpid()))
print("initializing engine", flush=True)
time.sleep(600)
"""


def _proc_for(script: str, *args: str, port: int, window_s: float):
    from gen_worker.runtimes.server import ServerProcess

    return ServerProcess(
        [sys.executable, "-u", "-c", script, *args],
        health_url=f"http://127.0.0.1:{port}/health",
        stall_window_s=window_s,
    )


def test_a_talking_engine_boots_however_long_it_takes(caplog) -> None:
    """The stall window is a FIFTH of the boot; only silence may fail it."""
    port = _free_port()
    boot = _proc_for(_CHATTY_ENGINE, str(port), "1.5", port=port, window_s=0.3)

    caplog.set_level("INFO", logger="gen_worker.runtimes.server")
    start = time.monotonic()
    handle = boot.start()
    try:
        elapsed = time.monotonic() - start
        assert elapsed > 0.3 * 3  # outlived several windows while advancing
        assert handle.alive
        # The engine's own boot log now reaches the worker log, where an
        # operator can see it — it used to go to an unread inherited stdout.
        assert any("[engine] loading weights shard" in r.getMessage()
                   for r in caplog.records)
    finally:
        handle.stop()
    assert not handle.alive


def test_a_silent_engine_fails_on_the_stall_window_and_is_killed(tmp_path) -> None:
    from gen_worker.runtimes.server import ServerBootError

    pidfile = tmp_path / "engine.pid"
    port = _free_port()
    boot = _proc_for(_WEDGED_ENGINE, str(pidfile), port=port, window_s=0.5)

    start = time.monotonic()
    with pytest.raises(ServerBootError) as excinfo:
        boot.start()
    elapsed = time.monotonic() - start

    assert "produced no output" in str(excinfo.value)
    assert elapsed < 30.0  # NOT the child's own 600s sleep
    pid = int(pidfile.read_text())
    for _ in range(100):  # SIGTERM -> exit is not instantaneous
        try:
            os.kill(pid, 0)
        except (ProcessLookupError, PermissionError):
            break
        time.sleep(0.1)
    else:
        pytest.fail(f"wedged engine pid {pid} survived the boot failure")


def test_the_boot_gate_keys_on_the_GAP_not_on_total_runtime() -> None:
    """Control for the two tapes above: the same chatty engine, with a window
    narrower than its 0.05s inter-line gap, DOES fail — so the gate is live,
    and what it measures is silence between lines rather than elapsed time."""
    from gen_worker.runtimes.server import ServerBootError

    port = _free_port()
    boot = _proc_for(_CHATTY_ENGINE, str(port), "5.0", port=port, window_s=0.01)
    with pytest.raises(ServerBootError) as excinfo:
        boot.start()
    assert "produced no output" in str(excinfo.value)


def test_degrading_boot_degrades_on_silence_not_on_a_clock(tmp_path) -> None:
    from gen_worker.runtimes.server import DegradingBoot

    dead_port, live_port = _free_port(), _free_port()
    boot = DegradingBoot([
        _proc_for(_WEDGED_ENGINE, str(tmp_path / "p"), port=dead_port, window_s=0.4),
        _proc_for(_CHATTY_ENGINE, str(live_port), "0.3", port=live_port, window_s=0.4),
    ])
    handle = boot.start()
    try:
        assert handle.base_url.endswith(str(live_port))
        assert handle.alive
    finally:
        handle.stop()


# ---------------------------------------------------------------------------
# C — models/cozy_cas.py: the CAS retry give-up is byte progress
# ---------------------------------------------------------------------------


class _FlakyBlobServer:
    """Serves ``blob`` with Range resume, severing the connection after
    ``sever_after`` bytes of every response (or answering ``fail_status``)."""

    def __init__(self, blob: bytes, *, sever_after: int = 0, fail_status: int = 0) -> None:
        self.blob, self.calls = blob, 0
        outer = self

        class _H(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_GET(self) -> None:
                outer.calls += 1
                if fail_status:
                    self.send_response(fail_status)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                start = 0
                rng = str(self.headers.get("Range") or "")
                if rng.startswith("bytes="):
                    start = int(rng.split("=", 1)[1].split("-", 1)[0])
                body = outer.blob[start:]
                self.send_response(206 if start else 200)
                if start:
                    self.send_header(
                        "Content-Range",
                        f"bytes {start}-{len(outer.blob) - 1}/{len(outer.blob)}",
                    )
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                if sever_after and len(body) > sever_after:
                    # Truncate mid-body and drop the socket: exactly what a
                    # flaky pod uplink does, and the client resumes by Range.
                    self.wfile.write(body[:sever_after])
                    self.wfile.flush()
                    self.close_connection = True
                    self.connection.close()
                else:
                    self.wfile.write(body)

            def log_message(self, *a: Any) -> None:
                pass

        self.port = _free_port()
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", self.port), _H)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/blob"

    def __enter__(self) -> "_FlakyBlobServer":
        self.thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()


def _fetch(url: str, dst: Path, blob: bytes) -> None:
    from gen_worker.models import cozy_cas

    asyncio.run(cozy_cas._download_one_file(
        url, dst, len(blob), "sha256:" + hashlib.sha256(blob).hexdigest(),
    ))


@pytest.fixture()
def fast_cas_retries(monkeypatch) -> None:
    from gen_worker.models import cozy_cas

    monkeypatch.setattr(cozy_cas, "_RETRY_BACKOFF_CAP_S", 0.01)
    monkeypatch.setattr(cozy_cas, "_RETRY_PROGRESS_FLOOR_BYTES", 4 * 1024 * 1024)
    monkeypatch.setattr(cozy_cas, "_TRANSIENT_MAX_TRIES", 2)


def test_a_flaky_but_advancing_cas_fetch_outlives_its_retry_cap(
    tmp_path, fast_cas_retries,
) -> None:
    """Two severed connections against a cap of TWO — and it still lands,
    because each attempt put 8 MiB on disk and cleared the byte floor. The
    old code counted failures (and wall seconds) regardless of bytes."""
    blob = os.urandom(24 * 1024 * 1024)
    dst = tmp_path / "blob.bin"
    with _FlakyBlobServer(blob, sever_after=8 * 1024 * 1024) as srv:
        _fetch(srv.url, dst, blob)
        assert srv.calls == 3  # 8 MiB + 8 MiB + the resumed tail
    assert dst.read_bytes() == blob


def test_a_cas_fetch_that_delivers_nothing_gives_up_on_the_count(
    tmp_path, fast_cas_retries,
) -> None:
    blob = os.urandom(4096)
    with _FlakyBlobServer(blob, fail_status=500) as srv:
        with pytest.raises(requests.RequestException):
            _fetch(srv.url, tmp_path / "blob.bin", blob)
        assert srv.calls == 2  # exactly _TRANSIENT_MAX_TRIES, no byte progress


# ---------------------------------------------------------------------------
# E — presigned_upload.py: a live hub-side assembly is never a job failure
# ---------------------------------------------------------------------------


@pytest.fixture()
def fast_finalize_polls(monkeypatch) -> None:
    from gen_worker import presigned_upload as pu

    monkeypatch.setattr(pu, "_COMPLETE_IN_PROGRESS_POLL_S", 0.05)
    monkeypatch.setattr(pu, "_COMPLETE_SILENCE_WINDOW_S", 0.4)


def _poll_finalize(url: str) -> dict:
    from gen_worker.presigned_upload import _poll_until_finalized

    return _poll_until_finalized(
        complete_url=url,
        complete_headers={"Content-Type": "application/json"},
        payload={"parts": []},
        cancel_check=None,
        session=requests.Session(),
    )


def test_a_long_hub_side_assembly_is_waited_out_not_failed(fast_finalize_polls) -> None:
    """Twenty in-progress answers spanning ~2.5x the give-up window — the job
    must survive, because the hub was working the whole time."""
    ok = _json_body({"ref": "cas/abc", "blake3": "ab", "size_bytes": 1})
    with _Server(_n_then_ok(
        20, (409, _IN_PROGRESS, None), (200, ok, {"Content-Type": "application/json"}),
    )) as srv:
        start = time.monotonic()
        out = _poll_finalize(f"{srv.base}/complete")
        assert time.monotonic() - start > 0.4 * 2
    assert out["ref"] == "cas/abc"


def test_a_hub_that_stops_answering_fails_the_finalize(fast_finalize_polls) -> None:
    from gen_worker.api.errors import ArtifactTransferError

    dead = f"http://127.0.0.1:{_free_port()}/complete"  # nothing listening
    with pytest.raises(ArtifactTransferError) as excinfo:
        _poll_finalize(dead)
    assert "stopped answering" in str(excinfo.value)
    assert excinfo.value.retryable is True


# ---------------------------------------------------------------------------
# D — convert/hub.py: publish-complete waits while the hub verifies
# ---------------------------------------------------------------------------


@pytest.fixture()
def fast_hub_retries(monkeypatch) -> None:
    from gen_worker.convert import hub

    monkeypatch.setattr(hub, "_RETRY_BASE_DELAY_S", 0.01)
    monkeypatch.setattr(hub, "_RETRY_MAX_DELAY_S", 0.02)
    # pgw#796: `_post` funnels through `_send_with_retries`, whose OWN silence
    # window (pgw#738/#743, added after this fixture was written) was left at
    # its 600s production value — so the dead-port tape below sat in that inner
    # loop for ten real minutes before the outer loop ever saw a failure. One
    # test was 600.0s of a 1341.7s publish gate. Scaling it with the other four
    # changes nothing about the mechanism under test; only the clock it runs on.
    monkeypatch.setattr(hub, "_SEND_SILENCE_WINDOW_S", 0.4)


from gen_worker.convert.hub import HubPublishError


def _hub_client(base: str):
    from gen_worker.convert.hub import HubClient

    return HubClient(base_url=base, token="t", owner="o")


def test_v2_complete_retries_only_when_nothing_was_heard(fast_hub_retries) -> None:
    """gw#666 on the surviving completion path. pgw#807 deleted the v1
    `_post_complete` (and with it the 409 `upload_complete_in_progress` poll —
    a v2 publish is NOT idempotent to re-complete, so a blind retry there can
    only destroy the diagnosis). What the rule still requires: a completion
    that heard NOTHING keeps trying on liveness rather than a stopwatch."""
    client = _hub_client(f"http://127.0.0.1:{_free_port()}")  # nothing listening
    start = time.monotonic()
    with pytest.raises(HubPublishError):
        client._post_v2_complete("/complete")
    # It did not give up on the first refused connection: the silence window
    # (0.4 s here, 600 s in production) is what bounds it, not an attempt count.
    assert time.monotonic() - start >= 0.4


def test_v2_complete_returns_any_hub_answer_without_retrying(fast_hub_retries) -> None:
    """A refusal is a th#1301 PROJECTION, not an error envelope. Retrying it
    found the session already terminal and reported `publish_repudiated` — a
    consequence in place of a cause. Any HTTP answer comes back as-is, once."""
    body = _json_body({"status": {"stage": "repudiated", "terminal": True,
                                  "failure": {"code": "invalid_manifest_for_kind",
                                              "retryable": False}}})
    with _Server(_n_then_ok(
        0, (409, body, {"Content-Type": "application/json"}),
        (409, body, {"Content-Type": "application/json"}),
    )) as srv:
        resp = _hub_client(srv.base)._post_v2_complete("/complete")
    assert resp.status_code == 409
    assert resp.json()["status"]["failure"]["code"] == "invalid_manifest_for_kind"


# ---------------------------------------------------------------------------
# F — request_context/_datasets.py: materialize polls while the hub answers
# ---------------------------------------------------------------------------


@pytest.fixture()
def fast_dataset_polls(monkeypatch) -> None:
    from gen_worker.request_context import _datasets

    monkeypatch.setattr(_datasets, "_POLL_BACKOFF_START_S", 0.02)
    monkeypatch.setattr(_datasets, "_POLL_BACKOFF_CAP_S", 0.05)


def _materialize(base: str, window_s: float):
    from gen_worker.request_context._datasets import fetch_materialize_manifest

    return fetch_materialize_manifest(
        base, "tok", "11111111-1111-1111-1111-111111111111",
        hub_silence_window_s=window_s,
    )


def test_a_building_snapshot_is_polled_for_as_long_as_the_hub_answers(
    fast_dataset_polls,
) -> None:
    building = _json_body({"status": "building", "state_version": 7, "retry_after": 0})
    ready = _json_body({
        "snapshot_id": "dsnap-1",
        "entries": [{"path": "rows.jsonl", "inline_text": "{}"}],
    })
    with _Server(_n_then_ok(
        20, (202, building, None), (200, ready, {"Content-Type": "application/json"}),
    )) as srv:
        start = time.monotonic()
        snapshot_id, entries = _materialize(srv.base, 0.3)
        # The build outran the give-up window several times over: a 202 is the
        # hub saying the build is LIVE, so it refreshes the window.
        assert time.monotonic() - start > 0.3 * 2
    assert snapshot_id == "dsnap-1" and len(entries) == 1


def test_materialize_gives_up_only_when_the_hub_says_nothing_definite(
    fast_dataset_polls,
) -> None:
    with _Server(lambda _r: (502, b"", None)) as srv:
        with pytest.raises(RuntimeError) as excinfo:
            _materialize(srv.base, 0.3)
    assert "said nothing definite" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Source guard: none of the five may regrow a wall clock
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "relpath, gone",
    [
        ("runtimes/server.py", "_DEFAULT_BOOT_TIMEOUT_S"),
        ("models/cozy_cas.py", "_MAX_RETRY_TIME_S"),
        ("convert/hub.py", "_COMPLETE_NETWORK_MAX_WAIT_S"),
        ("presigned_upload.py", "_COMPLETE_IN_PROGRESS_MAX_WAIT_S"),
        ("request_context/_datasets.py", "_MATERIALIZE_BUDGET_S"),
    ],
)
def test_the_wall_clock_constants_are_gone(relpath: str, gone: str) -> None:
    """No module-level ASSIGNMENT may reappear (the comments that explain why
    each one died still name them, and should)."""
    source = (SRC / relpath).read_text()
    assert re.search(rf"^{gone}\s*=", source, re.MULTILINE) is None


# ---------------------------------------------------------------------------
# pgw#795: the guard covers TEST code too, because a release gate runs it
# ---------------------------------------------------------------------------
#
# The five tapes above pin five NAMED constants in five NAMED src files. That
# shape missed pgw#795 twice over: it never looks at ``tests/`` at all, and it
# is a regression pin rather than a pattern detector, so even in ``src/`` a
# freshly-invented wall clock walks past it.
#
# "Tests are exempt" is the position that needed defending, and it does not
# survive the evidence. A magic timeout in a test cannot kill a tenant's job —
# that asymmetry is real — but the publish workflow runs `pytest tests/` as the
# gate on every upload to PyPI, so a test that fails on the runner's mood is a
# release-blocking defect. It cost three consecutive v0.78.0 publish jobs, each
# queued ~4.6 h and running ~22 min, and blocked three downstream lanes; the
# blast radius simply arrives through the release rather than through a pod.
#
# A blanket ban would be wrong, though, because three fixed durations in tests
# are legitimate and none of them is a give-up decision:
#
#   1. INDUCED durations — the fake handler that sleeps 0.5s IS the stimulus.
#   2. ABSENCE probes — `wait_for(..., timeout=2.0)` inside `except
#      TimeoutError`, where expiry makes the test PASS. A slow runner weakens
#      the assertion; it cannot flake it.
#   3. HANG bounds with an order of magnitude of headroom — `elapsed < 15`
#      proving a connection-refused path did not become a multi-minute
#      pod-billing hang. The bound is the property.
#
# The dangerous shape is precisely the fourth: a fixed duration whose EXPIRY
# FAILS the test — a deadline on an awaited event, or an assertion comparing
# measured time to a constant. Those two are what these guards detect, and the
# inventories below are burn-down lists, not exemptions: they may shrink, and
# the guard fails if they grow.

TESTS = Path(__file__).resolve().parent

_CLOCK = r"time\.(?:monotonic|time|perf_counter)\(\)"

_LITERAL_DEADLINE = re.compile(rf"{_CLOCK}\s*\+\s*[0-9]")

# pgw#845: the two regexes below used to require a DIGIT and a small token
# vocabulary, and both holes were load-bearing.
#
#   `deadline = time.monotonic() + _TIMEOUT`  (_TIMEOUT = 15.0)
#
# is the pgw#795 shape with the number spelled as a name, and it walked past
# `_LITERAL_DEADLINE` — including inside tests/harness/, which made
# `test_the_shared_test_harness_has_no_wall_clock_deadline` a guard that could
# not fail. A deadline built from a PARAMETER or from a harness quantity is a
# different thing (the caller chose it, and anchoring to a measured or
# configured quantity is the recommended fix), so only names bound to a literal
# duration IN THE SAME MODULE count.
_NAMED_DEADLINE = re.compile(rf"{_CLOCK}\s*\+\s*([A-Za-z_][A-Za-z0-9_]*)\b")
_DURATION_CONST = re.compile(
    r"^([A-Za-z_][A-Za-z0-9_]*)\s*[:=][^=][^#\n]*?=?\s*([0-9][0-9_]*(?:\.[0-9]*)?)\s*(?:#.*)?$"
)
_SIMPLE_CONST = re.compile(r"^([A-Z_][A-Z0-9_]*)\s*=\s*([0-9][0-9_]*(?:\.[0-9]*)?)\s*(?:#.*)?$")

# ...and a measured duration compared to a constant is dangerous in BOTH
# directions: `< N` fails on a slow runner, `>= N` fails on a fast one. The
# vocabulary now includes the clock call itself (`assert time.monotonic() -
# start < 60` named nothing) plus latency/rtt/duration. A comparison against
# literal zero is a positivity check, not a budget, and is excluded.
_MEASURED = rf"(?:{_CLOCK}|elapsed|wall\b|latency|rtt|_ms\b|_s\b|took)"
_ELAPSED_UPPER_BOUND = re.compile(
    rf"^\s*assert\s+[^#\n]*{_MEASURED}[^#\n]*[<>]=?\s*([0-9][0-9_]*(?:\.[0-9]*)?)"
)

#: Wait loops in test modules that still race a fixed clock instead of waiting
#: on progress (``harness.progress_wait.await_count`` is the replacement). Every
#: one of them can fail a publish the way pgw#628 did. Owned by pgw#795's
#: follow-up; delete lines from this list, never add them.
_DEADLINE_BURNDOWN = {
    "test_activity_gw601.py",
    "test_cancel_unwind_pgw687.py",
    "test_mint_gate_pgw677.py",
    "test_mint_reopen_pgw677.py",
    "test_rotation_preload_pgw674.py",
    "test_sp_group_isolation_pgw773_774.py",
}

#: Assertions that compare measured time to a constant. Each survives review as
#: a HANG bound (class 3 above) with at least an order of magnitude of headroom
#: over the observed value — except the two mint_reopen entries, which are
#: latency budgets and are the next ones to anchor to the harness's own
#: configured quantities the way pgw#677's were.
_ELAPSED_ASSERT_BURNDOWN = {
    "test_boot_smoke_hardware_report.py",  # hang bound: refused connect << 15s
    "test_hardware_report.py",             # hang bound: refused connect << 10s
    "test_mint_process_pgw784.py",         # hang bound: reap must not wait out a 600s child
    "test_subprocess_stall_gw665.py",      # hang bound: stall window << 30s
    "test_mint_reopen_pgw677.py",          # LATENCY BUDGET — anchor it next
    # pgw#845 — exposed by the widened pattern below, each reviewed:
    "test_p6_cancel_stream_backpressure.py",  # LOWER bound on 2x an induced 0.5s
                                              # handler sleep: only OVERLAP can
                                              # fail it, slowness only raises it
    "test_progress_gw621.py",                 # LOWER bound on the fixture's own
                                              # configured freeze window
    "test_mint_gate_pgw677.py",               # LOWER bounds on induced seed work
    "test_procsplit_pgw763.py",               # hang bound: cancel across the seam
                                              # << 3s against a 50ms handler poll
    "test_gw640_postmortem.py",               # hang bound: shutdown << 45s for a
                                              # named grace. OWNED by the gw#640
                                              # lane while it lands the SIGTERM
                                              # deadlock fix — anchor to `grace`
                                              # there, not here
}

#: Fixtures in tests/harness/ that spend a fixed duration PRODUCING the stimulus
#: (class 1), rather than giving up on an awaited event. They are named here so
#: the harness guard states a true claim it can still fail: a NEW clock in the
#: harness, or a clock in a waiting path, is a failure.
_HARNESS_INDUCED_DURATION = {
    "harness/progress_endpoints.py",        # ticks a progress counter for TICK_S
    "harness/stubborn_endpoints_pgw687.py",  # self-cap on a deliberately wedged
                                             # handler, so a bug cannot hang the
                                             # suite forever
}


def _code_lines(path: Path) -> list:
    return [line for line in path.read_text().splitlines()
            if not line.lstrip().startswith("#")]


def _scan(root: Path, pattern: "re.Pattern[str]", *, match: bool = False) -> set:
    hits = set()
    for path in sorted(root.rglob("*.py")):
        if path.name == Path(__file__).name:
            continue
        for line in _code_lines(path):
            found = pattern.match(line) if match else pattern.search(line)
            if not found:
                continue
            if pattern is _ELAPSED_UPPER_BOUND and float(found.group(1)) == 0:
                continue  # `> 0` is a positivity check, not a budget
            hits.add(path.relative_to(TESTS).as_posix())
    return hits


def _deadlines_from_named_constants(root: Path) -> set:
    """Files where a deadline is built from a module-level literal duration."""
    hits = set()
    for path in sorted(root.rglob("*.py")):
        if path.name == Path(__file__).name:
            continue
        lines = _code_lines(path)
        constants = {
            m.group(1) for m in (_SIMPLE_CONST.match(line) for line in lines) if m
        }
        for line in lines:
            named = _NAMED_DEADLINE.search(line)
            if named is not None and named.group(1) in constants:
                hits.add(path.relative_to(TESTS).as_posix())
    return hits


def test_the_shared_test_harness_has_no_wall_clock_deadline() -> None:
    """``tests/harness/`` is the substrate every hub-double test waits through;
    one wall clock there is a wall clock in ~37 files. It had three (a 15s
    ``DEFAULT_TIMEOUT_S`` on ``wait_for`` / ``wait_for_count`` /
    ``wait_connection``), and they are gone — waits there are progress-gated
    unless the caller passes an explicit ``timeout=`` to probe for absence."""
    found = (
        _scan(TESTS / "harness", _LITERAL_DEADLINE)
        | _deadlines_from_named_constants(TESTS / "harness")
    )
    new = found - _HARNESS_INDUCED_DURATION
    assert not new, (
        f"wall-clock deadline(s) in a shared harness WAIT: {sorted(new)} — a "
        f"clock here is a clock in every hub-double test; pgw#795"
    )
    assert not (_HARNESS_INDUCED_DURATION - found), (
        f"gone! remove {sorted(_HARNESS_INDUCED_DURATION - found)} from the list"
    )


def test_no_new_fixed_deadline_wait_appears_in_a_test() -> None:
    found = {
        p for p in _scan(TESTS, _LITERAL_DEADLINE)
        | _deadlines_from_named_constants(TESTS)
        if not p.startswith("harness/")
    }
    new = found - _DEADLINE_BURNDOWN
    assert not new, (
        f"new fixed-deadline wait(s) in {sorted(new)} — wait on progress "
        f"(harness.progress_wait.await_count) rather than on a clock; pgw#795"
    )
    assert not (_DEADLINE_BURNDOWN - found), (
        f"fixed! remove {sorted(_DEADLINE_BURNDOWN - found)} from the burn-down list"
    )


def test_no_new_wall_clock_assertion_appears_in_a_test() -> None:
    found = _scan(TESTS, _ELAPSED_UPPER_BOUND, match=True)
    new = found - _ELAPSED_ASSERT_BURNDOWN
    assert not new, (
        f"new elapsed-vs-constant assertion(s) in {sorted(new)} — assert the "
        f"property (the work's product, an ordering, a share of a measured "
        f"baseline), not the runner's speed; pgw#795"
    )
    assert not (_ELAPSED_ASSERT_BURNDOWN - found), (
        f"fixed! remove {sorted(_ELAPSED_ASSERT_BURNDOWN - found)} from the list"
    )
