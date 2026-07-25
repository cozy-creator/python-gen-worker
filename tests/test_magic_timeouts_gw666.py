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
from blake3 import blake3

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
        url, dst, len(blob), blake3(blob).hexdigest(),
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

    monkeypatch.setattr(hub, "_COMPLETE_IN_PROGRESS_POLL_S", 0.05)
    monkeypatch.setattr(hub, "_COMPLETE_NETWORK_RETRY_DELAY_S", 0.05)
    monkeypatch.setattr(hub, "_COMPLETE_SILENCE_WINDOW_S", 0.4)
    monkeypatch.setattr(hub, "_RETRY_BASE_DELAY_S", 0.01)
    monkeypatch.setattr(hub, "_RETRY_MAX_DELAY_S", 0.02)


def _hub_client(base: str):
    from gen_worker.convert.hub import HubClient

    return HubClient(base_url=base, token="t", owner="o")


def test_publish_complete_waits_while_the_hub_keeps_verifying(fast_hub_retries) -> None:
    ok = _json_body({"ok": True})
    with _Server(_n_then_ok(
        15, (409, _IN_PROGRESS, None), (200, ok, {"Content-Type": "application/json"}),
    )) as srv:
        start = time.monotonic()
        resp = _hub_client(srv.base)._post_complete("/complete", {})
        assert time.monotonic() - start > 0.4  # outlived the give-up window
    assert resp.status_code == 200


def test_publish_complete_gives_up_once_the_hub_goes_silent(fast_hub_retries) -> None:
    from gen_worker.convert.hub import HubPublishError

    client = _hub_client(f"http://127.0.0.1:{_free_port()}")  # nothing listening
    with pytest.raises(HubPublishError):
        client._post_complete("/complete", {})


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
