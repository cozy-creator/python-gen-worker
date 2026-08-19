"""pgw#1421 — the engine-hosted (external-binary) runtime, ISOLATED.

Paul's 2026-08-19 working loop: isolate the unit, RUN it, introspect deeply,
then widen. The unit here is process supervision, and the thing it supervises
is deliberately NOT llama.cpp or vLLM: it is a trivial stand-in HTTP server
whose behaviour each arm dictates — serves, dies on exec, dies after N
seconds, goes silent, or talks slowly. That isolation is legitimate because
the boundary is strict and side-effect-free: ``EngineSpec`` hands
:func:`boot_engine` an argv and a health route and learns nothing else about
the engine, so an arm that proves the supervisor against a stand-in proves it
against llama-server.

$0, no GPU, no weights, no network beyond loopback. What it does NOT prove —
and this is the widening a real pod owes — is that ``vllm serve`` and
``llama-server`` accept the argv these specs build. Those are their
CONTRACTS, not this code's behaviour, and the qwen3.6 activation lane owns
that leg.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterator, List

import pytest

from gen_worker import activity
from gen_worker.serving.engine_runtime import (
    EngineBootError,
    EngineCommand,
    EngineHandle,
    EngineSpec,
    LlamaServer,
    VllmServer,
    boot_engine,
    free_port,
)

# --------------------------------------------------------------------------
# The stand-in engine: an HTTP server whose boot behaviour the arm dictates.
# --------------------------------------------------------------------------

_STAND_IN = textwrap.dedent(
    '''
    import http.server, sys, threading, time

    port = int(sys.argv[1])
    # seconds of "loading" before /health starts answering
    warmup = float(sys.argv[2])
    # seconds between progress lines; 0 = say NOTHING at all
    chatter = float(sys.argv[3])
    # exit with this code instead of serving; -1 = serve
    die = int(sys.argv[4])

    if die >= 0:
        print("stand-in: refusing to serve", flush=True)
        sys.exit(die)

    ready = threading.Event()

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health" and not ready.is_set():
                self.send_error(503)
                return
            body = ("READY " + self.path).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    deadline = time.monotonic() + warmup
    while time.monotonic() < deadline:
        if chatter > 0:
            print("stand-in: loading...", flush=True)
            time.sleep(min(chatter, max(0.0, deadline - time.monotonic())))
        else:
            time.sleep(0.05)
    ready.set()
    print("stand-in: ready", flush=True)
    while True:
        time.sleep(3600)
    '''
)


@pytest.fixture(scope="module")
def stand_in(tmp_path_factory: Any) -> Path:
    path = tmp_path_factory.mktemp("engine") / "stand_in.py"
    path.write_text(_STAND_IN)
    return path


class StandIn(EngineSpec, frozen=True, kw_only=True):
    """A one-rung spec around the stand-in. ``script`` is the only thing this
    spec knows that a real one gets from the platform."""

    runtime = "stand-in"
    health_path = "/health"

    script: str = ""
    warmup_s: float = 0.0
    chatter_s: float = 0.1
    die_code: int = -1

    def ladder(self, checkpoint_dir: Path) -> List[EngineCommand]:
        port = self._port()
        return [EngineCommand(
            argv=(sys.executable, self.script, str(port), str(self.warmup_s),
                  str(self.chatter_s), str(self.die_code)),
            port=port,
        )]


class StandInLadder(StandIn, frozen=True, kw_only=True):
    """Two rungs: the first dies, the second serves — the degrade shape."""

    def ladder(self, checkpoint_dir: Path) -> List[EngineCommand]:
        bad, good = free_port(), free_port()
        return [
            EngineCommand(
                argv=(sys.executable, self.script, str(bad), "0", "0.1", "3"),
                port=bad, label="planned",
            ),
            EngineCommand(
                argv=(sys.executable, self.script, str(good), "0", "0.1", "-1"),
                port=good, label="cpu-only",
            ),
        ]


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> Iterator[List[Any]]:
    """Every typed event this test's code emits, in order.

    Introspection is the point: a boot that "worked" but reported nothing is
    a boot no operator can read, so the arms assert the PHASES as hard as
    they assert the process.
    """
    captured: List[Any] = []
    monkeypatch.setattr(activity, "_sink", captured.append)
    yield captured
    monkeypatch.setattr(activity, "_sink", None)


def _phases(events: List[Any], kind: str) -> List[str]:
    return [e.phase for e in events if e.kind == kind]


def _dead(proc: subprocess.Popen, *, within_s: float = 10.0) -> bool:
    deadline = time.monotonic() + within_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return True
        time.sleep(0.05)
    return False


# --------------------------------------------------------------------------
# 1. start -> health -> dispatch -> terminate, with the typed events
# --------------------------------------------------------------------------


def test_boot_health_dispatch_terminate(
    stand_in: Path, tmp_path: Path, events: List[Any]
) -> None:
    handle = boot_engine(
        StandIn(script=str(stand_in), warmup_s=0.5), tmp_path
    )
    try:
        assert isinstance(handle, EngineHandle)
        assert handle.alive
        # HEALTH: boot_engine returned only because /health answered 2xx.
        with urllib.request.urlopen(handle.base_url + "/health", timeout=5) as r:
            assert r.status == 200
        # DISPATCH: an entrypoint's POST target is reachable on base_url.
        with urllib.request.urlopen(handle.base_url + "/v1/chat", timeout=5) as r:
            assert r.read() == b"READY /v1/chat"
    finally:
        handle.stop()

    # TERMINATE, and the process is genuinely reaped rather than orphaned.
    assert not handle.alive
    assert _dead(handle.process)
    handle.stop()  # idempotent: the host's structural stop cannot double-kill

    phases = _phases(events, activity.KIND_ENGINE_BOOT)
    assert phases == [
        "engine_planned", "engine_started", "engine_healthy", "engine_stopped",
    ], phases
    stopped = [
        e for e in events
        if e.kind == activity.KIND_ENGINE_BOOT and e.phase == "engine_stopped"
    ]
    # HOW it stopped, not just that it did: `manner=kill` is the row an
    # operator needs, because an engine that had to be SIGKILLed is the shape
    # that strands VRAM anywhere less careful than here.
    assert "manner=term" in stopped[0].detail
    healthy = [
        e for e in events
        if e.kind == activity.KIND_ENGINE_BOOT and e.phase == "engine_healthy"
    ]
    # The boot wall is a NUMBER in the duration column, not a sentence: a
    # reader can percentile it. The stand-in warms for 0.5s, so a zero here
    # would mean the span was never measured.
    assert healthy[0].duration_ms >= 400, healthy[0].duration_ms
    assert handle.base_url in healthy[0].detail


# --------------------------------------------------------------------------
# 2. the boot ABORTS on engine death — and leaves nothing running
# --------------------------------------------------------------------------


def test_boot_aborts_when_the_engine_exits(
    stand_in: Path, tmp_path: Path, events: List[Any]
) -> None:
    with pytest.raises(EngineBootError) as excinfo:
        boot_engine(StandIn(script=str(stand_in), die_code=3), tmp_path)
    assert "exited during boot (code 3)" in str(excinfo.value)
    assert "engine_boot_failed" in _phases(events, activity.KIND_ENGINE_BOOT)
    # The failure is not a HEALTHY row with a sad detail — nothing claimed
    # readiness.
    assert "engine_healthy" not in _phases(events, activity.KIND_ENGINE_BOOT)


def test_boot_refuses_an_engine_that_cannot_exec(tmp_path: Path) -> None:
    class Missing(StandIn, frozen=True, kw_only=True):
        def ladder(self, checkpoint_dir: Path) -> List[EngineCommand]:
            port = free_port()
            return [EngineCommand(
                argv=("cozy-no-such-engine-binary",), port=port
            )]

    with pytest.raises(EngineBootError) as excinfo:
        boot_engine(Missing(script=""), tmp_path)
    assert "could not exec" in str(excinfo.value)


# --------------------------------------------------------------------------
# 3. THE DOCTRINE ARM: a boot is bounded by SILENCE, not by a clock.
#    Red and green differ ONLY in whether the child keeps talking.
# --------------------------------------------------------------------------


def test_a_silent_engine_is_wedged(stand_in: Path, tmp_path: Path) -> None:
    started = time.monotonic()
    with pytest.raises(EngineBootError) as excinfo:
        boot_engine(
            StandIn(script=str(stand_in), warmup_s=60.0, chatter_s=0.0,
                    stall_window_s=1.5),
            tmp_path,
        )
    assert "produced no output" in str(excinfo.value)
    # It gave up on SILENCE, long before the 60s the engine claimed to need.
    assert time.monotonic() - started < 20.0


def test_a_talking_engine_outlives_the_window(
    stand_in: Path, tmp_path: Path
) -> None:
    """The control for the arm above, and the whole point of the design: this
    boot takes 3s with a 1.5s window and SUCCEEDS, because every line the
    engine prints is proof it is still loading. A wall-clock budget would
    have killed it; that is the mistake this module refuses to make."""
    handle = boot_engine(
        StandIn(script=str(stand_in), warmup_s=3.0, chatter_s=0.25,
                stall_window_s=1.5),
        tmp_path,
    )
    try:
        assert handle.alive
    finally:
        handle.stop()


# --------------------------------------------------------------------------
# 4. degrade, never OOM — and the degraded rung CONFESSES
# --------------------------------------------------------------------------


def test_the_ladder_degrades_and_says_so(
    stand_in: Path, tmp_path: Path, events: List[Any]
) -> None:
    handle = boot_engine(StandInLadder(script=str(stand_in)), tmp_path)
    try:
        assert handle.alive
        assert handle.rung == "cpu-only"
    finally:
        handle.stop()

    boot = _phases(events, activity.KIND_ENGINE_BOOT)
    assert boot.count("engine_boot_failed") == 1
    assert boot.count("engine_healthy") == 1
    # Two channels on purpose (the z-image finding: report_* does not subsume
    # emit_event). The degrade is a COUNTABLE row on the quality channel, not
    # only a phase inside the boot's own story.
    degrades = [e for e in events if e.kind == activity.KIND_SERVE_DEGRADE]
    assert len(degrades) == 1
    assert degrades[0].phase == "engine_degraded"
    assert "rung=cpu-only" in degrades[0].detail


def test_every_rung_failing_raises_the_last_failure(
    stand_in: Path, tmp_path: Path
) -> None:
    class AllBad(StandInLadder, frozen=True, kw_only=True):
        def ladder(self, checkpoint_dir: Path) -> List[EngineCommand]:
            rungs = []
            for code, label in ((3, "planned"), (7, "cpu-only")):
                port = free_port()
                rungs.append(EngineCommand(
                    argv=(sys.executable, self.script, str(port),
                          "0", "0.1", str(code)),
                    port=port, label=label,
                ))
            return rungs

    with pytest.raises(EngineBootError) as excinfo:
        boot_engine(AllBad(script=str(stand_in)), tmp_path)
    # The LAST rung's failure, not the first: a caller that swallowed this
    # would serve requests against a base URL nothing is listening on.
    assert "code 7" in str(excinfo.value)
