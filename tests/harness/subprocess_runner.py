"""Real ``python -m gen_worker.entrypoint`` subprocess boot harness.

Extracted from ``tests/test_boot_smoke_gw591.py`` (gw#591) per th#960/pgw#609:
a real subprocess re-imports the package fresh, catching import-time
landmines an in-process ``entrypoint._run_main()`` call cannot (th#766 class).
No GPU, no network — an unroutable TEST-NET-1 address is the hello target so
any escape past cache/cuda-probe preflight fails fast instead of hanging.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

import msgspec

from gen_worker.stall import SilenceWindow

from .hardware_report_hub import closed_port_addr

#: An unroutable RFC 5737 TEST-NET-1 address: a dial to it is SILENT, so the
#: caller pays the boot's full hardware-report budget (~7s: two 3s RPC deadlines
#: plus backoff) before the process exits. That is the right stimulus for a test
#: whose subject is "an unreachable, silent hub does not hang the boot", and
#: pure waste for one whose subject is the startup phase contract — pgw#796
#: measured it at 7.0s of each 13.9s boot, paid three times over.
BLACKHOLE_ADDR = "192.0.2.1:1"


def write_manifest(path: Path, functions: List[Dict[str, Any]]) -> None:
    path.write_bytes(msgspec.toml.encode({"functions": functions}))


def gpu_manifest_compiled_graph(*, module: str = "harness_smoke_nonexistent_module") -> Dict[str, Any]:
    return {
        "name": "gen", "module": module, "kind": "inference",
        "resources": {"gpu": True},
    }


def cpu_manifest_compiled_graph(*, module: str = "harness_smoke_nonexistent_module") -> Dict[str, Any]:
    return {"name": "gen", "module": module, "kind": "inference", "resources": {}}


def run_entrypoint(
    tmp_path: Path,
    *,
    functions: List[Dict[str, Any]],
    env_overrides: Optional[Dict[str, str]] = None,
    timeout: float = 25.0,
    total_budget_s: float = 300.0,
) -> subprocess.CompletedProcess[str]:
    """Boot a real entrypoint subprocess and collect its output.

    pgw#795: ``timeout`` is a SILENCE window, not a total budget. It was a
    ``subprocess.run(timeout=...)`` wall clock and it failed a full-suite run at
    25s on a loaded box — a boot that is merely slow is not a boot that is
    wedged, and the entrypoint says so continuously (``worker.startup.phase``
    lines throughout). So a boot that keeps talking runs as long as it needs,
    and only silence gives up. Same rule ``gen_worker.stall`` gives production
    code, and the same reason: on a shared runner a fixed budget decides the
    machine's speed, not the code's behaviour.

    pgw#826: ``total_budget_s`` is the COEXISTING backstop the silence window
    cannot be — a crash LOOP is never silent (every respawn prints), so
    without a total-runtime cap a looping boot presents as liveness and burns
    the whole CI job. Generous (order-of-magnitude over a slow boot), never a
    speed judgment.
    """
    manifest_path = tmp_path / "endpoint.lock"
    write_manifest(manifest_path, functions)

    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
        # Nothing is listening here (bind, read port, close): any escape to the
        # network hello is REFUSED instantly, and reaching it at all already
        # fails the assertions below. Callers whose subject is a silent hub pass
        # ``BLACKHOLE_ADDR`` explicitly.
        "ORCHESTRATOR_PUBLIC_ADDR": closed_port_addr(),
        "TENSORHUB_CACHE_DIR": str(tmp_path / "cache"),
        "ENDPOINT_LOCK_PATH": str(manifest_path),
        # gw#640: the supervisor's boot record must not be shared between runs
        # (its default is a fixed container-local path), or one boot reports
        # the previous one's death and pays the report budget for it.
        "GEN_WORKER_BOOT_RECORD": str(tmp_path / "boot-record.json"),
    }
    env.update(env_overrides or {})

    proc = subprocess.Popen(
        [sys.executable, "-m", "gen_worker.entrypoint"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    # A caller's number is a FLOOR on tolerated silence, never a cap: it can
    # only make the wait more patient. Measured why (pgw#795 round 4): on a
    # contended box a starved boot went 25s without printing, and 25.0 was the
    # number the call site happened to carry.
    from .progress_wait import Cadence

    return _collect_until_silent(
        proc, silence_window_s=max(timeout, Cadence().floor_s),
        total_budget_s=total_budget_s,
    )


class _WentSilent(subprocess.TimeoutExpired):
    """The child stopped SAYING anything — not "the child ran too long".

    Subclasses ``TimeoutExpired`` so existing handlers keep working, but says
    what actually happened: ``TimeoutExpired``'s stock message ("timed out
    after 25.0 seconds") reads as a total budget and sent this lane looking for
    a slow boot rather than a silent one.
    """

    def __str__(self) -> str:
        tail = "".join((self.output or "").splitlines(keepends=True)[-3:]).strip()
        return (
            f"the boot subprocess produced no output for {self.timeout}s "
            f"(a SILENCE window, not a time budget — the process was killed). "
            f"Last output: {tail!r}"
        )


class _RanTooLong(subprocess.TimeoutExpired):
    """pgw#826: the child kept TALKING past the total-runtime cap — the shape
    of a crash loop, which no silence window can end (every respawn prints)."""

    def __str__(self) -> str:
        tail = "".join((self.output or "").splitlines(keepends=True)[-3:]).strip()
        return (
            f"the boot subprocess was still running (and talking) after "
            f"{self.timeout}s total — a crash loop presents as liveness, so the "
            f"total-runtime cap killed it. Last output: {tail!r}"
        )


def _collect_until_silent(
    proc: "subprocess.Popen[str]", *, silence_window_s: float,
    total_budget_s: float = 300.0,
) -> "subprocess.CompletedProcess[str]":
    import time as _time

    window = SilenceWindow(silence_window_s)
    started = _time.monotonic()
    chunks: Dict[str, List[str]] = {"out": [], "err": []}

    def _drain(stream: Any, key: str) -> None:
        for line in iter(stream.readline, ""):
            chunks[key].append(line)
            window.touch()
        stream.close()

    readers = [
        threading.Thread(target=_drain, args=(proc.stdout, "out"), daemon=True),
        threading.Thread(target=_drain, args=(proc.stderr, "err"), daemon=True),
    ]
    for reader in readers:
        reader.start()
    while True:
        try:
            proc.wait(timeout=0.25)
            break
        except subprocess.TimeoutExpired:
            stalled = window.stalled()
            ran_too_long = _time.monotonic() - started > total_budget_s
            if not stalled and not ran_too_long:
                continue
            proc.kill()
            proc.wait()
            for reader in readers:
                reader.join(timeout=5.0)
            exc = _WentSilent if stalled else _RanTooLong
            raise exc(
                proc.args, silence_window_s if stalled else total_budget_s,
                output="".join(chunks["out"]), stderr="".join(chunks["err"]),
            ) from None
    for reader in readers:
        reader.join(timeout=5.0)
    return subprocess.CompletedProcess(
        proc.args, proc.returncode,
        "".join(chunks["out"]), "".join(chunks["err"]),
    )


def startup_phase_lines(output: str) -> List[Dict[str, Any]]:
    phases = []
    for line in output.splitlines():
        idx = line.find("worker.startup.phase ")
        if idx == -1:
            idx = line.find("worker.fatal ")
            if idx == -1:
                continue
            idx += len("worker.fatal ")
        else:
            idx += len("worker.startup.phase ")
        try:
            phases.append(json.loads(line[idx:]))
        except (ValueError, json.JSONDecodeError):
            continue
    return phases


def assert_no_unhandled_crash(
    result: "subprocess.CompletedProcess[str]", phases: List[Dict[str, Any]],
) -> None:
    """A raw traceback on stderr is only acceptable ahead of a matching
    structured ``worker_fatal`` phase; otherwise the process crashed outside
    the clean-failure contract (th#766-class import/boot landmine)."""
    combined = result.stdout + result.stderr
    has_raw_traceback = any(
        line.startswith("Traceback (most recent call last):") for line in combined.splitlines()
    )
    fatal = next((p for p in phases if p.get("phase") == "worker_fatal"), None)
    if has_raw_traceback:
        assert fatal is not None, (
            f"raw traceback with no structured worker_fatal phase — unhandled crash:\n{combined}"
        )
