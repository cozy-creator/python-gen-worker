"""Real ``python -m gen_worker.entrypoint`` subprocess boot harness."""

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

BLACKHOLE_ADDR = "192.0.2.1:1"


def write_manifest(path: Path, functions: List[Dict[str, Any]]) -> None:
    path.write_bytes(msgspec.toml.encode({"entrypoints": functions}))


def gpu_manifest_entry(*, module: str = "harness_smoke_nonexistent_module") -> Dict[str, Any]:
    return {
        "name": "gen", "module": module, "kind": "inference",
        "resources": {"gpu": True},
    }


def cpu_manifest_entry(*, module: str = "harness_smoke_nonexistent_module") -> Dict[str, Any]:
    return {"name": "gen", "module": module, "kind": "inference", "resources": {}}


def run_entrypoint(
    tmp_path: Path,
    *,
    functions: List[Dict[str, Any]],
    env_overrides: Optional[Dict[str, str]] = None,
    timeout: float = 25.0,
    total_budget_s: float = 300.0,
) -> subprocess.CompletedProcess[str]:
    """Boot a real entrypoint subprocess and collect its output."""
    manifest_path = tmp_path / "endpoint.lock"
    write_manifest(manifest_path, functions)

    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
        "ORCHESTRATOR_PUBLIC_ADDR": closed_port_addr(),
        "TENSORHUB_CACHE_DIR": str(tmp_path / "cache"),
        "ENDPOINT_LOCK_PATH": str(manifest_path),
        "GEN_WORKER_BOOT_RECORD": str(tmp_path / "boot-record.json"),
    }
    env.update(env_overrides or {})

    proc = subprocess.Popen(
        [sys.executable, "-m", "gen_worker.entrypoint"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    from .progress_wait import Cadence

    return _collect_until_silent(
        proc, silence_window_s=max(timeout, Cadence().floor_s),
        total_budget_s=total_budget_s,
    )


class _WentSilent(subprocess.TimeoutExpired):

    def __str__(self) -> str:
        tail = "".join((self.output or "").splitlines(keepends=True)[-3:]).strip()
        return (
            f"the boot subprocess produced no output for {self.timeout}s "
            f"(a SILENCE window, not a time budget — the process was killed). "
            f"Last output: {tail!r}"
        )


class _RanTooLong(subprocess.TimeoutExpired):

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
    combined = result.stdout + result.stderr
    has_raw_traceback = any(
        line.startswith("Traceback (most recent call last):") for line in combined.splitlines()
    )
    fatal = next((p for p in phases if p.get("phase") == "worker_fatal"), None)
    if has_raw_traceback:
        assert fatal is not None, (
            f"raw traceback with no structured worker_fatal phase — unhandled crash:\n{combined}"
        )
