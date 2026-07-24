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
from pathlib import Path
from typing import Any, Dict, List, Optional

import msgspec


def write_manifest(path: Path, functions: List[Dict[str, Any]]) -> None:
    path.write_bytes(msgspec.toml.encode({"functions": functions}))


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
) -> subprocess.CompletedProcess[str]:
    manifest_path = tmp_path / "endpoint.lock"
    write_manifest(manifest_path, functions)

    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(Path(__file__).resolve().parents[2] / "src"),
        # Unroutable (RFC 5737 TEST-NET-1): any escape to the network hello
        # connects fail-fast/hangs rather than reaching a real host — the
        # timeout is the backstop, but reaching here at all already fails
        # the assertions below.
        "ORCHESTRATOR_PUBLIC_ADDR": "192.0.2.1:1",
        "TENSORHUB_CACHE_DIR": str(tmp_path / "cache"),
        "ENDPOINT_LOCK_PATH": str(manifest_path),
        # gw#640: the supervisor's boot record must not be shared between runs
        # (its default is a fixed container-local path), or one boot reports
        # the previous one's death and pays the report budget for it.
        "GEN_WORKER_BOOT_RECORD": str(tmp_path / "boot-record.json"),
    }
    env.update(env_overrides or {})

    return subprocess.run(
        [sys.executable, "-m", "gen_worker.entrypoint"],
        env=env, capture_output=True, text=True, timeout=timeout,
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
