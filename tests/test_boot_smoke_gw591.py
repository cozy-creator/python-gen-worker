"""gw#591 regression: `python -m gen_worker.entrypoint` must boot to config
validation (settings -> manifest -> cache preflight -> CUDA probe decision)
as a REAL subprocess, with no GPU and no network, and never crash with a raw
unhandled traceback. Existing test_cuda_probe.py exercises `_run_main()`
in-process with `entrypoint.Worker`/`entrypoint.probe_cuda` monkeypatched —
useful for probe-decision logic, but it never re-imports the package fresh,
so it cannot catch an import-time landmine (e.g. a grpcio/protobuf floor
mismatch against the generated pb stubs, th#766's crashloop class). A real
subprocess does.

gw#591 (0.37.5/0.38.0 boot regression) traced the actual production failure
to a host/driver CUDA-version mismatch outside gen-worker (RunPod hosts
without a CUDA-13-capable driver serving a CUDA-13-linked torch image) —
cuda_probe's clean, structured failure is *working as designed* per gw#529.
This suite locks in that "fails cleanly, never crashes" contract so a future
real import/boot regression is caught in CI before it reaches a pod.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import msgspec


def _write_manifest(path: Path, *, gpu: bool) -> None:
    manifest = {
        "functions": [
            {
                "name": "gen",
                # Deliberately unimportable: the boot-smoke contract only
                # needs the process to reach (and cleanly fail past) cuda
                # probe / module import — never a real network hello. A
                # missing module still exercises the real import machinery
                # (registry.collect_endpoints) instead of a mocked stand-in.
                "module": "gw591_nonexistent_smoke_module",
                "kind": "inference",
                "resources": {"gpu": True} if gpu else {},
            }
        ]
    }
    path.write_bytes(msgspec.toml.encode(manifest))


def _run_entrypoint(tmp_path: Path, *, gpu: bool, timeout: float = 25.0) -> subprocess.CompletedProcess[str]:
    manifest_path = tmp_path / "endpoint.lock"
    _write_manifest(manifest_path, gpu=gpu)

    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(Path(__file__).resolve().parents[1] / "src"),
        # Unroutable (RFC 5737 TEST-NET-1): if boot ever reaches the network
        # hello, connect fails fast/hangs rather than escaping to a real
        # host — the test's timeout is the backstop, but reaching here at
        # all already fails the assertions below.
        "ORCHESTRATOR_PUBLIC_ADDR": "192.0.2.1:1",
        "TENSORHUB_CACHE_DIR": str(tmp_path / "cache"),
        "ENDPOINT_LOCK_PATH": str(manifest_path),
    }

    return subprocess.run(
        [sys.executable, "-m", "gen_worker.entrypoint"],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _startup_phase_lines(output: str) -> list[dict[str, Any]]:
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


def _assert_no_unhandled_crash(result: subprocess.CompletedProcess[str], phases: list[dict[str, Any]]) -> None:
    """A raw traceback on stderr is only acceptable when it's the expected
    diagnostic `logger.exception(...)` call ahead of a matching, structured
    `worker_fatal` phase (e.g. the import-failure path below). A traceback
    with NO corresponding worker_fatal phase means the process crashed
    outside our own clean-failure contract — the th#766-class import/boot
    landmine this suite exists to catch."""
    combined = result.stdout + result.stderr
    has_raw_traceback = any(
        line.startswith("Traceback (most recent call last):") for line in combined.splitlines()
    )
    fatal = next((p for p in phases if p.get("phase") == "worker_fatal"), None)
    if has_raw_traceback:
        assert fatal is not None, (
            f"raw traceback with no structured worker_fatal phase — unhandled crash:\n{combined}"
        )


def test_boot_smoke_gpu_manifest_fails_cleanly_without_gpu(tmp_path: Path) -> None:
    """A GPU-required manifest on a GPU-less/driver-incompatible host must
    exit nonzero via the structured cuda_probe fatal path, not a crash —
    this is the exact th#874 "container exits deterministically before
    hello" signature, reproduced with zero mocking."""
    result = _run_entrypoint(tmp_path, gpu=True)
    combined = result.stdout + result.stderr

    phases = _startup_phase_lines(combined)
    _assert_no_unhandled_crash(result, phases)
    assert result.returncode == 1

    phase_names = [p.get("phase") for p in phases]
    assert "manifest_loaded" in phase_names
    assert "cache_preflight_ok" in phase_names

    fatal = next((p for p in phases if p.get("phase") == "worker_fatal"), None)
    assert fatal is not None, f"expected a worker_fatal phase; got {phase_names}"
    assert fatal.get("phase_context") == "cuda_probe"
    assert fatal.get("exit_code") == 1
    # cuda_probe's RuntimeError is synthesized (never raised), so its fatal
    # carries no raw traceback at all — a stronger signal than the generic
    # crash check above, and specific to this path.
    assert not any(
        line.startswith("Traceback (most recent call last):") for line in combined.splitlines()
    )
    # Reaching cuda_probe (and failing) proves settings/manifest/cache
    # preflight and every eager import along that path (grpc/pb included,
    # since Worker is imported at module load) succeeded cleanly, and that
    # the process never attempted the orchestrator handshake (gw#529: hello
    # is strictly gated on a passing probe) — the "Starting worker..." block
    # (which logs "Orchestrator Public Address: ...") is never reached.
    assert "Starting worker..." not in combined


def test_boot_smoke_cpu_manifest_reaches_module_import_with_no_gpu_probe(tmp_path: Path) -> None:
    """An accelerator=none manifest must skip the CUDA probe entirely (no
    torch/driver dependency at all) and fail cleanly at module import
    (the deliberately-missing user module) — never a crash, never a GPU
    touch, never a network call."""
    result = _run_entrypoint(tmp_path, gpu=False)
    combined = result.stdout + result.stderr
    phases = _startup_phase_lines(combined)

    _assert_no_unhandled_crash(result, phases)
    assert result.returncode == 1

    phase_names = [p.get("phase") for p in phases]
    assert "manifest_loaded" in phase_names
    assert "cache_preflight_ok" in phase_names
    assert "cuda_probe_ok" not in phase_names, "accelerator=none manifest must never probe CUDA"
    assert "GEN_WORKER_CUDA_PROBE_FAILED" not in combined

    fatal = next((p for p in phases if p.get("phase") == "worker_fatal"), None)
    assert fatal is not None, f"expected a worker_fatal phase; got {phase_names}"
    assert fatal.get("phase_context") == "import"
    assert fatal.get("exit_code") == 1
