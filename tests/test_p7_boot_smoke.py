"""P7 (th#960/pgw#609 design table): ``python -m gen_worker.entrypoint`` as a
real subprocess, table-driven over ``tests/harness/subprocess_runner.py``
(gw#591 pattern, kept per the design: "already greenfield-shaped"). No GPU,
no network. See ``tests/test_boot_smoke_gw591.py`` for the original
same-assertion suite this harness was extracted from (kept, not deleted
this phase); this file proves the extraction preserves behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness.subprocess_runner import (
    assert_no_unhandled_crash,
    cpu_manifest_entry,
    gpu_manifest_entry,
    run_entrypoint,
    startup_phase_lines,
)


def test_gpu_manifest_fails_cleanly_without_gpu(tmp_path: Path) -> None:
    """th#874 signature: a GPU-required manifest on a GPU-less/driver-
    incompatible host exits nonzero via the structured cuda_probe fatal
    path, never a crash, never reaching the network hello."""
    result = run_entrypoint(tmp_path, functions=[gpu_manifest_entry()])
    combined = result.stdout + result.stderr
    phases = startup_phase_lines(combined)
    assert_no_unhandled_crash(result, phases)
    assert result.returncode == 1

    phase_names = [p.get("phase") for p in phases]
    assert "manifest_loaded" in phase_names
    assert "cache_preflight_ok" in phase_names

    fatal = next((p for p in phases if p.get("phase") == "worker_fatal"), None)
    assert fatal is not None, f"expected a worker_fatal phase; got {phase_names}"
    assert fatal.get("phase_context") == "cuda_probe"
    assert fatal.get("exit_code") == 1
    assert not any(
        line.startswith("Traceback (most recent call last):") for line in combined.splitlines()
    )
    assert "Starting worker..." not in combined


def test_cpu_manifest_reaches_module_import_with_no_gpu_probe(tmp_path: Path) -> None:
    """An accelerator=none manifest skips CUDA probing entirely and fails
    cleanly at module import (the deliberately-missing user module) — never
    a crash, never a GPU touch, never a network call."""
    result = run_entrypoint(tmp_path, functions=[cpu_manifest_entry()])
    combined = result.stdout + result.stderr
    phases = startup_phase_lines(combined)
    assert_no_unhandled_crash(result, phases)
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


@pytest.mark.parametrize("bad_manifest", [
    [{"name": "gen", "kind": "not-a-real-kind", "module": "nope"}],
    [],
])
def test_malformed_or_empty_manifest_fails_cleanly(tmp_path: Path, bad_manifest) -> None:
    """A malformed or empty functions list must never crash the process —
    boot fails structured, same contract as the GPU/CPU rows above."""
    result = run_entrypoint(tmp_path, functions=bad_manifest)
    combined = result.stdout + result.stderr
    phases = startup_phase_lines(combined)
    assert_no_unhandled_crash(result, phases)
    assert result.returncode != 0
