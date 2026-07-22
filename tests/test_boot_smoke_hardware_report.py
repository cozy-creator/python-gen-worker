"""gw#619/th#988 end-to-end: a REAL ``python -m gen_worker.entrypoint``
subprocess whose boot-time CUDA probe fails must dial a real hub socket with
a HardwareUnsuitable report before exiting — the exact wire path production
pods take, not an in-process shortcut. Extends the gw#591 boot-smoke
contract (tests/test_boot_smoke_gw591.py): probe failure still exits 1
cleanly, and now also reports why first.
"""

from __future__ import annotations

import time
from pathlib import Path

from harness.hardware_report_hub import closed_port_addr, recording_hub
from harness.subprocess_runner import (
    assert_no_unhandled_crash,
    gpu_manifest_entry,
    run_entrypoint,
    startup_phase_lines,
)


def test_probe_failure_boot_dials_hub_sends_report_and_exits_cleanly(tmp_path: Path) -> None:
    with recording_hub() as (servicer, addr):
        result = run_entrypoint(
            tmp_path,
            functions=[gpu_manifest_entry()],
            env_overrides={
                "ORCHESTRATOR_PUBLIC_ADDR": addr,
                "WORKER_ID": "gw619-smoke-worker",
            },
        )
        combined = result.stdout + result.stderr
        phases = startup_phase_lines(combined)
        assert_no_unhandled_crash(result, phases)
        assert result.returncode == 1

        phase_names = [p.get("phase") for p in phases]
        assert "cuda_probe_hardware_report" in phase_names, phase_names
        report_phase = next(p for p in phases if p.get("phase") == "cuda_probe_hardware_report")
        assert report_phase.get("delivered") is True, report_phase

        msg = servicer.wait_for_message(timeout=5.0)
        assert msg.WhichOneof("msg") == "hardware_unsuitable"
        hw = msg.hardware_unsuitable
        assert hw.worker_id == "gw619-smoke-worker"
        # This box's own torch/driver mismatch reproduces the real th#591/
        # th#979 signature end to end — no mocking needed.
        assert hw.reason_class in ("cuda_unavailable", "driver_too_old")
        assert hw.detail
        assert hw.torch_version


def test_probe_failure_hub_unreachable_still_exits_without_hanging(tmp_path: Path) -> None:
    start = time.monotonic()
    result = run_entrypoint(
        tmp_path,
        functions=[gpu_manifest_entry()],
        env_overrides={"ORCHESTRATOR_PUBLIC_ADDR": closed_port_addr()},
        timeout=20.0,
    )
    elapsed = time.monotonic() - start
    combined = result.stdout + result.stderr
    phases = startup_phase_lines(combined)
    assert_no_unhandled_crash(result, phases)
    assert result.returncode == 1

    report_phase = next(p for p in phases if p.get("phase") == "cuda_probe_hardware_report")
    assert report_phase.get("delivered") is False, report_phase
    # The silent-exit fallback must still fire promptly — a refused
    # connection must not turn into a multi-minute pod-billing hang.
    assert elapsed < 15.0
