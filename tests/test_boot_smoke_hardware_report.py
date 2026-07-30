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
    BLACKHOLE_ADDR,
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
        timeout=90.0,
    )
    elapsed = time.monotonic() - start
    combined = result.stdout + result.stderr
    phases = startup_phase_lines(combined)
    assert_no_unhandled_crash(result, phases)
    assert result.returncode == 1

    report_phase = next(p for p in phases if p.get("phase") == "cuda_probe_hardware_report")
    assert report_phase.get("delivered") is False, report_phase
    # The silent-exit fallback must still fire — a refused connection must not
    # turn into a multi-minute pod-billing hang.
    #
    # pgw#796 RAISED this from 15.0s after measuring the boot at 14.74s and
    # 14.92s on a loaded box (this box, load avg 15). pgw#795's own taxonomy
    # admits a hang bound only "with an order of magnitude of headroom", and
    # 15.0 over a ~14.9s observation is a coin flip, not a bound — the same
    # shape, and the same release-blocking failure mode, as the 15s deadline
    # that stopped v0.78.0. What is being excluded is MINUTES; say minutes.
    assert elapsed < 90.0


def test_probe_failure_silent_hub_exits_on_the_report_budget_not_a_hang(
    tmp_path: Path,
) -> None:
    """The harder half of the row above: a hub that is not merely refusing but
    SILENT (unroutable TEST-NET-1 — SYNs vanish, nothing ever answers).

    A refused connect returns an error immediately; the boot only has to not
    mishandle it. A blackhole is the shape that actually hangs things, and the
    only thing that ends it is the report path's own attempt/backoff budget.
    ``run_entrypoint``'s ``timeout`` is the hang bound: if the budget ever stops
    bounding the dial, this fails as a timeout rather than passing quietly.

    pgw#796: this property used to be covered by accident — every
    ``run_entrypoint`` caller dialled the blackhole and paid ~7s for it, three
    times in ``test_p7_boot_smoke`` alone, where the subject is the phase
    contract and the hub is irrelevant. It is asserted once, here, on purpose.
    """
    result = run_entrypoint(
        tmp_path,
        functions=[gpu_manifest_entry()],
        env_overrides={"ORCHESTRATOR_PUBLIC_ADDR": BLACKHOLE_ADDR},
        timeout=60.0,
    )
    combined = result.stdout + result.stderr
    phases = startup_phase_lines(combined)
    assert_no_unhandled_crash(result, phases)
    assert result.returncode == 1

    report_phase = next(p for p in phases if p.get("phase") == "cuda_probe_hardware_report")
    assert report_phase.get("delivered") is False, report_phase
    fatal = next((p for p in phases if p.get("phase") == "worker_fatal"), None)
    assert fatal is not None and fatal.get("phase_context") == "cuda_probe", phases
