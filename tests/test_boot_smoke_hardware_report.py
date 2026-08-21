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

        report_phase = next(p for p in phases if p.get("phase") == "cuda_probe_boot_fatal")
        assert report_phase.get("relayed") is True, report_phase

        servicer.wait_for_message()
        reports = [
            m.hardware_unsuitable for m in servicer.received
            if m.WhichOneof("msg") == "hardware_unsuitable"
        ]
        hw = next(
            (r for r in reports
             if r.reason_class in ("cuda_unavailable", "driver_too_old")),
            None,
        )
        assert hw is not None, [r.reason_class for r in reports]
        assert hw.worker_id == "gw619-smoke-worker"
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

    report_phase = next(p for p in phases if p.get("phase") == "cuda_probe_boot_fatal")
    assert report_phase.get("relayed") is True, report_phase
    assert "report_delivered=False" in combined
    assert elapsed < 90.0


def test_probe_failure_silent_hub_exits_on_the_report_budget_not_a_hang(
    tmp_path: Path,
) -> None:
    """The harder half of the row above: a hub that is not merely refusing but SILENT (unroutable TEST-NET-1 — SYNs vanish, nothing ever answers)."""
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

    report_phase = next(p for p in phases if p.get("phase") == "cuda_probe_boot_fatal")
    assert report_phase.get("relayed") is True, report_phase
    assert "report_delivered=False" in combined
    fatal = next((p for p in phases if p.get("phase") == "worker_fatal"), None)
    assert fatal is not None and fatal.get("phase_context") == "cuda_probe", phases


def test_probe_failure_is_terminal_no_respawn_exits_1(tmp_path: Path) -> None:
    result = run_entrypoint(
        tmp_path,
        functions=[gpu_manifest_entry()],
        env_overrides={
            "ORCHESTRATOR_PUBLIC_ADDR": closed_port_addr(),
        },
        timeout=90.0,
    )
    combined = result.stdout + result.stderr
    assert result.returncode == 1
    verdict = combined.find("reported a TERMINAL boot verdict")
    assert verdict != -1, combined
    assert "spawning compute child" not in combined[verdict:], combined
    assert "cuda_probe_boot_fatal" in combined
    assert "compute_boot_fatal" in combined


def test_probe_failure_parent_relays_the_typed_report(tmp_path: Path) -> None:
    with recording_hub() as (servicer, addr):
        result = run_entrypoint(
            tmp_path,
            functions=[gpu_manifest_entry()],
            env_overrides={
                "ORCHESTRATOR_PUBLIC_ADDR": addr,
                "WORKER_ID": "pgw826-split-worker",
            },
        )
        assert result.returncode == 1
        servicer.wait_for_message()
        reports = [
            m.hardware_unsuitable for m in servicer.received
            if m.WhichOneof("msg") == "hardware_unsuitable"
        ]
        hw = next(
            (r for r in reports
             if r.reason_class in ("cuda_unavailable", "driver_too_old")),
            None,
        )
        assert hw is not None, [r.reason_class for r in reports]
        assert hw.worker_id == "pgw826-split-worker"
        assert hw.detail
