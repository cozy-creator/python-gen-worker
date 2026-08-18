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

        # The probe fails in the compute child, which holds no
        # credential — it hands the typed report to the control parent, which
        # relays it to the hub and exits 1.
        report_phase = next(p for p in phases if p.get("phase") == "cuda_probe_boot_fatal")
        assert report_phase.get("relayed") is True, report_phase

        # No explicit bound — this message MUST arrive, and the boot
        # that produces it was measured taking 25.11s on a loaded runner while
        # 5.0 was the number here.
        servicer.wait_for_message()
        reports = [
            m.hardware_unsuitable for m in servicer.received
            if m.WhichOneof("msg") == "hardware_unsuitable"
        ]
        # This box's own torch/driver mismatch reproduces the real th#591/
        # th#979 signature end to end — no mocking needed.
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

    # The child's verdict reaches the parent, whose relay to
    # the unreachable hub fails — and the parent still exits 1, no respawn.
    report_phase = next(p for p in phases if p.get("phase") == "cuda_probe_boot_fatal")
    assert report_phase.get("relayed") is True, report_phase
    assert "report_delivered=False" in combined
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

    this property used to be covered by accident — every
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

    # The parent's relay to the blackholed hub is what the
    # report budget must bound; it fails undelivered and the pod still ends.
    report_phase = next(p for p in phases if p.get("phase") == "cuda_probe_boot_fatal")
    assert report_phase.get("relayed") is True, report_phase
    assert "report_delivered=False" in combined
    fatal = next((p for p in phases if p.get("phase") == "worker_fatal"), None)
    assert fatal is not None and fatal.get("phase_context") == "cuda_probe", phases


def test_probe_failure_is_terminal_no_respawn_exits_1(tmp_path: Path) -> None:
    """pgw#826 regression, the exact shape that wedged the 0.85.0 cut: the
    compute child's CUDA probe failure must be TERMINAL — the parent exits 1
    instead of respawning a fresh child every ~55s forever. The harness's
    total-runtime cap makes a regressed crash loop fail fast rather than burn
    the CI job (no silence window can end it: every respawn prints)."""
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
    # A hardware verdict is never crash-to-retry: NOTHING respawns after it.
    #
    # Asserting a single TOTAL spawn instead was over-strict and did not test
    # this property — any unrelated pre-Hello crash breaks it while pgw#826
    # holds perfectly. Measured on the 0.91.0 promotion: child #1 aborted at
    # 0.8s with SIGABRT out of gRPC's fork handler ("Other threads are
    # currently calling into gRPC" / "Epoll1Poller ... Bad file descriptor")
    # before the CUDA probe ever ran. The parent rightly respawned that
    # pre-Hello crash; child #2 then reached the probe, relayed the terminal
    # verdict, and the parent exited 1 without respawning — the invariant this
    # test names, passing, while the old assertion failed the promotion.
    verdict = combined.find("reported a TERMINAL boot verdict")
    assert verdict != -1, combined
    assert "spawning compute child" not in combined[verdict:], combined
    assert "cuda_probe_boot_fatal" in combined
    assert "compute_boot_fatal" in combined


def test_probe_failure_parent_relays_the_typed_report(tmp_path: Path) -> None:
    """pgw#826: the child holds no credential, so the PARENT must deliver the
    typed HardwareUnsuitable report to the hub before exiting 1."""
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
        # The parent also dials worker_fatal-carrier postmortems on the same
        # stream shape; the typed hardware verdict must be among them.
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
