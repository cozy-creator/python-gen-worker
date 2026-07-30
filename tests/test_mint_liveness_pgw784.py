"""pgw#784 — THE liveness proof. Paul's contract, measured on the real loop.

    "THE WORKER IS AVAILABLE THE ENTIRE TIME WHILE IT IS MINTING!"
    "Workers report every 10 seconds, and get killed after 5 or 6 heartbeat
     misses. Hard stop." (WORKER-CONTRACTS.md §1-2)

th#1299's tape: an sd15 pod entered ``self_mint_compile phase=warmup_forward``,
went 72s without an app heartbeat, and the hub terminated it mid-mint —
correctly. Read at source, that pod was never hung: at 12:30:02/03/11 it
reported again with an evidence counter advancing at 500/s. It was STARVED. The
hub's information was genuinely absent, so the fix cannot be hub patience; a
worker whose reporting its own compute can mute is a broken worker.

This file measures the thing the incident measured, on a REAL worker (real
``Lifecycle._heartbeat_loop``, real ``Executor``, real gRPC socket to the
hub-double) with a REAL mint child:

* **the green arm** — the mint in its own OS process: every beat gap stays
  inside the hub's tolerance and eager jobs complete THROUGHOUT;
* **the red arm** — the same work on the worker's loop: gaps blow past the
  6-miss window, which is what proves this instrument can see the defect.

MEASURED at Paul's literal numbers (``PGW784_REAL_CADENCE=1``, 10.00s beat,
140s mint, 60.00s kill line)::

    GREEN  beats=229  worst_gap=10.16s   eager_completed=14
    RED    beats=13   worst_gap=90.02s   eager_completed=0

The green arm's worst gap is one nominal interval plus 160ms across a mint more
than twice the kill window. The red arm reproduces th#1299 to the second — the
incident measured 72s of silence; this measures 90s — and shows the half the
incident also reported: jobs did not merely crawl, none finished at all.

A measurement worth recording, taken while authoring this (32-core box, CPython
3.12): a pure-Python GIL-holding burn in ONE thread stretches a 0.25s nominal
beat to 0.256s — no starvation at all — and it takes ~16 contending threads to
reach 1.17s (4.7x). The real incident lost 72-126s. So the live mechanism is
strictly worse than any pure-Python synthetic reproduces (inductor's codegen
and its own compile-worker processes), and that is precisely why the fix is
structural — get the compile out of the interpreter — instead of "make the
compile yield more often". It also means the red arm has to burn ON the loop to
be deterministic; a thread-shaped red arm would be flaky on this hardware and
is deliberately not asserted.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Tuple

import msgspec
import pytest

import gen_worker.lifecycle as lifecycle_mod
from gen_worker import mint_process as mp
from gen_worker.pb import worker_scheduler_pb2 as pb
from harness.hub_double import hub_double, is_ready, is_result_for

STUB_MODULE = "harness.mint_child_stub"

#: The fast arm scales the beat so the whole proof runs in seconds while the
#: RATIOS the contract is written in (a beat every interval, death at 6 misses)
#: stay exactly what they are. `PGW784_REAL_CADENCE=1` runs the same proof at
#: the literal 10s / >2min numbers.
FAST_INTERVAL_MS = 250
FAST_MINT_S = 12.0
REAL_INTERVAL_MS = 10_000
REAL_MINT_S = 140.0

#: The hub kills at 5-6 missed beats. Asserting 2 keeps a wide margin between
#: "passes" and "the hub would have killed this pod" — a proof that only just
#: cleared the kill line would not be a proof.
MAX_GAP_INTERVALS = 2.0
KILL_INTERVALS = 6.0


def _cadence() -> Tuple[int, float]:
    if os.environ.get("PGW784_REAL_CADENCE", "").strip() == "1":
        return REAL_INTERVAL_MS, REAL_MINT_S
    return FAST_INTERVAL_MS, FAST_MINT_S


def _payload(obj: object) -> bytes:
    return msgspec.msgpack.encode(obj)


def _beat_gaps(stamps: List[float]) -> List[float]:
    return [b - a for a, b in zip(stamps, stamps[1:])]


def _sample_beats(conn, until: float, stamps: List[float]) -> None:
    """Record when each StateDelta ARRIVES at the hub.

    Sampled from the test thread, which is the hub's own vantage point: the
    hub does not know when a worker intended to beat, only when a beat landed.
    """
    seen = 0
    while time.monotonic() < until:
        now = len(conn.received)
        if now > seen:
            for msg in conn.received[seen:now]:
                if msg.WhichOneof("msg") == "state_delta":
                    stamps.append(time.monotonic())
            seen = now
        time.sleep(0.02)


def _run_arm(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, function: str,
) -> Tuple[List[float], int, list]:
    """Drive one arm: start a long mint, then hammer eager jobs at it.

    Returns (beat arrival stamps, eager completions, mint outcomes).
    """
    from harness import mint_endpoints_pgw784 as endpoints

    interval_ms, mint_s = _cadence()
    monkeypatch.setattr(lifecycle_mod, "HEARTBEAT_INTERVAL_MS", interval_ms)
    monkeypatch.setattr(mp, "MINT_CHILD_MODULE", STUB_MODULE)
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(
        [str(root / "src"), str(root / "tests")]))
    monkeypatch.setenv("PGW784_WORKDIR", str(tmp_path))
    monkeypatch.setattr(endpoints, "WORKDIR", str(tmp_path))
    endpoints.reset()

    interval_s = interval_ms / 1000.0
    stamps: List[float] = []
    completed = 0

    # Two slots so the tenant tick is never merely waiting on the mint's
    # admission: whatever it does or does not complete is about the LOOP.
    with hub_double(
        modules=("harness.mint_endpoints_pgw784",), gpu_slots=2,
    ) as (sched, _h):
        conn = sched.wait_connection(0)
        conn.wait_for(is_ready, timeout=30)
        # Beats before the mint are the baseline this arm is measured against.
        _sample_beats(conn, time.monotonic() + interval_s * 3, stamps)

        conn.send(run_job=pb.RunJob(
            request_id="mint", attempt=1, function_name=function,
            input_payload=_payload(endpoints.MintIn(seconds=mint_s))))
        mint_started = time.monotonic()
        deadline = mint_started + mint_s

        # Tenant traffic for the mint's whole duration. Each tick is a REAL
        # dispatch through the real executor: a completed result is proof that
        # eager serving never stopped, not just that a timer fired.
        n = 0
        while time.monotonic() < deadline:
            n += 1
            rid = f"tick-{n}"
            conn.send(run_job=pb.RunJob(
                request_id=rid, attempt=1, function_name="eager-tick",
                input_payload=_payload(endpoints.TickIn(n=n))))
            try:
                res = conn.wait_for(
                    is_result_for(rid), timeout=max(2.0, interval_s * 8),
                ).job_result
                if res.status == pb.JOB_STATUS_OK:
                    completed += 1
            except (TimeoutError, AssertionError):
                pass  # a starved worker never answers — that IS the finding
            _sample_beats(
                conn, min(deadline, time.monotonic() + interval_s), stamps)

        # Let the mint finish and the last beats land.
        _sample_beats(conn, time.monotonic() + interval_s * 4, stamps)
        outcomes = list(endpoints.MINT_OUTCOMES)
    return stamps, completed, outcomes


def test_beats_never_miss_and_eager_serving_continues_through_a_long_mint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """THE proof. A mint far longer than the hub's kill window runs in its own
    process while this worker keeps beating and keeps serving."""
    interval_ms, mint_s = _cadence()
    interval_s = interval_ms / 1000.0
    stamps, completed, outcomes = _run_arm(
        monkeypatch, tmp_path, function="mint-out-of-process")

    assert mint_s > interval_s * KILL_INTERVALS, (
        "the mint must outlast the hub's kill window or this proves nothing")

    gaps = _beat_gaps(stamps)
    print(
        f"\n[pgw#784 GREEN] interval={interval_s:.2f}s mint={mint_s:.0f}s "
        f"beats={len(stamps)} worst_gap={max(gaps):.2f}s "
        f"(kill line {interval_s * KILL_INTERVALS:.2f}s) "
        f"eager_completed={completed}")
    assert len(stamps) >= int(mint_s / interval_s) * 0.5, (
        f"only {len(stamps)} beats arrived across a {mint_s:.0f}s mint at a "
        f"{interval_s:.2f}s cadence")
    worst = max(gaps)
    assert worst < interval_s * MAX_GAP_INTERVALS, (
        f"worst beat gap {worst:.2f}s exceeded {MAX_GAP_INTERVALS} intervals "
        f"({interval_s * MAX_GAP_INTERVALS:.2f}s); the hub kills at "
        f"{KILL_INTERVALS} ({interval_s * KILL_INTERVALS:.2f}s). Beat gaps: "
        f"{[round(g, 2) for g in gaps]}")

    # Eager serving, not just liveness: real dispatches completed throughout.
    assert completed >= 3, (
        f"only {completed} eager jobs completed during the mint — the worker "
        "must be AVAILABLE the entire time, not merely alive")

    # And the mint itself really ran in a child, for the whole window.
    assert outcomes, "the mint never reported an outcome"
    outcome = outcomes[0]
    assert outcome.status == mp.MINTED, outcome.detail
    assert outcome.elapsed_s >= mint_s * 0.5, (
        f"the child only ran {outcome.elapsed_s:.1f}s of a {mint_s:.0f}s mint "
        "— it was not the long compile this test claims to have covered")


def test_the_detector_sees_the_th1299_shape_when_the_mint_is_in_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Calibration, and the reason the green arm above is believable.

    Run the SAME work inside the serving process and the beat stops dead: gaps
    blow through the 6-miss window the hub kills on. This is th#1299 in a test
    instead of on a billed pod.
    """
    interval_ms, mint_s = _cadence()
    interval_s = interval_ms / 1000.0
    stamps, completed, _ = _run_arm(
        monkeypatch, tmp_path, function="mint-in-process")

    gaps = _beat_gaps(stamps)
    worst = max(gaps) if gaps else float("inf")
    print(
        f"\n[pgw#784 RED] interval={interval_s:.2f}s mint={mint_s:.0f}s "
        f"beats={len(stamps)} worst_gap={worst:.2f}s "
        f"(kill line {interval_s * KILL_INTERVALS:.2f}s) "
        f"eager_completed={completed}")
    assert worst > interval_s * KILL_INTERVALS, (
        f"the in-process arm's worst beat gap was only {worst:.2f}s — this "
        f"instrument cannot see a {KILL_INTERVALS}-miss stall, so the green "
        f"arm proves nothing. Beat gaps: {[round(g, 2) for g in gaps]}")
    # The other half of the incident: jobs crawled. Eager serving stops too.
    assert completed <= 2, (
        f"{completed} eager jobs completed during an in-process mint; the "
        "red arm is supposed to show serving stopping as well")
