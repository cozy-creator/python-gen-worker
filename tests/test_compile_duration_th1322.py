"""th#1322: compile duration is a NUMBER on the wire, for both mint routes.

The gap this closes (audited 2026-07-30, `BENCHMARK-TELEMETRY-COMPLETENESS.md`):

* **AOT mint** already reached the hub as a typed `aot_mint_phases` event, but
  the whole phase table arrived interpolated into the free-text `detail`. A
  measurement lane had to run `substring(detail from 'total_s.: ?([0-9.]+)')`
  to get a duration — not indexable, not percentile-able without a cast per
  row, and silently NULL the first time the formatting changes.
* **JIT mint** was worse: `compile_cache` logged `"compiled %s in %.0fs"` and
  nothing else, and a serve pod exposes no logs (pgw#760). The single number an
  AOT-vs-JIT comparison most needs did not exist off-pod in any form.

So: `ActivityUpdate.duration_ms` (proto field 17) + a typed `jit_compile` event
with the SAME event shape as `aot_mint_phases` — `phase=minted` carries the
roll-up, finer phases carry the spans inside it. Comparing the two routes is one
grouped query, not two incompatible sources.

What is real here
-----------------
* The events cross a REAL TCP gRPC socket to the hub-double, emitted through the
  REAL `activity.bind_sink` the real `Worker` transport installed — the same leg
  the pgw#789 dimension assertion uses. Read back off the wire, never from a
  spy.
* The emitters are the PRODUCTION functions: `aot_mint._emit_phase_event`,
  `compile_cache.emit_jit_compile_event`, `mint_delegate._emit_jit_compile`.
  Nothing is re-implemented for the test — the pgw#789 lesson was a module that
  was correct, unit-tested, and imported by nothing but its own test.
* The `mint_child` phase clock is driven through the real `frame()` funnel, in a
  real subprocess-free path, and the resulting `MintReport` is what the parent
  emitter consumes.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from gen_worker import activity as activity_mod  # noqa: E402
from gen_worker import compile_cache as cc  # noqa: E402
from gen_worker import mint_delegate  # noqa: E402
from gen_worker import mint_process  # noqa: E402
from gen_worker.pb import worker_scheduler_pb2 as pb  # noqa: E402
from harness.hub_double import hub_double  # noqa: E402


def _updates(conn: Any, kind: str) -> List[pb.ActivityUpdate]:
    return [
        m.activity_update for m in list(conn.received)
        if m.WhichOneof("msg") == "activity_update"
        and m.activity_update.kind == kind
    ]


def _by_phase(updates: List[pb.ActivityUpdate]) -> Dict[str, pb.ActivityUpdate]:
    return {u.phase: u for u in updates}


def _wait_for_phases(
    conn: Any, kind: str, phases: set, timeout_s: float = 15.0,
) -> Dict[str, pb.ActivityUpdate]:
    """Poll the wire until every named phase for ``kind`` has arrived.

    Bounded by a test deadline, not by a guessed sleep: the assertion is
    "these events arrive", and the failure message names what did not.
    """
    deadline = time.monotonic() + timeout_s
    seen: Dict[str, pb.ActivityUpdate] = {}
    while time.monotonic() < deadline:
        seen = _by_phase(_updates(conn, kind))
        if phases <= set(seen):
            return seen
        time.sleep(0.02)
    raise AssertionError(
        f"{kind}: missing {sorted(phases - set(seen))} on the wire "
        f"(saw {sorted(seen)})")


# --------------------------------------------------------------------------
# The wire contract
# --------------------------------------------------------------------------


def test_duration_ms_is_a_wire_field_not_prose() -> None:
    """The field exists, is field 17, and survives a serialize round trip."""
    field = pb.ActivityUpdate.DESCRIPTOR.fields_by_name["duration_ms"]
    assert field.number == 17, (
        "th#1322 took field 17; a renumber would silently mis-decode every "
        "worker already shipping the old number")
    decoded = pb.ActivityUpdate.FromString(
        pb.ActivityUpdate(
            kind=activity_mod.KIND_JIT_COMPILE,
            phase=activity_mod.PHASE_MINTED,
            duration_ms=1_234_567,
        ).SerializeToString())
    assert decoded.duration_ms == 1_234_567


def test_emit_event_never_invents_a_duration() -> None:
    """0 means 'not measured here', and a negative is clamped rather than
    written — a reader filters on `duration_ms > 0`, so a bogus value would
    read as a real (and tiny) measurement."""
    sent: List[pb.WorkerMessage] = []
    loop = asyncio.new_event_loop()

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    try:
        activity_mod.bind_sink(_send, loop)
        activity_mod.emit_event("self_mint_abort", "refused", phase="pack_failed")
        activity_mod.emit_event("x", "d", phase="p", duration_ms=-5)
        loop.run_until_complete(asyncio.sleep(0.02))
        durations = [
            m.activity_update.duration_ms for m in sent
            if m.WhichOneof("msg") == "activity_update"
        ]
        assert durations == [0, 0], durations
    finally:
        activity_mod.bind_sink(None, None)
        loop.close()


# --------------------------------------------------------------------------
# The mint_child phase clock (the JIT route's own measurement)
# --------------------------------------------------------------------------


def test_child_phase_clock_measures_spans_not_frames(capsys: Any) -> None:
    """`frame()` is the single funnel every child phase transition goes
    through, so the span table comes out of it. Repeat frames for the same
    phase (step/note updates) must NOT restart or split that phase's span —
    `warmup_forward` emits one frame per warm job."""
    child = importlib.reload(importlib.import_module("gen_worker.mint_child"))

    child.frame(phase="load", note="env seal")
    time.sleep(0.05)
    child.frame(phase="warmup_forward", step=0, total=2)
    child.frame(phase="warmup_forward", step=1, total=2, note="txt2img")
    time.sleep(0.05)
    child.frame(phase="warmup_forward", step=2, total=2, note="turbo")
    child.frame(phase="seal_publish", note="packing")
    phases = child._close_phases()

    assert set(phases) == {"load", "warmup_forward", "seal_publish"}, phases
    assert phases["load"] >= 0.05, phases
    assert phases["warmup_forward"] >= 0.05, phases
    # One span for the phase, not three: the frames are progress, the phase is
    # the span.
    capsys.readouterr()


def test_child_report_carries_the_phase_table() -> None:
    """A MintReport round-trips its phase table, so the parent reads the
    CHILD's measurement rather than deriving spans from pipe receipts."""
    import msgspec

    report = mint_process.MintReport(
        status="minted", elapsed_s=612.5,
        phases={"load": 33.1, "warmup_forward": 540.2, "seal_publish": 39.2})
    round_tripped = msgspec.json.decode(
        msgspec.json.encode(report), type=mint_process.MintReport)
    assert round_tripped.phases == report.phases
    # Absent by default, and empty rather than zero for a child that died
    # before writing one: no measurement is not a measurement of nothing.
    assert mint_process.MintReport(status="failed").phases == {}


# --------------------------------------------------------------------------
# Both routes, over a real socket, in the same shape
# --------------------------------------------------------------------------



def test_both_mint_routes_report_duration_over_real_grpc() -> None:
    """The headline assertion: an AOT mint and a JIT mint each land a numeric
    `duration_ms` on the hub, under the same `phase=minted` roll-up, with their
    per-entry / per-phase spans as their own events.

    That is what makes "AOT vs JIT mint duration" one grouped query over
    `worker_activity_events` instead of a regex over one side's prose and a grep
    of the other side's (nonexistent) pod log.
    """
    aot_mint = importlib.import_module("gen_worker.aot_mint")

    with hub_double(modules=("harness.toy_endpoints",)) as (sched, _harness):
        conn = sched.wait_connection(0)

        # ---- AOT route: the REAL _emit_phase_event over a REAL phase table.
        # The table shape is aot_mint._mint_phase_table's own output contract
        # (v1: totals + phases + per-entry timings), not an invention.
        table = {
            "v": 1,
            "n_entries": 2,
            "autotune": {"mode": "max-autotune-no-cudagraphs"},
            "inductor_configs": {"compile_threads": 8},
            "totals": {
                "export_s": 120.0, "compile_s": 900.0, "warm_s": 60.0,
                "package_s": 12.0, "declare_s": 3.0, "total_s": 1101.5,
            },
            "phases": {"inductor_compile": 900.0},
            "entries": {
                "unet_cfg": {"export_s": 80.0, "compile_s": 700.0, "warm_s": 40.0},
                "vae_decode": {"export_s": 40.0, "compile_s": 200.0, "warm_s": 20.0},
            },
        }

        class _Spec:
            family = "sdxl"

            @staticmethod
            def lane_label() -> str:
                return "w8a8"

        aot_mint._emit_phase_event(_Spec(), table)

        # ---- JIT route (fleet half — the one a hub-spawned pod runs): the REAL
        # mint_delegate emitter, fed a real MintOutcome carrying a real child
        # MintReport.
        outcome = mint_process.MintOutcome(
            status=mint_process.MINTED,
            report=mint_process.MintReport(
                status="minted", elapsed_s=612.5,
                phases={"load": 33.1, "warmup_forward": 540.2,
                        "seal_publish": 39.2}),
            elapsed_s=615.0)
        mint_delegate._emit_jit_compile(
            outcome, family="sdxl", lane="w8a8", key="ck1-" + "a" * 64,
            attempt=1)

        aot = _wait_for_phases(
            conn, aot_mint.MINT_PHASES_KIND,
            {"minted", "entry:unet_cfg", "entry:vae_decode"})
        jit = _wait_for_phases(
            conn, activity_mod.KIND_JIT_COMPILE,
            {"minted", "child:load", "child:warmup_forward",
             "child:seal_publish"})

    # The roll-ups: the numbers a comparison groups on, in a column. ONE query
    # over (kind, phase='minted') now yields both, in the same unit.
    assert aot["minted"].duration_ms == 1_101_500
    assert jit["minted"].duration_ms == 612_500

    # The per-entry / per-phase spans: Paul's "why is AOT mint so much slower
    # than JIT mint?" is answered by these, not by the totals.
    assert aot["entry:unet_cfg"].duration_ms == 820_000   # 80 + 700 + 40
    assert aot["entry:vae_decode"].duration_ms == 260_000  # 40 + 200 + 20
    # Entry spans deliberately do NOT reconstruct total_s: package/declare/pack
    # are per-mint, not per-entry, so a reader must never sum children into a
    # roll-up.
    assert (aot["entry:unet_cfg"].duration_ms
            + aot["entry:vae_decode"].duration_ms) < aot["minted"].duration_ms

    assert jit["child:warmup_forward"].duration_ms == 540_200
    assert jit["child:load"].duration_ms == 33_100

    # Every timed event is COMPLETED, which is what makes it durable hub-side
    # (th#1250 records terminal updates unconditionally).
    for update in list(aot.values()) + list(jit.values()):
        assert update.state == pb.ActivityState.ACTIVITY_STATE_COMPLETED
        assert update.duration_ms > 0, update


def test_the_producer_warm_loop_reports_its_per_shape_compile_time() -> None:
    """`compile_cache`'s own warm loops (`build`, `_compile_and_warm`) are the
    site of the log-only line th#1322 retires — `compile_cache.py:3803`,
    ``"compiled %s in %.0fs"``. The same numbers now ride the wire per shape,
    plus a roll-up.
    """
    with hub_double(modules=("harness.toy_endpoints",)) as (sched, _harness):
        conn = sched.wait_connection(0)
        cc.emit_jit_compile_event(
            {"1024x1024": 41.5, "768x768": 12.25},
            family="sdxl", lane="w8a8", route="compile_and_warm", n_graphs=6)
        jit = _wait_for_phases(
            conn, activity_mod.KIND_JIT_COMPILE,
            {"minted", "shape:1024x1024", "shape:768x768"})

    assert jit["shape:1024x1024"].duration_ms == 41_500
    assert jit["shape:768x768"].duration_ms == 12_250
    # The roll-up IS the sum of the shapes here (unlike the AOT route, whose
    # total also covers per-mint package/declare/pack work).
    assert jit["minted"].duration_ms == 53_750
    assert "route=compile_and_warm" in jit["minted"].detail
    assert "n_graphs=6" in jit["minted"].detail



def test_a_failed_jit_mint_reports_its_cost_but_not_as_minted() -> None:
    """A mint that burned 10 minutes and produced nothing spent real seconds —
    they are recorded. But they must NOT enter an AOT-vs-JIT comparison as if a
    cell came out, so the roll-up phase is `aborted`, never `minted`."""
    with hub_double(modules=("harness.toy_endpoints",)) as (sched, _harness):
        conn = sched.wait_connection(0)
        outcome = mint_process.MintOutcome(
            status=mint_process.CRASHED,
            detail="child exited 1",
            report=mint_process.MintReport(
                status="failed", detail="CUDA out of memory",
                elapsed_s=602.0, phases={"load": 30.0, "warmup_forward": 572.0}),
            exit_code=1, elapsed_s=605.0)
        mint_delegate._emit_jit_compile(
            outcome, family="sdxl", lane="w8a8", key="ck1-" + "b" * 64,
            attempt=2)
        jit = _wait_for_phases(
            conn, activity_mod.KIND_JIT_COMPILE,
            {"aborted", "child:load", "child:warmup_forward"})

    assert "minted" not in jit
    assert jit["aborted"].duration_ms == 602_000
    assert "status=crashed" in jit["aborted"].detail



def test_a_child_that_died_before_reporting_still_costs_measured_time() -> None:
    """No report at all: the parent's own wall clock is the honest cost of that
    attempt, and it is still a number on the wire rather than a lost minute."""
    with hub_double(modules=("harness.toy_endpoints",)) as (sched, _harness):
        conn = sched.wait_connection(0)
        mint_delegate._emit_jit_compile(
            mint_process.MintOutcome(
                status=mint_process.CRASHED, report=None, elapsed_s=7.5),
            family="sdxl", lane="", key="ck1-" + "c" * 64, attempt=1)
        jit = _wait_for_phases(conn, activity_mod.KIND_JIT_COMPILE, {"aborted"})

    assert jit["aborted"].duration_ms == 7_500


def test_telemetry_never_fails_the_compile_it_measures() -> None:
    """Every emitter is wrapped: a broken sink, a bad table, a report with junk
    in it — none of them may raise into a mint. The compile is the product; the
    measurement is not allowed to cost it."""
    aot_mint = importlib.import_module("gen_worker.aot_mint")

    class _Exploding:
        family = "sdxl"

        @staticmethod
        def lane_label() -> str:
            raise RuntimeError("boom")

    aot_mint._emit_phase_event(_Exploding(), {"totals": {"total_s": 1.0}})
    cc.emit_jit_compile_event({"a": "not-a-number"}, family="sdxl")  # type: ignore[dict-item]
    mint_delegate._emit_jit_compile(
        mint_process.MintOutcome(status=mint_process.MINTED, report=None),
        family="sdxl", lane="", key="k", attempt=1)
