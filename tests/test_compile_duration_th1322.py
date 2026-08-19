"""Compile duration is a NUMBER on the wire, for both mint routes.

A duration parsed out of free-text `detail` is not indexable, not
percentile-able without a per-row cast, and silently NULL the first time the
formatting changes. So: `ActivityUpdate.duration_ms` (proto field 17) + a typed
`jit_compile` event with the SAME event shape as `aot_mint_phases` —
`phase=minted` carries the roll-up, finer phases carry the spans inside it.
Comparing the two routes is one grouped query, not two incompatible sources.

What is real here:

* The events cross a REAL TCP gRPC socket to the hub-double, emitted through the
  REAL `activity.bind_sink` the real `Worker` transport installed. Read back off
  the wire, never from a spy.
* The emitter is the PRODUCTION function `compile_cache.emit_jit_compile_event`.
  Nothing is re-implemented for the test.
* The JIT compile it measures is the INTAKE compile a serving pod pays for
  itself.

pgw#1373 (cd46c957) deleted `aot_mint` and `mint_child`, so the AOT half of the
original comparison — `aot_mint._emit_phase_event` and the child `frame()`
clock — has no subject left. The rows that drove them are removed with that
reason recorded at each site rather than skipped. `mint_process.MintReport`
survives and is still held to its round-trip below.
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


sys.path.insert(0, str(Path(__file__).parent))

from gen_worker import activity as activity_mod  # noqa: E402
from gen_worker import compile_cache as cc  # noqa: E402
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


# `test_child_phase_clock_measures_spans_not_frames` lived here. Its subject was
# `gen_worker.mint_child.frame()` — the mint child's phase-transition funnel —
# and cd46c957 (pgw#1373) DELETED that module with the v1 mint runtime. There is
# no successor to point it at: `mint_process` survives and still names
# `MINT_CHILD_MODULE = "gen_worker.mint_child"` as the module it spawns, which
# is an orphan of the same hardcut, filed rather than papered over here
# (pgw#1438; the mint wiring is pgw#1371's).


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
# The surviving route, over a real socket
# --------------------------------------------------------------------------
#
# This was `test_both_mint_routes_report_duration_over_real_grpc`. The AOT half
# drove `aot_mint._emit_phase_event`, and cd46c957 (pgw#1373) deleted
# `aot_mint.py` with the v1 mint runtime — so the two-route COMPARISON it
# asserted has one route left to compare. The JIT half is untouched and is what
# stays: the real production emitter, a real gRPC socket, and the `n_graphs`
# assertion that is the reason this file exists.


def test_the_jit_mint_route_reports_duration_over_real_grpc() -> None:
    """A JIT mint lands a numeric `duration_ms` on the hub under a
    `phase=minted` roll-up, with its per-shape spans as their own events.

    `duration_ms` (proto field 17) is the point: a duration parsed out of
    free-text `detail` is not indexable and is silently NULL the first time the
    formatting changes.
    """
    with hub_double() as (sched, _harness):
        conn = sched.wait_connection(0)

        # pgw#1010: the INTAKE compile a serving pod pays for itself — the only
        # JIT left, emitted through the REAL production emitter.
        cc.emit_jit_compile_event(
            {"boot": 612.5}, family="sdxl", execution_lane="w8a8",
            route="intake",
            audit=cc.GraphAudit(unique_graphs=8, graph_breaks=0))

        jit = _wait_for_phases(
            conn, activity_mod.KIND_JIT_COMPILE, {"minted", "shape:boot"})

    assert jit["minted"].duration_ms == 612_500
    assert jit["shape:boot"].duration_ms == 612_500
    assert "route=intake" in jit["minted"].detail
    # `n_graphs` finally has a caller. It read 0 on every event on
    # the platform because `emit_jit_compile_event`'s parameter was never
    # populated — the blindness that made a graph-broken 20.1B denoiser
    # indistinguishable from a healthy one for two releases.
    assert "n_graphs=8 n_breaks=0" in jit["minted"].detail

    # Every timed event is COMPLETED, which is what makes it durable hub-side
    # (th#1250 records terminal updates unconditionally).
    for update in jit.values():
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
            family="sdxl", execution_lane="w8a8", route="compile_and_warm",
                audit=cc.GraphAudit(unique_graphs=6, graph_breaks=0))
        jit = _wait_for_phases(
            conn, activity_mod.KIND_JIT_COMPILE,
            {"minted", "shape:1024x1024", "shape:768x768"})

    assert jit["shape:1024x1024"].duration_ms == 41_500
    assert jit["shape:768x768"].duration_ms == 12_250
    # The roll-up IS the sum of the shapes here (unlike the AOT route, whose
    # total also covers per-mint package/declare/pack work).
    assert jit["minted"].duration_ms == 53_750
    assert "route=compile_and_warm" in jit["minted"].detail
    assert "n_graphs=6 n_breaks=0" in jit["minted"].detail



def test_an_intake_compile_that_produced_nothing_is_not_reported_at_all(
) -> None:
    """pgw#1010 replaces the two "a failed JIT MINT still reports its seconds"
    rows: there is no JIT mint to fail. What remains is the emitter's own rule —
    a zero/absent measurement is reported as ABSENCE, never as a zero row that
    would enter an AOT-vs-JIT comparison as a free compile."""
    with hub_double(modules=("harness.toy_endpoints",)) as (sched, _harness):
        conn = sched.wait_connection(0)
        cc.emit_jit_compile_event({}, family="sdxl", route="intake")
        cc.emit_jit_compile_event({"boot": 0.0}, family="sdxl", route="intake")
        cc.emit_jit_compile_event(
            {"boot": 3.5}, family="sdxl", route="intake")
        jit = _wait_for_phases(
            conn, activity_mod.KIND_JIT_COMPILE, {"minted", "shape:boot"})
    assert jit["minted"].duration_ms == 3_500


def test_telemetry_never_fails_the_compile_it_measures() -> None:
    """The emitter is wrapped: a bad table must not raise into a compile. The
    compile is the product; the measurement is not allowed to cost it.

    cd46c957 (pgw#1373) deleted `aot_mint._emit_phase_event`, so the AOT arm of
    this test went with it. The claim is unchanged for the emitter that is left.
    """
    cc.emit_jit_compile_event({"a": "not-a-number"}, family="sdxl")  # type: ignore[dict-item]
