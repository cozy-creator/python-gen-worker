"""pgw#797: the boot span ladder, asserted from the REAL boot sequence.

WHY THIS FILE EXISTS AND WHY IT LOOKS LIKE THIS
-----------------------------------------------
pgw#789 shipped `weights_fetch`, `pipeline_load` and `warm_complete`, its test
suite passed, and on gen-worker 0.78.0 in production those spans emitted
NOTHING. Three real hub-spawned L4 boots on chaos recorded two rows each —
`first_request_servable` and `hello`, in that order, the boot closing 4s before
the stream to the hub even existed.

The reason the tests missed it is visible in `test_benchmark_telemetry_pgw789.py`:
every boot assertion calls the emitters directly —
`boot_phases.span(PHASE_PIPELINE_LOAD, ...)`, `boot_phases.mark(...)` — or
constructs a `Lifecycle` and calls one method on it. That proves the RECORDER
works. It cannot notice that the real boot never reaches the recorder, which was
the actual defect: a spec declaring `Compile` or `Slot` is routed to `dynamic`
in `Lifecycle.startup()` and set up later from hub delivery, so it never passed
either instrumented call site, while the milestone that CLOSES the boot fired at
the end of `startup()` — which `Worker.arun` runs concurrently with the
transport — and `in_boot()` then suppressed every span that came after.

So this file drives the real entrypoint sequence and reads boot rows OFF THE
WIRE. `harness.hub_double` runs an actual `Worker` against an actual gRPC
server; the hub stamps a Hot `DesiredInstance` exactly as the orchestrator does;
weights are real bytes over a real blob host. Nothing here calls a boot emitter.
If the production path stops reaching an emitter again, these go red.

The endpoint is deliberately a SLOT endpoint: `spec.slots or spec.compile is not
None` is the one branch that made the released ladder empty, so the regression
test has to boot on that branch and not the plain one.
"""

from __future__ import annotations

from typing import Dict, List

import pytest

from gen_worker import boot_phases
from gen_worker.api.binding import wire_ref
from gen_worker.pb import worker_scheduler_pb2 as pb

from harness.blob_host import BlobHost
from harness.hub_double import hub_double, is_ready
from harness.boot_ladder_endpoints_pgw797 import COMPILE_MODEL


# --------------------------------------------------------------------------
# reading the ladder off the wire
# --------------------------------------------------------------------------

def _boot_rows(conn) -> List[pb.BootPhase]:
    return [
        m.boot_phase for m in list(conn.received)
        if m.WhichOneof("msg") == "boot_phase"
    ]


def _terminal(rows: List[pb.BootPhase], phase: str) -> List[pb.BootPhase]:
    return [r for r in rows if r.phase == phase and r.terminal]


def _one(rows: List[pb.BootPhase], phase: str) -> pb.BootPhase:
    got = _terminal(rows, phase)
    assert got, (
        f"no terminal {phase!r} row on the wire — the ladder has: "
        f"{sorted({r.phase for r in rows})}"
    )
    return got[0]


def _detail(row: pb.BootPhase) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for tok in row.detail.split():
        if "=" in tok:
            k, _, v = tok.partition("=")
            out[k] = v
    return out


@pytest.fixture(scope="module")
def ladder(tmp_path_factory) -> List[pb.BootPhase]:
    """ONE real hub-delivered boot; every assertion below reads its ladder.

    One boot rather than one per test on purpose. `boot_phases` state is
    process-global (BOOT_ID is generated at import, `_rows` is a module list),
    so a second boot in the same interpreter would start with `in_boot()`
    already False and record nothing — the very failure under test, arriving as
    a test artifact. Booting once, off one wire capture, is also what production
    gives a reader: one boot's ladder, whole.
    """
    boot_phases.reset_for_tests()
    tmp_path = tmp_path_factory.mktemp("pgw797")
    blobs = BlobHost(tmp_path)
    model_ref = wire_ref(COMPILE_MODEL)
    model_snap = blobs.one_file_snapshot(
        "snap-model", "model", b"pgw797-compile-model-bytes" * 512)
    try:
        with hub_double(
            modules=("harness.boot_ladder_endpoints_pgw797",),
        ) as (scheduler, _harness):
            conn = scheduler.wait_connection(0)
            conn.wait_for(is_ready)
            conn.send(hello_ack=pb.HelloAck(
                protocol_version=pb.PROTOCOL_VERSION_CURRENT,
                desired_residency=pb.DesiredResidency(
                    generation=1,
                    disk_refs=[model_ref],
                    snapshots={model_ref: model_snap},
                    hot=[pb.DesiredInstance(
                        function_name="compile-echo",
                        models=[pb.ModelBinding(slot="model", ref=model_ref)],
                    )],
                ),
            ))
            # Advertising the function IS the servable fact, so the boot close
            # must follow it. Waiting for the close (not a clock) is what makes
            # this deterministic.
            conn.wait_for(
                lambda m: m.WhichOneof("msg") == "state_delta"
                and "compile-echo" in m.state_delta.available_functions
            )
            conn.wait_for(
                lambda m: m.WhichOneof("msg") == "boot_phase"
                and m.boot_phase.phase == boot_phases.PHASE_FIRST_REQUEST_SERVABLE
                and m.boot_phase.terminal
            )
            rows = _boot_rows(conn)
    finally:
        blobs.shutdown()
    return rows


# --------------------------------------------------------------------------
# the defect: the released ladder was two rows, in the wrong order
# --------------------------------------------------------------------------

def test_a_real_slot_boot_emits_the_whole_span_ladder(ladder) -> None:
    """On 0.78.0 this produced exactly `first_request_servable` and `hello` and
    nothing else. Every phase below is one the released SDK claimed to emit."""
    rows = ladder
    phases = sorted({r.phase for r in rows if r.terminal})
    for required in (
        boot_phases.PHASE_HELLO,
        boot_phases.PHASE_WEIGHTS_FETCH,
        boot_phases.PHASE_PIPELINE_LOAD,
        boot_phases.PHASE_FIRST_REQUEST_SERVABLE,
    ):
        assert required in phases, (
            f"{required} absent from a REAL hub-delivered boot; got {phases}"
        )


def test_the_boot_close_never_precedes_hello(ladder) -> None:
    """The 0.78.0 inversion, as a regression.

    Both rows are cumulative off the same origin, so `servable - hello` is a
    phase of the boot and cannot be negative. Live on chaos it was: servable
    8,951ms at ordinal 1, hello 13,165ms at ordinal 2 — the boot closed before
    the worker had a stream to close it on.
    """
    rows = ladder
    hello = _one(rows, boot_phases.PHASE_HELLO)
    servable = _one(rows, boot_phases.PHASE_FIRST_REQUEST_SERVABLE)
    assert hello.cumulative and servable.cumulative
    assert servable.ordinal > hello.ordinal, (
        "the boot close was RECORDED before hello "
        f"(servable ordinal {servable.ordinal}, hello {hello.ordinal})"
    )
    assert servable.process_uptime_ms >= hello.process_uptime_ms, (
        "servable precedes hello on the shared process-start clock: "
        f"{servable.process_uptime_ms}ms < {hello.process_uptime_ms}ms"
    )


def test_the_boot_close_is_last_so_no_span_is_suppressed(ladder) -> None:
    """`in_boot()` gates span recording, and it closes at the servable
    milestone — so a premature close does not merely mislabel one row, it
    silently deletes the rest of the ladder. Assert the close is last."""
    rows = ladder
    servable = _one(rows, boot_phases.PHASE_FIRST_REQUEST_SERVABLE)
    later = [
        r for r in rows
        if r.ordinal > servable.ordinal
        and r.phase not in (boot_phases.PHASE_WARM_COMPLETE,)
    ]
    assert not later, (
        "phases recorded after the boot closed (they would have been "
        f"suppressed by in_boot()): {[r.phase for r in later]}"
    )


def test_cumulative_milestones_are_never_a_span_s_child(ladder) -> None:
    """A milestone measured from process start is not part of any span, and a
    reader subtracts a child's duration from its parent's exclusive time — so a
    whole-boot number parented onto an 870ms `pipeline_load` zeroes that phase.
    Observed live while authoring this: the servable mark rides a StateDelta
    task created inside the setup span and inherited its context."""
    for row in ladder:
        if row.cumulative:
            assert row.parent_ordinal == 0, (
                f"cumulative {row.phase} is a child of ordinal "
                f"{row.parent_ordinal}"
            )


def test_weights_fetch_carries_bytes_and_source(ladder) -> None:
    fetches = _terminal(ladder, boot_phases.PHASE_WEIGHTS_FETCH)
    assert fetches, "no weights_fetch span on a boot that fetched weights"
    assert any(f.bytes > 0 for f in fetches), (
        "weights_fetch reported no bytes; a fetch rate cannot be derived "
        f"from {[(f.ref, f.bytes, f.source) for f in fetches]}"
    )
    assert any(f.source for f in fetches), (
        "weights_fetch named no source — a cold pull and a warm volume hit "
        "are the same phase with wildly different rates"
    )


# --------------------------------------------------------------------------
# pgw#797 proper: the warmup split
# --------------------------------------------------------------------------

def test_warmup_is_a_span_nested_inside_pipeline_load(ladder) -> None:
    """`pipeline_load` was load+warmup as one number. The split has to nest, or
    the ladder stops reconciling (both would charge the same seconds)."""
    rows = ladder
    load = _one(rows, boot_phases.PHASE_PIPELINE_LOAD)
    warms = _terminal(rows, boot_phases.PHASE_WARMUP)
    assert warms, "no warmup span — warmup is still swallowed by pipeline_load"
    nested = [w for w in warms if w.parent_ordinal == load.ordinal]
    assert nested, (
        f"warmup {[ (w.ordinal, w.parent_ordinal) for w in warms ]} is not "
        f"a child of pipeline_load (ordinal {load.ordinal})"
    )
    assert nested[0].duration_ms <= load.duration_ms, (
        "a nested warmup cannot be longer than the load that contains it"
    )
    # The POINT of the split: `pipeline_load` reads as weights->VRAM once its
    # warmup children are subtracted. Before pgw#797 it was load+warmup as one
    # number and this quantity did not exist.
    load_only = load.duration_ms - sum(w.duration_ms for w in nested)
    assert load_only >= 0
    servable = _one(rows, boot_phases.PHASE_FIRST_REQUEST_SERVABLE)
    assert f"class.load={load_only}" in servable.detail, (
        "the reconciliation does not charge the warmup to the CHILD: expected "
        f"class.load={load_only} in {servable.detail!r}"
    )


def test_a_warmup_row_says_whether_a_cell_was_armed(ladder) -> None:
    """"Warmup before arm" and "warmup after arm" must be separable ROWS —
    otherwise they are two code paths that happen to share a name and the
    "what does a cell save" question stays unanswerable."""
    warms = _terminal(ladder, boot_phases.PHASE_WARMUP)
    assert warms
    for w in warms:
        tags = _detail(w)
        assert "armed" in tags, f"warmup row does not say armed=: {w.detail!r}"
        assert tags["armed"] in ("0", "1")


def test_per_iteration_warm_cost_is_recorded(ladder) -> None:
    """The first forward dominates (allocator growth, CUDA module load, algo
    selection). A total without the sequence cannot say what a cell removes."""
    rows = ladder
    warms = _terminal(rows, boot_phases.PHASE_WARMUP)
    iters = _terminal(rows, boot_phases.PHASE_WARMUP_ITERATION)
    assert iters, (
        "no warmup_iteration rows — the harness endpoint declares "
        "Resources(gpu=True) precisely so `warmup.plan` schedules a real warm "
        "job; without them this issue's per-iteration question is unanswerable"
    )
    warm_ordinals = {w.ordinal for w in warms}
    assert all(i.parent_ordinal in warm_ordinals for i in iters), (
        "warmup_iteration rows are not children of their warmup span"
    )
    seq = [int(_detail(i).get("iteration", "0")) for i in iters]
    assert seq == sorted(seq) and seq[0] == 1, (
        f"iteration indices are not a 1-based ordered sequence: {seq}"
    )


def test_warm_complete_is_emitted_on_a_boot_with_no_background_mint(ladder) -> None:
    """pgw#789 put `warm_complete` in `_background_mint`'s finally only, and
    `rec.background_mint` is set ONLY under eager-first — so an inline-mint,
    cell-adopt or plain eager boot emitted nothing and was read as "compiled
    serving never reached" when the truth was "never measured"."""
    row = _one(ladder, boot_phases.PHASE_WARM_COMPLETE)
    assert row.cumulative, "warm_complete is a milestone off process start"
    assert row.outcome in (
        boot_phases.OUTCOME_OK, boot_phases.OUTCOME_REFUSED)
    if row.outcome == boot_phases.OUTCOME_REFUSED:
        assert row.reason == "serving_eager"


def test_the_ladder_reconciles_against_the_boot_total(ladder) -> None:
    """The th#1111 property: nested spans charge the CHILD, so measured phases
    plus residual equal the whole boot window. A ladder that double-counts is
    worse than no ladder."""
    rows = ladder
    servable = _one(rows, boot_phases.PHASE_FIRST_REQUEST_SERVABLE)
    children: Dict[int, int] = {}
    for r in rows:
        if r.terminal and not r.cumulative and r.parent_ordinal:
            children[r.parent_ordinal] = (
                children.get(r.parent_ordinal, 0) + r.duration_ms)
    exclusive = sum(
        max(0, r.duration_ms - children.get(r.ordinal, 0))
        for r in rows if r.terminal and not r.cumulative
    )
    assert exclusive <= servable.duration_ms, (
        f"measured phases ({exclusive}ms) exceed the boot total "
        f"({servable.duration_ms}ms) — the ladder double-counts"
    )
    assert "residual_ms" in servable.detail, (
        "the boot close must carry its reconciliation: " + servable.detail
    )


# --------------------------------------------------------------------------
# the ordering invariant, at the recorder
# --------------------------------------------------------------------------

def test_the_recorder_makes_the_inversion_unrepresentable() -> None:
    """The caller fix alone is a fix; this is the structural half. A boot close
    recorded before `hello` is HELD, not emitted, and released with its time
    re-read — because a worker the hub cannot reach is not servable."""
    boot_phases.reset_for_tests()
    try:
        boot_phases.mark(
            boot_phases.PHASE_FIRST_REQUEST_SERVABLE, since_process_start=True)
        assert boot_phases.recorded_rows() == [], (
            "a boot close before hello reached the wire"
        )
        assert boot_phases.in_boot() is True, (
            "a held close still closed the boot window, so the spans that "
            "follow would be suppressed — the exact 0.78.0 failure"
        )
        boot_phases.mark_once(boot_phases.PHASE_HELLO, since_process_start=True)
        rows = boot_phases.recorded_rows()
        phases = [r.phase for r in rows]
        assert phases == [
            boot_phases.PHASE_HELLO, boot_phases.PHASE_FIRST_REQUEST_SERVABLE,
        ], phases
        assert rows[1].process_uptime_ms >= rows[0].process_uptime_ms
        assert boot_phases.in_boot() is False
    finally:
        boot_phases.reset_for_tests()


def test_nesting_survives_concurrent_tasks() -> None:
    """The span stack is a ContextVar, not a thread-local: setup, mint and
    adopt run interleaved on the one worker thread, and a shared stack would
    record task B's span as task A's CHILD — which subtracts B's time from A's
    exclusive total and silently breaks reconciliation."""
    import asyncio

    boot_phases.reset_for_tests()

    async def main() -> None:
        async def leaf(name: str) -> int:
            with boot_phases.span(name) as sp:
                await asyncio.sleep(0.01)
                return sp.ordinal

        with boot_phases.span(boot_phases.PHASE_PIPELINE_LOAD) as outer:
            # A sibling task started from inside the span inherits it...
            inner = await asyncio.create_task(leaf(boot_phases.PHASE_WARMUP))
            outer_ordinal = outer.ordinal
        # ...and one started outside it does not.
        detached = await asyncio.create_task(leaf(boot_phases.PHASE_CELL_FETCH))
        return outer_ordinal, inner, detached

    try:
        outer_ordinal, inner, detached = asyncio.run(main())
        rows = {r.ordinal: r for r in boot_phases.recorded_rows() if r.terminal}
        assert rows[inner].parent_ordinal == outer_ordinal
        assert rows[detached].parent_ordinal == 0, (
            "a task started after the span closed inherited it as a parent"
        )
    finally:
        boot_phases.reset_for_tests()
