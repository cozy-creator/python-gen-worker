"""pgw#1483 / th#2204: the v2 worker ANSWERS the hub's residency goal.

th#2204 measured the absence on a live rented H100 at $3.29/hr: the hub sent
`HelloAck(desired_generation=3 disk=2 hot=0 resolutions=0)`, the worker acked the
generation, and then nothing on the pod ever acted — `placement declined …
reason=model_download_pending … download=[no_position_reported]` at 1 Hz for the
life of the rental, twice interrupted by th#1142 releasing a reservation whose
owner had "accepted the goal 6m14s ago with NO DOWNLOAD EVER OPENED" and
re-electing the same sole worker.

These drive the REAL consumer over the REAL `ModelStore` funnel against a real
trickling HTTP origin and a real CAS — the harness pgw#1455 built and pgw#1485
extended. $0, no GPU, no pod.

Two arms, and they are the two answers the hub must be able to tell apart:

* **goal names an already-resident ref** → the SATISFIED answer. An
  `already_resident` row on the `weight_fetch` position stream, an `ON_DISK`
  ModelEvent so hub-side placement sees DISK locality and dispatches, and NO
  download record (pgw#1485: a record opened for bytes the pod holds can never
  close honestly).
* **goal names an absent ref** → the funnel opens, positions advance, the
  record closes. This is pgw#1485's control, re-run through the goal consumer
  rather than through a bare `ensure_local`, because the thing under test is
  the JOIN and a join can be wrong in either direction.

The echo is asserted in both: `observed_generation` only advances once every
declared ref reached a terminal answer. A goal still fetching is ACCEPTED, not
ANSWERED, and the hub reads those two states differently.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from gen_worker import activity
from gen_worker.models.refs import WireRef
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.residency_goal import ResidencyGoal
from gen_worker.weight_position import MIB, PHASE_ALREADY_RESIDENT

# The pgw#1455/#1485 harness: `tests` is on `pythonpath` (pyproject), so this is
# the same real origin, real CAS, real ModelStore every position test uses.
from test_weight_position import (  # type: ignore[import-not-found]
    OBJECT_BYTES,
    OBJECTS,
    _Origin,
    _Wire,
    _pb_snapshot,
    _store,
)

_REF = WireRef("acme/model-a")


def _goal(generation: int, ref: WireRef, snapshot: pb.Snapshot) -> pb.DesiredResidency:
    """The hub's wire form, built exactly as `hello_ack.go` builds it:
    `disk_refs` in orchestrator priority order and `snapshots` keyed by ref."""
    desired = pb.DesiredResidency(generation=generation, disk_refs=[str(ref)])
    desired.snapshots[str(ref)].CopyFrom(snapshot)
    return desired


async def _drain(goal: ResidencyGoal) -> None:
    """Let the reconcile task and the activity sink's `create_task` hops run."""
    task = goal._task  # the lane under test owns its task; nothing else does
    assert task is not None, "apply() created no reconcile task"
    await task
    for _ in range(8):
        await asyncio.sleep(0)


def test_a_goal_for_a_resident_ref_is_ANSWERED_and_opens_no_record(
    tmp_path: Path, _fine_cadence: Any
) -> None:
    """th#2204's own shape: a WARM pod whose weights are already on the
    attached volume.

    Before this, the worker acked the generation and did nothing — no download,
    no event, no position — so the hub's `no_position_reported` was the honest
    reading of a genuine silence, and placement had nothing that could ever
    terminate the wait.
    """
    origin = _Origin()
    try:
        blobs = [
            (f"unet/shard-{i}.safetensors", bytes([i + 1]) * OBJECT_BYTES)
            for i in range(OBJECTS)
        ]
        files = [(p, b, origin.put(b)) for p, b in blobs]
        snapshot = _pb_snapshot(files)
        wire = _Wire()
        store = _store(wire, tmp_path / "cas")

        async def run() -> None:
            activity.bind_sink(wire.send, asyncio.get_running_loop())
            # Generation 1 stages the bytes — this is the COLD arm and it is
            # here so the resident arm is measured on a pod that genuinely
            # holds them, not on a mocked residency.
            goal = ResidencyGoal(store)
            goal.apply(_goal(1, _REF, snapshot))
            await _drain(goal)
            assert goal.observed_generation == 1

            # Everything the cold generation said is now history; the resident
            # generation must speak for itself.
            wire.updates.clear()
            wire.model_events.clear()

            # Generation 2: the identical goal, re-declared. This is the warm
            # re-dispatch / reconnect the hub performs on every HelloAck.
            goal.apply(_goal(2, _REF, snapshot))
            await _drain(goal)
            assert goal.observed_generation == 2

        asyncio.run(run())

        phases = [u.phase for u in wire.positions()]
        assert PHASE_ALREADY_RESIDENT in phases, (
            "a goal naming a ref the pod already holds must ANSWER with the "
            f"satisfied fact; the position stream said {phases}"
        )
        states = [e.state for e in wire.model_events]
        assert pb.MODEL_STATE_DOWNLOADING not in states, (
            "the pod holds these bytes: declaring a transfer opens a hub-side "
            f"liability that can never close honestly; got {states}"
        )
        assert pb.MODEL_STATE_ON_DISK in states, (
            "the satisfied answer must reach the hub as residency, or placement "
            "still sees a pod that said nothing and parks forever "
            f"(th#2204, $3.29/hr); got {states}"
        )
        assert wire.open_download_records() == {}, (
            "a resident ref must leave no open download record")
    finally:
        origin.close()


def test_a_goal_for_an_absent_ref_OPENS_the_instrumented_funnel(
    tmp_path: Path, _fine_cadence: Any
) -> None:
    """pgw#1485's control, driven through the goal consumer.

    The fix must not buy silence-on-resident by making the consumer refuse to
    fetch: an absent ref still opens, advances and closes, with positions.
    """
    origin = _Origin()
    try:
        blobs = [
            (f"unet/shard-{i}.safetensors", bytes([i + 1]) * OBJECT_BYTES)
            for i in range(OBJECTS)
        ]
        files = [(p, b, origin.put(b)) for p, b in blobs]
        snapshot = _pb_snapshot(files)
        wire = _Wire()
        store = _store(wire, tmp_path / "cas")

        async def run() -> None:
            activity.bind_sink(wire.send, asyncio.get_running_loop())
            goal = ResidencyGoal(store)
            goal.apply(_goal(7, _REF, snapshot))
            # Not answered until the transfer reaches a terminal answer: a goal
            # in flight is ACCEPTED, and the hub must not read a 134 GB fetch
            # as satisfied.
            assert goal.observed_generation == 0
            assert goal.accepted_generation == 7
            await _drain(goal)
            assert goal.observed_generation == 7

        asyncio.run(run())

        states = [e.state for e in wire.model_events]
        assert pb.MODEL_STATE_DOWNLOADING in states, (
            f"an absent ref must declare its transfer; got {states}")
        assert pb.MODEL_STATE_ON_DISK in states
        assert wire.open_download_records() == {}
        steps = [r.step for r in wire.positions()]
        assert steps and steps[-1] == (OBJECTS * OBJECT_BYTES) // MIB, (
            f"the funnel's byte positions must advance to the whole; got {steps}")
        assert steps[-1] > steps[0]
    finally:
        origin.close()


def test_a_generation_bump_over_UNCHANGED_bytes_declares_no_transfer(
    tmp_path: Path, _fine_cadence: Any
) -> None:
    """The defect this lane found by measuring, not by reading.

    pgw#1485's resident pre-check compared the whole identity TUPLE
    `(digest, generation)`. The generation is the hub's causal fence for
    EVENTS, not a property of the bytes, and every HelloAck bumps it — so
    re-declaring an unchanged goal made the pod open a `DOWNLOADING` record
    for weights it already held, resolve every object out of the local CAS
    (zero bytes moved, so `_progress` never fires once) and hand the hub the
    exact `0 of N bytes @ 0.00/s` phantom that th#2205 measured for 1h54m and
    th#2204 measured as a placement livelock.

    Driven through `ensure_local` deliberately, BELOW `announce_resident`, so
    the predicate is proved on its own rather than through the fast path.
    """
    origin = _Origin()
    try:
        blobs = [
            (f"unet/shard-{i}.safetensors", bytes([i + 1]) * OBJECT_BYTES)
            for i in range(OBJECTS)
        ]
        files = [(p, b, origin.put(b)) for p, b in blobs]
        snapshot = _pb_snapshot(files)
        wire = _Wire()
        store = _store(wire, tmp_path / "cas")

        async def run() -> None:
            activity.bind_sink(wire.send, asyncio.get_running_loop())
            store.replace_desired_snapshots({_REF: snapshot}, generation=1)
            await store.ensure_local(_REF)
            for _ in range(8):
                await asyncio.sleep(0)
            wire.updates.clear()
            wire.model_events.clear()
            # Same bytes, same digest, next generation — the ordinary reconnect.
            store.replace_desired_snapshots({_REF: snapshot}, generation=2)
            await store.ensure_local(_REF)
            for _ in range(8):
                await asyncio.sleep(0)

        asyncio.run(run())

        states = [e.state for e in wire.model_events]
        assert pb.MODEL_STATE_DOWNLOADING not in states, (
            "a generation bump moved no bytes: declaring a transfer here opens "
            "the 0-of-N hub row that parks placement and vetoes idle retire; "
            f"got {states}"
        )
        phases = [u.phase for u in wire.positions()]
        assert PHASE_ALREADY_RESIDENT in phases, (
            f"the unchanged bytes must be reported as resident; got {phases}")
    finally:
        origin.close()


def test_an_unset_or_rewound_generation_is_not_a_goal(tmp_path: Path) -> None:
    """Generation 0 is the wire's "unset", and the hub never rewinds a live
    generation — a rewind on the wire is a replayed frame, not an instruction.
    Neither may start a reconcile, because either would make the echo lie."""
    wire = _Wire()
    store = _store(wire, tmp_path / "cas")
    goal = ResidencyGoal(store)

    goal.apply(None)
    assert goal._task is None
    goal.apply(pb.DesiredResidency(generation=0, disk_refs=[str(_REF)]))
    assert goal._task is None
    assert goal.accepted_generation == 0

    async def run() -> None:
        goal.apply(pb.DesiredResidency(generation=5))
        await _drain(goal)
        assert goal.observed_generation == 5
        # A stale frame arriving after generation 5 must not reopen it.
        goal.apply(pb.DesiredResidency(generation=4))
        assert goal.accepted_generation == 5

    asyncio.run(run())


@pytest.fixture
def _fine_cadence(monkeypatch: pytest.MonkeyPatch) -> Any:
    """pgw#1455's stride/floor make a 9 MiB fixture emit one row; the position
    mechanics are asserted in `test_weight_position.py` and re-asserted here,
    so the cadence is tightened the same way it is there."""
    import gen_worker.weight_position as wp

    monkeypatch.setattr(wp, "STRIDE_MIB", 1)
    monkeypatch.setattr(wp, "MIN_INTERVAL_S", 0.0)
    return None
