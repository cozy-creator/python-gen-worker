"""BOOT materializes what the config names, and readiness waits for it.

# pgw#1490 / th#2204, after Paul's "simplify first" ruling

th#2204 measured the old shape on a live rented H100 at $3.29/hr: the hub sent
`HelloAck(desired_generation=3 disk=2 hot=0)`, the worker acked the generation
and then nothing on the pod ever acted, so `placement declined …
reason=model_download_pending … download=[no_position_reported]` ran at 1 Hz for
the life of the rental while th#1142 re-elected the same sole worker every six
minutes. The first fix gave the hub's reconcile protocol its missing worker
half. The ruling deletes the protocol instead: *the production system should
use the exact same code paths as we are using and testing here*.

So boot IS `up`. The worker pulls what its runtime config names, and does not
advertise a function until it holds the weights — the same reason `run`
requires `up` locally. The hub then routes on advertised readiness like any web
service, with no residency accounting, no parking, no reservations and no
transfer-owner election anywhere in the picture.

These drive the REAL `ModelStore` funnel against a real trickling HTTP origin
and a real CAS — the harness pgw#1455 built and pgw#1485 extended. $0, no GPU,
no pod.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, List

import pytest

from gen_worker import activity
from gen_worker.boot_materialize import (
    STATE_FAILED,
    STATE_MATERIALIZING,
    STATE_READY,
    CheckpointConfig,
    CheckpointMaterialization,
)
from gen_worker.models.refs import WireRef
from gen_worker.pb import worker_scheduler_pb2 as pb
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


def _config(version: int, ref: WireRef, snapshot: pb.Snapshot) -> CheckpointConfig:
    """The config as the pod receives it — built through `from_wire`, so the
    wire shape the hub already sends is what the tests exercise."""
    desired = pb.DesiredResidency(generation=version, disk_refs=[str(ref)])
    desired.snapshots[str(ref)].CopyFrom(snapshot)
    return CheckpointConfig.from_wire(desired)


async def _settle(mat: CheckpointMaterialization) -> None:
    """Let the pull task and the activity sink's `create_task` hops run."""
    task = mat._task  # this lane owns the task; nothing else does
    assert task is not None, "configure() started no materialization"
    await task
    for _ in range(8):
        await asyncio.sleep(0)


def _blobs(origin: Any) -> pb.Snapshot:
    files = [
        (path, body, origin.put(body))
        for path, body in (
            (f"unet/shard-{i}.safetensors", bytes([i + 1]) * OBJECT_BYTES)
            for i in range(OBJECTS)
        )
    ]
    return _pb_snapshot(files)


# ---------------------------------------------------------------------------
# the warm pod — th#2204's ACTUAL rental
# ---------------------------------------------------------------------------


def test_a_FRESH_process_on_a_warm_volume_is_ready_without_moving_a_byte(
    tmp_path: Path, _fine_cadence: Any
) -> None:
    """Pod orq6abdjo28it6 booted onto a volume that already held all 134 GB.

    Its first config named weights it already had, in a brand-new process with
    no banked identity to recognise them by. Under the deleted protocol that
    pod answered nothing and was re-elected for 13 minutes. Under this one it
    reads its config, finds the bytes, and becomes READY — the staged volume
    makes the boot pull a near-no-op, which is the whole reason volume staging
    survives as an ops-side pre-warm.
    """
    origin = _Origin()
    try:
        snapshot = _blobs(origin)
        cas = tmp_path / "endpoint-volume-cas"

        async def stage() -> None:
            wire = _Wire()
            activity.bind_sink(wire.send, asyncio.get_running_loop())
            mat = CheckpointMaterialization(_store(wire, cas))
            mat.configure(_config(1, _REF, snapshot))
            await _settle(mat)

        asyncio.run(stage())

        warm = _Wire()

        async def boot() -> None:
            store = _store(warm, cas)
            activity.bind_sink(warm.send, asyncio.get_running_loop())
            store.rescan_disk()  # boot-time truth, exactly as the Worker does
            mat = CheckpointMaterialization(store)
            mat.configure(_config(1, _REF, snapshot))
            await _settle(mat)
            assert mat.state == STATE_READY
            assert mat.phase() == pb.WORKER_PHASE_READY

        asyncio.run(boot())

        phases = [u.phase for u in warm.positions()]
        assert phases == [PHASE_ALREADY_RESIDENT], (
            f"a warm volume answers, it does not fetch; positions said {phases}")
        states = [e.state for e in warm.model_events]
        assert pb.MODEL_STATE_DOWNLOADING not in states, (
            f"nothing to transfer, so nothing to declare; got {states}")
        assert pb.MODEL_STATE_ON_DISK in states
        assert warm.open_download_records() == {}
    finally:
        origin.close()


# ---------------------------------------------------------------------------
# the cold pod — readiness is WITHHELD, and the positions are the download's
# ---------------------------------------------------------------------------


def test_a_cold_boot_PULLS_and_is_not_ready_until_it_lands(
    tmp_path: Path, _fine_cadence: Any
) -> None:
    """The whole invariant, in one test.

    While the pull runs the worker is `materializing`, its wire phase is
    DOWNLOADING_MODELS and `ready` is false — which is what makes "never
    serialize a multi-GB fetch inside a user request" true BY CONSTRUCTION,
    with no hub-side parking to enforce it. The byte positions are the
    download's own instrument (`weight_position` inside the funnel), not a
    reporter bolted on beside it.
    """
    origin = _Origin()
    try:
        snapshot = _blobs(origin)
        wire = _Wire()
        store = _store(wire, tmp_path / "cas")
        seen: List[str] = []

        async def run() -> None:
            activity.bind_sink(wire.send, asyncio.get_running_loop())
            mat = CheckpointMaterialization(
                store, announce=lambda: _record(seen, mat),
            )
            mat.configure(_config(7, _REF, snapshot))
            assert mat.state == STATE_MATERIALIZING, (
                "the config is taken synchronously so no window exists in "
                "which the pod is both unmaterialized and advertising ready")
            assert not mat.ready
            assert mat.phase() == pb.WORKER_PHASE_DOWNLOADING_MODELS
            await _settle(mat)
            assert mat.state == STATE_READY
            assert mat.phase() == pb.WORKER_PHASE_READY

        asyncio.run(run())

        assert seen[0] == STATE_MATERIALIZING and seen[-1] == STATE_READY, (
            "readiness must be ANNOUNCED on both edges — withheld at the start "
            f"and granted at the end; the announcements were {seen}")
        states = [e.state for e in wire.model_events]
        assert pb.MODEL_STATE_DOWNLOADING in states, (
            f"an absent ref must declare its transfer; got {states}")
        assert pb.MODEL_STATE_ON_DISK in states
        assert wire.open_download_records() == {}
        steps = [r.step for r in wire.positions()]
        assert steps and steps[-1] == (OBJECTS * OBJECT_BYTES) // MIB, (
            f"the download's own byte positions must reach the whole; got {steps}")
        assert steps[-1] > steps[0]
    finally:
        origin.close()


async def _record(seen: List[str], mat: CheckpointMaterialization) -> None:
    seen.append(mat.state)


# ---------------------------------------------------------------------------
# the reconnect
# ---------------------------------------------------------------------------


def test_an_unchanged_config_repush_moves_no_bytes_and_keeps_readiness(
    tmp_path: Path, _fine_cadence: Any
) -> None:
    """Every reconnect re-delivers the same config.

    Under the deleted protocol the generation bumped on every HelloAck, and a
    bumped generation over unchanged bytes made the pod open a DOWNLOADING
    record for weights it held — the 0-of-N phantom th#2205 measured for 1h54m.
    Config identity is (ref, digest): the version moves, the bytes do not, and
    a re-push is a no-op that does not even withdraw readiness.
    """
    origin = _Origin()
    try:
        snapshot = _blobs(origin)
        wire = _Wire()
        store = _store(wire, tmp_path / "cas")

        async def run() -> None:
            activity.bind_sink(wire.send, asyncio.get_running_loop())
            mat = CheckpointMaterialization(store)
            mat.configure(_config(1, _REF, snapshot))
            await _settle(mat)
            assert mat.state == STATE_READY
            wire.updates.clear()
            wire.model_events.clear()

            before = mat._task
            mat.configure(_config(2, _REF, snapshot))  # same bytes, next version
            assert mat._task is before, "an unchanged config started a new pull"
            assert mat.state == STATE_READY, (
                "a re-push of the config this pod already satisfied must not "
                "take the pod out of service")

        asyncio.run(run())

        assert [e.state for e in wire.model_events] == [], (
            "an unchanged config re-push must produce no model events at all")
        assert [u.phase for u in wire.positions()] == []
    finally:
        origin.close()


# ---------------------------------------------------------------------------
# the two degenerate configs
# ---------------------------------------------------------------------------


def test_a_config_naming_no_refs_is_ready_immediately(tmp_path: Path) -> None:
    """A weightless release has nothing to wait for.

    The readiness gate must never be able to strand a pod that has no weights —
    that would turn this fix into a new way of serving nothing.
    """
    wire = _Wire()
    store = _store(wire, tmp_path / "cas")

    async def run() -> None:
        mat = CheckpointMaterialization(store)
        mat.configure(CheckpointConfig.from_wire(None))
        assert mat.state == STATE_READY
        assert mat._task is None
        mat.configure(CheckpointConfig.from_wire(pb.DesiredResidency(generation=4)))
        assert mat.state == STATE_READY

    asyncio.run(run())


def test_a_failed_pull_is_a_TYPED_LOUD_STATE_and_not_a_retry(
    tmp_path: Path, _fine_cadence: Any
) -> None:
    """A pull that cannot succeed must END, visibly.

    `ensure_local` already retries transient failures with backoff and raises
    terminal ones. When it raises, the worker enters `failed`, reports
    `FnUnavailable{model_unavailable}` for every function it would have served
    and stays connected-and-unroutable. A pod that retried forever in the
    background would be th#2204's livelock wearing a different hat — a state
    with no exit and nothing that can ever finish.
    """
    origin = _Origin()
    try:
        snapshot = _blobs(origin)
        # The origin is closed BEFORE the pull, so every object 404s/refuses:
        # a terminal failure, not a slow one.
        origin.close()
        wire = _Wire()
        store = _store(wire, tmp_path / "cas")
        seen: List[str] = []

        async def run() -> None:
            activity.bind_sink(wire.send, asyncio.get_running_loop())
            mat = CheckpointMaterialization(
                store, announce=lambda: _record(seen, mat),
            )
            mat.configure(_config(3, _REF, snapshot))
            await _settle(mat)
            # The failure is a STATE, reached without raising out of the task:
            # a pull that dies silently and a pull that keeps trying are the
            # same thing to everyone outside this process.
            assert mat.state == STATE_FAILED, (
                f"a hopeless pull must terminate in `failed`; state={mat.state}")
            assert mat.phase() == pb.WORKER_PHASE_ERROR
            assert not mat.ready
            assert _REF in mat.failure and mat.failure.count(":") >= 2, (
                "the failure must NAME the ref and the exception type; "
                f"got {mat.failure!r}")

        asyncio.run(run())

        assert seen and seen[-1] == STATE_FAILED, (
            f"the failure must be ANNOUNCED, not merely logged; saw {seen}")
    finally:
        origin.close()


# ---------------------------------------------------------------------------
# and then the dispatch can FIND it
# ---------------------------------------------------------------------------


def test_the_dispatch_resolves_the_tree_BOOT_materialized(
    tmp_path: Path, _fine_cadence: Any
) -> None:
    """The other half of "same code paths", and a latent defect on its own.

    Boot putting the tree on disk is worth nothing if `ctx.load` cannot find
    it, and on v2 it could not, for two independent reasons:

    * `ModelBinding.manifest_digest` has never had a sender (its own proto
      comment says so), so every dispatch hit "carries no manifest_digest; the
      worker has no other fetch pointer" before it looked at anything; and
    * and a digest can arrive in two spellings. MEASURED on the standing
      stack rather than assumed: the hub's volume manifest is 38/38 BARE hex
      and every tree in this box's CAS is bare, so the resolver's old strip
      was a no-op — but the hub's artifact metadata is 1999/1999
      `sha256:`-tagged, so both spellings are live somewhere. The lookup tries
      both rather than picking a side.

    So the resolver asks the store that materialized the ref. This test drives
    the real pull and then the real `tree_for` over a dispatch whose binding
    carries what the hub actually sends today: no digest at all.
    """
    from gen_worker.worker import _DISPATCH, _Pick, HubBindingResolver, _DispatchPicks

    origin = _Origin()
    try:
        snapshot = _blobs(origin)
        wire = _Wire()
        cas = tmp_path / "cas"
        store = _store(wire, cas)

        async def run() -> None:
            activity.bind_sink(wire.send, asyncio.get_running_loop())
            mat = CheckpointMaterialization(store)
            mat.configure(_config(1, _REF, snapshot))
            await _settle(mat)
            assert mat.state == STATE_READY

        asyncio.run(run())

        # Exactly the binding tensorhub sends: a ref, no manifest_digest.
        pick = _Pick(
            slot="model", ref=str(_REF), manifest_digest="",
            model="", inference_defaults="",
        )
        token = _DISPATCH.set(
            _DispatchPicks(by_ref={str(_REF): pick}, by_slot={"model": str(_REF)})
        )
        try:
            resolver = HubBindingResolver(snapshots_root=cas / "snapshots")
            resolver.bind_store(store)
            tree = resolver.tree_for(type("Sd15", (), {}), str(_REF))
        finally:
            _DISPATCH.reset(token)

        assert tree.is_dir(), f"{tree} is not a materialized tree"
        assert tree.parent.name == "snapshots"
        # The WRITER names the tree; the resolver must find it whichever
        # spelling the config carried. This fixture's snapshot is tagged, and
        # production's is bare — both have to resolve, which is the assertion.
        digest = str(snapshot.digest)
        assert tree.name in (digest, digest.split(":", 1)[-1]), (
            f"tree {tree.name} is neither spelling of {digest}")
    finally:
        origin.close()


@pytest.fixture
def _fine_cadence(monkeypatch: pytest.MonkeyPatch) -> Any:
    """pgw#1455's stride/floor make a 9 MiB fixture emit one row; the position
    mechanics are asserted in `test_weight_position.py` and re-asserted here,
    so the cadence is tightened the same way it is there."""
    import gen_worker.weight_position as wp

    monkeypatch.setattr(wp, "STRIDE_MIB", 1)
    monkeypatch.setattr(wp, "MIN_INTERVAL_S", 0.0)
    return None
