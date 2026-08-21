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
from gen_worker import boot_phases
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


def test_the_dispatch_supplies_its_own_fetch_pointer(tmp_path: Path) -> None:
    """`RunJob.snapshots` is keyed by ref and ships on every dispatch.

    `ModelBinding.manifest_digest` has no sender, but the dispatch is not
    actually pointer-less: the hub already sends the ref's snapshot beside the
    binding (pgw#1475 depends on exactly that for reserved repos) and the
    worker was dropping it on the floor for checkpoints. Reading it is not
    dispatch-time materialization — it is the IDENTITY of a tree, not a
    request for one.
    """
    from gen_worker.worker import _picks_of

    run = pb.RunJob()
    binding = run.models.add()
    binding.slot = "model"
    binding.ref = str(_REF)
    run.snapshots[str(_REF)].digest = "b7f2c1a0" * 8

    picks = _picks_of(run)
    assert picks.by_ref[str(_REF)].manifest_digest == "b7f2c1a0" * 8, (
        "the dispatch's own snapshot must supply the fetch pointer the wire's "
        "digest field never carries")


@pytest.fixture
def _fine_cadence(monkeypatch: pytest.MonkeyPatch) -> Any:
    """pgw#1455's stride/floor make a 9 MiB fixture emit one row; the position
    mechanics are asserted in `test_weight_position.py` and re-asserted here,
    so the cadence is tightened the same way it is there."""
    import gen_worker.weight_position as wp

    monkeypatch.setattr(wp, "STRIDE_MIB", 1)
    monkeypatch.setattr(wp, "MIN_INTERVAL_S", 0.0)
    return None


# ---------------------------------------------------------------------------
# pgw#1511: a tree that EXISTS is not a tree that is RESIDENT
# ---------------------------------------------------------------------------


def test_a_TRUNCATED_warm_tree_is_refused_quarantined_and_refetched(
    tmp_path: Path, _fine_cadence: Any
) -> None:
    """The fleet incident: two endpoints, two volumes, 20x apart, identical
    `SafetensorError: header too large` on read.

    Every write path in the stack is airtight — the CAS hashes and size-checks
    every object before committing it, `project_snapshot` builds in a scratch
    directory and renames, and tensorfs' materializer re-hashes each object AND
    the whole file and refuses a short read. So the bytes were never
    short-WRITTEN. What happened is that a tree left incomplete by an
    interrupted materialization is still a DIRECTORY THAT EXISTS, and
    `announce_resident` tested exactly that and nothing else: it answered
    `already_resident`, published ON_DISK, marked the ref `_verified` (which
    suppresses the first-use digest check for the rest of the process), and the
    worker advertised ready. The first component to notice was the tenant's
    loader.

    So: stage a good tree, TRUNCATE one weight file in it the way an
    interrupted write would, and boot a second store on that CAS. The pod must
    refuse the residency rather than advertise it.
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

        # The damage, applied to the STAGED tree exactly as a half-finished
        # write leaves it: the file is present, and short.
        trees = [p for p in (cas / "snapshots").iterdir() if p.is_dir()]
        assert len(trees) == 1, f"expected one staged tree, got {trees}"
        weights = sorted(trees[0].rglob("*.safetensors"))
        assert weights, f"no weight files in {trees[0]}"
        victim = weights[0]
        full = victim.stat().st_size
        # The tree's files are read-only HARD LINKS into the CAS objects, so
        # this truncates the object itself — a more faithful reproduction than
        # damaging a private copy would be, and it means recovery has to
        # re-download rather than re-link the same bad bytes.
        mode = victim.stat().st_mode
        victim.chmod(0o644)
        with victim.open("r+b") as handle:
            handle.truncate(max(1, full // 3))
        victim.chmod(mode)
        assert victim.stat().st_size < full

        warm = _Wire()
        outcome: List[str] = []

        async def boot() -> None:
            store = _store(warm, cas)
            activity.bind_sink(warm.send, asyncio.get_running_loop())
            store.rescan_disk()
            mat = CheckpointMaterialization(store)
            mat.configure(_config(1, _REF, snapshot))
            await _settle(mat)
            outcome.append(mat.state)

        asyncio.run(boot())

        phases = [u.phase for u in warm.positions()]
        assert PHASE_ALREADY_RESIDENT not in phases, (
            "a truncated tree was answered as ALREADY RESIDENT — that is the "
            "incident: the pod advertises weights it does not have, and the "
            f"first reader gets `header too large`. positions={phases}")
        assert outcome == [STATE_READY], (
            f"the pod must recover by re-fetching, not stall; state={outcome}")
        # And the recovery is real, judged by the SAME verifier that rejected
        # the damage — not by raw file size, because a projected tree's files
        # are pointer stubs by design and sizing them proves nothing.
        final = [p for p in (cas / "snapshots").iterdir() if p.is_dir()]
        assert len(final) == 1, f"expected one tree after recovery, got {final}"
        checker = _store(_Wire(), cas)
        ok, bad = checker._verify_snapshot_tree(final[0], snapshot)
        assert ok, f"the recovered tree still fails verification: {bad}"
    finally:
        origin.close()


# ---------------------------------------------------------------------------
# pgw#1555 — THE BOOT LADDER CAN SEE THE FETCH
#
# The instrument that exists to answer "where did the boot's seconds go" was
# blind to weights on every pod. `in_boot()` latched shut at the FIRST servable
# mark, `on_hello_ack` marks servable the instant `materialization.ready` is
# true, and a config naming NO refs makes that true immediately — so a fleet
# whose first ack was empty closed its boot window ~1 ms after `hello` and then
# downloaded gigabytes with nothing recording. Read off the standing stack:
# 15/15 recent boots hold only `hello` + `first_request_servable` (+ sometimes
# `eager_ready`) with `bytes = 0` in every row, while `worker_activity_events`
# shows the same pods moving 2,742,235,508 bytes of sd15 fourteen seconds later.
# ---------------------------------------------------------------------------


def _fetch_rows(phase: str) -> List[Any]:
    return [r for r in boot_phases.recorded_rows()
            if r.phase == phase and r.terminal]


def test_a_pull_that_follows_an_EMPTY_config_still_lands_in_the_boot_ladder(
    tmp_path: Path, _fine_cadence: Any
) -> None:
    """The field sequence, exactly: empty config, servable, THEN the real one.

    The hub's first ack named no refs, so the pod was legitimately ready with
    nothing to hold and `first_request_servable` was stamped. The config that
    named the checkpoint arrived on a later ack. Everything after that point
    is the boot this table exists to measure, and before this fix none of it
    was recorded — not one `weights_fetch` row, not one byte.
    """
    origin = _Origin()
    try:
        snapshot = _blobs(origin)
        wire = _Wire()
        store = _store(wire, tmp_path / "cas")

        async def run() -> None:
            activity.bind_sink(wire.send, asyncio.get_running_loop())
            mat = CheckpointMaterialization(store)
            # The recorder asks the same object the hub routes on, which is
            # what `Worker.arun` binds in production.
            boot_phases.bind_servable_probe(lambda: mat.ready)

            # Ack #1: no refs. Ready with nothing to materialize, and the
            # worker marks the servable milestone — both correct.
            mat.configure(CheckpointConfig.from_wire(pb.DesiredResidency(generation=1)))
            assert mat.ready
            boot_phases.mark_once(boot_phases.PHASE_HELLO)
            boot_phases.mark_once(boot_phases.PHASE_FIRST_REQUEST_SERVABLE)
            assert boot_phases.servable_ms() is not None

            # Ack #2: the checkpoint. The pod stops being routable, so it is
            # not in steady state and the fetch is a boot fact.
            mat.configure(_config(2, _REF, snapshot))
            assert not mat.ready
            assert boot_phases.in_boot(), (
                "a worker that is advertising `loading_functions` and holds "
                "none of its configured weights is NOT past its boot — that "
                "reading is what blinded the ladder on every field pod")
            await _settle(mat)
            assert mat.ready

        asyncio.run(run())

        rows = _fetch_rows(boot_phases.PHASE_WEIGHTS_FETCH)
        assert rows, (
            "the pull produced NO `weights_fetch` row. This is the fleet "
            "defect verbatim: 2.74 GB moved and the boot ladder recorded "
            f"nothing. rows={[r.phase for r in boot_phases.recorded_rows()]}")
        assert sum(r.bytes for r in rows) > 0, (
            "a `weights_fetch` row with bytes=0 over a real transfer is the "
            "other half of the field reading — every stored row had bytes=0")
        # And the window closing again is what keeps the table bounded: a
        # steady-state materialization hours later must NOT land in the ladder.
        assert not boot_phases.in_boot()
    finally:
        origin.close()


def test_a_WARM_volume_boot_is_a_TIMED_ROW_and_not_a_hole_in_the_ladder(
    tmp_path: Path, _fine_cadence: Any
) -> None:
    """The flagship case, and the one with no instrument at all.

    `_materialize` `continue`s past `ensure_local` when the tree is already
    resident, so the row `ensure_local` would have opened never exists — a
    warm 134 GB pod rendered as an EMPTY boot ladder. The check it does
    instead is not free (a verified manifest match, pgw#1511), and on a warm
    volume it IS the boot, so it gets the row.
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
        boot_phases.reset_for_tests()

        warm = _Wire()

        async def boot() -> None:
            store = _store(warm, cas)
            activity.bind_sink(warm.send, asyncio.get_running_loop())
            store.rescan_disk()
            mat = CheckpointMaterialization(store)
            boot_phases.bind_servable_probe(lambda: mat.ready)
            mat.configure(_config(1, _REF, snapshot))
            await _settle(mat)
            assert mat.state == STATE_READY

        asyncio.run(boot())

        rows = _fetch_rows(boot_phases.PHASE_RESIDENCY_CHECK)
        assert len(rows) == 1, (
            "one row per configured ref, whichever way the answer goes; got "
            f"{[(r.phase, r.reason) for r in boot_phases.recorded_rows()]}")
        assert rows[0].reason == "resident", (
            f"a warm volume answers `resident`; row said {rows[0].reason!r}")
        assert rows[0].bytes == 0, (
            "the bytes column means BYTES MOVED and a resident tree moves "
            "none — putting the tree size here would make a warm boot read as "
            "the fastest transfer the fleet ever did")
        assert f"tree_bytes={OBJECTS * OBJECT_BYTES}" in rows[0].detail, (
            "the size it did NOT have to move belongs on the row, in a field "
            f"that is not `bytes`; detail={rows[0].detail!r}")
        assert boot_phases.phase_class(rows[0].phase) == boot_phases.CLASS_FETCH
    finally:
        origin.close()


def test_an_ABSENT_ref_records_the_check_it_lost_before_the_fetch_it_starts(
    tmp_path: Path, _fine_cadence: Any
) -> None:
    """Both arms, or the row is a survivorship filter.

    If only the `resident` arm emitted, the ladder would show a population in
    which residency checks are always free and always win — and the cold boot's
    check time would silently join `residual_ms`. The reconciliation this
    module promises is a union of MEASURED spans; an unmeasured one is a lie
    of omission, not a gap.
    """
    origin = _Origin()
    try:
        snapshot = _blobs(origin)
        wire = _Wire()
        store = _store(wire, tmp_path / "cas")

        async def run() -> None:
            activity.bind_sink(wire.send, asyncio.get_running_loop())
            mat = CheckpointMaterialization(store)
            boot_phases.bind_servable_probe(lambda: mat.ready)
            mat.configure(_config(3, _REF, snapshot))
            await _settle(mat)

        asyncio.run(run())

        rows = _fetch_rows(boot_phases.PHASE_RESIDENCY_CHECK)
        assert len(rows) == 1 and rows[0].reason == "absent", (
            "a cold pod's residency check must leave a row saying it lost; "
            f"got {[(r.phase, r.reason) for r in rows]}")
        assert _fetch_rows(boot_phases.PHASE_WEIGHTS_FETCH), (
            "and the fetch it fell through to must be its own row")
    finally:
        origin.close()


# ---------------------------------------------------------------------------
# pgw#1613 — the fetch must be an OPEN ACTIVITY, because the compute-child
# watchdog decides a silent event loop by what is open
# ---------------------------------------------------------------------------


def _hang_verdict(*, activity_kind: str, now: float = 100.0) -> str | None:
    """Run the REAL `_ChildSlot._hang_verdict` against a slot that is burning
    CPU (fresh evidence) with a silent loop, and `activity_kind` open.

    Called unbound on a stand-in so the assertion is about the production
    decision function itself, not a copy of its logic.
    """
    from types import SimpleNamespace

    from gen_worker.procsplit.parent import _ChildSlot

    slot = SimpleNamespace(
        # evidence is FRESH: the child's tree is accruing kernel-accounted work,
        # which is what a CAS fill looks like from the parent's /proc sampler.
        liveness_evidence=1234.5,
        liveness_evidence_at=now - 0.5,
        liveness_activity=activity_kind,
        p=SimpleNamespace(_evidence_hold_window=15.0),
    )
    return _ChildSlot._hang_verdict(slot, now)


def test_the_watchdog_HOLDS_a_cpu_burning_child_only_because_an_activity_is_open() -> None:
    """The two halves of pgw#1613, asserted against the real verdict function.

    A child mid-materialization is burning CPU with a starved event loop. What
    decides its life is whether anything is OPEN. This is the RED ARM for the
    fix below: delete the `activity.running(...)` scope in `_materialize` and
    the second assertion is what production does instead — SIGKILL on a live
    pull.
    """
    assert _hang_verdict(activity_kind=activity.KIND_BOOT_MATERIALIZE) == "held", (
        "a fetch that declares itself must survive a starved event loop"
    )
    assert _hang_verdict(activity_kind="") == "loop_wedged_no_activity", (
        "with nothing open the same child is killed — this is the defect, and "
        "it is why the fetch has to open an activity at all"
    )


def test_a_cold_fetch_holds_a_weight_fetch_activity_OPEN_for_its_whole_pull(
    tmp_path: Path, _fine_cadence: Any
) -> None:
    """Measured on minimax-h3 twice, at two pins: a ~105 GB fill silenced the
    child's loop and the watchdog killed it at 356 s / 687 s with `cause=
    watchdog_hang`, no OOM, while the process burned CPU. Nothing was open.

    So: while the pull is in flight, `activity.current()` must be the fetch.
    The observation is taken from INSIDE the funnel — the store's own byte
    callback — because that is the only place that is provably mid-pull.
    """
    origin = _Origin()
    try:
        snapshot = _blobs(origin)
        wire = _Wire()
        seen: List[Any] = []

        async def boot() -> None:
            activity.bind_sink(wire.send, asyncio.get_running_loop())
            store = _store(wire, tmp_path / "cas")
            inner = store.ensure_local

            async def watched(*a: Any, **kw: Any) -> Any:
                # mid-pull, by construction: the funnel has been entered and
                # has not returned.
                seen.append(activity.current())
                return await inner(*a, **kw)

            store.ensure_local = watched  # type: ignore[method-assign]
            mat = CheckpointMaterialization(store)
            mat.configure(_config(1, _REF, snapshot))
            await _settle(mat)
            assert mat.state == STATE_READY

        asyncio.run(boot())

        assert seen, "the cold path never entered the funnel; test proves nothing"
        kinds = [getattr(a, "kind", None) for a in seen]
        assert all(k == activity.KIND_BOOT_MATERIALIZE for k in kinds), (
            "the pull must run under an open weight_fetch activity or the "
            f"compute-child watchdog SIGKILLs it (pgw#1613); saw {kinds}"
        )
    finally:
        origin.close()


def test_a_failed_fetch_ends_the_activity_FAILED_not_completed(
    tmp_path: Path, _fine_cadence: Any
) -> None:
    """`activity.py`'s contract: a silent death is a bug. A materialization
    that dies must not leave a COMPLETED fetch behind for the hub to read as a
    successful pull."""
    origin = _Origin()
    try:
        snapshot = _blobs(origin)
        wire = _Wire()

        async def boot() -> None:
            activity.bind_sink(wire.send, asyncio.get_running_loop())
            store = _store(wire, tmp_path / "cas")

            async def boom(*a: Any, **kw: Any) -> Any:
                raise RuntimeError("origin went away mid-pull")

            store.ensure_local = boom  # type: ignore[method-assign]
            mat = CheckpointMaterialization(store)
            mat.configure(_config(1, _REF, snapshot))
            await _settle(mat)
            assert mat.state == STATE_FAILED
            assert activity.current() is None, "the fetch activity outlived its pull"

        asyncio.run(boot())

        fetches = [
            u for u in wire.updates
            if u.kind == activity.KIND_BOOT_MATERIALIZE
        ]
        assert fetches, "no weight_fetch activity was reported at all"
        assert fetches[-1].state == pb.ActivityState.ACTIVITY_STATE_FAILED, (
            "a dead pull must terminate FAILED, not COMPLETED; last state was "
            f"{fetches[-1].state}"
        )
    finally:
        origin.close()
