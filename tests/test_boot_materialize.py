"""BOOT materializes what the config names, and readiness waits for it."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, List, cast

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
    desired = pb.DesiredResidency(generation=version, disk_refs=[str(ref)])
    desired.snapshots[str(ref)].CopyFrom(snapshot)
    return CheckpointConfig.from_wire(desired)


async def _settle(mat: CheckpointMaterialization) -> None:
    task = mat._task
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


def test_a_FRESH_process_on_a_warm_volume_is_ready_without_moving_a_byte(
    tmp_path: Path, _fine_cadence: Any
) -> None:
    """Pod orq6abdjo28it6 booted onto a volume that already held all 134 GB."""
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
            store.rescan_disk()
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


def test_a_cold_boot_PULLS_and_is_not_ready_until_it_lands(
    tmp_path: Path, _fine_cadence: Any
) -> None:
    """The whole invariant, in one test."""
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


def test_an_unchanged_config_repush_moves_no_bytes_and_keeps_readiness(
    tmp_path: Path, _fine_cadence: Any
) -> None:
    """Every reconnect re-delivers the same config."""
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
            mat.configure(_config(2, _REF, snapshot))
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


def test_a_config_naming_no_refs_is_ready_immediately(tmp_path: Path) -> None:
    """A weightless release has nothing to wait for."""
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
    """A pull that cannot succeed must END, visibly."""
    origin = _Origin()
    try:
        snapshot = _blobs(origin)
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


def test_the_dispatch_resolves_the_tree_BOOT_materialized(
    tmp_path: Path, _fine_cadence: Any
) -> None:
    """The other half of "same code paths", and a latent defect on its own."""
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
        digest = str(snapshot.digest)
        assert tree.name in (digest, digest.split(":", 1)[-1]), (
            f"tree {tree.name} is neither spelling of {digest}")
    finally:
        origin.close()


def test_the_dispatch_supplies_its_own_fetch_pointer(tmp_path: Path) -> None:
    """`RunJob.snapshots` is keyed by ref and ships on every dispatch."""
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
    import gen_worker.weight_position as wp

    monkeypatch.setattr(wp, "STRIDE_MIB", 1)
    monkeypatch.setattr(wp, "MIN_INTERVAL_S", 0.0)
    return None


def test_a_TRUNCATED_warm_tree_is_refused_quarantined_and_refetched(
    tmp_path: Path, _fine_cadence: Any
) -> None:
    """The fleet incident: two endpoints, two volumes, 20x apart, identical `SafetensorError: header too large` on read."""
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

        trees = [p for p in (cas / "snapshots").iterdir() if p.is_dir()]
        assert len(trees) == 1, f"expected one staged tree, got {trees}"
        weights = sorted(trees[0].rglob("*.safetensors"))
        assert weights, f"no weight files in {trees[0]}"
        victim = weights[0]
        full = victim.stat().st_size
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
        final = [p for p in (cas / "snapshots").iterdir() if p.is_dir()]
        assert len(final) == 1, f"expected one tree after recovery, got {final}"
        checker = _store(_Wire(), cas)
        ok, bad = checker._verify_snapshot_tree(final[0], snapshot)
        assert ok, f"the recovered tree still fails verification: {bad}"
    finally:
        origin.close()


def _fetch_rows(phase: str) -> List[Any]:
    return [r for r in boot_phases.recorded_rows()
            if r.phase == phase and r.terminal]


def test_a_pull_that_follows_an_EMPTY_config_still_lands_in_the_boot_ladder(
    tmp_path: Path, _fine_cadence: Any
) -> None:
    """The field sequence, exactly: empty config, servable, THEN the real one."""
    origin = _Origin()
    try:
        snapshot = _blobs(origin)
        wire = _Wire()
        store = _store(wire, tmp_path / "cas")

        async def run() -> None:
            activity.bind_sink(wire.send, asyncio.get_running_loop())
            mat = CheckpointMaterialization(store)
            boot_phases.bind_servable_probe(lambda: mat.ready)

            mat.configure(CheckpointConfig.from_wire(pb.DesiredResidency(generation=1)))
            assert mat.ready
            boot_phases.mark_once(boot_phases.PHASE_HELLO)
            boot_phases.mark_once(boot_phases.PHASE_FIRST_REQUEST_SERVABLE)
            assert boot_phases.servable_ms() is not None

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
        assert not boot_phases.in_boot()
    finally:
        origin.close()


def test_a_WARM_volume_boot_is_a_TIMED_ROW_and_not_a_hole_in_the_ladder(
    tmp_path: Path, _fine_cadence: Any
) -> None:
    """The flagship case, and the one with no instrument at all."""
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
    """Both arms, or the row is a survivorship filter."""
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


def _hang_verdict(*, activity_kind: str, now: float = 100.0) -> str | None:
    from types import SimpleNamespace

    from gen_worker.procsplit.parent import _ChildSlot

    slot = SimpleNamespace(
        liveness_evidence=1234.5,
        liveness_evidence_at=now - 0.5,
        liveness_activity=activity_kind,
        p=SimpleNamespace(_evidence_hold_window=15.0),
    )
    return _ChildSlot._hang_verdict(cast(Any, slot), now)


def test_the_watchdog_HOLDS_a_cpu_burning_child_only_because_an_activity_is_open() -> None:
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
    """Measured on minimax-h3 twice, at two pins: a ~105 GB fill silenced the child's loop and the watchdog killed it at 356 s / 687 s with `cause= watchdog_hang`, no OOM, while the process burned CPU."""
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
    """`activity.py`'s contract: a silent death is a bug."""
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
