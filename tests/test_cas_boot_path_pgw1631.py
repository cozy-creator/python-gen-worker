"""pgw#1631: the CAS boot path on a pod bought for ONE release.

All three of the 2026-08-21 store incidents lived in the bookkeeping ABOVE the
byte bank — a stale-view precondition (pgw#1596), a compat copier (th#2246), an
unclassified errno (pgw#1612) — and none of them in content addressing, which is
what saved z-image and what makes the volume tier possible. So the bank stays
and the multi-tenant machinery around it goes:

* **the headroom gate consumes the fill PLAN**, so a divergent precondition is
  unwritable rather than merely fixed;
* **no GC during boot** — a boot that needs eviction to fit is a sizing bug
  upstream, and evict-and-retry turns it into a slow expensive one. The gate
  refuses with a typed `insufficient_disk` naming the mount;
* **supersede never cancels byte movement** — objects are content-keyed, so
  bytes an old plan is mid-way through fetching are bytes the new plan will find
  present. What supersede means is that the old task's VERDICT stops counting;
* **one writer per object** — tmp+rename under the CAS's own per-object lock is
  the only commit, so th#2246's "every loser rmtree's and restarts from zero" is
  not representable;
* **an ENOSPC is a claim about the SHAPE** (pgw#1612), reported on the reason
  the hub already has a migration path behind.

Everything drives the real store, the real CAS and a real HTTP origin.
"""

from __future__ import annotations

import asyncio
import errno
import shutil
import threading
from pathlib import Path
from typing import Any, Iterator

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
from gen_worker.capability import InsufficientDiskError
from gen_worker.models import cozy_snapshot, disk_errors, fill_plan
from gen_worker.models.refs import WireRef
from gen_worker.models.store import _DISK_GC_MARGIN_BYTES, ModelStore
from gen_worker.pb import worker_scheduler_pb2 as pb

from fill_fixture import (
    FILL_OPS,
    FillContext,
    Origin,
    Tree,
    build_tree,
    reset_fill_memos,
    resident_bytes,
    run_fill,
    sha,
)

OBJECTS = 16
OBJECT_BYTES = 128 * 1024
_REF = WireRef("acme/boot-model")


@pytest.fixture(autouse=True)
def _clean_activity() -> Iterator[None]:
    activity.reset_for_tests()
    yield
    activity.reset_for_tests()


@pytest.fixture(scope="module")
def origin() -> Iterator[Origin]:
    served = Origin()
    try:
        yield served
    finally:
        served.close()


@pytest.fixture(scope="module")
def tree(origin: Origin) -> Tree:
    return build_tree(origin, objects=OBJECTS, object_bytes=OBJECT_BYTES)


def _ctx(tmp_path: Path, tree: Tree, origin: Origin) -> FillContext:
    return FillContext(cache_dir=tmp_path / "cas", tree=tree, origin=origin, ref=_REF)


async def _noop_emit(_msg: Any) -> None:
    return None


class _InBoot:
    """The boot window, expressed the way `boot_phases` really decides it.

    `in_boot()` is "this worker CANNOT SERVE" — true while `_servable_ms` is
    unset. A fresh recorder is therefore already in boot, which is also why the
    steady-state control below has to mark the milestone explicitly rather than
    assume it.
    """

    def __enter__(self) -> "_InBoot":
        boot_phases.reset_for_tests()
        assert boot_phases.in_boot()
        return self

    def __exit__(self, *_exc: object) -> None:
        boot_phases.reset_for_tests()


class _Servable:
    """Past the boot window: the worker has been servable, so this is steady state."""

    def __enter__(self) -> "_Servable":
        boot_phases.reset_for_tests()
        # `first_request_servable` is HELD until `hello` — a worker the hub
        # cannot reach is not servable (pgw#797) — so the milestone needs the
        # hello row first, exactly as a real boot does.
        boot_phases.mark(boot_phases.PHASE_HELLO)
        boot_phases.mark(boot_phases.PHASE_FIRST_REQUEST_SERVABLE)
        assert not boot_phases.in_boot()
        return self

    def __exit__(self, *_exc: object) -> None:
        boot_phases.reset_for_tests()


# ---------------------------------------------------------------------------
# 1. The gate consumes the plan — one predicate, no second derivation
# ---------------------------------------------------------------------------


def test_the_fill_and_the_gate_call_the_same_predicate(
    tmp_path: Path, tree: Tree, origin: Origin
) -> None:
    """`cozy_snapshot`'s skip and the gate's plan are ONE function.

    pgw#1596 happened because there were two: the gate walked the manifest, the
    fill walked the store. This is the structural half — both routes are proven
    to go through `fill_plan.is_present` by making that one function lie and
    watching BOTH answers move together.
    """

    from gen_worker._vendor.tensorfs import LocalCAS

    cas = LocalCAS(tmp_path / "cas")
    body = tree.files[0][1]
    cas.put_bytes(body)

    plan = fill_plan.plan_fill(cas, [
        type("F", (), {"digest": sha(body), "size_bytes": len(body), "path": "a"})(),
    ])
    assert plan.present_bytes == len(body) and plan.missing_bytes == 0

    # `resident()` inside `_ensure_objects` is the fill's own skip; it is the
    # same call, so a store that reports absent makes both say "fetch".
    assert fill_plan.is_present(cas, sha(body), len(body))
    assert not fill_plan.is_present(cas, sha(body), len(body) + 1), (
        "a declared size mismatch is ABSENT — a truncated object must not be "
        "skipped by either side"
    )


def test_the_fill_loop_routes_its_skip_through_fill_plan() -> None:
    """The lint half: `_ensure_objects` may not spell the predicate itself."""

    import inspect

    source = inspect.getsource(cozy_snapshot.CozySnapshotDownloader._ensure_objects)
    assert "fill_plan.is_present" in source, (
        "the fill's skip must call the shared predicate"
    )
    assert "cas.contains(" not in source, (
        "a second spelling of the skip predicate is how pgw#1596 happened"
    )


# ---------------------------------------------------------------------------
# 2. No GC during boot
# ---------------------------------------------------------------------------


def test_a_boot_that_does_not_fit_REFUSES_instead_of_evicting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE RULING. Boot never evicts to make room for itself.

    A boot that needs eviction to fit is a sizing bug upstream (th#2264).
    Evicting turns a fast, actionable refusal into th#2246's shape: two A100
    pods, ~$1.72, and a human cancelling a retry loop that was doomed before it
    was paid for. The refusal names the mount and its statvfs totals so the hub
    can demote the shape rather than re-buy it.
    """

    calls: list[int] = []
    store = ModelStore(_noop_emit, cache_dir=tmp_path / "cas", disk_free_bytes_fn=lambda: 0)
    monkeypatch.setattr(
        ModelStore, "gc_disk",
        lambda self, target, exclude=(): calls.append(int(target)),
    )
    plan = fill_plan.FillPlan(
        missing=(fill_plan.PlannedObject("sha256:" + "a" * 64, 4096),)
    )

    with _InBoot():
        with pytest.raises(InsufficientDiskError) as caught:
            asyncio.run(store._ensure_disk_headroom(_REF, plan))

    assert calls == [], f"gc_disk ran during boot: targets={calls}"
    text = str(caught.value)
    assert "mount=" in text and "statvfs_total=" in text and "statvfs_free=" in text
    assert "4096 missing" in text
    assert "no GC" in text, "the refusal must say WHY it did not evict"
    assert caught.value.required_bytes == 4096


def test_steady_state_still_evicts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """THE CONTROL. Outside boot, eviction is what the disk tier is FOR.

    Deleting GC everywhere would be a different bug: a long-lived pod whose LRU
    never releases anything fills its disk and refuses work it could have done.
    """

    calls: list[int] = []
    freed = {"after": 0}
    store = ModelStore(
        _noop_emit,
        cache_dir=tmp_path / "cas",
        disk_free_bytes_fn=lambda: freed["after"],
    )

    def _gc(self: Any, target: int, exclude: Any = ()) -> None:
        calls.append(int(target))
        freed["after"] = int(target)  # the eviction worked

    monkeypatch.setattr(ModelStore, "gc_disk", _gc)
    plan = fill_plan.FillPlan(
        missing=(fill_plan.PlannedObject("sha256:" + "a" * 64, 4096),)
    )

    with _Servable():
        asyncio.run(store._ensure_disk_headroom(_REF, plan))
    assert calls == [4096 + _DISK_GC_MARGIN_BYTES], (
        "steady state must still try to make room, and must ask for exactly "
        f"the plan's missing bytes plus the margin; got {calls}"
    )


# ---------------------------------------------------------------------------
# 3. Supersede never cancels byte movement
# ---------------------------------------------------------------------------


def _config(version: int, ref: WireRef, snapshot: pb.Snapshot) -> CheckpointConfig:
    desired = pb.DesiredResidency(generation=version, disk_refs=[str(ref)])
    desired.snapshots[str(ref)].CopyFrom(snapshot)
    return CheckpointConfig.from_wire(desired)


def test_an_unchanged_config_is_a_no_op_and_starts_nothing(
    tmp_path: Path, tree: Tree, origin: Origin
) -> None:
    """A RECONNECT re-delivers the same config. It must move no bytes.

    The re-entry that armed pgw#1596 was a supersede that changed nothing about
    the ref — so this is the arm that keeps the whole class from starting.
    """

    ctx = _ctx(tmp_path, tree, origin)
    snapshot = tree.snapshot()

    async def run() -> tuple[str, int]:
        store = ctx.store()
        mat = CheckpointMaterialization(store)
        mat.configure(_config(1, _REF, snapshot))
        task = mat._task
        assert task is not None
        await task
        for _ in range(8):
            await asyncio.sleep(0)
        after_first = origin.wire_bytes
        # Same identity, a NEW version number: the version is not part of
        # identity, so this changes nothing.
        mat.configure(_config(2, _REF, snapshot))
        assert mat._task is task, "an unchanged config started a second fill"
        return mat.state, after_first

    origin.reset()
    state, first = asyncio.run(run())
    assert state == STATE_READY
    assert first == tree.total_bytes
    assert origin.wire_bytes == first, "an unchanged re-push moved bytes"


def test_a_supersede_lets_the_in_flight_fill_DRAIN(
    tmp_path: Path, tree: Tree, origin: Origin
) -> None:
    """THE CHANGE. A new config does not cancel the old one's byte movement.

    Objects are release-agnostic by content key, so bytes the old plan is
    mid-way through fetching are bytes the NEW plan's fill finds present.
    Cancelling them buys nothing and costs the re-fetch — the class th#2204
    measured as phantom downloads and pgw#1596 turned into a disk-capacity
    incident.

    Asserted three ways: the old task is not cancelled, its bytes are in the
    CAS when it finishes, and its verdict does not touch readiness.
    """

    ctx = _ctx(tmp_path, tree, origin)
    snapshot = tree.snapshot()
    other = tree.snapshot()
    other.digest = "sha256:" + "d" * 64  # a genuinely different plan

    async def run() -> dict[str, Any]:
        store = ctx.store()
        mat = CheckpointMaterialization(store)
        # Slow the origin so the fill is genuinely MID-FLIGHT when the
        # supersede lands. The wait below is on BYTES SERVED, not on a clock:
        # "in flight" is a progress fact, and asserting it off elapsed time is
        # the magic-timeout shape this repo bans.
        origin.delay_s = 0.02
        mat.configure(_config(1, _REF, snapshot))
        first = mat._task
        assert first is not None
        await origin.wait_until_served(OBJECT_BYTES)
        assert not first.done(), "the fill finished before it could be superseded"
        mat.configure(_config(2, WireRef("acme/other-model"), other))
        second = mat._task
        assert second is not None and second is not first, (
            "a changed config must start a new fill"
        )
        assert not first.cancelled() and not first.done(), (
            "the superseded fill was cancelled — its bytes are content-keyed "
            "and the successor wants them"
        )
        state_at_supersede = mat.state
        served_at_supersede = origin.wire_bytes
        origin.delay_s = 0.0
        await asyncio.gather(first, second, return_exceptions=True)
        for _ in range(8):
            await asyncio.sleep(0)
        return {
            "first_cancelled": first.cancelled(),
            "state_at_supersede": state_at_supersede,
            "served_at_supersede": served_at_supersede,
            "state": mat.state,
        }

    origin.reset()
    out = asyncio.run(run())
    assert 0 < out["served_at_supersede"] < tree.total_bytes, (
        "the supersede did not land mid-flight, so this proves nothing about "
        "cancelling byte movement"
    )

    assert out["first_cancelled"] is False
    assert out["state_at_supersede"] == STATE_MATERIALIZING
    # The superseded plan's bytes LANDED — that is the whole point.
    assert resident_bytes(ctx.cache_dir, tree) == tree.total_bytes, (
        "the drained fill did not bank its objects"
    )
    # ...and a successor over the same objects now fetches nothing.
    origin.reset()
    reset_fill_memos()
    run_fill(_ctx_at(ctx), FILL_OPS[0])
    assert origin.wire_bytes == 0, (
        f"the successor re-fetched {origin.wire_bytes} bytes the superseded "
        f"fill had already banked"
    )


def _ctx_at(ctx: FillContext) -> FillContext:
    """A fresh process view of the SAME store."""
    return FillContext(
        cache_dir=ctx.cache_dir, tree=ctx.tree, origin=ctx.origin, ref=_REF,
    )


def test_a_superseded_fill_decides_nothing_about_readiness(
    tmp_path: Path, tree: Tree, origin: Origin
) -> None:
    """A stale verdict must not strand — or falsely advertise — this pod.

    Draining the old fill is only safe if its ANSWER stops counting. A stale
    FAILED would strand a pod whose live config is fine; a stale READY would
    advertise weights the live config does not name.
    """

    ctx = _ctx(tmp_path, tree, origin)
    doomed = tree.snapshot()
    doomed.digest = "sha256:" + "e" * 64
    for f in doomed.files:
        f.url = f.url.rsplit("/", 1)[0] + "/sha256:" + "f" * 64  # 404s

    live = tree.snapshot()

    async def run() -> str:
        store = ctx.store()
        mat = CheckpointMaterialization(store)
        mat.configure(_config(1, WireRef("acme/doomed"), doomed))
        first = mat._task
        assert first is not None
        mat.configure(_config(2, _REF, live))
        second = mat._task
        assert second is not None
        await asyncio.gather(first, second, return_exceptions=True)
        for _ in range(8):
            await asyncio.sleep(0)
        return mat.state

    origin.reset()
    state = asyncio.run(run())
    assert state == STATE_READY, (
        f"the superseded fill's failure decided readiness: state={state}"
    )


# ---------------------------------------------------------------------------
# 4. One writer per object
# ---------------------------------------------------------------------------


def test_concurrent_writers_of_one_object_converge_without_restarting(
    tmp_path: Path,
) -> None:
    """th#2246's "every loser rmtree's and restarts from zero", made unrepresentable.

    The unit of work is ONE object under the CAS's own per-object lock, and the
    only commit is tmp+link/replace. Contenders therefore converge: every writer
    succeeds, exactly one object exists, its bytes are correct, and no writer
    has to discard work and start over. No extra machinery was needed for this —
    it is a property of the bank the review said to keep, and this test is here
    so a change that breaks it cannot land quietly.
    """

    from gen_worker._vendor.tensorfs import LocalCAS

    cas = LocalCAS(tmp_path / "cas")
    payload = b"contended-object-" + b"z" * 200_000
    digest = sha(payload)
    barrier = threading.Barrier(8)
    errors: list[BaseException] = []

    def writer(i: int) -> None:
        source = tmp_path / f"src-{i}.bin"
        source.write_bytes(payload)
        try:
            barrier.wait(30)
            cas.put_file(source, expected=digest, size=len(payload))
        except BaseException as exc:  # noqa: BLE001 — the test IS the report
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)

    assert not errors, f"a contender failed rather than converging: {errors}"
    assert cas.contains(digest, size=len(payload))
    assert cas.object_path(digest).read_bytes() == payload
    # Nothing left behind in the temp lane: a loser that rmtree'd a shared
    # directory is exactly the shape that made th#2246 restart from zero.
    assert list(cas.tmp.glob("*")) == []


# ---------------------------------------------------------------------------
# 5. pgw#1612 — an ENOSPC is a claim about the SHAPE
# ---------------------------------------------------------------------------


def test_an_enospc_anywhere_in_the_chain_is_classified() -> None:
    """Errno 28, however it is wrapped. Reading only the outermost type is how
    a deterministic disk failure reads as a generic one."""

    bare = OSError(errno.ENOSPC, "No space left on device", "/tmp/x/y.safetensors")
    assert disk_errors.out_of_space(bare) is bare

    wrapped = RuntimeError("load failed")
    wrapped.__cause__ = bare
    assert disk_errors.out_of_space(wrapped) is bare

    ctx_only = RuntimeError("export failed")
    ctx_only.__context__ = bare
    assert disk_errors.out_of_space(ctx_only) is bare

    inside_shutil = shutil.Error([("a", "b", bare)])
    assert disk_errors.out_of_space(inside_shutil) is bare

    # And nothing else is a capacity claim. Inventing one out of an unrelated
    # error is worse than the generic bucket: the hub ACTS on this token.
    assert disk_errors.out_of_space(OSError(errno.EACCES, "denied")) is None
    assert disk_errors.out_of_space(RuntimeError("disk is full, honest")) is None


def test_the_classified_reason_names_the_mount_and_its_totals(tmp_path: Path) -> None:
    """Carry the FACT, not just the token.

    "the container disk was 100 GB and the boot needed 121 GB" has to be
    readable off the wire, or a lane re-derives it weeks later — which is
    exactly what th#2246 had to do.
    """

    victim = tmp_path / "cas" / "objects" / "aa" / "blob"
    victim.parent.mkdir(parents=True)
    exc = OSError(errno.ENOSPC, "No space left on device", str(victim))
    typed = disk_errors.as_insufficient_disk(
        exc, doing="materializing acme/x", fallback_path=tmp_path
    )
    assert typed is not None
    text = str(typed)
    assert "materializing acme/x" in text
    assert str(victim) in text
    assert "statvfs_total=" in text and "statvfs_free=" in text


def test_the_hub_reason_is_the_bare_token_and_the_facts_ride_the_activity_stream(
    tmp_path: Path,
) -> None:
    """RED-ARMED WIRE CONTRACT. `connect_worker.go:3737` reads
    `strings.TrimSpace(ev.GetError())` and compares it for EXACT equality
    against `insufficient_disk`, so appending detail after a colon — the way
    `download_failed` does — silently disables the entire migration path.

    The facts therefore go where they are readable: a typed `residency_fault`
    activity event, which lands in `worker_activity_events`. pgw#1620's lesson —
    a confession that only reaches a pod's stdout reaches nobody.
    """

    sent: list[pb.WorkerMessage] = []

    async def emit(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    async def run() -> None:
        activity.bind_sink(emit, asyncio.get_running_loop())
        store = ModelStore(emit, cache_dir=tmp_path / "cas")
        await store.report_insufficient_disk(
            _REF, "no space left while materializing acme/x; mount=/ statvfs_total=7"
        )
        for _ in range(8):
            await asyncio.sleep(0)

    asyncio.run(run())

    failures = [
        m.model_event for m in sent
        if m.WhichOneof("msg") == "model_event"
        and m.model_event.state == pb.MODEL_STATE_FAILED
    ]
    assert len(failures) == 1
    assert failures[0].error == "insufficient_disk", (
        f"the hub compares this for EXACT equality; got {failures[0].error!r}"
    )

    facts = [
        m.activity_update for m in sent
        if m.WhichOneof("msg") == "activity_update"
        and m.activity_update.kind == activity.KIND_RESIDENCY_FAULT
    ]
    assert facts, "the mount facts reached nobody"
    assert facts[0].phase == "insufficient_disk"
    assert "statvfs_total=7" in facts[0].detail


def test_a_boot_enospc_reaches_the_hub_as_insufficient_disk(
    tmp_path: Path, tree: Tree, origin: Origin, monkeypatch: pytest.MonkeyPatch
) -> None:
    """pgw#1612's ask, end to end through the boot seam.

    th#2246: `qwen-image` ENOSPC'd on `8gpqows0j349gm` (A100-SXM4-80GB, 100 GB)
    and requeued onto `3zod6pwvn10f4y` — another A100-SXM4-80GB with the same
    100 GB — at $1.59/hr until a human cancelled it. Deterministic failure,
    unbounded retry, real money. The pre-fix code reports this as a generic
    failure; the failure below is a LOAD-side ENOSPC, not the download's, which
    is the half that had no classifier at all.
    """

    ctx = _ctx(tmp_path, tree, origin)
    sent: list[pb.WorkerMessage] = []

    async def emit(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    boom = OSError(errno.ENOSPC, "No space left on device", str(tmp_path / "cas" / "x"))

    async def run() -> str:
        store = ModelStore(emit, cache_dir=ctx.cache_dir)
        # An ENOSPC raised INSIDE the materialization, from something that is
        # not the downloader — a cache write, an export, a projection.
        async def _explode(*_a: Any, **_k: Any) -> Path:
            raise RuntimeError("staging the tree failed") from boom

        monkeypatch.setattr(ModelStore, "ensure_local", _explode)
        mat = CheckpointMaterialization(store)
        mat.configure(_config(1, _REF, tree.snapshot()))
        task = mat._task
        assert task is not None
        await asyncio.gather(task, return_exceptions=True)
        for _ in range(8):
            await asyncio.sleep(0)
        return mat.state

    state = asyncio.run(run())
    assert state == STATE_FAILED

    reasons = [
        m.model_event.error for m in sent
        if m.WhichOneof("msg") == "model_event"
        and m.model_event.state == pb.MODEL_STATE_FAILED
    ]
    assert "insufficient_disk" in reasons, (
        f"a boot ENOSPC reached the hub as {reasons} — the hub requeues that "
        f"onto a machine with the identical container_disk_gb_requested"
    )
