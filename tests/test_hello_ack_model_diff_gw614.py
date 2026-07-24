"""gw#614(B)/gw#623: on_hello_ack model-set-diff handling.

The th#961 livelock cancelled the in-flight self_mint_compile at phase=load
4,602 times because EVERY ack replaced (cancelled + restarted) the
residency-reconcile task. gw#614 kept the reconcile when the semantic set is
unchanged; gw#623 tightens the cancel test to the ACTIVE work item: an ack
may only cancel an in-flight load when the model that load is FOR left the
set (or changed snapshot identity / resolution). Any other churn — presigned
URL rotation (the hub re-signs snapshot URLs on every ~15s config rebuild),
sibling refs added or removed, plan rewrites — lets the load finish and
re-converges afterwards (level-triggered loop). The live signature this
kills: `worker activity {warmup,self_mint_compile} failed at phase load:
CancelledError` on nearly every first boot, requeue cycles exhausting 60-min
request deadlines."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from gen_worker.executor import Executor, ModelStore
from gen_worker.lifecycle import Lifecycle
from gen_worker.pb import worker_scheduler_pb2 as pb

_REF_A = "acme/model-a"
_REF_B = "acme/model-b"
_DIGEST = "blake3:" + "a" * 64


async def _noop_send(msg) -> None:  # pragma: no cover
    pass


class _FakeTransport:
    def __init__(self) -> None:
        self.connected = True
        self.sent: list[pb.WorkerMessage] = []
        self.queue = SimpleNamespace(pending_result_keys=set())

    async def send(self, msg: pb.WorkerMessage) -> None:
        self.sent.append(msg)

    async def prepend_reconnect(self, messages) -> None:
        self.sent.extend(messages)


def _lifecycle(tmp_path: Path) -> Lifecycle:
    store = ModelStore(_noop_send, cache_dir=tmp_path)
    ex = Executor([], _noop_send, store=store)
    lc = Lifecycle(
        SimpleNamespace(worker_jwt="", worker_id="w-ack-diff",
                        runpod_pod_id="", worker_image_digest=""),
        ex,
    )
    lc.transport = _FakeTransport()
    return lc


def _ack(
    generation: int, refs: list[str], *, resolved: str = "", url: str = "",
) -> pb.HelloAck:
    ack = pb.HelloAck(
        protocol_version=pb.PROTOCOL_VERSION_CURRENT,
        desired_residency=pb.DesiredResidency(
            generation=generation,
            disk_refs=refs,
            snapshots={
                r: pb.Snapshot(digest=_DIGEST, files=[pb.SnapshotFile(
                    path="weights.safetensors", size_bytes=8,
                    blake3="b" * 64, url=url or "https://cas/stable",
                )])
                for r in refs
            },
        ),
    )
    if resolved:
        ack.resolutions.add(ref=_REF_A, resolved_ref=resolved, lane="w8a8")
    return ack


def test_identical_model_set_keeps_the_running_reconcile(tmp_path) -> None:
    """Red-first: a benign ack (same refs/snapshots/hot, bumped generation —
    the th#961 +2/cycle shape) must NOT cancel an in-flight reconcile; the
    self-mint riding it needs ~190s uninterrupted."""
    lc = _lifecycle(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()
    cancelled: list[str] = []
    ensured: list[str] = []

    async def _blocking_ensure(ref, snapshot=None, *, binding=None):
        ensured.append(ref)
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.append(ref)
            raise
        return tmp_path

    async def _no_revalidate(ref, snapshot=None):
        return None

    lc.executor.store.ensure_local = _blocking_ensure  # type: ignore[method-assign]
    lc.executor.revalidate_snapshot_identity = _no_revalidate  # type: ignore[method-assign]

    async def run() -> None:
        await lc.on_hello_ack(_ack(1, [_REF_A]))
        await asyncio.wait_for(started.wait(), timeout=5)
        first_task = lc._residency_task
        assert first_task is not None and not first_task.done()

        # Benign rewrite: same semantic model set, generation bumped.
        await lc.on_hello_ack(_ack(3, [_REF_A]))
        await asyncio.sleep(0)
        assert lc._residency_task is first_task, (
            "an ack with an IDENTICAL model set must not replace the "
            "running reconcile (th#961: each replace kills the in-flight "
            "self-mint at phase=load)")
        assert not first_task.cancelled() and cancelled == []
        # Non-model deltas still applied: the observed generation advanced.
        assert lc._observed_residency_generation == 3

        # gw#623: a sibling ref joining the set is NOT a reason to discard
        # the in-flight load of _REF_A — the loop converges to the new set
        # (including _REF_B) after the active item completes.
        await lc.on_hello_ack(_ack(4, [_REF_A, _REF_B]))
        await asyncio.sleep(0)
        assert lc._residency_task is first_task, (
            "adding a sibling ref must not cancel the active load")
        assert cancelled == []
        release.set()
        await asyncio.wait_for(first_task, timeout=5)
        assert _REF_B in ensured, (
            "after the active load finishes the loop must converge to the "
            "updated set (level-triggered re-pass)")

    asyncio.run(run())


def test_url_rotation_is_not_a_model_change(tmp_path) -> None:
    """gw#623 root cause: the hub re-presigns snapshot file URLs on every
    release-config rebuild (~15s TTL) and deliberately EXCLUDES them from
    its own HelloAck semantic hash. Hashing them worker-side cancelled the
    in-flight warmup load on every URL refresh that rode a resent ack
    (live: CancelledError at phase=load every ~14s during ack storms,
    every ~10min otherwise)."""
    lc = _lifecycle(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()
    cancelled: list[str] = []

    async def _blocking_ensure(ref, snapshot=None, *, binding=None):
        started.set()
        try:
            await release.wait()
        except asyncio.CancelledError:
            cancelled.append(ref)
            raise
        return tmp_path

    async def _no_revalidate(ref, snapshot=None):
        return None

    lc.executor.store.ensure_local = _blocking_ensure  # type: ignore[method-assign]
    lc.executor.revalidate_snapshot_identity = _no_revalidate  # type: ignore[method-assign]

    async def run() -> None:
        await lc.on_hello_ack(_ack(1, [_REF_A], url="https://cas/sig-1"))
        await asyncio.wait_for(started.wait(), timeout=5)
        first_task = lc._residency_task
        assert first_task is not None and not first_task.done()

        await lc.on_hello_ack(_ack(3, [_REF_A], url="https://cas/sig-2"))
        await asyncio.sleep(0)
        assert lc._residency_task is first_task, (
            "a presigned-URL rotation is not a model change and must not "
            "touch the running reconcile")
        assert cancelled == []
        release.set()
        await asyncio.wait_for(first_task, timeout=5)

    asyncio.run(run())


def test_active_ref_leaving_the_set_cancels(tmp_path) -> None:
    """The one case that MUST still cancel: the model the in-flight load is
    FOR left the desired set."""
    lc = _lifecycle(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()
    cancelled: list[str] = []

    async def _blocking_ensure(ref, snapshot=None, *, binding=None):
        if ref == _REF_A:
            started.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.append(ref)
                raise
        return tmp_path

    async def _no_revalidate(ref, snapshot=None):
        return None

    lc.executor.store.ensure_local = _blocking_ensure  # type: ignore[method-assign]
    lc.executor.revalidate_snapshot_identity = _no_revalidate  # type: ignore[method-assign]

    async def run() -> None:
        await lc.on_hello_ack(_ack(1, [_REF_A]))
        await asyncio.wait_for(started.wait(), timeout=5)
        first_task = lc._residency_task

        await lc.on_hello_ack(_ack(2, [_REF_B]))
        await asyncio.sleep(0)
        assert lc._residency_task is not first_task
        assert cancelled == [_REF_A]
        task = lc._residency_task
        assert task is not None
        await asyncio.wait_for(task, timeout=5)

    asyncio.run(run())


def test_changed_resolutions_count_as_a_model_set_change(tmp_path) -> None:
    """A precision-ladder repick for the ACTIVELY-LOADING ref IS a semantic
    model change: the in-flight load is for the wrong flavor and must be
    cancelled and restarted."""
    lc = _lifecycle(tmp_path)
    started = asyncio.Event()
    release = asyncio.Event()

    async def _blocking_ensure(ref, snapshot=None, *, binding=None):
        started.set()
        await release.wait()
        return tmp_path

    async def _no_revalidate(ref, snapshot=None):
        return None

    lc.executor.store.ensure_local = _blocking_ensure  # type: ignore[method-assign]
    lc.executor.revalidate_snapshot_identity = _no_revalidate  # type: ignore[method-assign]

    async def run() -> None:
        await lc.on_hello_ack(_ack(1, [_REF_A], resolved=f"{_REF_A}#fp8-w8a8"))
        await asyncio.wait_for(started.wait(), timeout=5)
        first_task = lc._residency_task

        await lc.on_hello_ack(_ack(2, [_REF_A], resolved=f"{_REF_A}#nvfp4"))
        assert lc._residency_task is not first_task, (
            "a repicked resolution changes what the worker must serve — "
            "the reconcile restarts")
        release.set()
        task = lc._residency_task
        assert task is not None
        await asyncio.wait_for(task, timeout=5)

    asyncio.run(run())
