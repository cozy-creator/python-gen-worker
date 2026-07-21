"""gw#614(B): on_hello_ack model-set-diff cancel — th#961 defense in depth.

The th#961 livelock cancelled the in-flight self_mint_compile at phase=load
4,602 times because EVERY ack replaced (cancelled + restarted) the
residency-reconcile task. The hub no longer rewrites plans benignly mid-mint
(tensorhub cb85c690), but the worker defends itself too: an ack whose
semantic model set (resolutions + disk_refs + snapshots + hot) is unchanged
must keep the running reconcile; a genuinely changed set cancels as before.
"""

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


def _ack(generation: int, refs: list[str], *, resolved: str = "") -> pb.HelloAck:
    ack = pb.HelloAck(
        protocol_version=pb.PROTOCOL_VERSION_CURRENT,
        desired_residency=pb.DesiredResidency(
            generation=generation,
            disk_refs=refs,
            snapshots={r: pb.Snapshot(digest=_DIGEST) for r in refs},
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

        # A genuinely changed model set cancels as before.
        await lc.on_hello_ack(_ack(4, [_REF_A, _REF_B]))
        await asyncio.sleep(0)
        assert lc._residency_task is not first_task
        assert cancelled == [_REF_A], "the changed set must cancel the old task"
        release.set()
        task = lc._residency_task
        assert task is not None
        await asyncio.wait_for(task, timeout=5)

    asyncio.run(run())


def test_changed_resolutions_count_as_a_model_set_change(tmp_path) -> None:
    """A precision-ladder repick (same disk refs, different resolution map)
    IS a semantic model change: the reconcile must restart on it."""
    lc = _lifecycle(tmp_path)
    release = asyncio.Event()

    async def _blocking_ensure(ref, snapshot=None, *, binding=None):
        await release.wait()
        return tmp_path

    async def _no_revalidate(ref, snapshot=None):
        return None

    lc.executor.store.ensure_local = _blocking_ensure  # type: ignore[method-assign]
    lc.executor.revalidate_snapshot_identity = _no_revalidate  # type: ignore[method-assign]

    async def run() -> None:
        await lc.on_hello_ack(_ack(1, [_REF_A], resolved=f"{_REF_A}#fp8-w8a8"))
        first_task = lc._residency_task
        await asyncio.sleep(0)

        await lc.on_hello_ack(_ack(2, [_REF_A], resolved=f"{_REF_A}#nvfp4"))
        assert lc._residency_task is not first_task, (
            "a repicked resolution changes what the worker must serve — "
            "the reconcile restarts")
        release.set()
        task = lc._residency_task
        assert task is not None
        await asyncio.wait_for(task, timeout=5)

    asyncio.run(run())
