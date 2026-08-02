"""th#1330 (th#1316 worker half): a declared disk_ref the hub's OWN
resolutions have replaced is not materialized, and is not GC-protected.

Measured on prod (release tensorhub/sdxl 0.2.23, L4, pod tp10exbz4vnv7q):
the hub's desired disk set carried BOTH `tensorhub/nova-anime-xl:prod` and
`tensorhub/nova-anime-xl:prod#fp8` while the worker's own resolutions mapped
the first onto the second. The worker executed the list verbatim and serially,
so a 6.94 GB bf16 base landed at +178 s ahead of the 4.38 GB fp8 variant it
was meant to replace (+230 s) — 144 s of a 270 s cold boot, for weights that
were never loaded (`lane=fp8-w8a16`).

th#1316 fixed the hub-side cause of the pick being retracted. It did NOT give
the fossil an exit: declared and effective spellings collapse to one key in
the hub's removal layer, so once both are desired neither can be dropped
alone. This is the worker-side backstop for the whole class, and it is exact:
a declared ref is skipped only when the ref it resolves to is desired in the
SAME generation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from gen_worker import activity as activity_mod
from gen_worker.executor import Executor, ModelStore
from gen_worker.lifecycle import Lifecycle
from gen_worker.pb import worker_scheduler_pb2 as pb

_BARE = "tensorhub/nova-anime-xl:prod"
_FP8 = "tensorhub/nova-anime-xl:prod#fp8"
_VAE = "tensorhub/sdxl-vae-fp16-fix:prod"
_DIGEST = "blake3:" + "a" * 64


async def _noop_send(msg) -> None:  # pragma: no cover - never asserted on
    pass


class _FakeTransport:
    def __init__(self) -> None:
        self.connected = True
        self.sent: list = []
        self.queue = SimpleNamespace(pending_result_keys=set())

    async def send(self, msg) -> None:
        self.sent.append(msg)

    async def prepend_reconnect(self, messages) -> None:
        self.sent.extend(messages)


def _lifecycle(tmp_path: Path) -> Lifecycle:
    store = ModelStore(_noop_send, cache_dir=tmp_path)
    ex = Executor([], _noop_send, store=store)
    lc = Lifecycle(
        SimpleNamespace(bootstrap_worker_jwt="", worker_id="w-th1330",
                        runpod_pod_id="", worker_image_digest=""),
        ex,
    )
    lc.transport = _FakeTransport()
    return lc


def _ack(generation: int, refs: list, resolutions: dict) -> pb.HelloAck:
    ack = pb.HelloAck(
        protocol_version=pb.PROTOCOL_VERSION_CURRENT,
        desired_residency=pb.DesiredResidency(
            generation=generation,
            disk_refs=refs,
            snapshots={
                r: pb.Snapshot(digest=_DIGEST, files=[pb.SnapshotFile(
                    path="weights.safetensors", size_bytes=8,
                    blake3="b" * 64, url="https://cas/" + r,
                )])
                for r in refs
            },
        ),
    )
    for declared, resolved in resolutions.items():
        ack.resolutions.add(ref=declared, resolved_ref=resolved,
                            lane="fp8-w8a16")
    return ack


def _drive(lc: Lifecycle, ack: pb.HelloAck) -> list:
    """Run one hello-ack + one full reconcile pass; return refs fetched."""
    ensured: list = []

    async def _ensure(ref, snapshot=None, *, binding=None):
        ensured.append(ref)
        return Path("/tmp")

    async def _no_revalidate(ref, snapshot=None):
        return None

    lc.executor.store.ensure_local = _ensure  # type: ignore[method-assign]
    lc.executor.revalidate_snapshot_identity = _no_revalidate  # type: ignore

    async def run() -> None:
        await lc.on_hello_ack(ack)
        task = lc._residency_task
        if task is not None:
            await asyncio.wait_for(task, timeout=10)

    asyncio.run(run())
    return ensured


def test_superseded_declared_ref_is_not_fetched(tmp_path) -> None:
    lc = _lifecycle(tmp_path)
    events: list = []
    previous = activity_mod._sink
    activity_mod._sink = events.append
    try:
        fetched = _drive(lc, _ack(1, [_BARE, _FP8, _VAE], {_BARE: _FP8}))
    finally:
        activity_mod._sink = previous

    assert _BARE not in fetched, (
        "the declared bf16 spelling was materialized even though this "
        "worker's own resolution maps it onto the fp8 variant desired in "
        "the same generation — this is the 6.94 GB / 144 s th#1316 waste")
    assert _FP8 in fetched and _VAE in fetched, (
        "only the superseded spelling may be skipped; the pick and every "
        "unresolved ref must still converge")

    # The superseded ref must also lose its GC protection, or the unused
    # base outranks genuinely cold refs in the preserve set.
    assert _BARE not in lc.executor.store.keep
    assert _FP8 in lc.executor.store.keep and _VAE in lc.executor.store.keep

    # Never silent: one typed hub-visible event per (declared -> resolved).
    kinds = [(e.kind, e.phase) for e in events]
    assert ("residency_ref_superseded", "skipped") in kinds, kinds
    superseded_events = [e for e in events
                         if e.kind == "residency_ref_superseded"]
    assert len(superseded_events) == 1
    assert _BARE in superseded_events[0].detail
    assert _FP8 in superseded_events[0].detail


def test_declared_ref_without_its_resolved_twin_is_still_fetched(tmp_path) -> None:
    """The guard is exact. A lane override that keeps the canonical spelling
    (th#913 cross-family) lists ONLY the declared ref — skipping it would
    strand the request that asked for it."""
    lc = _lifecycle(tmp_path)
    fetched = _drive(lc, _ack(1, [_BARE, _VAE], {_BARE: _FP8}))
    assert _BARE in fetched, (
        "the resolved twin is NOT desired, so the declared spelling is the "
        "only artifact the hub asked for and must be materialized")
    assert _BARE in lc.executor.store.keep


def test_no_resolutions_fetches_every_declared_ref(tmp_path) -> None:
    lc = _lifecycle(tmp_path)
    fetched = _drive(lc, _ack(1, [_BARE, _VAE], {}))
    assert sorted(fetched) == sorted([_BARE, _VAE])
