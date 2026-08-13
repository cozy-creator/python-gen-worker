"""pgw#674 rotation preload: the NEXT checkpoint stages while jobs compute.

WORKER-RESIDENCY-DESIGN "Rotating double-buffer serving" (Paul-ratified):
load model-B while model-A runs inference; rotate on completion; the GPU
stays hot. Pinned here, through the REAL executor/preloader code paths
(fakes only at the download and CUDA boundaries):

  1. the preloader stages a desired NEXT instance to a READY record while
     the executor is NOT idle (a live unfinished job) — the old
     reconcile-only path is tenant-idle-gated, so this is the behavior
     that did not exist before;
  2. a later dispatch of that instance is a pure cache hit: no download,
     no load, no warm run — visible swap ~0 (double-buffer);
  3. the pgw#638 fence carve-out: a ref RESIDENT under a different
     identity than the desired snapshot is never touched by the preloader;
  4. when VRAM cannot hold both (fits() False), staging is COMPONENT-FIRST:
     exclusive components load on CPU into the shared cache (by content
     digest) and dispatch-time injection consumes them — from_pretrained
     skips those components' disk loads;
  5. the pinned pool bounds pinned host memory (refusal degrades to
     pageable, accounting balances);
  6. the packaged benchmark harness + diagnostics endpoint extract as
     ordinary specs (the th#1198 payload path) and their pure planning
     parts run off-pod.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import hashlib
import msgspec
import pytest

from gen_worker import Hub, RequestContext, Slot, endpoint, worker_function
from gen_worker.executor import Executor, _Job
from gen_worker.models import staging as staging_mod
from gen_worker.models.pinned_swap import prestage_module
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import extract_specs
from gen_worker.models import store as store_mod

_GiB = 1024 ** 3


class GenIn(msgspec.Struct):
    prompt: str
    model: str = ""


class Out(msgspec.Struct):
    y: str = "ok"


CALLS: List[Tuple[str, str]] = []


@endpoint(models={"pipeline": Slot(str, selected_by="model")})
class Family:
    def setup(self, pipeline: str) -> None:
        self.pipeline = pipeline

    @worker_function()
    def generate(self, ctx: RequestContext, p: GenIn) -> Out:
        CALLS.append(("generate", self.pipeline))
        return Out()


# ---------------------------------------------------------------------------
# Composed toy pipeline for the component-first host-staging case.
# ---------------------------------------------------------------------------


class ToyDenoiser:
    """Component class named by the toy tree's model_index.json."""

    def __init__(self, content: str) -> None:
        self.content = content

    @classmethod
    def from_pretrained(cls, path: str, **_kw: object) -> "ToyDenoiser":
        return cls((Path(path) / "weights.bin").read_text())


class ToyStagePipeline:
    """Diffusers-shaped fake: records whether its denoiser was INJECTED
    (the gw#479/pgw#674 ``components=`` mechanism) or loaded from disk."""

    def __init__(self, denoiser: ToyDenoiser, injected: bool) -> None:
        self.denoiser = denoiser
        self.denoiser_injected = injected

    @classmethod
    def from_pretrained(cls, path: str, **kwargs: object) -> "ToyStagePipeline":
        injected = kwargs.get("denoiser")
        if isinstance(injected, ToyDenoiser):
            return cls(injected, injected=True)
        return cls(
            ToyDenoiser.from_pretrained(str(Path(path) / "denoiser")),
            injected=False,
        )

    def to(self, device: str) -> "ToyStagePipeline":
        return self


@endpoint(models={"pipeline": Slot(ToyStagePipeline, selected_by="model")})
class ComposedFamily:
    def setup(self, pipeline: ToyStagePipeline) -> None:
        self.pipe = pipeline

    @worker_function()
    def render(self, ctx: RequestContext, p: GenIn) -> Out:
        return Out(y=f"injected={self.pipe.denoiser_injected}")


#: pgw#1148: a QUANTIZED lane is declared as a CAST on the binding now — the
#: `#fp8` ref that used to say it is deleted (§1.32(d)), and a dispatch-named
#: ref carries no precision at all. So the staging skip is exercised through
#: a DECLARED cast binding, which is the shape that can still state one.
@endpoint(models={"pipeline": Hub("acme/qwen-finetune", storage_dtype="fp8")})
class ComposedCastFamily:
    def setup(self, pipeline: ToyStagePipeline) -> None:
        self.pipe = pipeline

    @worker_function()
    def render(self, ctx: RequestContext, p: GenIn) -> Out:
        return Out(y=f"injected={self.pipe.denoiser_injected}")


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


DOWNLOADS: List[str] = []

_PLAIN_BYTES = b"cozy!"


def _plain_tree_writer(ref: str, p: Path) -> None:
    (p / "model.safetensors").write_bytes(_PLAIN_BYTES)


def _executor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cls: type,
    *,
    tree_writer: Any = None,
) -> Executor:
    sent: List[pb.WorkerMessage] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    ex = Executor(extract_specs(cls), _send)
    ex.store._cache_dir = tmp_path / "cas"
    DOWNLOADS.clear()

    async def _fake_download(ref: str, **kwargs: Any) -> Path:
        DOWNLOADS.append(ref)
        snap = kwargs.get("snapshot")
        digest = str(getattr(snap, "snapshot_digest", "") or "")
        name = (
            digest.split(":", 1)[-1].strip().lower()
            or ref.replace("/", "_").replace(":", "_").replace("#", "_")
        )
        p = tmp_path / name
        p.mkdir(parents=True, exist_ok=True)
        writer = tree_writer or _plain_tree_writer
        writer(ref, p)
        return p

    import gen_worker.executor as ex_mod

    monkeypatch.setattr(store_mod, "ensure_local", _fake_download)
    return ex


def _orders(run):
    """pgw#904: the driver reads neutral slot orders, never the wire message."""
    from gen_worker import dispatch

    return {
        b.slot: dispatch.SlotOrder(
            ref=b.ref.strip(),
            components=tuple(sorted(
                (str(k).strip(), str(v).strip())
                for k, v in b.components.items())),
        )
        for b in run.models if b.slot
    }


def _pick(ex: Executor, name: str, ref: str) -> Any:
    spec = ex.specs[name]
    run = pb.RunJob(
        function_name=name,
        models=[pb.ModelBinding(slot="pipeline", ref=ref)],
    )
    return ex._dispatched_spec(spec, _orders(run))


def _snapshots(ref: str, digest: str) -> Dict[str, pb.Snapshot]:
    return {ref: pb.Snapshot(digest=digest, files=[pb.SnapshotFile(
        path="model.safetensors", size_bytes=len(_PLAIN_BYTES),
        digest="sha256:" + hashlib.sha256(_PLAIN_BYTES).hexdigest(),
        url="http://r2.invalid/presigned")])}


def _instance(fn: str, ref: str) -> pb.DesiredInstance:
    return pb.DesiredInstance(
        function_name=fn,
        models=[pb.ModelBinding(slot="pipeline", ref=ref)],
    )


def _seed_preloader(
    ex: Executor,
    instances: List[pb.DesiredInstance],
    snapshots: Dict[str, pb.Snapshot],
) -> None:
    """Deterministic state injection: tests drive ``_pass()`` directly
    instead of racing the background task."""
    pl = ex.preloader
    pl._hot = tuple(instances)
    pl._snapshots = dict(snapshots)
    pl._generation = 1


def _fake_busy_job(ex: Executor) -> None:
    """An unfinished tenant job: the executor is NOT idle. Any staging path
    that waits on tenant idle (the pre-pgw#674 reconcile shape) would hang
    below and fail the wait_for guards."""
    job = _Job(request_id="busy", attempt=1, spec=None, intent_id="")
    ex.jobs[("busy", 1)] = job
    ex._idle.clear()


# ---------------------------------------------------------------------------
# 1 + 2: double-buffer — background setup while busy; dispatch = cache hit
# ---------------------------------------------------------------------------


def test_preload_double_buffers_next_instance_while_busy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    CALLS.clear()
    ex = _executor(tmp_path, monkeypatch, Family)
    # Fake CUDA budget: both instances fit -> tier 2 (true double-buffer).
    ex.store.residency._vram_budget = 64 * _GiB

    async def _run() -> None:
        _fake_busy_job(ex)
        ck = "acme/next-ckpt"
        _seed_preloader(ex, [_instance("generate", ck)], _snapshots(ck, "d2" * 16))
        did = await asyncio.wait_for(ex.preloader._pass(), timeout=60)
        assert did is True

        eff = _pick(ex, "generate", ck)
        rec = ex._classes.get(eff.instance_key)
        assert rec is not None and rec.ready and not rec.stale, (
            "the desired NEXT instance must be fully set up in the "
            "background while the worker is busy"
        )
        assert DOWNLOADS == [ck]

        # Rotation: dispatching the preloaded instance is a pure cache hit —
        # no download, no load, no warm run. Visible swap ~0.
        CALLS.clear()
        await asyncio.wait_for(
            ex.ensure_setup(eff, _snapshots(ck, "d2" * 16)), timeout=60)
        assert CALLS == []
        assert DOWNLOADS == [ck]

        # A second pass is idempotent: the instance is hot, nothing to do.
        assert await asyncio.wait_for(ex.preloader._pass(), timeout=60) is False

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 3: the pgw#638 fence carve-out
# ---------------------------------------------------------------------------


def test_preload_skips_ref_resident_under_moved_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    CALLS.clear()
    ex = _executor(tmp_path, monkeypatch, Family)
    ex.store.residency._vram_budget = 64 * _GiB

    async def _run() -> None:
        ck = "acme/moving-tag"
        eff = _pick(ex, "generate", ck)
        await ex.ensure_setup(eff, _snapshots(ck, "d1" * 16))
        rec = ex._classes[eff.instance_key]
        assert rec.ready
        # The hub now names NEW bytes for the same ref (mutable tag move).
        rec.stale = True
        before = list(DOWNLOADS)

        vacated: List[str] = []

        async def _spy(instance: Any, snapshots: Any) -> None:
            vacated.append(instance.function_name)

        monkeypatch.setattr(ex, "ensure_desired_instance", _spy)
        _fake_busy_job(ex)
        _seed_preloader(ex, [_instance("generate", ck)], _snapshots(ck, "d2" * 16))
        did = await asyncio.wait_for(ex.preloader._pass(), timeout=60)
        assert did is False
        assert DOWNLOADS == before, (
            "identity moves belong to the idle-gated reconcile (pgw#638 "
            "fence); the preloader must not fetch the new bytes"
        )
        assert vacated == []

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 4: component-first host staging + injection consumption
# ---------------------------------------------------------------------------


_DENOISER_BYTES = b"denoiser-weights-B" * (2 * 1024 * 1024)  # ~36 MiB
_VAE_BYTES = b"tiny-vae"


def _composed_tree_writer(ref: str, p: Path) -> None:
    (p / "model_index.json").write_text(json.dumps({
        "_class_name": "ToyStagePipeline",
        "denoiser": ["test_rotation_preload_pgw674", "ToyDenoiser"],
        "vae": ["test_rotation_preload_pgw674", "ToyDenoiser"],
    }))
    d = p / "denoiser"
    d.mkdir(exist_ok=True)
    (d / "weights.bin").write_bytes(_DENOISER_BYTES)
    v = p / "vae"
    v.mkdir(exist_ok=True)
    (v / "weights.bin").write_bytes(_VAE_BYTES)


def _composed_snapshot(ref: str, digest: str) -> Dict[str, pb.Snapshot]:
    def _dig(data: bytes) -> str:
        return "sha256:" + hashlib.sha256(data).hexdigest()

    return {ref: pb.Snapshot(digest=digest, files=[
        pb.SnapshotFile(path="denoiser/weights.bin",
                        size_bytes=len(_DENOISER_BYTES),
                        digest=_dig(_DENOISER_BYTES),
                        url="http://r2.invalid/a"),
        pb.SnapshotFile(path="vae/weights.bin",
                        size_bytes=len(_VAE_BYTES),
                        digest=_dig(_VAE_BYTES),
                        url="http://r2.invalid/b"),
    ])}


def test_preload_component_staging_feeds_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ex = _executor(
        tmp_path, monkeypatch, ComposedFamily, tree_writer=_composed_tree_writer)
    # NO vram budget: free VRAM measures 0 (no CUDA) -> fits() is False ->
    # tier 3, the RAM-staged single-buffer.

    async def _run() -> None:
        _fake_busy_job(ex)
        ck = "acme/qwen-finetune"
        snaps = _composed_snapshot(ck, "e1" * 16)
        _seed_preloader(ex, [_instance("render", ck)], snaps)
        did = await asyncio.wait_for(ex.preloader._pass(), timeout=60)
        assert did is True
        assert DOWNLOADS == [ck]

        # The exclusive weight-bearing component (>=32MiB) is seeded into
        # the shared cache by CONTENT digest; the config-ish vae (< floor)
        # is not.
        res = ex.store.residency
        stats = res.shared_stats()
        comps = {e["ref"].split("::")[1] for e in stats["entries"]}
        assert "denoiser" in comps
        assert "vae" not in comps
        staged = [e for e in stats["entries"] if "denoiser" in e["ref"]]
        assert staged[0]["tier"] == "RAM"
        assert staged[0]["holders"] == 0  # seed-then-release: LRU-reclaimable

        # Dispatch-time setup consumes the staged component: from_pretrained
        # receives it via components= and skips that disk load.
        eff = _pick(ex, "render", ck)
        await asyncio.wait_for(ex.ensure_setup(eff, snaps), timeout=60)
        rec = ex._classes[eff.instance_key]
        assert rec.ready
        pipe = rec.instance.pipe  # type: ignore[union-attr]
        assert pipe.denoiser_injected is True, (
            "the staged component must be consumed by injection, not "
            "re-loaded from disk"
        )
        assert pipe.denoiser.content == _DENOISER_BYTES.decode()
        # Consumption holds the shared entry (acquire_shared handoff).
        stats = res.shared_stats()
        staged = [e for e in stats["entries"] if "denoiser" in e["ref"]]
        assert staged[0]["holders"] >= 1

        # Idempotent afterward.
        assert await asyncio.wait_for(ex.preloader._pass(), timeout=60) is False

    asyncio.run(_run())


def test_preload_component_staging_skips_quantized_execution_lanes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Quantized bindings load through special lanes a vanilla component
    load cannot reproduce: staging stops at the disk tier.

    pgw#1148: the signal is the declared CAST (`storage_dtype`), not a
    `#fp8` in the ref — §1.32(d) deleted that address, and the preload
    driver no longer has a flavor field to read."""
    ex = _executor(
        tmp_path, monkeypatch, ComposedCastFamily,
        tree_writer=_composed_tree_writer)

    async def _run() -> None:
        ck = "acme/qwen-finetune"
        snaps = _composed_snapshot(ck, "e2" * 16)
        _seed_preloader(ex, [_instance("render", ck)], snaps)
        await asyncio.wait_for(ex.preloader._pass(), timeout=60)
        assert DOWNLOADS == [ck]  # tier 1 still happened
        stats = ex.store.residency.shared_stats()
        assert stats["entries"] == []  # no host staging on the cast lane

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# 5: pinned pool bounds
# ---------------------------------------------------------------------------


def test_pinned_pool_reserve_release_accounting() -> None:
    pool = staging_mod.PinnedPool(budget_fn=lambda: 100)
    assert pool.try_reserve(60) is True
    assert pool.reserved_bytes() == 60
    assert pool.try_reserve(200) is False
    pool.release(60)
    assert pool.reserved_bytes() == 0
    assert pool.try_reserve(0) is True  # zero-byte asks are free


def test_prestage_refused_by_pool_leaves_module_pageable() -> None:
    torch = pytest.importorskip("torch")
    prev = staging_mod.set_pinned_pool(staging_mod.PinnedPool(budget_fn=lambda: 0))
    try:
        mod = torch.nn.Linear(8, 8)
        assert prestage_module(mod) == 0
        assert not mod.weight.is_pinned()
    finally:
        staging_mod.set_pinned_pool(prev)


def test_alloc_pinned_like_releases_on_failure() -> None:
    torch = pytest.importorskip("torch")
    pool = staging_mod.PinnedPool(budget_fn=lambda: 10 * _GiB)
    prev = staging_mod.set_pinned_pool(pool)
    try:
        t = torch.zeros(16)
        host = staging_mod.alloc_pinned_like(torch, t)
        if host is None:
            # No CUDA on this box: pin_memory fails -> the reservation must
            # have been released, never leaked.
            assert pool.reserved_bytes() == 0
        else:
            assert pool.reserved_bytes() == t.numel() * t.element_size()
            del host
            import gc

            gc.collect()
            assert pool.reserved_bytes() == 0  # finalizer released it
    finally:
        staging_mod.set_pinned_pool(prev)


def test_copy_stream_is_none_off_cuda() -> None:
    torch = pytest.importorskip("torch")
    if torch.cuda.is_available():  # pragma: no cover - GPU boxes
        pytest.skip("CPU-only assertion")
    assert staging_mod.copy_stream() is None
    with staging_mod.copy_stream_ctx() as stream:
        assert stream is None


# ---------------------------------------------------------------------------
# wiring: poke/update/stop lifecycle
# ---------------------------------------------------------------------------


def test_update_desired_spawns_and_stop_kills_the_driver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ex = _executor(tmp_path, monkeypatch, Family)
    ex.store.residency._vram_budget = 64 * _GiB

    async def _run() -> None:
        ck = "acme/spawned"
        ex.preloader.update_desired(
            [_instance("generate", ck)], _snapshots(ck, "d9" * 16), 1)
        task = ex.preloader._task
        assert task is not None
        deadline = time.monotonic() + 60
        eff = _pick(ex, "generate", ck)
        while time.monotonic() < deadline:
            rec = ex._classes.get(eff.instance_key)
            if rec is not None and rec.ready:
                break
            await asyncio.sleep(0.02)
        rec = ex._classes.get(eff.instance_key)
        assert rec is not None and rec.ready

        ex.preloader.stop()
        await asyncio.sleep(0)
        assert ex.preloader._task is None
        # A post-stop update is inert (drain must not restage).
        ex.preloader.update_desired(
            [_instance("generate", "acme/late")],
            _snapshots("acme/late", "da" * 16), 2)
        assert ex.preloader._task is None

    asyncio.run(_run())


def test_stale_generation_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    ex = _executor(tmp_path, monkeypatch, Family)
    pl = ex.preloader
    snaps_new: Dict[str, pb.Snapshot] = {}
    pl._generation = 5
    pl._hot = (_instance("generate", "acme/current"),)
    pl.update_desired([_instance("generate", "acme/old")], snaps_new, 3)
    assert pl._generation == 5
    assert pl._hot[0].models[0].ref == "acme/current"
