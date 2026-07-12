"""gw#465: boot-prefetch/preposition ModelOps must never cascade or conflate.

The J23 failure: a snapshot-less ModelOp{LOAD} for a SHARED companion ref (one
vae bound to every variant of an SDXL family) cold-set-up the first non-ready
variant spec; its checkpoint slot had no snapshot, the worker can't resolve
tensorhub refs itself, and the deterministic local miss burned the 1s+4s retry
loop before being mislabeled ``download_failed`` (paired with the companion's
``load_failed``). Demand ops for the same refs — which carry snapshots —
succeeded seconds later.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import msgspec
import pytest

from gen_worker.api.binding import Hub, ModelRef
from gen_worker.executor import Executor
from gen_worker.models.errors import MissingSnapshotError
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec


class _In(msgspec.Struct):
    x: str


class _Out(msgspec.Struct):
    y: str


VAE = Hub("tensorhub/sdxl-vae-fp16-fix")
VARIANT_A = Hub("acme/variant-a")
VARIANT_B = Hub("acme/variant-b")


def _spec(name: str, checkpoint: ModelRef, calls: List[str]) -> EndpointSpec:
    class Endpoint:
        def setup(self, model: str, vae: str) -> None:
            calls.append(name)

        def run(self, ctx, payload: _In) -> _Out:  # pragma: no cover
            return _Out(y=payload.x)

    return EndpointSpec(
        name=name, method=Endpoint.run, kind="inference",
        payload_type=_In, output_mode="single", cls=Endpoint,
        attr_name="run", models={"model": checkpoint, "vae": VAE},
    )


def _snapshot(digest: str = "ab" * 32) -> pb.Snapshot:
    return pb.Snapshot(digest=digest, files=[pb.SnapshotFile(
        path="model.safetensors", size_bytes=5, blake3="cd" * 32,
        url="http://r2.invalid/presigned")])


def _harness(tmp_path: Path, calls: List[str], *, on_disk: List[str] = ()):
    """Executor with the REAL ModelStore/handle_model_op orchestration; only
    the network-touching download primitive is faked (existing test pattern).
    """
    sent: List[pb.WorkerMessage] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    ex = Executor([_spec("gen-a", VARIANT_A, calls), _spec("gen-b", VARIANT_B, calls)], _send)

    downloads: List[Dict[str, Any]] = []

    async def _fake_download(ref: str, **kwargs: Any) -> Path:
        downloads.append({"ref": ref, **kwargs})
        if kwargs.get("snapshot") is None:
            raise MissingSnapshotError(f"tensorhub ref {ref!r} needs a snapshot")
        p = tmp_path / ref.replace("/", "_")
        p.mkdir(parents=True, exist_ok=True)
        return p

    import gen_worker.executor as ex_mod
    for ref in on_disk:
        p = tmp_path / ref.replace("/", "_")
        p.mkdir(parents=True, exist_ok=True)
        ex.store.residency.track_disk(ref, p)
        ex.store._verified.add(ref)
    return ex, sent, downloads, ex_mod, _fake_download


def _failed_events(sent: List[pb.WorkerMessage]) -> List[pb.ModelEvent]:
    return [m.model_event for m in sent
            if m.WhichOneof("msg") == "model_event"
            and m.model_event.state == pb.MODEL_STATE_FAILED]


def test_load_of_shared_companion_does_not_cascade(tmp_path, monkeypatch) -> None:
    """A LOAD for the shared vae while instance A is ready must touch/promote
    A only — NOT cold-set-up variant B (whose checkpoint has no snapshot)."""
    calls: List[str] = []
    ex, sent, downloads, ex_mod, fake = _harness(
        tmp_path, calls, on_disk=["acme/variant-a", "tensorhub/sdxl-vae-fp16-fix"])
    monkeypatch.setattr(ex_mod, "ensure_local", fake)

    async def _run() -> None:
        spec_a = ex.specs["gen-a"]
        await ex.ensure_setup(spec_a)  # instance A resident
        assert calls == ["gen-a"]
        await ex.handle_model_op(pb.ModelOp(
            op=pb.MODEL_OP_KIND_LOAD, ref="tensorhub/sdxl-vae-fp16-fix"))

    asyncio.run(_run())
    assert calls == ["gen-a"], f"LOAD(vae) cascaded into cold setups: {calls}"
    assert not _failed_events(sent), (
        f"LOAD(vae) with a ready owner emitted failures: {_failed_events(sent)}")


def test_cold_load_missing_sibling_snapshot_fails_fast_and_typed(tmp_path, monkeypatch) -> None:
    """Cold worker, LOAD(vae): every candidate spec needs a checkpoint the
    worker cannot materialize. The op must fail FAST (no 1s+4s local retry
    burn) with ``missing_snapshot`` naming the blocking refs — never a phantom
    ``download_failed`` — and must not disable any function."""
    calls: List[str] = []
    ex, sent, downloads, ex_mod, fake = _harness(
        tmp_path, calls, on_disk=["tensorhub/sdxl-vae-fp16-fix"])
    monkeypatch.setattr(ex_mod, "ensure_local", fake)

    t0 = time.monotonic()
    asyncio.run(ex.handle_model_op(pb.ModelOp(
        op=pb.MODEL_OP_KIND_LOAD, ref="tensorhub/sdxl-vae-fp16-fix")))
    elapsed = time.monotonic() - t0

    assert elapsed < 1.0, f"deterministic local miss burned retries ({elapsed:.1f}s)"
    assert calls == [], f"unmaterializable specs were cold-set-up: {calls}"
    failed = _failed_events(sent)
    errors = {(e.ref, e.error) for e in failed}
    assert ("acme/variant-a", "missing_snapshot") in errors
    assert ("acme/variant-b", "missing_snapshot") in errors
    assert ("tensorhub/sdxl-vae-fp16-fix", "missing_snapshot") in errors
    assert all(e.error != "download_failed" for e in failed), errors
    assert not ex.unavailable, f"functions disabled by a transient miss: {ex.unavailable}"


def test_snapshot_memory_enables_companion_setup(tmp_path, monkeypatch) -> None:
    """The store remembers digest-carrying snapshots (gw#465): after the hub's
    DOWNLOAD(variant-b, snapshot), a snapshot-less LOAD(vae) can set up the
    variant-b spec — the boot-prefetch batch lands instead of wedging."""
    calls: List[str] = []
    ex, sent, downloads, ex_mod, fake = _harness(
        tmp_path, calls, on_disk=["tensorhub/sdxl-vae-fp16-fix"])
    monkeypatch.setattr(ex_mod, "ensure_local", fake)

    async def _run() -> None:
        await ex.handle_model_op(pb.ModelOp(
            op=pb.MODEL_OP_KIND_DOWNLOAD, ref="acme/variant-b",
            snapshot=_snapshot()))
        assert ex.store.has_snapshot("acme/variant-b")
        await ex.handle_model_op(pb.ModelOp(
            op=pb.MODEL_OP_KIND_LOAD, ref="tensorhub/sdxl-vae-fp16-fix"))

    asyncio.run(_run())
    assert calls == ["gen-b"], f"companion LOAD did not set up variant-b: {calls}"
    assert not _failed_events(sent), _failed_events(sent)


def test_store_missing_snapshot_waits_then_raises_typed(tmp_path, monkeypatch) -> None:
    """ModelStore.ensure_local on a snapshot-less tensorhub ref (th#763):
    emits FAILED ``missing_snapshot`` immediately (the hub's re-mint
    trigger), then BLOCKS for the re-minted snapshot; when nothing arrives
    it raises the typed error — no DOWNLOADING ghost, never a phantom
    ``download_failed``."""
    import gen_worker.executor as ex_mod
    from gen_worker.executor import ModelStore

    monkeypatch.setattr(ex_mod, "_MISSING_SNAPSHOT_WAIT_S", 0.2)
    sent: List[pb.WorkerMessage] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    store = ModelStore(_send, cache_dir=tmp_path)
    # endpoint refs are registered at boot (#377) — that's what makes the
    # provider classification confident enough to classify the miss
    store.register_binding("acme/never-resolved", Hub("acme/never-resolved"))

    async def _run() -> None:
        with pytest.raises(MissingSnapshotError):
            await store.ensure_local("acme/never-resolved")

    t0 = time.monotonic()
    asyncio.run(_run())
    elapsed = time.monotonic() - t0
    assert 0.2 <= elapsed < 2.0, f"must block for the re-mint window, took {elapsed:.2f}s"
    states = [m.model_event.state for m in sent if m.WhichOneof("msg") == "model_event"]
    assert pb.MODEL_STATE_DOWNLOADING not in states
    failed = _failed_events(sent)
    assert failed and failed[-1].error == "missing_snapshot"


def test_store_cold_ref_blocks_until_remint_then_serves(tmp_path, monkeypatch) -> None:
    """th#763 block-and-serve: a cold ensure_local reports missing_snapshot
    and WAITS; the hub's re-minted DOWNLOAD (a concurrent ensure_local WITH
    the snapshot) wakes it and the original call returns the materialized
    path — the first user request per unseen ref succeeds instead of
    fataling as the sacrificial cache warmer."""
    import gen_worker.executor as ex_mod
    from gen_worker.executor import ModelStore

    monkeypatch.setattr(ex_mod, "_MISSING_SNAPSHOT_WAIT_S", 10.0)
    sent: List[pb.WorkerMessage] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    async def _fake_download(ref: str, **kwargs: Any) -> Path:
        assert kwargs.get("snapshot") is not None, "download must carry the re-minted snapshot"
        p = tmp_path / ref.replace("/", "_")
        p.mkdir(parents=True, exist_ok=True)
        return p

    monkeypatch.setattr(ex_mod, "ensure_local", _fake_download)
    store = ModelStore(_send, cache_dir=tmp_path)
    store.register_binding("acme/cold-ref", Hub("acme/cold-ref"))

    async def _run() -> None:
        cold = asyncio.create_task(store.ensure_local("acme/cold-ref"))
        # Hub reacts to the FAILED(missing_snapshot) event with a DOWNLOAD op.
        for _ in range(100):
            if _failed_events(sent):
                break
            await asyncio.sleep(0.01)
        assert _failed_events(sent) and _failed_events(sent)[-1].error == "missing_snapshot"
        assert not cold.done(), "cold caller must still be waiting on the re-mint"
        remint = asyncio.create_task(store.ensure_local("acme/cold-ref", _snapshot()))
        path = await asyncio.wait_for(cold, 5.0)
        assert path == tmp_path / "acme_cold-ref"
        assert await asyncio.wait_for(remint, 5.0) == path

    asyncio.run(_run())


def test_missing_snapshot_maps_retryable_never_fatal() -> None:
    """th#763: a job that hits MissingSnapshotError must come back
    JOB_STATUS_RETRYABLE — a cold worker mid-resolution must never fatal a
    user request."""
    from gen_worker.executor import _map_exception

    status, msg = _map_exception(MissingSnapshotError("tensorhub ref needs a snapshot"))
    assert status == pb.JOB_STATUS_RETRYABLE
    assert "snapshot" in msg
