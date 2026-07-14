"""Snapshot-bank and re-mint behavior for cold Tensorhub refs."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any, List

import pytest

from gen_worker.api.binding import Hub
from gen_worker.models.errors import MissingSnapshotError
from gen_worker.pb import worker_scheduler_pb2 as pb


def _snapshot(digest: str = "ab" * 32) -> pb.Snapshot:
    return pb.Snapshot(digest=digest, files=[pb.SnapshotFile(
        path="model.safetensors", size_bytes=5, blake3="cd" * 32,
        url="http://r2.invalid/presigned")])


def _failed_events(sent: List[pb.WorkerMessage]) -> List[pb.ModelEvent]:
    return [m.model_event for m in sent
            if m.WhichOneof("msg") == "model_event"
            and m.model_event.state == pb.MODEL_STATE_FAILED]


def test_store_missing_snapshot_waits_then_raises_typed(tmp_path, monkeypatch) -> None:
    """A snapshot-less Tensorhub ref waits for a refresh, then fails typed."""
    import gen_worker.executor as ex_mod
    from gen_worker.executor import ModelStore

    monkeypatch.setattr(ex_mod, "_MISSING_SNAPSHOT_WAIT_S", 0.2)
    sent: List[pb.WorkerMessage] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    store = ModelStore(_send, cache_dir=tmp_path)
    store.register_binding("acme/never-resolved", Hub("acme/never-resolved"))

    async def _run() -> None:
        with pytest.raises(MissingSnapshotError):
            await store.ensure_local("acme/never-resolved")

    t0 = time.monotonic()
    asyncio.run(_run())
    elapsed = time.monotonic() - t0
    assert 0.2 <= elapsed < 2.0, f"must block for the refresh window, took {elapsed:.2f}s"
    states = [m.model_event.state for m in sent if m.WhichOneof("msg") == "model_event"]
    assert pb.MODEL_STATE_DOWNLOADING not in states
    failed = _failed_events(sent)
    assert failed and failed[-1].error == "missing_snapshot"


def test_store_cold_ref_blocks_until_remint_then_serves(tmp_path, monkeypatch) -> None:
    """A refreshed desired/job snapshot wakes and serves the cold caller."""
    import gen_worker.executor as ex_mod
    from gen_worker.executor import ModelStore

    monkeypatch.setattr(ex_mod, "_MISSING_SNAPSHOT_WAIT_S", 10.0)
    sent: List[pb.WorkerMessage] = []

    async def _send(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    async def _fake_download(ref: str, **kwargs: Any) -> Path:
        assert kwargs.get("snapshot") is not None, "download must carry the refreshed snapshot"
        p = tmp_path / ref.replace("/", "_")
        p.mkdir(parents=True, exist_ok=True)
        return p

    monkeypatch.setattr(ex_mod, "ensure_local", _fake_download)
    store = ModelStore(_send, cache_dir=tmp_path)
    store.register_binding("acme/cold-ref", Hub("acme/cold-ref"))

    async def _run() -> None:
        cold = asyncio.create_task(store.ensure_local("acme/cold-ref"))
        for _ in range(100):
            if _failed_events(sent):
                break
            await asyncio.sleep(0.01)
        assert _failed_events(sent) and _failed_events(sent)[-1].error == "missing_snapshot"
        assert not cold.done(), "cold caller must still be waiting on the refresh"
        refresh = asyncio.create_task(store.ensure_local("acme/cold-ref", _snapshot()))
        path = await asyncio.wait_for(cold, 5.0)
        assert path == tmp_path / "acme_cold-ref"
        assert await asyncio.wait_for(refresh, 5.0) == path

    asyncio.run(_run())


def test_missing_snapshot_maps_retryable_never_fatal() -> None:
    """A cold worker mid-resolution asks the scheduler to retry the job."""
    from gen_worker.executor import _map_exception

    status, msg = _map_exception(MissingSnapshotError("tensorhub ref needs a snapshot"))
    assert status == pb.JOB_STATUS_RETRYABLE
    assert "snapshot" in msg
