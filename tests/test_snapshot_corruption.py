"""Corrupt disk snapshots are quarantined + re-materialized (gw#408).

J17 flood: a pod-churn-interrupted write left truncated safetensors in the
disk tier; every later load fataled ``OSError: Unable to load weights from
checkpoint file`` (9 user-visible failures) and the snapshot was trusted
forever. Now: snapshots are integrity-verified on first use per boot, a
corruption-shaped load failure digest-verifies + quarantines + re-downloads
+ retries once, cached blobs are size-checked before reuse, and merged
single-file checkpoints are structurally revalidated before reuse.
"""

from __future__ import annotations

import asyncio
import json
import struct
from pathlib import Path

import msgspec
import pytest
from blake3 import blake3

import gen_worker.models.cozy_snapshot as snap_mod
from gen_worker.api.binding import Hub
from gen_worker.api.decorators import Resources
from gen_worker.executor import Executor, ModelStore, _is_corrupt_load_error
from gen_worker.models.loading import _single_file_checkpoint, safetensors_file_valid
from gen_worker.pb import worker_scheduler_pb2 as pb
from gen_worker.registry import EndpointSpec


def _tiny_safetensors(tag: bytes = b"\x00\x01\x02\x03") -> bytes:
    header = {"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, len(tag)]}}
    hb = json.dumps(header, separators=(",", ":")).encode()
    return struct.pack("<Q", len(hb)) + hb + tag


def _snapshot(digest_hex: str, content: bytes) -> pb.Snapshot:
    return pb.Snapshot(
        digest=f"blake3:{digest_hex}",
        files=[pb.SnapshotFile(
            path="model.safetensors",
            size_bytes=len(content),
            blake3=blake3(content).hexdigest(),
            url="http://example.invalid/blob",
        )],
    )


def _wire_download(monkeypatch, content: bytes) -> dict:
    calls = {"n": 0}

    async def _fake_dl(url, dst, expected_size, expected_blake3, on_bytes=None):
        calls["n"] += 1
        dst.write_bytes(content)

    monkeypatch.setattr(snap_mod, "_download_one_file", _fake_dl)
    return calls


async def _noop_emit(msg: pb.WorkerMessage) -> None:
    pass


# --------------------------------------------------------------------------- #
# First-use-per-boot verification: truncation is quarantined + re-materialized
# --------------------------------------------------------------------------- #


def test_truncated_snapshot_is_quarantined_and_rematerialized(tmp_path: Path, monkeypatch) -> None:
    content = _tiny_safetensors()
    snap = _snapshot("11" * 32, content)
    calls = _wire_download(monkeypatch, content)

    store1 = ModelStore(_noop_emit, cache_dir=tmp_path)
    path = asyncio.run(store1.ensure_local("e2e/sdxl-a", snap))
    assert (path / "model.safetensors").read_bytes() == content
    assert calls["n"] == 1

    # Pod churn persists a truncated file (rename landed, data pages lost).
    # The snapshot file hardlinks the CAS blob, so the blob is equally bad.
    (path / "model.safetensors").write_bytes(content[:10])

    # Fresh boot: rescan re-registers disk truth; first use re-verifies.
    store2 = ModelStore(_noop_emit, cache_dir=tmp_path)
    store2.rescan_disk()
    path2 = asyncio.run(store2.ensure_local("e2e/sdxl-a", snap))
    assert calls["n"] == 2  # quarantined + re-downloaded, no manual delete
    assert (path2 / "model.safetensors").read_bytes() == content


def test_clean_snapshot_verified_once_per_boot(tmp_path: Path, monkeypatch) -> None:
    content = _tiny_safetensors()
    snap = _snapshot("22" * 32, content)
    calls = _wire_download(monkeypatch, content)

    store = ModelStore(_noop_emit, cache_dir=tmp_path)
    asyncio.run(store.ensure_local("e2e/sdxl-b", snap))
    assert calls["n"] == 1

    verifies = {"n": 0}
    orig = ModelStore._verify_snapshot_tree

    def _counting(self, path, snapshot):
        verifies["n"] += 1
        return orig(self, path, snapshot)

    monkeypatch.setattr(ModelStore, "_verify_snapshot_tree", _counting)

    # Same boot: already verified -> no re-hash, no re-download.
    asyncio.run(store.ensure_local("e2e/sdxl-b", snap))
    assert verifies["n"] == 0 and calls["n"] == 1

    # New boot: exactly one verification for the cached tree.
    store2 = ModelStore(_noop_emit, cache_dir=tmp_path)
    store2.rescan_disk()
    asyncio.run(store2.ensure_local("e2e/sdxl-b", snap))
    assert verifies["n"] == 1 and calls["n"] == 1


# --------------------------------------------------------------------------- #
# Load-failure path: digest verify -> quarantine -> re-download -> retry once
# --------------------------------------------------------------------------- #


class _In(msgspec.Struct):
    x: str


class _WeightsPipe:
    """Loads only when the on-disk weights match `expected` — the exact J17
    failure shape otherwise."""

    expected: bytes = b""

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        data = (Path(path) / "model.safetensors").read_bytes()
        if data != cls.expected:
            raise OSError("Unable to load weights from checkpoint file")
        return cls()

    def to(self, device):
        return self


class _WeightsEndpoint:
    def setup(self, m: _WeightsPipe) -> None:
        self.m = m

    def run(self, ctx, payload: _In):  # pragma: no cover
        return payload


class _AlwaysOSError:
    @classmethod
    def from_pretrained(cls, path, **kwargs):
        raise OSError("Unable to load weights from checkpoint file")

    def to(self, device):  # pragma: no cover
        return self


class _AlwaysOSErrorEndpoint:
    def setup(self, m: _AlwaysOSError) -> None:  # pragma: no cover
        self.m = m

    def run(self, ctx, payload: _In):  # pragma: no cover
        return payload


def test_corrupt_load_failure_refetches_and_retries_once(tmp_path: Path, monkeypatch) -> None:
    content = _tiny_safetensors()
    snap = _snapshot("33" * 32, content)
    calls = _wire_download(monkeypatch, content)
    ref = "e2e/sdxl-c"
    monkeypatch.setattr(_WeightsPipe, "expected", content)

    spec = EndpointSpec(
        name="ep", method=_WeightsEndpoint.run, kind="inference", payload_type=_In,
        output_mode="single", cls=_WeightsEndpoint, attr_name="run",
        models={"m": Hub(ref)}, resources=Resources(),
    )

    async def _run() -> None:
        store = ModelStore(_noop_emit, cache_dir=tmp_path)
        ex = Executor([spec], _noop_emit, store=store)
        # Materialize once, then corrupt the weights IN PLACE and drop the
        # verified marker (as a fresh boot would).
        path = await store.ensure_local(ref, snap)
        (path / "model.safetensors").write_bytes(b"garbage-that-parses-as-nothing")
        store._verified.discard(ref)

        # ensure_local short-circuits (digest name matches), the load blows
        # up with the exact J17 OSError, and the executor digest-verifies,
        # quarantines, re-downloads, and retries ONCE — setup succeeds.
        inst = await ex.ensure_setup(spec, {ref: snap})
        assert isinstance(inst.m, _WeightsPipe)
        assert (path / "model.safetensors").read_bytes() == content

    asyncio.run(_run())
    assert calls["n"] == 2


def test_non_corrupt_load_failure_is_reraised_not_quarantined(tmp_path: Path, monkeypatch) -> None:
    """A clean tree + corruption-shaped error re-raises: the verify gate keeps
    code bugs from triggering pointless quarantine/re-download loops."""
    content = _tiny_safetensors()
    snap = _snapshot("44" * 32, content)
    calls = _wire_download(monkeypatch, content)
    ref = "e2e/sdxl-d"

    spec = EndpointSpec(
        name="ep", method=_AlwaysOSErrorEndpoint.run, kind="inference", payload_type=_In,
        output_mode="single", cls=_AlwaysOSErrorEndpoint, attr_name="run",
        models={"m": Hub(ref)}, resources=Resources(),
    )

    async def _run() -> None:
        store = ModelStore(_noop_emit, cache_dir=tmp_path)
        ex = Executor([spec], _noop_emit, store=store)
        with pytest.raises(OSError, match="Unable to load weights"):
            await ex.ensure_setup(spec, {ref: snap})

    asyncio.run(_run())
    assert calls["n"] == 1  # tree verified clean: NO re-download happened


def test_corrupt_load_error_classifier() -> None:
    import errno

    assert _is_corrupt_load_error(OSError("Unable to load weights from checkpoint file"))
    assert _is_corrupt_load_error(FileNotFoundError("model_index.json"))
    assert _is_corrupt_load_error(json.JSONDecodeError("x", "doc", 0))
    assert _is_corrupt_load_error(struct.error("unpack"))
    nospace = OSError(errno.ENOSPC, "no space left on device")
    assert not _is_corrupt_load_error(nospace)
    assert not _is_corrupt_load_error(ValueError("bad dtype string"))
    assert not _is_corrupt_load_error(RuntimeError("CUDA out of memory"))


# --------------------------------------------------------------------------- #
# Cached blobs and merged checkpoints are never trusted blindly
# --------------------------------------------------------------------------- #


def test_truncated_cached_blob_is_redownloaded_at_build(tmp_path: Path, monkeypatch) -> None:
    content = _tiny_safetensors()
    digest = blake3(content).hexdigest()
    calls = _wire_download(monkeypatch, content)

    # A pre-durability boot left a truncated blob at the final CAS path.
    blob = tmp_path / "blobs" / "blake3" / digest[:2] / digest[2:4] / digest
    blob.parent.mkdir(parents=True)
    blob.write_bytes(content[:7])

    from gen_worker.models.cozy_snapshot import ensure_snapshot_async
    from gen_worker.models.refs import TensorhubRef

    from gen_worker.models.hub_client import WorkerResolvedRepo, WorkerResolvedRepoFile

    resolved = WorkerResolvedRepo(
        snapshot_digest="55" * 32,
        files=[WorkerResolvedRepoFile(
            path="model.safetensors", size_bytes=len(content),
            blake3=digest, url="http://example.invalid/blob")],
    )
    out = asyncio.run(ensure_snapshot_async(
        base_dir=tmp_path, ref=TensorhubRef(owner="e2e", repo="tiny"),
        resolved=resolved,
    ))
    assert calls["n"] == 1  # size-mismatched cached blob was NOT reused
    assert (out / "model.safetensors").read_bytes() == content


def _sharded_dir(tmp_path: Path) -> Path:
    d = tmp_path / "snap"
    d.mkdir()
    for name, tensor in (("shard-00001.safetensors", "a"), ("shard-00002.safetensors", "b")):
        header = {tensor: {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
        hb = json.dumps(header, separators=(",", ":")).encode()
        (d / name).write_bytes(struct.pack("<Q", len(hb)) + hb + b"\x0a\x0b\x0c\x0d")
    (d / "model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"a": "shard-00001.safetensors", "b": "shard-00002.safetensors"},
    }))
    return d


def test_truncated_merged_checkpoint_is_remerged(tmp_path: Path) -> None:
    d = _sharded_dir(tmp_path)
    merged = _single_file_checkpoint(d)
    assert merged is not None and merged.name == "model.safetensors"
    assert safetensors_file_valid(merged)
    good = merged.read_bytes()

    # Pod kill mid-writeback: merged file exists but is truncated. Before the
    # fix this file was returned forever ("Unable to load weights" loop).
    merged.write_bytes(good[:20])
    assert not safetensors_file_valid(merged)

    again = _single_file_checkpoint(d)
    assert again is not None
    assert safetensors_file_valid(again)
    assert again.read_bytes() == good


def test_safetensors_file_valid(tmp_path: Path) -> None:
    good = tmp_path / "ok.safetensors"
    good.write_bytes(_tiny_safetensors())
    assert safetensors_file_valid(good)

    truncated = tmp_path / "short.safetensors"
    truncated.write_bytes(_tiny_safetensors()[:12])
    assert not safetensors_file_valid(truncated)

    garbage = tmp_path / "garbage.safetensors"
    garbage.write_bytes(b"not a safetensors file at all")
    assert not safetensors_file_valid(garbage)

    empty = tmp_path / "empty.safetensors"
    empty.write_bytes(b"")
    assert not safetensors_file_valid(empty)
