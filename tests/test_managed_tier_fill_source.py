"""th#850 managed-tier ruling (gw#599): the CAS root stays on local/pod disk
as a managed, bounded LRU tier. A RunPod endpoint volume, when attached, is
FILL SOURCE #1 (checked before R2, FILL SOURCE #2); an R2 fill writes
through to the volume so the next same-endpoint pod finds it warm. This
supersedes the CAS-root-on-volume shape
(test_shared_cas_root_multiwriter.py covers that mechanism's multi-writer
temp-file safety, which write-through publishing still relies on).

Outcome-level tests only, against the real ``ensure_snapshot_async`` CAS
path with the R2 transport stubbed — no mocks of the fill-source mechanism
itself, since it is just filesystem copy+verify.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

from blake3 import blake3

import gen_worker.executor as executor_mod
import gen_worker.models.cozy_snapshot as snap_mod
from gen_worker.executor import ModelStore
from gen_worker.models.cache_paths import tensorhub_fill_source_dir
from gen_worker.models.cozy_snapshot import NetworkBytesScope, ensure_snapshot_async
from gen_worker.models.hub_client import WorkerResolvedRepo, WorkerResolvedRepoFile
from gen_worker.models.refs import TensorhubRef
from gen_worker.pb import worker_scheduler_pb2 as pb

_PAYLOAD = b"managed-tier-fill-source-payload" * 4096  # ~128KB
_BLAKE3 = blake3(_PAYLOAD).hexdigest()
_SNAPSHOT = "c7" * 32


def _resolved() -> WorkerResolvedRepo:
    return WorkerResolvedRepo(
        snapshot_digest=_SNAPSHOT,
        files=[
            WorkerResolvedRepoFile(
                path="model.safetensors",
                size_bytes=len(_PAYLOAD),
                blake3=_BLAKE3,
                url="https://tensorhub.invalid/authorized-blob",
            )
        ],
    )


def _blob_at(cas_root: Path, digest: str) -> Path:
    return cas_root / "blobs" / "blake3" / digest[:2] / digest[2:4] / digest


def _blob(cas_root: Path) -> Path:
    return _blob_at(cas_root, _BLAKE3)


def _stub_r2(monkeypatch, calls: list) -> None:
    async def _public_get(
        _url: str, dst: Path, expected_size: int, expected_blake3: str,
        on_bytes=None,
    ) -> None:
        del expected_size, expected_blake3
        calls.append(1)
        dst.write_bytes(_PAYLOAD)
        if on_bytes is not None:
            on_bytes(len(_PAYLOAD))

    monkeypatch.setattr(snap_mod, "_download_one_file", _public_get)


# ---------------------------------------------------------------------------
# Fill-source ordering (cozy_snapshot layer)
# ---------------------------------------------------------------------------

def test_volume_blob_preferred_over_r2(tmp_path: Path, monkeypatch) -> None:
    calls: list = []
    _stub_r2(monkeypatch, calls)
    volume = tmp_path / "volume"
    local = tmp_path / "local"
    blob = _blob(volume)
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(_PAYLOAD)

    ref = TensorhubRef(owner="org", repo="model")
    with NetworkBytesScope() as scope:
        snap = asyncio.run(ensure_snapshot_async(
            base_dir=local, ref=ref, resolved=_resolved(), fill_source_dir=volume,
        ))
    assert (snap / "model.safetensors").read_bytes() == _PAYLOAD
    assert calls == []  # no R2 fetch — the volume already had it
    assert scope.network_bytes == 0
    assert _blob(local).read_bytes() == _PAYLOAD  # copied into local CAS


def test_r2_fetch_writes_through_to_volume(tmp_path: Path, monkeypatch) -> None:
    calls: list = []
    _stub_r2(monkeypatch, calls)
    volume = tmp_path / "volume"
    local = tmp_path / "local"
    ref = TensorhubRef(owner="org", repo="model")

    with NetworkBytesScope() as scope:
        snap = asyncio.run(ensure_snapshot_async(
            base_dir=local, ref=ref, resolved=_resolved(), fill_source_dir=volume,
        ))
    assert (snap / "model.safetensors").read_bytes() == _PAYLOAD
    assert calls == [1]  # exactly one R2 fetch
    assert scope.network_bytes == len(_PAYLOAD)
    assert _blob(volume).read_bytes() == _PAYLOAD  # warmed for the next pod


def test_no_fill_source_is_byte_identical_to_pre_th850(
    tmp_path: Path, monkeypatch,
) -> None:
    """cozy-local / no-volume degenerate case: straight to R2, no new branch
    taken, no volume path ever touched."""
    calls: list = []
    _stub_r2(monkeypatch, calls)
    local = tmp_path / "local"
    ref = TensorhubRef(owner="org", repo="model")

    with NetworkBytesScope() as scope:
        snap = asyncio.run(ensure_snapshot_async(
            base_dir=local, ref=ref, resolved=_resolved(),
        ))
    assert (snap / "model.safetensors").read_bytes() == _PAYLOAD
    assert calls == [1]
    assert scope.network_bytes == len(_PAYLOAD)


def test_corrupt_volume_blob_falls_through_to_r2(tmp_path: Path, monkeypatch) -> None:
    """Revert-turns-red guard: digest-verification of volume-read blobs is
    mandatory (Paul's ruling) — a same-SIZE, wrong-content volume blob must
    never be silently trusted just because it's the right length."""
    calls: list = []
    _stub_r2(monkeypatch, calls)
    volume = tmp_path / "volume"
    local = tmp_path / "local"
    blob = _blob(volume)
    blob.parent.mkdir(parents=True, exist_ok=True)
    corrupt = bytes(b ^ 0xFF for b in _PAYLOAD)  # same length, different bytes
    assert len(corrupt) == len(_PAYLOAD)
    blob.write_bytes(corrupt)

    ref = TensorhubRef(owner="org", repo="model")
    with NetworkBytesScope() as scope:
        snap = asyncio.run(ensure_snapshot_async(
            base_dir=local, ref=ref, resolved=_resolved(), fill_source_dir=volume,
        ))
    assert (snap / "model.safetensors").read_bytes() == _PAYLOAD  # real bytes
    assert calls == [1]  # fell through to R2, not the corrupt volume copy
    assert scope.network_bytes == len(_PAYLOAD)


# ---------------------------------------------------------------------------
# tensorhub_fill_source_dir(): ismount-guarded, env-driven
# ---------------------------------------------------------------------------

def test_fill_source_dir_unset_is_none(monkeypatch) -> None:
    monkeypatch.delenv("TENSORHUB_FILL_SOURCE_DIR", raising=False)
    assert tensorhub_fill_source_dir() is None


def test_fill_source_dir_requires_a_real_mount(tmp_path: Path, monkeypatch) -> None:
    """A plain directory (baked into the image, or a stray path) must never
    be mistaken for the real per-endpoint volume."""
    plain_dir = tmp_path / "not-a-mount"
    plain_dir.mkdir()
    monkeypatch.setenv("TENSORHUB_FILL_SOURCE_DIR", str(plain_dir))
    assert tensorhub_fill_source_dir() is None  # ismount() is False -> rejected

    monkeypatch.setattr(os.path, "ismount", lambda p: str(p) == str(plain_dir))
    assert tensorhub_fill_source_dir() == plain_dir


# ---------------------------------------------------------------------------
# Disk-residency network_bytes reaches the wire (executor layer)
# ---------------------------------------------------------------------------

def test_network_bytes_reaches_on_disk_model_event(tmp_path: Path, monkeypatch) -> None:
    calls: list = []
    _stub_r2(monkeypatch, calls)
    volume = tmp_path / "volume"
    local = tmp_path / "local"
    sent: list = []

    async def _emit(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    store = ModelStore(_emit, cache_dir=local, fill_source_dir=volume)

    async def _run() -> None:
        await store.ensure_local(
            "org/model",
            pb.Snapshot(digest=_SNAPSHOT, files=[
                pb.SnapshotFile(
                    path="model.safetensors", size_bytes=len(_PAYLOAD),
                    blake3=_BLAKE3, url="https://tensorhub.invalid/authorized-blob",
                ),
            ]),
        )

    asyncio.run(_run())
    on_disk = [
        m.model_event for m in sent
        if m.WhichOneof("msg") == "model_event"
        and m.model_event.state == pb.MODEL_STATE_ON_DISK
    ]
    assert on_disk, "expected at least one ON_DISK ModelEvent"
    # Residency's own generic transition event (network_bytes-blind) and the
    # executor's explicit evidence event (network_bytes-carrying) may land
    # in either order — protocol-v3 events are observation, not
    # convergence, so a receiver reads the most informative one it saw.
    assert max(e.network_bytes for e in on_disk) == len(_PAYLOAD)  # fetched from R2
    assert _blob(volume).read_bytes() == _PAYLOAD  # write-through happened

    # A second, fresh ref whose blob is already warm on the volume reports
    # network_bytes == 0 — the "warm boot ⇒ ~0 R2 bytes" signal.
    calls.clear()
    sent.clear()
    payload2 = _PAYLOAD + b"-2"
    digest2 = blake3(payload2).hexdigest()
    dst2 = _blob_at(volume, digest2)
    dst2.parent.mkdir(parents=True, exist_ok=True)
    dst2.write_bytes(payload2)

    async def _run2() -> None:
        await store.ensure_local(
            "org/model2",
            pb.Snapshot(digest="d2" * 32, files=[
                pb.SnapshotFile(
                    path="model.safetensors", size_bytes=len(payload2),
                    blake3=digest2, url="https://tensorhub.invalid/authorized-blob-2",
                ),
            ]),
        )

    asyncio.run(_run2())
    on_disk2 = [
        m.model_event for m in sent
        if m.WhichOneof("msg") == "model_event"
        and m.model_event.state == pb.MODEL_STATE_ON_DISK
    ]
    assert on_disk2
    assert max(e.network_bytes for e in on_disk2) == 0
    assert calls == []  # no R2 fetch at all — warm from the volume


def test_network_bytes_is_a_running_total_on_downloading_ticks(
    tmp_path: Path, monkeypatch,
) -> None:
    """tensorhub th#850/PR#493 reads network_bytes off the DOWNLOADING
    events' running value (WorkerModelDownloadState.NetworkBytes, populated
    the same way as BytesDownloaded/BytesTotal), not only the terminal
    ON_DISK event — both must carry it or the hub's accounting silently
    stays zero. Disable the progress-event debounce so every chunk reaches
    the wire, and stream the fake download in chunks (not one write) so a
    mid-flight DOWNLOADING event genuinely sees a PARTIAL network_bytes."""
    monkeypatch.setattr(executor_mod, "_PROGRESS_EVENT_MIN_INTERVAL_S", 0.0)
    chunk = _PAYLOAD[: len(_PAYLOAD) // 4]
    n_chunks = 4
    assert chunk * n_chunks == _PAYLOAD

    async def _chunked_get(
        _url: str, dst: Path, expected_size: int, expected_blake3: str,
        on_bytes=None,
    ) -> None:
        del expected_size, expected_blake3
        with open(dst, "wb") as f:
            for _ in range(n_chunks):
                f.write(chunk)
                if on_bytes is not None:
                    on_bytes(len(chunk))
                    await asyncio.sleep(0)  # let the executor's callback run

    monkeypatch.setattr(snap_mod, "_download_one_file", _chunked_get)

    local = tmp_path / "local"
    sent: list = []

    async def _emit(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    store = ModelStore(_emit, cache_dir=local)

    async def _run() -> None:
        await store.ensure_local(
            "org/model",
            pb.Snapshot(digest=_SNAPSHOT, files=[
                pb.SnapshotFile(
                    path="model.safetensors", size_bytes=len(_PAYLOAD),
                    blake3=_BLAKE3, url="https://tensorhub.invalid/authorized-blob",
                ),
            ]),
        )

    asyncio.run(_run())
    downloading = [
        m.model_event for m in sent
        if m.WhichOneof("msg") == "model_event"
        and m.model_event.state == pb.MODEL_STATE_DOWNLOADING
    ]
    partial = [e.network_bytes for e in downloading if 0 < e.network_bytes < len(_PAYLOAD)]
    assert partial, (
        "expected at least one DOWNLOADING event with a PARTIAL running "
        f"network_bytes total; got {[e.network_bytes for e in downloading]}"
    )


def test_downloading_progress_reports_populated_bytes_done_and_total(
    tmp_path: Path, monkeypatch,
) -> None:
    """ie#522 (Paul, 2026-07-21): the hub must see a MOVING counter during a
    long fill ("stalled at 12.3/69GB for 9min" vs silence), not just a
    started-then-on_disk pair with bytes_total unpopulated. This rides the
    existing DOWNLOADING ModelEvent channel (_materialize_local's
    _progress -> self._event(..., bytes_done=, bytes_total=)) — no new
    protocol surface. Same chunked-fake-download shape as the
    network_bytes sibling test above, but asserts bytes_done/bytes_total
    directly: total must be populated from the snapshot's known size (not
    0), and at least one mid-flight tick must show a PARTIAL bytes_done
    (0 < done < total), proving real progress reaches the wire in between
    started and on_disk."""
    monkeypatch.setattr(executor_mod, "_PROGRESS_EVENT_MIN_INTERVAL_S", 0.0)
    chunk = _PAYLOAD[: len(_PAYLOAD) // 4]
    n_chunks = 4
    assert chunk * n_chunks == _PAYLOAD

    async def _chunked_get(
        _url: str, dst: Path, expected_size: int, expected_blake3: str,
        on_bytes=None,
    ) -> None:
        del expected_size, expected_blake3
        with open(dst, "wb") as f:
            for _ in range(n_chunks):
                f.write(chunk)
                if on_bytes is not None:
                    on_bytes(len(chunk))
                    await asyncio.sleep(0)

    monkeypatch.setattr(snap_mod, "_download_one_file", _chunked_get)

    local = tmp_path / "local"
    sent: list = []

    async def _emit(msg: pb.WorkerMessage) -> None:
        sent.append(msg)

    store = ModelStore(_emit, cache_dir=local)

    async def _run() -> None:
        await store.ensure_local(
            "org/model",
            pb.Snapshot(digest=_SNAPSHOT, files=[
                pb.SnapshotFile(
                    path="model.safetensors", size_bytes=len(_PAYLOAD),
                    blake3=_BLAKE3, url="https://tensorhub.invalid/authorized-blob",
                ),
            ]),
        )

    asyncio.run(_run())
    downloading = [
        m.model_event for m in sent
        if m.WhichOneof("msg") == "model_event"
        and m.model_event.state == pb.MODEL_STATE_DOWNLOADING
    ]
    assert len(downloading) >= 2, (
        f"expected a started tick plus at least one progress tick; "
        f"got {len(downloading)} DOWNLOADING events"
    )
    # The very first DOWNLOADING event (fired before the retry loop even
    # starts) legitimately carries no byte counts yet — the regression this
    # guards is EVERY event after it also reading total=0/done=0.
    later = downloading[1:]
    assert all(e.bytes_total == len(_PAYLOAD) for e in later), (
        f"bytes_total not populated from the known snapshot size on a "
        f"progress tick; got {[e.bytes_total for e in later]}"
    )
    partial_done = [e.bytes_done for e in later if 0 < e.bytes_done < len(_PAYLOAD)]
    assert partial_done, (
        "expected at least one mid-flight DOWNLOADING event with a "
        f"PARTIAL bytes_done; got {[e.bytes_done for e in later]}"
    )
