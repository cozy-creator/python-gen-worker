"""th#850 managed-tier ruling: the CAS root stays on local/pod disk
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
import hashlib
import os
from pathlib import Path

from gen_worker._vendor.tensorfs import CASRef
from gen_worker.transfer.grants import DEFAULT_PARALLEL, TransferReport

import projection_fixture
import gen_worker.models.cozy_snapshot as snap_mod
from gen_worker import config as gw_config
from gen_worker.models.cache_paths import open_worker_cas, tensorhub_fill_source_dir
from gen_worker.models.cozy_snapshot import NetworkBytesScope, ensure_snapshot_async
from gen_worker.models.hub_client import WorkerResolvedRepo, WorkerResolvedRepoFile
from gen_worker.models.refs import TensorhubRef
from gen_worker.models.store import ModelStore
from gen_worker.pb import worker_scheduler_pb2 as pb

_PAYLOAD = b"managed-tier-fill-source-payload" * 4096  # ~128KB
_HEX = hashlib.sha256(_PAYLOAD).hexdigest()
_DIGEST = "sha256:" + _HEX
_SNAPSHOT = "c7" * 32


def _resolved() -> WorkerResolvedRepo:
    return WorkerResolvedRepo(
        snapshot_digest=_SNAPSHOT,
        files=[
            WorkerResolvedRepoFile(
                path="model.safetensors",
                size_bytes=len(_PAYLOAD),
                digest=_DIGEST,
                url="https://tensorhub.invalid/authorized-blob",
            )
        ],
    )


def _blob_at(cas_root: Path, digest: str) -> Path:
    return open_worker_cas(cas_root).object_path(CASRef(digest))


def _blob(cas_root: Path) -> Path:
    return _blob_at(cas_root, _HEX)


def _stub_r2(monkeypatch, calls: list) -> None:
    def _download(grants, cas, *, progress=None) -> TransferReport:
        for grant in grants:
            calls.append(1)
            cas.put_bytes(_PAYLOAD, expected=grant.digest)
            if progress is not None:
                progress(grant.digest, grant.size_bytes)
        return TransferReport(
            examined=len(grants),
            succeeded=len(grants),
            bytes_transferred=sum(grant.size_bytes for grant in grants),
        )

    monkeypatch.setattr(snap_mod, "download", _download)


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
    assert projection_fixture.bytes_at(snap, "model.safetensors") == _PAYLOAD
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
    assert projection_fixture.bytes_at(snap, "model.safetensors") == _PAYLOAD
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
    assert projection_fixture.bytes_at(snap, "model.safetensors") == _PAYLOAD
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
    assert projection_fixture.bytes_at(snap, "model.safetensors") == _PAYLOAD  # real bytes
    assert calls == [1]  # fell through to R2, not the corrupt volume copy
    assert scope.network_bytes == len(_PAYLOAD)


# ---------------------------------------------------------------------------
# pgw#1556 — HOW FAST the volume fill is, as behaviour rather than as a comment
#
# `test_corrupt_volume_blob_falls_through_to_r2` above is the revert-turns-red
# guard for the RULING (digest verification of volume-read blobs is mandatory)
# and it stays green through both changes below. That is the load-bearing fact:
# the hash pass that was deleted proved nothing the surviving one does not.
# ---------------------------------------------------------------------------


def test_a_volume_fill_hashes_the_source_ONCE(tmp_path: Path, monkeypatch) -> None:
    """Two full SHA-256 passes over every byte, on the flagship 134 GB path.

    `filled()` called `fill.verify_object` — a whole-file read and hash whose
    only product is a verdict — and then handed the same path to `put_file`,
    which reads it again and hashes it again while copying. Measured on this
    box over 48 synthetic 64 MiB objects through the REAL `LocalCAS` (paired
    alternation, min of 3): 203.6 MiB/s with both passes, 348.9 MiB/s with one,
    1.71x, before any fan-out.

    The verification is not relaxed: `put_file(expected=..., size=...)` hashes
    streaming alongside the copy and raises `DigestMismatch`, which is what
    keeps the corrupt-blob test above green.
    """
    calls: list = []
    _stub_r2(monkeypatch, calls)
    volume = tmp_path / "volume"
    local = tmp_path / "local"
    blob = _blob(volume)
    blob.parent.mkdir(parents=True, exist_ok=True)
    blob.write_bytes(_PAYLOAD)

    read_bytes = [0]
    real_open = Path.open

    def counting_open(self: Path, *args, **kwargs):  # type: ignore[no-untyped-def]
        handle = real_open(self, *args, **kwargs)
        if self == blob and "r" in (args[0] if args else kwargs.get("mode", "r")):
            class _Counted:
                def __init__(self, inner): self._inner = inner
                def read(self, *a):  # noqa: ANN
                    data = self._inner.read(*a)
                    read_bytes[0] += len(data)
                    return data
                def __getattr__(self, name): return getattr(self._inner, name)
                def __enter__(self): self._inner.__enter__(); return self
                def __exit__(self, *e): return self._inner.__exit__(*e)
            return _Counted(handle)
        return handle

    monkeypatch.setattr(Path, "open", counting_open)

    ref = TensorhubRef(owner="org", repo="model")
    snap = asyncio.run(ensure_snapshot_async(
        base_dir=local, ref=ref, resolved=_resolved(), fill_source_dir=volume,
    ))
    assert projection_fixture.bytes_at(snap, "model.safetensors") == _PAYLOAD
    assert calls == []
    assert read_bytes[0] == len(_PAYLOAD), (
        "the volume source was read "
        f"{read_bytes[0] / len(_PAYLOAD):.1f}x — a fill that hashes every byte "
        "twice is half the throughput of one that hashes it once, on the one "
        "path a staged endpoint volume exists to make fast")


def test_the_residency_and_fill_scan_runs_MORE_THAN_ONE_OBJECT_AT_A_TIME(
    tmp_path: Path, monkeypatch,
) -> None:
    """`await asyncio.to_thread(...)` inside a `for` buys zero parallelism.

    It is awaited in the loop body, so exactly one object is ever in flight —
    while the R2 fetch on the very same pod has run `DEFAULT_PARALLEL`-wide
    since pgw#1308. The barrier below is the proof, and it is the honest one:
    if the scan is serial, no two objects are ever inside it at once and the
    barrier times out rather than the test passing on a timing coincidence.
    """
    import threading as _threading

    calls: list = []
    _stub_r2(monkeypatch, calls)
    volume = tmp_path / "volume"
    local = tmp_path / "local"
    payloads = [_PAYLOAD + bytes([i]) for i in range(DEFAULT_PARALLEL)]
    files = []
    for body in payloads:
        digest = hashlib.sha256(body).hexdigest()
        blob = _blob_at(volume, digest)
        blob.parent.mkdir(parents=True, exist_ok=True)
        blob.write_bytes(body)
        files.append(WorkerResolvedRepoFile(
            path=f"shard-{digest[:8]}.safetensors",
            size_bytes=len(body),
            digest="sha256:" + digest,
            url="https://tensorhub.invalid/authorized-blob",
        ))
    resolved = WorkerResolvedRepo(snapshot_digest="d9" * 32, files=files)

    barrier = _threading.Barrier(len(payloads), timeout=15.0)
    peak = [0]
    live = [0]
    guard = _threading.Lock()
    real_put_file = snap_mod.LocalCAS.put_file

    def gated_put_file(self, source, **kwargs):  # type: ignore[no-untyped-def]
        with guard:
            live[0] += 1
            peak[0] = max(peak[0], live[0])
        try:
            # Every object must be inside the scan simultaneously, or this
            # raises BrokenBarrierError and the test names the reason.
            barrier.wait()
        except _threading.BrokenBarrierError as exc:  # pragma: no cover
            raise AssertionError(
                "the residency/fill scan is SERIAL: fewer than "
                f"{len(payloads)} objects were ever in flight at once "
                f"(peak {peak[0]}). One object at a time is the whole of the "
                "134 GB warm-volume boot's cost model."
            ) from exc
        finally:
            with guard:
                live[0] -= 1
        return real_put_file(self, source, **kwargs)

    monkeypatch.setattr(snap_mod.LocalCAS, "put_file", gated_put_file)

    ref = TensorhubRef(owner="org", repo="model")
    snap = asyncio.run(ensure_snapshot_async(
        base_dir=local, ref=ref, resolved=resolved, fill_source_dir=volume,
    ))
    assert calls == [], "every object came off the volume; nothing hit R2"
    assert peak[0] == len(payloads), (
        f"peak concurrency was {peak[0]}, expected {len(payloads)}")
    for body in payloads:
        digest = hashlib.sha256(body).hexdigest()
        assert projection_fixture.bytes_at(snap, f"shard-{digest[:8]}.safetensors") == body


# ---------------------------------------------------------------------------
# tensorhub_fill_source_dir(): ismount-guarded, Settings-driven
#
# "env-driven" was the old description and it is no longer accurate.
# The value reaches this helper through the `Settings` the process entry
# published, so a test that changes the environment must RELOAD — the same step
# a real deployment performs exactly once, at boot. Under the deleted
# `get_settings()` these tests passed without it, because a cleared cache would
# lazily re-read env from whatever depth first asked.
# ---------------------------------------------------------------------------

def test_fill_source_dir_unset_is_none(monkeypatch) -> None:
    monkeypatch.delenv("TENSORHUB_FILL_SOURCE_DIR", raising=False)
    gw_config.reload_for_test()
    assert tensorhub_fill_source_dir() is None


def test_fill_source_dir_requires_a_real_mount(tmp_path: Path, monkeypatch) -> None:
    """A plain directory (baked into the image, or a stray path) must never
    be mistaken for the real per-endpoint volume."""
    plain_dir = tmp_path / "not-a-mount"
    plain_dir.mkdir()
    monkeypatch.setenv("TENSORHUB_FILL_SOURCE_DIR", str(plain_dir))
    gw_config.reload_for_test()
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
                    digest=_DIGEST, url="https://tensorhub.invalid/authorized-blob",
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
    digest2 = hashlib.sha256(payload2).hexdigest()
    dst2 = _blob_at(volume, digest2)
    dst2.parent.mkdir(parents=True, exist_ok=True)
    dst2.write_bytes(payload2)

    async def _run2() -> None:
        await store.ensure_local(
            "org/model2",
            pb.Snapshot(digest="d2" * 32, files=[
                pb.SnapshotFile(
                    path="model.safetensors", size_bytes=len(payload2),
                    digest="sha256:" + digest2, url="https://tensorhub.invalid/authorized-blob-2",
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


# ---------------------------------------------------------------------------
# No fill source on a datacenter pod must be LOUD
# ---------------------------------------------------------------------------

def test_datacenter_pod_without_fill_source_logs(tmp_path: Path, monkeypatch, caplog) -> None:
    async def _emit(msg: pb.WorkerMessage) -> None:
        del msg

    monkeypatch.setenv("RUNPOD_POD_ID", "pod-guard-test")
    monkeypatch.delenv("RUNPOD_PROVIDER", raising=False)
    monkeypatch.delenv("TENSORHUB_FILL_SOURCE_DIR", raising=False)
    with caplog.at_level("WARNING", logger="gen_worker.executor"):
        ModelStore(_emit, cache_dir=tmp_path / "local")
    assert any("fill_source_disabled reason=unset" in r.message for r in caplog.records)


def test_datacenter_pod_with_unmounted_fill_source_logs(tmp_path: Path, monkeypatch, caplog) -> None:

    async def _emit(msg: pb.WorkerMessage) -> None:
        del msg

    plain_dir = tmp_path / "not-a-mount"
    plain_dir.mkdir()
    monkeypatch.setenv("RUNPOD_POD_ID", "pod-guard-test")
    monkeypatch.delenv("RUNPOD_PROVIDER", raising=False)
    monkeypatch.setenv("TENSORHUB_FILL_SOURCE_DIR", str(plain_dir))
    gw_config.reload_for_test()
    try:
        with caplog.at_level("WARNING", logger="gen_worker.executor"):
            ModelStore(_emit, cache_dir=tmp_path / "local")
    finally:
        gw_config.reload_for_test()
    assert any("fill_source_disabled reason=not_a_mount" in r.message for r in caplog.records)


def test_local_pod_without_fill_source_stays_quiet(tmp_path: Path, monkeypatch, caplog) -> None:
    async def _emit(msg: pb.WorkerMessage) -> None:
        del msg

    monkeypatch.delenv("RUNPOD_POD_ID", raising=False)
    monkeypatch.delenv("TENSORHUB_FILL_SOURCE_DIR", raising=False)
    with caplog.at_level("WARNING", logger="gen_worker.executor"):
        ModelStore(_emit, cache_dir=tmp_path / "local")
    assert not [r for r in caplog.records if "fill_source_disabled" in r.message]
