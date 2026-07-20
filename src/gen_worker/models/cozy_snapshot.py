from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set

from .cozy_cas import _download_one_file as _download_one_file
from .cozy_cas import _norm_rel_path, fsync_dir
from .download import components_present, select_component_paths
from .hub_client import WorkerResolvedRepo, WorkerResolvedRepoFile
from .refs import TensorhubRef
from ..capability import InsufficientDiskError
from ..s3_transfer import S3TransferGrant, download_file_with_grant

_log = logging.getLogger("gen_worker.download")

# Per-digest inflight downloads (gw#479 multi-lane fallout, th#757 forensics):
# two refs materializing concurrently SHARE blobs in the content-addressed
# store (split-vendor lanes: 9.7GB of identical encoder shards). Without a
# per-digest lock both tasks streamed into the SAME .part file, interleaved
# writes failed size/blake3 verification 3x, and the second ref died with
# download_failed on every attempt (J24M runs 10-12, three pods, ~2.5min in —
# while every blob verified byte-perfect in R2). First task downloads; the
# second awaits the lock, re-checks usability, and reuses the finished blob.
_INFLIGHT_BLOB_LOCKS: dict[str, "asyncio.Lock"] = {}
_INFLIGHT_BLOB_LOCKS_CAP = 8192


def _inflight_blob_lock(digest: str) -> "asyncio.Lock":
    if len(_INFLIGHT_BLOB_LOCKS) > _INFLIGHT_BLOB_LOCKS_CAP:
        _INFLIGHT_BLOB_LOCKS.clear()  # bounded memory; races just re-lock
    return _INFLIGHT_BLOB_LOCKS.setdefault(digest, asyncio.Lock())

ProgressFn = Callable[[int, Optional[int]], None]

# Free space that must remain after downloading the missing blobs.
_DISK_HEADROOM_BYTES = 1 << 30


# ---------------------------------------------------------------------------
# Snapshot build coordination (threading-based, works across event loops)
# ---------------------------------------------------------------------------

class _SnapshotEntry:
    """One builder, zero-or-more waiters."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.exception: Optional[BaseException] = None


_SNAP_LOCK = threading.Lock()
_SNAP_ENTRIES: Dict[str, _SnapshotEntry] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blob_path(blobs_root: Path, digest: str) -> Path:
    digest = (digest or "").strip().lower()
    if len(digest) < 4:
        raise ValueError(f"invalid blake3 digest: {digest!r}")
    return blobs_root / "blake3" / digest[:2] / digest[2:4] / digest


_PART_FILE_RE = re.compile(r"\.part\d{4}$")


def _strip_blake3_prefix(digest: str) -> str:
    """'blake3:abcd...' -> 'abcd...'"""
    d = (digest or "").strip().lower()
    if d.startswith("blake3:"):
        return d[7:]
    return d


def _is_part_file(path: str) -> bool:
    return bool(_PART_FILE_RE.search(path))


def _is_parts_manifest(path: str) -> bool:
    return path.endswith(".parts.json")


def _try_hardlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return
    except Exception as e:
        _log.debug("hardlink failed for %s -> %s: %s, trying symlink", src, dst, e)
    try:
        os.symlink(src, dst)
        return
    except Exception as e:
        _log.warning("symlink failed for %s -> %s: %s, falling back to copy", src, dst, e)
    shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Validate the typed resolved manifest (gw#497: WorkerResolvedRepo is THE
# shape — each wire boundary parses into it; no dict-or-object duck typing).
# ---------------------------------------------------------------------------

def _validate_resolved(ref: TensorhubRef, resolved: WorkerResolvedRepo) -> WorkerResolvedRepo:
    """Normalize digest prefixes and reject unusable manifests."""
    snapshot_digest = (resolved.snapshot_digest or "").strip()
    if not snapshot_digest:
        raise ValueError("resolved model missing snapshot_digest")
    snapshot_digest = _strip_blake3_prefix(snapshot_digest) or snapshot_digest

    files: List[WorkerResolvedRepoFile] = []
    for f in resolved.files:
        path = (f.path or "").strip()
        if not path:
            continue
        blake3_hex = _strip_blake3_prefix((f.blake3 or "").strip().lower())
        url = (f.url or "").strip() or None
        transfer_grant = f.transfer_grant if isinstance(f.transfer_grant, dict) else None
        if not blake3_hex or (not url and transfer_grant is None):
            raise ValueError(f"resolved model file missing blake3/transfer: {path}")
        files.append(
            WorkerResolvedRepoFile(
                path=path,
                size_bytes=int(f.size_bytes or 0),
                blake3=blake3_hex,
                url=url,
                transfer_grant=transfer_grant,
            )
        )

    if not files:
        raise ValueError("resolved model has empty files list")

    return WorkerResolvedRepo(snapshot_digest=snapshot_digest, files=files)


def snapshot_dir_key(snapshot_digest: str, components: Sequence[str] = ()) -> str:
    """On-disk snapshot-directory key (pgw#505): the bare digest normally, or
    ``<digest>__c<fingerprint>`` when ``components`` narrows the materialized
    tree to a SUBSET of the digest's full content. Keyed separately so a
    component-scoped (partial) materialization can never be mistaken for —
    or collide with — the full one under the same digest."""
    comps = sorted({c.strip() for c in components if c and str(c).strip()})
    if not comps:
        return snapshot_digest
    fingerprint = hashlib.sha1("+".join(comps).encode()).hexdigest()[:12]
    return f"{snapshot_digest}__c{fingerprint}"


def _filter_resolved_components(
    ref: TensorhubRef, res: WorkerResolvedRepo, components: Sequence[str],
) -> WorkerResolvedRepo:
    """Narrow a validated :class:`WorkerResolvedRepo` to the declared
    pipeline components (+ root config files) — the tensorhub-source twin of
    ``download.select_component_paths`` for the HF path."""
    paths = [f.path for f in res.files]
    if not components_present(paths, components):
        raise ValueError(
            f"components= {list(components)!r} matched nothing in "
            f"{ref.canonical()} snapshot {res.snapshot_digest[:16]}"
        )
    keep = select_component_paths(paths, components)
    files = [f for f in res.files if f.path in keep]
    return WorkerResolvedRepo(snapshot_digest=res.snapshot_digest, files=files)


# ---------------------------------------------------------------------------
# Main downloader
# ---------------------------------------------------------------------------

class CozySnapshotDownloader:
    """Downloads blobs into a CAS layout, reassembles chunked files, materializes snapshot.

    Layout under <base_dir>:
      blobs/blake3/<aa>/<bb>/<digest>
      snapshots/<snapshot_digest>/...            (whole repo)
      snapshots/<snapshot_digest>__c<fp>/...     (components=-scoped subset, pgw#505)
    """

    def __init__(self) -> None:
        pass

    async def ensure_snapshot(
        self,
        base_dir: Path,
        ref: TensorhubRef,
        *,
        resolved: Optional[WorkerResolvedRepo],
        progress: Optional[ProgressFn] = None,
        components: Sequence[str] = (),
    ) -> Path:
        blobs_root = base_dir / "blobs"
        snaps_root = base_dir / "snapshots"
        blobs_root.mkdir(parents=True, exist_ok=True)
        snaps_root.mkdir(parents=True, exist_ok=True)

        if resolved is None:
            # Workers don't resolve via HTTP — the orchestrator pre-resolves
            # every cozy ref a job needs and ships URLs via JobExecutionRequest.
            raise RuntimeError(
                "cozy snapshot requires orchestrator-resolved URLs (resolved=None)"
            )
        res = _validate_resolved(ref, resolved)
        if components:
            res = _filter_resolved_components(ref, res, components)

        # pgw#505: a components=-scoped fetch materializes a NARROWER tree
        # than the digest's full content, so it is keyed separately (never
        # under the bare digest — that name is reserved for the complete
        # snapshot).
        key = snapshot_dir_key(res.snapshot_digest, components)
        snap_dir = snaps_root / key
        if snap_dir.exists():
            _log.info("snapshot_cached key=%s", key[:24])
            return snap_dir

        # Coordinate concurrent builders via threading (works across event loops).
        loop = asyncio.get_running_loop()
        with _SNAP_LOCK:
            if snap_dir.exists():
                return snap_dir
            entry = _SNAP_ENTRIES.get(key)
            if entry is None:
                entry = _SnapshotEntry()
                _SNAP_ENTRIES[key] = entry
                is_builder = True
            else:
                is_builder = False

        if not is_builder:
            _log.info("snapshot_waiting key=%s (another builder active)", key[:24])
            await loop.run_in_executor(None, entry.event.wait)
            if entry.exception is not None:
                raise RuntimeError("concurrent snapshot build failed") from entry.exception
            return snap_dir

        try:
            _log.info("snapshot_build_start key=%s files=%d", key[:24], len(res.files))
            await self._ensure_blobs(
                blobs_root,
                res.files,
                progress=progress,
            )
            # Materialization copies/concatenates multi-GB trees — strictly
            # off the event loop (gw#407: a loop blocked for the duration of
            # a snapshot build cannot answer the hub; under page-cache
            # pressure that IO takes minutes).
            await asyncio.to_thread(
                self._materialize_snapshot, blobs_root, snaps_root, snap_dir, res
            )
            _log.info("snapshot_build_done key=%s", key[:24])
            return snap_dir
        except BaseException as exc:
            entry.exception = exc
            raise
        finally:
            # Digest-poisoning fix (#358): a FAILED build must not park a
            # set-event + stale exception under this digest forever. Evict the
            # entry so the next request creates a fresh builder and retries;
            # waiters already holding this entry still see its exception once.
            with _SNAP_LOCK:
                if _SNAP_ENTRIES.get(key) is entry:
                    del _SNAP_ENTRIES[key]
            entry.event.set()

    def _materialize_snapshot(
        self,
        blobs_root: Path,
        snaps_root: Path,
        snap_dir: Path,
        res: WorkerResolvedRepo,
    ) -> None:
        """Blocking build phase (worker thread): reassemble + hardlink into a
        writer-unique ``.building-<writer_id>`` dir, then atomically rename
        into place. Writer-unique (not a fixed ``.building`` name) so two
        pods racing to materialize the same snapshot key on a SHARED CAS
        root (e.g. one RunPod volume, th#850) never interleave writes into
        the same tree — only the atomic rename below, not the build, decides
        the winner."""
        writer_id = f"{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex}"
        tmp = snaps_root / f"{snap_dir.name}.building-{writer_id}"
        tmp.mkdir(parents=True, exist_ok=True)

        self._reassemble_chunked(blobs_root, tmp, res.files)
        self._materialize_regular(blobs_root, tmp, res.files)

        # Atomic rename; a concurrent writer may have already published.
        if snap_dir.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        else:
            try:
                tmp.rename(snap_dir)
            except OSError:
                shutil.rmtree(tmp, ignore_errors=True)
                if not snap_dir.exists():
                    raise
            else:
                fsync_dir(snaps_root)  # persist the rename itself (gw#408)

    # ------------------------------------------------------------------
    # Blob download (deduplicated, parallel)
    # ------------------------------------------------------------------

    async def _ensure_blobs(
        self,
        blobs_root: Path,
        files: List[WorkerResolvedRepoFile],
        *,
        progress: Optional[ProgressFn] = None,
    ) -> None:
        # Deduplicate by digest — same blob referenced by multiple paths (e.g.
        # fp16 and normal variants sharing the same part) is downloaded once.
        seen: Set[str] = set()
        unique: List[WorkerResolvedRepoFile] = []
        for f in files:
            digest = (f.blake3 or "").strip().lower()
            if not digest:
                raise ValueError(f"missing blake3 for {f.path}")
            if not f.url and not f.transfer_grant:
                raise ValueError(f"missing transfer for {f.path}")
            if digest not in seen:
                seen.add(digest)
                unique.append(f)

        _log.info("ensure_blobs total_entries=%d unique_blobs=%d", len(files), len(unique))

        cached_digests = {
            f.blake3.strip().lower() for f in unique
            if _blob_path(blobs_root, f.blake3.strip().lower()).exists()
        }
        missing_bytes = sum(
            int(f.size_bytes or 0) for f in unique
            if f.blake3.strip().lower() not in cached_digests
        )
        self._check_disk_headroom(blobs_root, missing_bytes)

        total = sum(int(f.size_bytes or 0) for f in unique) or None
        done = total - missing_bytes if total else 0
        network_bytes = 0  # th#850: bytes actually fetched over the network
        # this call, as opposed to bytes already present under blobs_root —
        # the signal a "volume-attached boot ⇒ ~0 network bytes" runtime
        # assertion needs. Only _dl_locked's real fetch increments it.
        done_lock = threading.Lock()

        def _on_bytes(n: int, *, network: bool = False) -> None:
            nonlocal done, network_bytes
            with done_lock:
                done += n
                if network:
                    network_bytes += n
                d = done if total is None else min(done, total)
            if progress is not None:
                try:
                    progress(d, total)
                except Exception:
                    pass

        if progress is not None:
            try:
                progress(min(done, total) if total else done, total)
            except Exception:
                pass

        # Sort largest first for better overlap, then download in parallel.
        unique.sort(key=lambda f: int(f.size_bytes or 0), reverse=True)

        max_conc = 4
        sem = asyncio.Semaphore(max_conc)

        async def _dl(f: WorkerResolvedRepoFile) -> None:
            digest = f.blake3.strip().lower()
            dst = _blob_path(blobs_root, digest)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if self._blob_usable(dst, f):
                _log.info("blob_cached path=%s digest=%s", f.path, digest[:16])
                return
            async with _inflight_blob_lock(digest):
                if self._blob_usable(dst, f):
                    _log.info("blob_shared_inflight path=%s digest=%s (sibling ref downloaded it)",
                              f.path, digest[:16])
                    return
                await _dl_locked(f, digest, dst)

        async def _dl_locked(f: WorkerResolvedRepoFile, digest: str, dst: Path) -> None:
            async with sem:
                if self._blob_usable(dst, f):
                    return
                _log.info("blob_download_start path=%s size=%s digest=%s",
                          f.path, f.size_bytes, digest[:16])
                if f.transfer_grant:
                    grant = S3TransferGrant.from_mapping(f.transfer_grant)
                    await asyncio.get_running_loop().run_in_executor(
                        None,
                        lambda: download_file_with_grant(
                            grant=grant,
                            dest_path=dst,
                            expected_size_bytes=int(f.size_bytes or 0) or None,
                            expected_blake3=digest,
                        ),
                    )
                    _on_bytes(int(f.size_bytes or 0), network=True)
                else:
                    assert f.url is not None  # validated above in _ensure_blobs loop
                    await _download_one_file(
                        f.url,
                        dst,
                        expected_size=int(f.size_bytes or 0),
                        expected_blake3=digest,
                        on_bytes=lambda n: _on_bytes(n, network=True),
                    )
                _log.info("blob_download_done path=%s digest=%s", f.path, digest[:16])

        await asyncio.gather(*(_dl(f) for f in unique))
        # th#850 runtime-assertion signal: on a volume-attached boot with
        # blobs already warm, network_bytes should land near zero. Log-only
        # for now (deliberately not threaded through the ProgressFn/
        # ModelEvent wire contract here — that needs a coordinated proto
        # change with the th#850 owner, see gw#597).
        cache_hit_pct = 100.0 * (1 - network_bytes / total) if total else 100.0
        _log.info(
            "ensure_blobs_summary total_bytes=%s network_bytes=%d cache_hit_pct=%.1f",
            total, network_bytes, cache_hit_pct,
        )

    @staticmethod
    def _blob_usable(dst: Path, f: WorkerResolvedRepoFile) -> bool:
        """A cached blob is only reusable at the manifest's size (gw#408): a
        truncated blob from a pre-durability build must be re-downloaded, not
        silently rebuilt into every future snapshot."""
        try:
            if not dst.exists():
                return False
            expected = int(f.size_bytes or 0)
            if expected and dst.stat().st_size != expected:
                _log.warning(
                    "blob_corrupt path=%s digest=%s size=%d expected=%d; re-downloading",
                    f.path, f.blake3[:16], dst.stat().st_size, expected,
                )
                dst.unlink(missing_ok=True)
                return False
            return True
        except OSError:
            return False

    @staticmethod
    def _check_disk_headroom(blobs_root: Path, missing_bytes: int) -> None:
        if missing_bytes <= 0:
            return
        try:
            free = shutil.disk_usage(blobs_root).free
        except OSError:
            return
        required = missing_bytes + _DISK_HEADROOM_BYTES
        if required > free:
            raise InsufficientDiskError(
                f"insufficient disk for snapshot download: need {required} bytes "
                f"({missing_bytes} blobs + headroom), {free} free at {blobs_root}",
                available_bytes=free,
                required_bytes=required,
                path=str(blobs_root),
            )

    # ------------------------------------------------------------------
    # Chunked file reassembly
    # ------------------------------------------------------------------

    def _reassemble_chunked(
        self, blobs_root: Path, tmp: Path, files: List[WorkerResolvedRepoFile]
    ) -> None:
        """Read .parts.json manifests and concatenate part blobs into original files."""
        for f in files:
            if not _is_parts_manifest(f.path):
                continue

            _log.info("reassemble_start manifest=%s", f.path)
            manifest_blob = _blob_path(blobs_root, f.blake3)
            manifest = json.loads(manifest_blob.read_bytes())

            original_path = str(manifest.get("original_path") or "").strip()
            if not original_path:
                raise ValueError(f"parts manifest {f.path} missing original_path")
            parts = manifest.get("parts") or []
            if not parts:
                raise ValueError(f"parts manifest {f.path} has no parts")

            dst = tmp / _norm_rel_path(original_path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                dst.unlink()

            total_written = 0
            with open(dst, "wb") as out_f:
                for i, part in enumerate(parts):
                    part_digest = _strip_blake3_prefix(str(part.get("digest") or ""))
                    if not part_digest:
                        raise ValueError(f"part[{i}] in {f.path} missing digest")
                    part_blob = _blob_path(blobs_root, part_digest)
                    part_size = part_blob.stat().st_size
                    _log.info("  concat part=%d/%d digest=%s size=%s",
                              i + 1, len(parts), part_digest[:16], part_size)
                    with open(part_blob, "rb") as in_f:
                        shutil.copyfileobj(in_f, out_f)
                    total_written += part_size
                out_f.flush()
                os.fsync(out_f.fileno())  # durable before the snapshot rename (gw#408)

            _log.info("reassemble_done file=%s total_size=%s", original_path, total_written)

    # ------------------------------------------------------------------
    # Regular (non-chunked) file materialization
    # ------------------------------------------------------------------

    def _materialize_regular(
        self, blobs_root: Path, tmp: Path, files: List[WorkerResolvedRepoFile]
    ) -> None:
        """Hardlink/copy non-chunked blobs into the snapshot tree."""
        part_paths = {f.path for f in files if _is_part_file(f.path)}
        for f in files:
            if _is_parts_manifest(f.path) or f.path in part_paths:
                continue
            dst = tmp / _norm_rel_path(f.path)
            dst.parent.mkdir(parents=True, exist_ok=True)
            src = _blob_path(blobs_root, f.blake3)
            _try_hardlink_or_copy(src, dst)



def delete_blobs(base_dir: Path, digests: Any) -> None:
    """Remove specific CAS blobs (quarantine of digest-mismatched content,
    gw#408) so a re-materialization re-downloads them instead of re-linking
    the same corrupt bytes."""
    blobs_root = Path(base_dir) / "blobs"
    for raw in digests or ():
        digest = _strip_blake3_prefix(str(raw or "")).strip().lower()
        if len(digest) < 4:
            continue
        try:
            _blob_path(blobs_root, digest).unlink(missing_ok=True)
        except OSError:
            continue


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

async def ensure_snapshot_async(
    *,
    base_dir: Path,
    ref: TensorhubRef,
    resolved: Optional[WorkerResolvedRepo],
    progress: Optional[ProgressFn] = None,
    components: Sequence[str] = (),
) -> Path:
    dl = CozySnapshotDownloader()
    return await dl.ensure_snapshot(
        base_dir, ref, resolved=resolved, progress=progress, components=components,
    )
