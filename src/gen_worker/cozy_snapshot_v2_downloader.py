from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import threading
from pathlib import Path
from typing import Any, Coroutine, Dict, List, Optional, Set

from .cozy_cas import _download_one_file as _download_one_file
from .cozy_cas import _norm_rel_path
from .tensorhub_policy import default_resolve_preferences, detect_worker_capabilities
from .tensorhub_v2 import CozyHubV2Client, CozyHubResolveArtifactResult, CozyHubSnapshotFile
from .model_refs import CozyRef

_log = logging.getLogger("gen_worker.download")


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


def _field(obj: Any, *keys: str) -> Any:
    """Read a field from dict or object, trying keys in order."""
    for k in keys:
        if isinstance(obj, dict):
            v = obj.get(k)
        else:
            v = getattr(obj, k, None)
        if v is not None:
            return v
    return None


def _try_hardlink_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return
    except Exception:
        pass
    try:
        os.symlink(src, dst)
        return
    except Exception:
        pass
    shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Coerce orchestrator wire format -> internal type
# ---------------------------------------------------------------------------

def _coerce_resolved_model(ref: CozyRef, resolved: Any) -> CozyHubResolveArtifactResult:
    """Handle both legacy (.files[]) and v2 (.entries[], blake3:-prefixed digests)."""
    snapshot_digest = str(_field(resolved, "snapshot_digest", "snapshotDigest") or "").strip()
    if not snapshot_digest:
        raise ValueError("resolved model missing snapshot_digest")
    snapshot_digest = _strip_blake3_prefix(snapshot_digest) or snapshot_digest

    # v2 uses "entries", legacy uses "files"
    files_raw = list(_field(resolved, "entries", "files") or [])
    files: List[CozyHubSnapshotFile] = []
    for ent in files_raw:
        path = str(_field(ent, "path") or "").strip()
        if not path:
            continue
        blake3_hex = str(_field(ent, "blake3", "BLAKE3") or "").strip().lower()
        if not blake3_hex:
            blake3_hex = _strip_blake3_prefix(str(_field(ent, "digest") or ""))
        url = str(_field(ent, "url") or "").strip() or None
        size_bytes = int(_field(ent, "size_bytes") or 0)
        if not blake3_hex or not url:
            raise ValueError(f"resolved model file missing blake3/url: {path}")
        files.append(CozyHubSnapshotFile(path=path, size_bytes=size_bytes, blake3=blake3_hex, url=url))

    if not files:
        raise ValueError("resolved model has empty files list")

    return CozyHubResolveArtifactResult(
        repo_revision_seq=0,
        snapshot_digest=snapshot_digest,
        artifact=None,
        files=files,
    )


# ---------------------------------------------------------------------------
# Main downloader
# ---------------------------------------------------------------------------

class CozySnapshotV2Downloader:
    """Downloads blobs into a CAS layout, reassembles chunked files, materializes snapshot.

    Layout under <base_dir>:
      blobs/blake3/<aa>/<bb>/<digest>
      snapshots/<snapshot_digest>/...
    """

    def __init__(self, client: Optional[CozyHubV2Client]) -> None:
        self._client = client

    async def ensure_snapshot(
        self,
        base_dir: Path,
        ref: CozyRef,
        *,
        resolved: Optional[Any] = None,
    ) -> Path:
        blobs_root = base_dir / "blobs"
        snaps_root = base_dir / "snapshots"
        blobs_root.mkdir(parents=True, exist_ok=True)
        snaps_root.mkdir(parents=True, exist_ok=True)

        if resolved is not None:
            res = _coerce_resolved_model(ref, resolved)
        else:
            res = await self._resolve(ref)

        snap_dir = snaps_root / res.snapshot_digest
        if snap_dir.exists():
            _log.info("snapshot_cached digest=%s", res.snapshot_digest[:16])
            return snap_dir

        # Coordinate concurrent builders via threading (works across event loops).
        loop = asyncio.get_running_loop()
        with _SNAP_LOCK:
            if snap_dir.exists():
                return snap_dir
            entry = _SNAP_ENTRIES.get(res.snapshot_digest)
            if entry is None:
                entry = _SnapshotEntry()
                _SNAP_ENTRIES[res.snapshot_digest] = entry
                is_builder = True
            else:
                is_builder = False

        if not is_builder:
            _log.info("snapshot_waiting digest=%s (another builder active)", res.snapshot_digest[:16])
            await loop.run_in_executor(None, entry.event.wait)
            if entry.exception is not None:
                raise RuntimeError(f"concurrent snapshot build failed") from entry.exception
            return snap_dir

        try:
            _log.info("snapshot_build_start digest=%s files=%d", res.snapshot_digest[:16], len(res.files))
            await self._ensure_blobs(blobs_root, res.files)

            tmp = snaps_root / f"{res.snapshot_digest}.building"
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir(parents=True, exist_ok=True)

            self._reassemble_chunked(blobs_root, tmp, res.files)
            self._materialize_regular(blobs_root, tmp, res.files)

            # Atomic rename; handle race with concurrent builder.
            if snap_dir.exists():
                shutil.rmtree(tmp, ignore_errors=True)
            else:
                try:
                    tmp.rename(snap_dir)
                except OSError:
                    shutil.rmtree(tmp, ignore_errors=True)
                    if not snap_dir.exists():
                        raise

            _log.info("snapshot_build_done digest=%s", res.snapshot_digest[:16])
            return snap_dir
        except BaseException as exc:
            entry.exception = exc
            raise
        finally:
            entry.event.set()

    # ------------------------------------------------------------------
    # Blob download (deduplicated, parallel)
    # ------------------------------------------------------------------

    async def _ensure_blobs(self, blobs_root: Path, files: List[CozyHubSnapshotFile]) -> None:
        # Deduplicate by digest — same blob referenced by multiple paths (e.g.
        # fp16 and normal variants sharing the same part) is downloaded once.
        seen: Set[str] = set()
        unique: List[CozyHubSnapshotFile] = []
        for f in files:
            digest = (f.blake3 or "").strip().lower()
            if not digest:
                raise ValueError(f"missing blake3 for {f.path}")
            if not f.url:
                raise ValueError(f"missing url for {f.path}")
            if digest not in seen:
                seen.add(digest)
                unique.append(f)

        _log.info("ensure_blobs total_entries=%d unique_blobs=%d", len(files), len(unique))

        # Sort largest first for better overlap, then download in parallel.
        unique.sort(key=lambda f: int(f.size_bytes or 0), reverse=True)

        max_conc = max(1, int(os.getenv("WORKER_MODEL_DOWNLOAD_CONCURRENCY", "4") or "4"))
        sem = asyncio.Semaphore(max_conc)

        async def _dl(f: CozyHubSnapshotFile) -> None:
            digest = f.blake3.strip().lower()
            dst = _blob_path(blobs_root, digest)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                _log.info("blob_cached path=%s digest=%s", f.path, digest[:16])
                return
            async with sem:
                if dst.exists():
                    return
                _log.info("blob_download_start path=%s size=%s digest=%s",
                          f.path, f.size_bytes, digest[:16])
                assert f.url is not None  # validated above in _ensure_blobs loop
                await _download_one_file(
                    f.url,
                    dst,
                    expected_size=int(f.size_bytes or 0),
                    expected_blake3=digest,
                )
                _log.info("blob_download_done path=%s digest=%s", f.path, digest[:16])

        await asyncio.gather(*(_dl(f) for f in unique))

    # ------------------------------------------------------------------
    # Chunked file reassembly
    # ------------------------------------------------------------------

    def _reassemble_chunked(
        self, blobs_root: Path, tmp: Path, files: List[CozyHubSnapshotFile]
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

            _log.info("reassemble_done file=%s total_size=%s", original_path, total_written)

    # ------------------------------------------------------------------
    # Regular (non-chunked) file materialization
    # ------------------------------------------------------------------

    def _materialize_regular(
        self, blobs_root: Path, tmp: Path, files: List[CozyHubSnapshotFile]
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

    # ------------------------------------------------------------------
    # Hub resolution
    # ------------------------------------------------------------------

    async def _resolve(self, ref: CozyRef) -> CozyHubResolveArtifactResult:
        if self._client is None:
            raise RuntimeError("cozy hub api resolve is disabled")
        prefs = default_resolve_preferences()
        caps = detect_worker_capabilities()
        return await self._client.resolve_artifact(
            owner=ref.owner,
            repo=ref.repo,
            tag=ref.tag,
            digest=ref.digest,
            include_urls=True,
            preferences=prefs,
            capabilities=caps.to_dict(),
        )


# ---------------------------------------------------------------------------
# Convenience wrappers
# ---------------------------------------------------------------------------

async def ensure_snapshot_async(
    *,
    base_dir: Path,
    ref: CozyRef,
    base_url: str,
    token: Optional[str],
    resolved: Optional[Any] = None,
) -> Path:
    client: Optional[CozyHubV2Client] = None
    if resolved is None:
        if not (base_url or "").strip():
            raise RuntimeError("cozy downloads require TENSORHUB_URL")
        client = CozyHubV2Client(base_url=base_url, token=token)
    dl = CozySnapshotV2Downloader(client)
    return await dl.ensure_snapshot(base_dir, ref, resolved=resolved)


def ensure_snapshot_sync(
    *,
    base_dir: Path,
    ref: CozyRef,
    base_url: str,
    token: Optional[str],
    resolved: Optional[Any] = None,
) -> Path:
    client: Optional[CozyHubV2Client] = None
    if resolved is None:
        if not (base_url or "").strip():
            raise RuntimeError("cozy downloads require TENSORHUB_URL")
        client = CozyHubV2Client(base_url=base_url, token=token)

    dl = CozySnapshotV2Downloader(client)

    async def _run() -> Path:
        return await dl.ensure_snapshot(base_dir, ref, resolved=resolved)

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            return Path(_run_in_thread(_run()))
    except RuntimeError:
        pass
    return asyncio.run(_run())


def _run_in_thread(coro: Coroutine[Any, Any, Path]) -> str:
    out: dict[str, str] = {}
    err: dict[str, BaseException] = {}

    def runner() -> None:
        try:
            out["v"] = asyncio.run(coro).as_posix()
        except BaseException as e:
            err["e"] = e

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    t.join()
    if "e" in err:
        raise err["e"]
    return out["v"]
