from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import threading
from pathlib import Path
from typing import Any, Coroutine, Dict, List, Optional

from .cozy_cas import _download_one_file as _download_one_file  # reuse verified Range-resume downloader
from .cozy_cas import _norm_rel_path
from .tensorhub_policy import default_resolve_preferences, detect_worker_capabilities
from .tensorhub_v2 import CozyHubV2Client, CozyHubResolveArtifactResult, CozyHubSnapshotFile
from .model_refs import CozyRef

# Module-global blob download locks shared across ALL CozySnapshotV2Downloader
# instances.  Without this, concurrent callers (startup prefetch, task handler,
# LoadModelCommand) each create their own downloader instance with separate
# _blob_locks dicts, allowing parallel writes to the same .part file and
# causing file corruption (interleaved appends → oversized / invalid blobs).
_GLOBAL_BLOB_LOCKS_LOCK = threading.Lock()
_GLOBAL_BLOB_LOCKS: Dict[str, asyncio.Lock] = {}
_GLOBAL_SNAPSHOT_LOCKS_LOCK = threading.Lock()
_GLOBAL_SNAPSHOT_LOCKS: Dict[str, asyncio.Lock] = {}


def _get_global_lock(lock_guard: threading.Lock, lock_map: Dict[str, asyncio.Lock], key: str) -> asyncio.Lock:
    """Return (or create) a per-key asyncio.Lock from a module-global map."""
    with lock_guard:
        lock = lock_map.get(key)
        if lock is None:
            lock = asyncio.Lock()
            lock_map[key] = lock
        return lock


def _blob_path(blobs_root: Path, digest: str) -> Path:
    digest = (digest or "").strip().lower()
    if len(digest) < 4:
        raise ValueError("invalid blake3 digest")
    return blobs_root / "blake3" / digest[:2] / digest[2:4] / digest


_PART_FILE_RE = re.compile(r"\.part\d{4}$")


def _strip_blake3_prefix(digest: str) -> str:
    """Strip the 'blake3:' scheme prefix if present, returning the bare hex."""
    d = (digest or "").strip().lower()
    if d.startswith("blake3:"):
        d = d[len("blake3:"):]
    return d


def _is_part_file(path: str) -> bool:
    """Return True if the path is a chunked part file (e.g. foo.part0001)."""
    return bool(_PART_FILE_RE.search(path))


def _is_parts_manifest(path: str) -> bool:
    """Return True if the path is a chunked-blob manifes (e.g. foo.parts.json)."""
    return path.endswith(".parts.json")


def _resolve_field(obj: Any, *keys: str) -> Any:
    """Get a field from either a dict or an attribute-bearing object, trying keys in order."""
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


def _coerce_resolved_model(ref: CozyRef, resolved: Any) -> CozyHubResolveArtifactResult:
    """Coerce an orchestrator-provided resolved model object into CozyHubResolveArtifactResult.

    Handles two wire shapes:
      - Legacy:  .snapshot_digest (bare hex) + .files[]
      - v2:      .snapshot_digest ('blake3:...' prefixed) + .entries[]  (new chunked-blob format)

    Both protobuf attribute access and plain-dict access are supported.
    """
    snapshot_digest = str(_resolve_field(resolved, "snapshot_digest", "snapshotDigest") or "").strip()
    if not snapshot_digest:
        raise ValueError("resolved model missing snapshot_digest")
    # Strip scheme prefix so the digest is a bare hex string suitable for path use.
    snapshot_digest = _strip_blake3_prefix(snapshot_digest) or snapshot_digest

    # New format uses "entries"; legacy format uses "files".
    files_raw = list(_resolve_field(resolved, "entries", "files") or [])
    files: List[CozyHubSnapshotFile] = []
    for ent in files_raw:
        path = str(_resolve_field(ent, "path") or "").strip()
        if not path:
            continue
        # Prefer bare "blake3" field; fall back to "digest" which may carry the prefix.
        blake3_hex = str(_resolve_field(ent, "blake3", "BLAKE3") or "").strip().lower()
        if not blake3_hex:
            digest_raw = str(_resolve_field(ent, "digest") or "").strip().lower()
            blake3_hex = _strip_blake3_prefix(digest_raw)
        url = str(_resolve_field(ent, "url") or "").strip() or None
        size_bytes = int(_resolve_field(ent, "size_bytes") or 0)
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


class CozySnapshotV2Downloader:
    """Cozy Hub v2 downloader.

    Normal mode:
      - resolve owner/repo:tag (or @digest) via resolve API
      - download all referenced blobs to a local blob store
      - materialize a snapshot checkout by hardlinking blobs into the snapshot tree

    Issue #92 mode:
      - if `resolved` is provided, skip Cozy Hub API calls and use the provided
        presigned URLs directly.

    On-disk layout under <base_dir>:
      - blobs/blake3/<aa>/<bb>/<digest>
      - snapshots/<snapshot_digest>/...
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
            return snap_dir

        lock = _get_global_lock(_GLOBAL_SNAPSHOT_LOCKS_LOCK, _GLOBAL_SNAPSHOT_LOCKS, res.snapshot_digest)
        async with lock:
            if snap_dir.exists():
                return snap_dir

            await self._ensure_blobs(blobs_root, res.files)

            tmp = snaps_root / f"{res.snapshot_digest}.building"
            tmp.mkdir(parents=True, exist_ok=True)

            # Reassemble any chunked files (produced by chunkedblob on ingest).
            # A ".parts.json" entry describes how to reassemble N part blobs into
            # the original file.  Part blobs and the manifest itself are all already
            # in the blob store at this point.
            parts_manifest_entries = [f for f in res.files if _is_parts_manifest(f.path)]
            part_file_paths = {f.path for f in res.files if _is_part_file(f.path)}

            import logging as _logging
            _log = _logging.getLogger("gen_worker.download")

            for pm_entry in parts_manifest_entries:
                _log.info("reassemble_start manifest=%s", pm_entry.path)
                print(f"DEBUG reassemble_start manifest={pm_entry.path}")
                parts_json_blob = _blob_path(blobs_root, pm_entry.blake3)
                try:
                    parts_manifest = json.loads(parts_json_blob.read_bytes())
                except Exception as exc:
                    raise ValueError(f"failed to parse parts manifest {pm_entry.path}: {exc}") from exc

                original_path = str(parts_manifest.get("original_path") or "").strip()
                if not original_path:
                    raise ValueError(f"parts manifest {pm_entry.path} missing original_path")
                parts = parts_manifest.get("parts") or []
                if not parts:
                    raise ValueError(f"parts manifest {pm_entry.path} has no parts")

                rel = _norm_rel_path(original_path)
                dst = tmp / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                if dst.exists():
                    dst.unlink()

                with open(dst, "wb") as out_f:
                    for i, part in enumerate(parts):
                        part_digest = _strip_blake3_prefix(
                            str(part.get("digest") or "").strip().lower()
                        )
                        if not part_digest:
                            raise ValueError(f"part entry in {pm_entry.path} missing digest")
                        part_blob = _blob_path(blobs_root, part_digest)
                        _log.info("  concat_part index=%d digest=%s exists=%s size=%s",
                                  i, part_digest[:16], part_blob.exists(),
                                  part_blob.stat().st_size if part_blob.exists() else -1)
                        print(f"DEBUG   concat_part index={i} digest={part_digest[:16]} exists={part_blob.exists()} size={part_blob.stat().st_size if part_blob.exists() else -1}")
                        with open(part_blob, "rb") as in_f:
                            shutil.copyfileobj(in_f, out_f)

            # Materialize regular files; skip part files and parts manifests since they
            # have already been consumed above during reassembly.
            for f in res.files:
                if _is_parts_manifest(f.path) or f.path in part_file_paths:
                    continue
                rel = _norm_rel_path(f.path)
                dst = tmp / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                src = _blob_path(blobs_root, f.blake3)
                _try_hardlink_or_copy(src, dst)

            # Another concurrent caller (e.g. a different downloader instance) may have
            # already materialized and renamed the snapshot while we were assembling ours.
            # In that case, discard our tmp dir and return the existing snapshot.
            if snap_dir.exists():
                shutil.rmtree(tmp, ignore_errors=True)
                return snap_dir
            try:
                tmp.rename(snap_dir)
            except OSError:
                # Lost the race — snap_dir was created between the exists() check and rename().
                shutil.rmtree(tmp, ignore_errors=True)
                if not snap_dir.exists():
                    raise
            return snap_dir

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

    async def _ensure_blobs(self, blobs_root: Path, files: List[CozyHubSnapshotFile]) -> None:
        import logging
        log = logging.getLogger("gen_worker.download")

        log.info("ensure_blobs total_files=%d", len(files))
        print(f"DEBUG ensure_blobs total_files={len(files)}")
        for f in files:
            log.info("  entry path=%s size=%s digest=%s url_present=%s", f.path, f.size_bytes, (f.blake3 or "")[:16], bool(f.url))
            print(f"DEBUG   entry path={f.path} size={f.size_bytes} digest={(f.blake3 or '')[:16]} url_present={bool(f.url)}")

        all_blobs: List[tuple[CozyHubSnapshotFile, str, Path]] = []
        for f in files:
            digest = (f.blake3 or "").strip().lower()
            if not digest:
                raise ValueError(f"missing blake3 for {f.path}")
            if not f.url:
                raise ValueError(f"missing url for {f.path}")
            dst = _blob_path(blobs_root, digest)
            dst.parent.mkdir(parents=True, exist_ok=True)
            all_blobs.append((f, digest, dst))

        # Parallelize shard/blob downloads to reduce first-load latency for
        # multi-file transformer checkpoints.
        max_conc = max(1, int(os.getenv("WORKER_MODEL_DOWNLOAD_CONCURRENCY", "4") or "4"))
        sem = asyncio.Semaphore(max_conc)

        async def _ensure_one(f: CozyHubSnapshotFile, digest: str, dst: Path) -> None:
            # Acquire the GLOBAL per-digest lock so that concurrent callers across
            # different downloader instances (startup prefetch, task request,
            # LoadModelCommand) don't write to the same .part file in parallel.
            lock = _get_global_lock(_GLOBAL_BLOB_LOCKS_LOCK, _GLOBAL_BLOB_LOCKS, digest)
            async with lock:
                if dst.exists():
                    return
                async with sem:
                    if dst.exists():
                        return
                    assert f.url is not None
                    await _download_one_file(
                        f.url,
                        dst,
                        expected_size=int(f.size_bytes or 0),
                        expected_blake3=digest,
                    )

        # Start larger blobs first for better overlap.
        all_blobs.sort(key=lambda row: int(row[0].size_bytes or 0), reverse=True)
        await asyncio.gather(*(_ensure_one(f, digest, dst) for f, digest, dst in all_blobs))




async def ensure_snapshot_async(
    *,
    base_dir: Path,
    ref: CozyRef,
    base_url: str,
    token: Optional[str],
    resolved: Optional[Any] = None,
) -> Path:
    """Async version of ensure_snapshot_sync for use in async contexts."""
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
