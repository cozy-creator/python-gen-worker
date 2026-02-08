from __future__ import annotations

import asyncio
import os
import shutil
import threading
from pathlib import Path
from typing import Dict, List, Optional

from .cozy_cas import _download_one_file as _download_one_file  # reuse verified Range-resume downloader
from .cozy_cas import _norm_rel_path
from .cozy_hub_policy import default_resolve_preferences, detect_worker_capabilities
from .cozy_hub_v2 import CozyHubV2Client, CozyHubResolveArtifactResult, CozyHubSnapshotFile
from .model_refs import CozyRef


def _blob_path(blobs_root: Path, digest: str) -> Path:
    digest = (digest or "").strip().lower()
    if len(digest) < 4:
        raise ValueError("invalid blake3 digest")
    return blobs_root / "blake3" / digest[:2] / digest[2:4] / digest


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


class CozySnapshotV2Downloader:
    """
    Cozy Hub v2 downloader:
      - resolve owner/repo:tag via resolve_artifact
      - download all referenced blobs to a local blob store
      - materialize a snapshot checkout by hardlinking blobs into the snapshot tree

    On-disk layout under <base_dir>/cozy:
      - blobs/blake3/<aa>/<bb>/<digest>
      - snapshots/<snapshot_digest>/...
    """

    def __init__(self, client: CozyHubV2Client) -> None:
        self._client = client
        self._locks_lock = threading.Lock()
        self._blob_locks: Dict[str, asyncio.Lock] = {}
        self._snapshot_locks: Dict[str, asyncio.Lock] = {}

    async def ensure_snapshot(self, base_dir: Path, ref: CozyRef) -> Path:
        cozy_root = base_dir / "cozy"
        blobs_root = cozy_root / "blobs"
        snaps_root = cozy_root / "snapshots"
        blobs_root.mkdir(parents=True, exist_ok=True)
        snaps_root.mkdir(parents=True, exist_ok=True)

        res = await self._resolve(ref)
        snap_dir = snaps_root / res.snapshot_digest
        if snap_dir.exists():
            return snap_dir

        lock = self._get_lock(self._snapshot_locks, res.snapshot_digest)
        async with lock:
            if snap_dir.exists():
                return snap_dir

            # Download blobs (singleflight per digest).
            await self._ensure_blobs(blobs_root, res.files)

            tmp = snaps_root / f"{res.snapshot_digest}.building"
            tmp.mkdir(parents=True, exist_ok=True)
            for f in res.files:
                rel = _norm_rel_path(f.path)
                dst = tmp / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                src = _blob_path(blobs_root, f.blake3)
                _try_hardlink_or_copy(src, dst)

            tmp.rename(snap_dir)
            return snap_dir

    async def _resolve(self, ref: CozyRef) -> CozyHubResolveArtifactResult:
        if ref.digest:
            # If the caller already pinned a snapshot digest, just fetch the snapshot manifest.
            files = await self._client.get_snapshot_manifest(owner=ref.owner, repo=ref.repo, digest=ref.digest)
            return CozyHubResolveArtifactResult(
                repo_revision_seq=0,
                snapshot_digest=ref.digest,
                artifact=None,  # type: ignore[arg-type]
                files=files,
            )

        prefs = default_resolve_preferences()
        caps = detect_worker_capabilities()
        return await self._client.resolve_artifact(
            owner=ref.owner,
            repo=ref.repo,
            tag=ref.tag,
            include_urls=True,
            preferences=prefs,
            capabilities=caps.to_dict(),
        )

    async def _ensure_blobs(self, blobs_root: Path, files: List[CozyHubSnapshotFile]) -> None:
        for f in files:
            digest = (f.blake3 or "").strip().lower()
            if not digest:
                raise ValueError(f"missing blake3 for {f.path}")
            if not f.url:
                raise ValueError(f"missing url for {f.path}")
            dst = _blob_path(blobs_root, digest)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                continue

            lock = self._get_lock(self._blob_locks, digest)
            async with lock:
                if dst.exists():
                    continue
                await _download_one_file(
                    f.url,
                    dst,
                    expected_size=int(f.size_bytes or 0),
                    expected_blake3=digest,
                )

    def _get_lock(self, mp: Dict[str, asyncio.Lock], key: str) -> asyncio.Lock:
        with self._locks_lock:
            lock = mp.get(key)
            if lock is None:
                lock = asyncio.Lock()
                mp[key] = lock
            return lock


def ensure_snapshot_sync(
    *,
    base_dir: Path,
    ref: CozyRef,
    base_url: str,
    token: Optional[str],
) -> Path:
    client = CozyHubV2Client(base_url=base_url, token=token)
    dl = CozySnapshotV2Downloader(client)

    async def _run() -> Path:
        return await dl.ensure_snapshot(base_dir, ref)

    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            return Path(_run_in_thread(_run()))
    except RuntimeError:
        pass
    return asyncio.run(_run())


def _run_in_thread(coro: "asyncio.Future[Path]") -> str:
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
