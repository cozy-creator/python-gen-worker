from __future__ import annotations

import asyncio
import json
import os
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import aiohttp
import backoff
from blake3 import blake3

from .refs import CozyRef


def _norm_rel_path(p: str) -> str:
    s = (p or "").strip().replace("\\", "/")
    s = s.lstrip("/")
    if not s or s == ".":
        raise ValueError("empty relative path")
    if ".." in s.split("/"):
        raise ValueError("path traversal not allowed")
    return s


def blake3_tree_hash(entries: Iterable[Tuple[str, int, str]]) -> str:
    """
    Compute a deterministic tree hash from (rel_path, size_bytes, blake3_hex) entries.

    Canonical encoding:
      - Normalize paths to UTF-8 with `/` separators, no leading `/`.
      - Sort by rel_path (bytewise).
      - Encode each entry as: "<rel_path>\t<size_bytes>\t<blake3_hex>\n" (UTF-8).
      - Digest = blake3(concat(all_entries)).
    """
    normalized: List[Tuple[str, int, str]] = []
    for rel, size, hx in entries:
        normalized.append((_norm_rel_path(rel), int(size), (hx or "").strip().lower()))
    normalized.sort(key=lambda t: t[0].encode("utf-8"))

    h = blake3()
    for rel, size, hx in normalized:
        if not hx:
            raise ValueError(f"missing blake3 for {rel}")
        h.update(rel.encode("utf-8"))
        h.update(b"\t")
        h.update(str(size).encode("utf-8"))
        h.update(b"\t")
        h.update(hx.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


@dataclass(frozen=True)
class CozyFileEntry:
    path: str
    size_bytes: int
    blake3_hex: str
    url: str


@dataclass(frozen=True)
class CozyObjectManifest:
    object_digest: str
    files: List[CozyFileEntry]


@dataclass(frozen=True)
class CozySnapshotManifest:
    snapshot_digest: str
    objects: Dict[str, str]  # object_name -> object_digest
    root_files: List[CozyFileEntry]


class CozyHubClient:
    """
    Minimal Cozy Hub client for snapshot/object resolution.

    Expected routes (see tensorhub issue id=40):
      - GET /api/v1/repos/<owner>/<repo>/resolve?tag=<tag> -> {"digest": "..."}
      - GET /api/v1/repos/<owner>/<repo>/snapshots/<digest>/manifest
      - GET /api/v1/objects/<object_digest>/manifest
    """

    def __init__(self, base_url: str, token: Optional[str] = None, timeout_s: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.token = (token or "").strip() or None
        self.timeout_s = timeout_s

    def _headers(self) -> Dict[str, str]:
        h: Dict[str, str] = {}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    async def _get_json(self, path: str) -> Mapping[str, Any]:
        url = f"{self.base_url}{path}"
        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        async with aiohttp.ClientSession(timeout=timeout, headers=self._headers()) as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                data = await resp.json()
                if not isinstance(data, dict):
                    raise ValueError("unexpected response shape")
                return data

    async def resolve_tag(self, owner: str, repo: str, tag: str) -> str:
        data = await self._get_json(f"/api/v1/repos/{owner}/{repo}/resolve?tag={aiohttp.helpers.quote(tag)}")
        digest = str(data.get("digest") or data.get("snapshot_digest") or "").strip()
        if not digest:
            raise ValueError("resolve response missing digest")
        return digest

    async def get_snapshot_manifest(self, owner: str, repo: str, digest: str) -> CozySnapshotManifest:
        data = await self._get_json(f"/api/v1/repos/{owner}/{repo}/snapshots/{digest}/manifest")
        snap = str(data.get("snapshot_digest") or data.get("digest") or digest).strip()
        objects_raw = data.get("objects") or {}
        if not isinstance(objects_raw, dict):
            raise ValueError("snapshot manifest missing objects")
        objects: Dict[str, str] = {str(k): str(v) for k, v in objects_raw.items() if k and v}
        files_raw = data.get("files") or data.get("root_files") or []
        root_files = _parse_files(files_raw)
        return CozySnapshotManifest(snapshot_digest=snap, objects=objects, root_files=root_files)

    async def get_object_manifest(self, object_digest: str) -> CozyObjectManifest:
        data = await self._get_json(f"/api/v1/objects/{object_digest}/manifest")
        od = str(data.get("object_digest") or data.get("digest") or object_digest).strip()
        files_raw = data.get("files") or []
        files = _parse_files(files_raw)
        return CozyObjectManifest(object_digest=od, files=files)


def _parse_files(v: Any) -> List[CozyFileEntry]:
    if not isinstance(v, list):
        return []
    out: List[CozyFileEntry] = []
    for ent in v:
        if not isinstance(ent, dict):
            continue
        path = str(ent.get("path") or "").strip()
        url = str(ent.get("url") or "").strip()
        if not path or not url:
            continue
        size = int(ent.get("size_bytes") or ent.get("size") or 0)
        b3 = str(ent.get("blake3") or ent.get("blake3_hex") or "").strip()
        if not b3:
            # allow missing hash for now, but keep empty to force download-time decision
            b3 = ""
        out.append(CozyFileEntry(path=path, size_bytes=size, blake3_hex=b3, url=url))
    return out


class CozySnapshotDownloader:
    """
    Downloads Cozy snapshots using Cozy Hub manifest APIs into a local CAS layout:

      <base>/objects/<object_digest>/...
      <base>/snapshots/<snapshot_digest>/...
    """

    def __init__(self, client: CozyHubClient) -> None:
        self._client = client
        self._object_locks: Dict[str, threading.Lock] = {}
        self._snapshot_locks: Dict[str, threading.Lock] = {}
        self._locks_lock = threading.Lock()

    async def canonicalize(self, ref: CozyRef) -> CozyRef:
        if ref.digest:
            return ref
        digest = await self._client.resolve_tag(ref.owner, ref.repo, ref.tag)
        return CozyRef(owner=ref.owner, repo=ref.repo, tag=ref.tag, digest=digest)

    async def ensure_snapshot(self, base_dir: Path, ref: CozyRef) -> Path:
        canon = await self.canonicalize(ref)
        snapshot_digest = canon.digest
        if not snapshot_digest:
            raise ValueError("snapshot digest missing after canonicalization")

        objects_root = base_dir / "objects"
        snaps_root = base_dir / "snapshots"
        objects_root.mkdir(parents=True, exist_ok=True)
        snaps_root.mkdir(parents=True, exist_ok=True)

        snap_dir = snaps_root / snapshot_digest
        if snap_dir.exists():
            return snap_dir

        lock = self._get_lock(self._snapshot_locks, snapshot_digest)
        with lock:
            if snap_dir.exists():
                return snap_dir
            manifest = await self._client.get_snapshot_manifest(canon.owner, canon.repo, snapshot_digest)

            # Ensure all referenced objects exist locally.
            resolved: Dict[str, str] = {}
            for name, od in manifest.objects.items():
                await self._ensure_object(objects_root, od)
                resolved[name] = od

            # Materialize snapshot directory with links + root files.
            tmp = snaps_root / f"{snapshot_digest}.building"
            # Keep partial builds to support resume; we overwrite links/files as needed.
            tmp.mkdir(parents=True, exist_ok=True)

            for name, od in resolved.items():
                target = objects_root / od
                link = tmp / name
                _safe_symlink_dir(target, link)

            if manifest.root_files:
                await self._download_files(tmp, manifest.root_files)

            # Write a resolved manifest for debugging/validation.
            (tmp / ".cozy-resolved.json").write_text(
                json.dumps(
                    {
                        "repo": canon.repo_id(),
                        "tag": canon.tag,
                        "snapshot_digest": snapshot_digest,
                        "objects": resolved,
                        "root_files": [f.path for f in manifest.root_files],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            tmp.rename(snap_dir)
            return snap_dir

    def _get_lock(self, mp: Dict[str, threading.Lock], key: str) -> threading.Lock:
        with self._locks_lock:
            lock = mp.get(key)
            if lock is None:
                lock = threading.Lock()
                mp[key] = lock
            return lock

    async def _ensure_object(self, objects_root: Path, object_digest: str) -> Path:
        obj_dir = objects_root / object_digest
        meta_path = obj_dir / ".cozy-object.json"
        if obj_dir.exists() and meta_path.exists():
            return obj_dir

        lock = self._get_lock(self._object_locks, object_digest)
        with lock:
            if obj_dir.exists() and meta_path.exists():
                return obj_dir

            manifest = await self._client.get_object_manifest(object_digest)

            tmp = objects_root / f"{object_digest}.downloading"
            # Keep partial downloads to support resume; we verify files individually.
            tmp.mkdir(parents=True, exist_ok=True)

            await self._download_files(tmp, manifest.files)

            meta_path_tmp = tmp / ".cozy-object.json"
            meta_path_tmp.write_text(
                json.dumps(
                    {
                        "object_digest": object_digest,
                        "files": [
                            {"path": f.path, "size_bytes": f.size_bytes, "blake3": f.blake3_hex}
                            for f in manifest.files
                        ],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            tmp.rename(obj_dir)
            return obj_dir

    async def _download_files(self, root: Path, files: List[CozyFileEntry]) -> None:
        # Download sequentially by default; caller can parallelize objects. Keep this simple.
        for f in files:
            rel = _norm_rel_path(f.path)
            dst = root / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            await _download_one_file(f.url, dst, expected_size=f.size_bytes, expected_blake3=f.blake3_hex)


def _safe_symlink_dir(target: Path, link: Path) -> None:
    if link.exists() or link.is_symlink():
        if link.is_dir() and not link.is_symlink():
            shutil.rmtree(link, ignore_errors=True)
        else:
            link.unlink(missing_ok=True)
    try:
        os.symlink(str(target), str(link), target_is_directory=True)
    except Exception:
        # Fallback: create a directory junction-like copy (expensive); keep minimal.
        shutil.copytree(target, link, dirs_exist_ok=True)


@backoff.on_exception(
    backoff.expo,
    (aiohttp.ClientError, asyncio.TimeoutError, ValueError, OSError),
    max_tries=30,
    max_time=3600,
    factor=1,
    max_value=30,  # cap backoff at 30s between retries
)
async def _download_one_file(url: str, dst: Path, expected_size: int, expected_blake3: str) -> None:
    """Download a single file with HTTP Range resume, size + blake3 validation.

    Fully async — no blocking calls that would stall the event loop.
    Caller is responsible for ensuring only one coroutine downloads a given
    dst at a time (dedup by digest in _ensure_blobs).
    """
    import logging
    log = logging.getLogger("gen_worker.download")

    def _human_size(n: int) -> str:
        if n >= 1 << 30:
            return f"{n / (1 << 30):.1f}GB"
        if n >= 1 << 20:
            return f"{n / (1 << 20):.1f}MB"
        if n >= 1 << 10:
            return f"{n / (1 << 10):.1f}KB"
        return f"{n}B"

    # Already downloaded and valid?
    if dst.exists():
        try:
            if expected_size and dst.stat().st_size != expected_size:
                raise ValueError("size mismatch")
            if expected_blake3:
                got = _blake3_file(dst)
                if got.lower() != expected_blake3.lower():
                    raise ValueError("blake3 mismatch")
            log.info("download_cached path=%s size=%s", dst.name, _human_size(dst.stat().st_size))
            return
        except Exception:
            pass  # re-download

    timeout = aiohttp.ClientTimeout(
        total=None,
        sock_connect=float(os.getenv("WORKER_MODEL_DOWNLOAD_SOCK_CONNECT_TIMEOUT_S", "60")),
        sock_read=float(os.getenv("WORKER_MODEL_DOWNLOAD_SOCK_READ_TIMEOUT_S", "180")),
    )
    tmp = dst.with_suffix(dst.suffix + ".part")

    # Resume from partial download if available.
    offset = 0
    if tmp.exists():
        try:
            offset = tmp.stat().st_size
        except OSError:
            offset = 0
        if expected_size and offset > expected_size:
            tmp.unlink(missing_ok=True)
            offset = 0

    # Partial file already complete? Validate and finalize.
    if offset and expected_size and offset == expected_size:
        got = _blake3_file(tmp)
        if expected_blake3 and got.lower() != expected_blake3.lower():
            log.warning("partial_corrupt path=%s (blake3 mismatch, restarting)", dst.name)
            tmp.unlink(missing_ok=True)
            offset = 0
        else:
            tmp.rename(dst)
            log.info("download_resumed_complete path=%s size=%s", dst.name, _human_size(expected_size))
            return

    headers: Dict[str, str] = {}
    mode = "wb"
    if offset and expected_size:
        headers["Range"] = f"bytes={offset}-"
        mode = "ab"
        log.info("download_resume path=%s offset=%s/%s (%s/%s)",
                 dst.name, offset, expected_size,
                 _human_size(offset), _human_size(expected_size))
    else:
        log.info("download_start path=%s size=%s blake3=%s",
                 dst.name, _human_size(expected_size) if expected_size else "unknown",
                 (expected_blake3 or "n/a")[:16])

    async def _stream(resp: aiohttp.ClientResponse, *, write_mode: str, start: int) -> None:
        downloaded = start
        last_log = start
        log_every = max(expected_size // 10, 50 << 20) if expected_size else (100 << 20)
        with open(tmp, write_mode) as f:
            async for chunk in resp.content.iter_chunked(1 << 20):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
                if expected_size and downloaded > expected_size:
                    raise ValueError(f"download exceeded expected size ({downloaded} > {expected_size})")
                if downloaded - last_log >= log_every:
                    pct = f" ({100 * downloaded // expected_size}%)" if expected_size else ""
                    log.info("download_progress path=%s downloaded=%s%s",
                             dst.name, _human_size(downloaded), pct)
                    last_log = downloaded

    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(url, headers=headers) as resp:
            content_range = str(resp.headers.get("Content-Range") or "").strip()

            # Server ignored Range or returned unexpected range start?
            # Restart from byte 0 to avoid corrupted appends.
            if offset and (
                resp.status == 200
                or (resp.status == 206 and not content_range.startswith(f"bytes {offset}-"))
            ):
                log.info("download_range_ignored path=%s status=%s (restarting from 0)", dst.name, resp.status)
                resp.release()
                async with session.get(url) as resp2:
                    resp2.raise_for_status()
                    await _stream(resp2, write_mode="wb", start=0)
            else:
                resp.raise_for_status()
                await _stream(resp, write_mode=mode, start=offset)

    # Validate.
    actual_size = tmp.stat().st_size
    if expected_size and actual_size != expected_size:
        log.error("download_size_mismatch path=%s expected=%s got=%s", dst.name, expected_size, actual_size)
        tmp.unlink(missing_ok=True)
        raise ValueError(f"size mismatch (expected {expected_size}, got {actual_size})")

    if expected_blake3:
        got = _blake3_file(tmp)
        if got.lower() != expected_blake3.lower():
            log.error("download_blake3_mismatch path=%s expected=%s got=%s",
                      dst.name, expected_blake3[:16], got[:16])
            tmp.unlink(missing_ok=True)
            raise ValueError("blake3 mismatch")

    # Atomic finalize.
    tmp.replace(dst)
    log.info("download_done path=%s size=%s", dst.name, _human_size(actual_size))


def _blake3_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = blake3()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()
