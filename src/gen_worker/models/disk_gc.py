"""Disk retention (#370): a persistent ref->bytes index + deletion helpers."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import shutil
import stat
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional, Tuple

from gen_worker._vendor.tensorfs import RefConflict
from gen_worker.cas.retention import collect_garbage

from .cache_paths import open_worker_cas

logger = logging.getLogger(__name__)

_INDEX_NAME = "ref-index.json"


class RefIndex:
    """Persistent {ref: {path, bytes, last_used}} with process-safe writes."""

    def __init__(self, cache_dir: Path) -> None:
        self._path = Path(cache_dir) / _INDEX_NAME
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, Any]] = {}
        with self._locked(exclusive=False):
            pass

    def _read_locked(self) -> Dict[str, Dict[str, Any]]:
        try:
            raw = json.loads(self._path.read_text("utf-8"))
            if isinstance(raw, dict):
                return {
                    str(k): v for k, v in raw.items()
                    if isinstance(v, dict) and v.get("path")
                }
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("ref-index unreadable (%s); starting empty", exc)
        return {}

    @contextmanager
    def _locked(self, *, exclusive: bool) -> Iterator[None]:
        with self._lock:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            directory = os.open(
                self._path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            )
            try:
                fcntl.flock(
                    directory, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
                )
                try:
                    self._data = self._read_locked()
                    yield
                finally:
                    fcntl.flock(directory, fcntl.LOCK_UN)
            finally:
                os.close(directory)

    def _save_locked(self) -> None:
        """The caller holds both the thread lock and the process-shared flock and has refreshed ``self._data`` from disk."""
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                dir=str(self._path.parent), prefix=self._path.name + ".",
                suffix=".tmp")
            tmp = Path(tmp_name)
            try:
                os.fchmod(fd, 0o644)
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(json.dumps(self._data))
                    fh.flush()
                    os.fsync(fh.fileno())
                tmp.replace(self._path)
                directory = os.open(self._path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            except BaseException:
                tmp.unlink(missing_ok=True)
                raise
        except Exception as exc:
            logger.warning("ref-index write failed: %s", exc)

    def record(self, ref: str, path: Path, size_bytes: int) -> None:
        with self._locked(exclusive=True):
            self._data[ref] = {
                "path": str(path), "bytes": int(size_bytes), "last_used": time.time(),
            }
            self._save_locked()

    def touch(self, ref: str) -> None:
        with self._locked(exclusive=True):
            e = self._data.get(ref)
            if e is not None:
                e["last_used"] = time.time()
                self._save_locked()

    def remove(self, ref: str) -> None:
        with self._locked(exclusive=True):
            if self._data.pop(ref, None) is not None:
                self._save_locked()

    def path(self, ref: str) -> Optional[Path]:
        with self._locked(exclusive=False):
            e = self._data.get(ref)
            return Path(e["path"]) if e else None

    def last_used(self, ref: str) -> float:
        with self._locked(exclusive=False):
            e = self._data.get(ref)
            return float(e.get("last_used") or 0.0) if e else 0.0

    def entries(self) -> Dict[str, Dict[str, Any]]:
        with self._locked(exclusive=False):
            return {k: dict(v) for k, v in self._data.items()}


def tree_bytes(path: Path) -> int:
    """Bytes under ``path`` (file or tree), hardlinked inodes counted once."""
    p = Path(path)
    try:
        from . import projection

        snapshot = projection.resolve_projection(p)
    except Exception:
        snapshot = None
    if snapshot is not None:
        from .materialized_view import view_root_for

        return sum(
            int(entry.size_bytes) for entry in snapshot.manifest.files
        ) + tree_bytes(view_root_for(p))
    try:
        if p.is_file():
            return int(p.stat().st_size)
        total = 0
        seen: set = set()
        for dirpath, _dirs, names in os.walk(p):
            for name in names:
                try:
                    st = os.stat(os.path.join(dirpath, name))
                except OSError:
                    continue
                key = (st.st_dev, st.st_ino)
                if key in seen:
                    continue
                seen.add(key)
                total += int(st.st_size)
        return total
    except OSError:
        return 0


def _regular_files(path: Path) -> Iterator[Tuple[Path, os.stat_result]]:
    root = Path(path)
    candidates: Iterable[Path]
    if root.is_dir():
        candidates = (
            Path(dirpath) / name
            for dirpath, _dirs, names in os.walk(root, followlinks=False)
            for name in names
        )
    else:
        candidates = iter((root,))
    for candidate in candidates:
        try:
            info = candidate.stat()
        except OSError:
            continue
        if stat.S_ISREG(info.st_mode):
            yield candidate, info


def reclaim_file_cache(
    path: Path, *, preserve_paths: Iterable[Path] = (),
) -> int:
    """Drop clean cached pages for an immutable local snapshot, best effort."""
    advise = getattr(os, "posix_fadvise", None)
    dontneed = getattr(os, "POSIX_FADV_DONTNEED", None)
    if advise is None or dontneed is None:
        return 0

    protected: set[Tuple[int, int]] = set()
    for protected_path in preserve_paths:
        for _file, info in _regular_files(Path(protected_path)):
            protected.add((int(info.st_dev), int(info.st_ino)))

    advised = 0
    seen: set[Tuple[int, int]] = set()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    for file_path, info in _regular_files(Path(path)):
        key = (int(info.st_dev), int(info.st_ino))
        if key in seen or key in protected:
            continue
        seen.add(key)
        try:
            fd = os.open(file_path, flags)
        except OSError:
            continue
        try:
            current = os.fstat(fd)
            current_key = (int(current.st_dev), int(current.st_ino))
            if current_key in protected or not stat.S_ISREG(current.st_mode):
                continue
            advise(fd, 0, 0, dontneed)
            advised += int(current.st_size)
        except OSError:
            continue
        finally:
            os.close(fd)
    return advised


def _retention_unit(path: Path, cas_dir: Path) -> Path:
    p = Path(path)
    snaps_root = Path(cas_dir) / "snapshots"
    try:
        rel = p.resolve().relative_to(snaps_root.resolve())
        return snaps_root / rel.parts[0]
    except (ValueError, OSError):
        pass
    for parent in (p, *p.parents):
        if parent.name.startswith("models--"):
            return parent
    return p


def delete_ref_bytes(ref: str, path: Path, cas_dir: Path) -> None:
    unit = _retention_unit(path, cas_dir)
    logger.info("disk-gc: deleting %s (%s)", ref, unit)
    snapshots = Path(cas_dir) / "snapshots"
    if unit.parent == snapshots:
        cas = open_worker_cas(cas_dir)
        name = f"snapshot:{unit.name}"
        current = cas.read_ref(name)
        if current is not None:
            try:
                cas.compare_and_swap_ref(name, None, expected=current)
            except RefConflict:
                logger.info("disk-gc: snapshot ref %s changed while deleting", name)
        from .materialized_view import view_root_for

        shutil.rmtree(view_root_for(unit), ignore_errors=True)
    if unit.is_dir():
        shutil.rmtree(unit, ignore_errors=True)
    else:
        unit.unlink(missing_ok=True)


def sweep_orphan_blobs(cas_dir: Path) -> int:
    """Collect unpinned tensorfs objects after the writer safety grace."""

    return int(
        collect_garbage(
            open_worker_cas(cas_dir), older_than=_STALE_WRITER_TEMP_AGE_S
        ).bytes_deleted
    )


_STALE_WRITER_TEMP_AGE_S = 6 * 3600


def sweep_stale_writer_temp(
    cas_dir: Path, *, older_than_s: float = _STALE_WRITER_TEMP_AGE_S,
) -> int:
    """Remove abandoned snapshot-materialization staging directories."""
    removed = 0
    root = Path(cas_dir)
    now = time.time()
    for base_name in ("snapshots",):
        base = root / base_name
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            for name in list(dirnames):
                if not (name.startswith(".") or ".building-" in name):
                    continue
                p = Path(dirpath) / name
                try:
                    if now - p.stat().st_mtime > older_than_s:
                        shutil.rmtree(p, ignore_errors=True)
                        removed += 1
                        dirnames.remove(name)
                except OSError:
                    continue
            for name in filenames:
                if ".part-" not in name:
                    continue
                p = Path(dirpath) / name
                try:
                    if now - p.stat().st_mtime > older_than_s:
                        p.unlink(missing_ok=True)
                        removed += 1
                except OSError:
                    continue
    return removed


__all__ = [
    "RefIndex",
    "tree_bytes",
    "reclaim_file_cache",
    "delete_ref_bytes",
    "sweep_orphan_blobs",
    "sweep_stale_writer_temp",
]
