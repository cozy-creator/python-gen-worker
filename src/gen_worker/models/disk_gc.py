"""Disk retention (#370): a persistent ref->bytes index + deletion helpers.

The CAS stores tensorhub models by snapshot digest, HF models under the HF
cache, civitai under version dirs — none of which map back to wire refs on
their own. ``RefIndex`` persists {ref: path, bytes, last_used} at
``<cache_dir>/ref-index.json`` so disk GC and the boot-time rescan can reason
in refs (the vocabulary of `keep`, Residency, and ModelEvents).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_INDEX_NAME = "ref-index.json"


class RefIndex:
    """Persistent {ref: {path, bytes, last_used}}. Thread-safe; every mutation
    is written through (small file, rare writes)."""

    def __init__(self, cache_dir: Path) -> None:
        self._path = Path(cache_dir) / _INDEX_NAME
        self._lock = threading.Lock()
        self._data: Dict[str, Dict[str, Any]] = {}
        try:
            raw = json.loads(self._path.read_text("utf-8"))
            if isinstance(raw, dict):
                self._data = {
                    str(k): v for k, v in raw.items()
                    if isinstance(v, dict) and v.get("path")
                }
        except FileNotFoundError:
            pass
        except Exception as exc:
            logger.warning("ref-index unreadable (%s); starting empty", exc)

    def _save_locked(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self._data), encoding="utf-8")
            tmp.replace(self._path)
        except Exception as exc:
            logger.warning("ref-index write failed: %s", exc)

    def record(self, ref: str, path: Path, size_bytes: int) -> None:
        with self._lock:
            self._data[ref] = {
                "path": str(path), "bytes": int(size_bytes), "last_used": time.time(),
            }
            self._save_locked()

    def touch(self, ref: str) -> None:
        with self._lock:
            e = self._data.get(ref)
            if e is not None:
                e["last_used"] = time.time()
                self._save_locked()

    def remove(self, ref: str) -> None:
        with self._lock:
            if self._data.pop(ref, None) is not None:
                self._save_locked()

    def path(self, ref: str) -> Optional[Path]:
        with self._lock:
            e = self._data.get(ref)
            return Path(e["path"]) if e else None

    def last_used(self, ref: str) -> float:
        with self._lock:
            e = self._data.get(ref)
            return float(e.get("last_used") or 0.0) if e else 0.0

    def entries(self) -> Dict[str, Dict[str, Any]]:
        with self._lock:
            return {k: dict(v) for k, v in self._data.items()}


def tree_bytes(path: Path) -> int:
    """Bytes under ``path`` (file or tree), hardlinked inodes counted once."""
    p = Path(path)
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


def _retention_unit(path: Path, cas_dir: Path) -> Path:
    """The directory/file that must be deleted to reclaim a ref's bytes:
    the snapshot dir for CAS refs, the ``models--*`` repo dir for HF refs,
    the tracked path otherwise."""
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
    if unit.is_dir():
        shutil.rmtree(unit, ignore_errors=True)
    else:
        unit.unlink(missing_ok=True)


def sweep_orphan_blobs(cas_dir: Path) -> int:
    """Delete CAS blobs no snapshot links anymore (st_nlink == 1). Snapshot
    trees hardlink into blobs/, so link count is the reference count."""
    blobs = Path(cas_dir) / "blobs"
    freed = 0
    if not blobs.is_dir():
        return 0
    for dirpath, _dirs, names in os.walk(blobs):
        for name in names:
            fp = os.path.join(dirpath, name)
            try:
                st = os.stat(fp)
                if st.st_nlink <= 1:
                    os.unlink(fp)
                    freed += int(st.st_size)
            except OSError:
                continue
    return freed


__all__ = ["RefIndex", "tree_bytes", "delete_ref_bytes", "sweep_orphan_blobs"]
