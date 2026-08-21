"""A real directory of real files, for a loader that will not read our store."""

from __future__ import annotations

import fcntl
import itertools
import logging
import os
import shutil
from pathlib import Path


from gen_worker._vendor.tensorfs import (
    FileEntry,
    LocalCAS,
    RepositoryManifest,
    TensorReader,
)

from . import projection

_log = logging.getLogger("gen_worker.models.materialized_view")
_SCRATCH = itertools.count()

VIEWS_DIR = "materialized"

_no_fill_serving = False

__all__ = [
    "VIEWS_DIR",
    "no_fill_serving",
    "serving_streams_weights",
    "third_party_dir",
    "view_root_for",
]


def no_fill_serving(active: bool = True) -> None:
    """Declare that this process serves with the streaming loader bound."""

    global _no_fill_serving
    _no_fill_serving = bool(active)


def serving_streams_weights() -> bool:

    return _no_fill_serving


def view_root_for(snapshot_root: Path | str) -> Path:
    """Where a snapshot's materialized view lives, whether or not it exists."""

    tree = Path(snapshot_root)
    return tree.parent.parent / VIEWS_DIR / tree.name


def _materialize(cas: LocalCAS, entry: FileEntry, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TensorReader(cas, RepositoryManifest((entry,))) as reader:
        reader.materialize(entry.path, destination)  # mixed-cas-hatch: author-slot-directory


def third_party_dir(path: Path | str, *, why: str) -> Path:
    """``path``, made real, for a consumer that cannot read the CAS."""

    target = Path(path)
    snapshot = projection.resolve_projection(target)
    root = target if snapshot is not None else projection.snapshot_root_of(target)
    if root is None:
        return target
    if snapshot is None:
        snapshot = projection.resolve_projection(root)
    if snapshot is None:
        return target

    try:
        rel = target.resolve().relative_to(root.resolve()).as_posix()
    except (OSError, ValueError):
        return target
    rel = "" if rel == "." else rel

    wanted = [
        entry
        for entry in snapshot.manifest.files
        if not rel or entry.path == rel or entry.path.startswith(rel + "/")
    ]
    if not wanted:
        raise projection.UnresolvedProjection(
            f"{target} is inside the projected tree {root} but its manifest "
            f"covers no file at {rel!r} ({why}). Refusing to hand a third "
            f"party a directory this store cannot fill."
        )

    view = view_root_for(root)
    out = view / rel if rel else view
    if out.exists():
        return out

    view.parent.mkdir(parents=True, exist_ok=True)
    lock_path = view.parent / f".{view.name}.lock"
    written = 0
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if out.exists():
                return out
            if len(wanted) == 1 and wanted[0].path == rel:
                out.parent.mkdir(parents=True, exist_ok=True)
                _materialize(snapshot.cas, wanted[0], out)
                written = wanted[0].size_bytes
            else:
                scratch = view.parent / f".building-{view.name}-{os.getpid()}-{next(_SCRATCH)}"
                shutil.rmtree(scratch, ignore_errors=True)
                try:
                    for entry in wanted:
                        suffix = entry.path[len(rel) + 1 :] if rel else entry.path
                        _materialize(snapshot.cas, entry, scratch / suffix)
                        written += entry.size_bytes
                    out.parent.mkdir(parents=True, exist_ok=True)
                    scratch.rename(out)
                except BaseException:
                    shutil.rmtree(scratch, ignore_errors=True)
                    raise
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    if _no_fill_serving:
        _log.error(
            "DEFECT materialized_view snapshot=%s rel=%s bytes=%d files=%d "
            "why=%s — this process serves with the pgw#1380 streaming loader "
            "bound, so a serving pytorch endpoint reached tier 3 for weights "
            "it already has in the chunk store. The 2026-08-19 no-fill ruling "
            "leaves tier 3 to external binaries and AOT .so delivery only; "
            "this copy is a bug in the caller, not a burn-down row.",
            root.name, rel or "(whole tree)", written, len(wanted), why,
        )
    else:
        _log.info(
            "materialized_view snapshot=%s rel=%s bytes=%d files=%d why=%s "
            "(pgw#1303 tier 3: the last resort of the access ladder)",
            root.name, rel or "(whole tree)", written, len(wanted), why,
        )
    return out
