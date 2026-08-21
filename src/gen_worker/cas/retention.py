"""Reachability collection over the local store."""

from __future__ import annotations

import json
import time
from collections.abc import Iterable
from dataclasses import dataclass

from .._vendor.tensorfs.local import LocalCAS, _fsync_dir
from .._vendor.tensorfs.manifest import RepositoryManifest
from .._vendor.tensorfs.refs import CASRef


@dataclass(frozen=True, slots=True)
class GCReport:
    """One reachability collection pass."""

    examined: int
    reachable: int
    deleted: int
    bytes_deleted: int


def _logical_roots(cas: LocalCAS) -> set[CASRef]:

    roots: set[CASRef] = set()
    for path in cas.refs.iterdir():
        if not path.is_file():
            continue
        if len(path.name) != 64 or any(char not in "0123456789abcdef" for char in path.name):
            continue
        raw = json.loads(path.read_bytes())
        if (
            not isinstance(raw, dict)
            or set(raw) != {"format", "name", "target"}
            or raw.get("format") != 1
            or not isinstance(raw.get("name"), str)
            or cas._ref_id(raw["name"]) != path.name
        ):
            raise ValueError(f"logical ref record {path.name!r} is malformed")
        roots.add(CASRef.parse(str(raw.get("target", ""))))
    return roots


def collect_garbage(
    cas: LocalCAS,
    reachable: Iterable[str | CASRef] = (),
    *,
    manifests: Iterable[RepositoryManifest] = (),
    older_than: float,
) -> GCReport:
    """Delete unreferenced immutable objects older than a caller cutoff."""

    if older_than <= 0:
        raise ValueError("garbage collection requires a positive age grace")
    with cas._store_lock(exclusive=True):
        keep = _logical_roots(cas)
        keep.update(CASRef.parse(ref) for ref in reachable)
        for manifest in manifests:
            for entry in manifest.files:
                keep.update(ref for ref, _size in entry.objects())

        for root in tuple(keep):
            path = cas.object_path(root)
            try:
                manifest = RepositoryManifest.from_bytes(path.read_bytes())
            except (OSError, UnicodeDecodeError, ValueError):
                continue
            for entry in manifest.files:
                keep.update(ref for ref, _size in entry.objects())

        cutoff = time.time() - older_than
        examined = deleted = bytes_deleted = 0
        namespace = cas.objects / "sha256"
        if namespace.exists():
            for path in namespace.glob("*/*/*"):
                if not path.is_file():
                    continue
                try:
                    ref = CASRef(path.name)
                except ValueError:
                    continue
                examined += 1
                stat = path.stat()
                if ref in keep or stat.st_mtime > cutoff:
                    continue
                with cas._object_lock(ref):
                    try:
                        current = path.stat()
                    except FileNotFoundError:
                        continue
                    if current.st_mtime > cutoff:
                        continue
                    path.unlink()
                    _fsync_dir(path.parent)
                    deleted += 1
                    bytes_deleted += current.st_size
        return GCReport(examined, len(keep), deleted, bytes_deleted)
