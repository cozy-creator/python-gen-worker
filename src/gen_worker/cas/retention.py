"""Reachability collection over the local store.

FIRST-PARTY (pgw#1575). Upstream never owned this: `LocalCAS.collect_garbage`'s
own docstring said "TensorFS deliberately owns no retention policy", and the
policy — what counts as reachable, and how long a freshly produced object is
protected — has always been `models/disk_gc.py`'s. The reachability WALK was
the only part that lived upstream, and it walks nothing but the ref record and
manifest formats, both of which are still upstream's and still vendored.

Master's `LocalCAS` has no successor for it: the Rust `ObjectStore` collects
abandoned TEMPORARIES (`collect_abandoned_temps`, which is a lease sweep, not a
reachability pass) and nothing else. So a bumped snapshot deletes this and puts
nothing in its place — which is exactly why it is here now.

`_store_lock` and `_object_lock` are `LocalCAS` privates. Reaching for them is
deliberate and is the whole reason this module sits beside the store rather
than in `models/`: collection has to exclude concurrent admission, and the
exclusive store lock is the only thing that does.
"""

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
    """Every current logical ref's target. Call under the store lock."""

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
    """Delete unreferenced immutable objects older than a caller cutoff.

    Current logical refs are always roots. The caller adds active byte refs and
    repository manifests. The age cutoff is required, and it protects freshly
    produced bytes during the gap between a writer finishing and its consumer
    installing the logical ref that makes them reachable.
    """

    if older_than <= 0:
        raise ValueError("garbage collection requires a positive age grace")
    with cas._store_lock(exclusive=True):
        keep = _logical_roots(cas)
        keep.update(CASRef.parse(ref) for ref in reachable)
        for manifest in manifests:
            for entry in manifest.files:
                keep.update(ref for ref, _size in entry.objects())

        # A logical ref commonly targets a stored repository manifest. Expand
        # valid manifests; arbitrary object bytes simply remain roots.
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
