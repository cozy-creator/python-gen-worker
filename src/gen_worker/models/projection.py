"""What a projected snapshot tree is, and how this worker reads one."""

from __future__ import annotations

import itertools
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from gen_worker._vendor.tensorfs import (
    CASRef,
    FileEntry,
    LocalCAS,
    RepositoryManifest,
    Stub,
    TensorReader,
    is_tensor_container,
    open_tensors,
    read_stub,
    stub_bytes,
)

_log = logging.getLogger("gen_worker.models.projection")

SNAPSHOTS_DIR = "snapshots"


_PIN_OUTCOMES: "dict[str, str]" = {}
_PIN_OUTCOMES_CAP = 64


def record_pin_outcome(tree_name: str, outcome: str) -> None:
    """Record which named exit `ensure_pinned` took for this tree."""
    if not tree_name:
        return
    if len(_PIN_OUTCOMES) >= _PIN_OUTCOMES_CAP and tree_name not in _PIN_OUTCOMES:
        _PIN_OUTCOMES.pop(next(iter(_PIN_OUTCOMES)), None)
    _PIN_OUTCOMES[tree_name] = outcome


def pin_outcome(tree_name: str) -> str:
    """The recorded exit, or `not attempted` — NEVER an empty string."""
    return _PIN_OUTCOMES.get(tree_name, "not attempted")
REF_PREFIX = "snapshot:"


class UnresolvedProjection(RuntimeError):
    """A projected tree was met where its manifest could not be recovered."""


def stub_at(path: Path | str) -> Optional[Stub]:
    """The pointer stub at ``path``, or ``None`` when that file is not one."""

    return read_stub(path)


@dataclass(frozen=True)
class ProjectedSnapshot:
    """A projected tree together with the store and manifest that back it."""

    root: Path
    cas: LocalCAS
    manifest: RepositoryManifest

    def entry(self, rel: str) -> FileEntry:
        for entry in self.manifest.files:
            if entry.path == rel:
                return entry
        raise UnresolvedProjection(
            f"{self.root}: {rel!r} is not in the manifest that backs this tree"
        )

    def entry_for(self, path: Path | str) -> FileEntry:
        """The manifest entry for an absolute path inside this tree."""

        rel = Path(path).resolve().relative_to(self.root.resolve()).as_posix()
        return self.entry(rel)

    def open_tensors(self, *, verify: bool = True) -> TensorReader:
        """Every tensor in this snapshot, read from CAS objects, no file."""

        return open_tensors(self.cas, self.manifest, verify=verify)

def resolve_projection(root: Path | str) -> Optional[ProjectedSnapshot]:
    """The ``(cas, manifest)`` backing a snapshot tree, or ``None``."""

    tree = Path(root)
    if tree.parent.name != SNAPSHOTS_DIR:
        return None
    base = tree.parent.parent
    if not (base / "refs").is_dir() or not (base / "objects").is_dir():
        return None
    try:
        cas = LocalCAS(base)
        ref = cas.read_ref(REF_PREFIX + tree.name)
    except (OSError, ValueError):
        return None
    if ref is None:
        return None
    try:
        manifest = cas.load_manifest(ref)
    except (OSError, ValueError) as exc:
        raise UnresolvedProjection(
            f"{tree}: manifest {ref} is pinned but unreadable: {exc}"
        ) from exc
    return ProjectedSnapshot(tree, cas, manifest)


def stub_at_any(root: Path | str) -> bool:
    """Whether ANY file under ``root`` is a pointer stub."""
    tree = Path(root)
    if not tree.is_dir():
        return False
    for path in tree.rglob("*"):
        if (path.is_file() or path.is_symlink()) and read_stub(path) is not None:
            return True
    return False


def collected_entries(root: Path | str) -> list[str]:
    """Projected entries under ``root`` whose CAS object is GONE."""
    tree = Path(root)
    gone: list[str] = []
    if not tree.is_dir():
        return gone
    for path in sorted(tree.rglob("*")):
        if path.is_symlink() and is_projection_artifact(path) and not path.exists():
            gone.append(path.relative_to(tree).as_posix())
    gone.sort(key=lambda rel: (rel != "model_index.json", rel))
    return gone


def collected_refusal(root: Path | str, entries: Sequence[str]) -> str:
    """THE wording for "this tree's bytes were collected", in one place."""
    shown = ", ".join(list(entries)[:3])
    more = "" if len(entries) <= 3 else f" (+{len(entries) - 3} more)"
    return (
        f"{Path(root)}: {len(entries)} of this tree's entries are projected "
        f"links whose CAS OBJECTS HAVE BEEN COLLECTED ({shown}{more}). The "
        f"tree's manifest pin is the only root those objects had, so losing it "
        f"makes a GC pass delete the bytes while leaving the tree standing. "
        f"These weights must be RE-FETCHED; this is not a re-pin. Do not read "
        f"this as a malformed checkpoint — a dangling `model_index.json` is "
        f"why a tree that HAS one gets reported as 'carries no "
        f"model_index.json'."
    )


def snapshot_root_of(path: Path | str) -> Optional[Path]:
    """The snapshot tree a file lives in, or ``None``."""

    try:
        resolved = Path(path).resolve()
    except OSError:
        return None
    for parent in resolved.parents:
        if parent.parent.name == SNAPSHOTS_DIR:
            return parent
    return None


def require_projection(root: Path | str, *, why: str) -> ProjectedSnapshot:
    """:func:`resolve_projection`, or a loud refusal naming what needed it."""

    resolved = resolve_projection(root)
    if resolved is None:
        raise UnresolvedProjection(
            f"{root} holds tensorfs pointer stubs, so its tensor bytes are in "
            f"the CAS and not at any file path, but the manifest backing it "
            f"cannot be recovered ({why}). Refusing to guess: on master this "
            f"path fell through to a default answer."
        )
    return resolved


def require_projection_for(
    path: Path | str, *, why: str
) -> tuple[ProjectedSnapshot, FileEntry]:
    """The snapshot and manifest entry backing ONE stubbed file, or a refusal."""

    file = Path(path)
    root = snapshot_root_of(file)
    if root is None:
        raise UnresolvedProjection(
            f"{file} is a tensorfs pointer stub, so its bytes are in the CAS "
            f"and not at this path, but it is not inside a snapshot tree this "
            f"worker published, so its manifest cannot be found ({why}). "
            f"Refusing to guess."
        )
    snapshot = require_projection(root, why=why)
    return snapshot, snapshot.entry_for(file)


def logical_size(path: Path | str) -> int:
    """How many bytes this file HOLDS, whether or not they are at this path."""

    stub = read_stub(path)
    if stub is not None:
        return stub.size
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


def object_of_symlink(path: Path | str) -> Optional[CASRef]:
    """The CAS object a projected symlink points at, or ``None``."""

    link = Path(path)
    if not link.is_symlink():
        return None
    try:
        parts = Path(os.path.realpath(link)).parts
    except OSError:
        return None
    if len(parts) < 5 or parts[-5] != "objects" or parts[-4] != "sha256":
        return None
    digest = parts[-1]
    if (parts[-3], parts[-2]) != (digest[:2], digest[2:4]):
        return None
    try:
        return CASRef(digest)
    except ValueError:
        return None


def is_projection_artifact(path: Path | str) -> bool:
    """Whether this path carries a POINTER instead of the bytes it names."""

    file = Path(path)
    return file.is_symlink() or read_stub(file) is not None


def projection_fault(path: Path | str, *, digest: str, size: int) -> Optional[str]:
    """Why the projection artifact at ``path`` is not the one the manifest names, or ``None`` when it is exactly right."""

    file = Path(path)
    stub = read_stub(file)
    if stub is not None:
        if stub.body_sha256 != digest:
            return (
                f"stub names body {stub.body_sha256[:16]}…, "
                f"manifest says {digest[:16]}…"
            )
        if size and stub.size != size:
            return f"stub declares {stub.size} bytes, manifest declares {size}"
        return None
    got = object_of_symlink(file)
    if got is None:
        return "symlink does not resolve into the CAS objects tree"
    if got.digest != digest:
        return f"links to object {got.digest[:16]}…, manifest says {digest[:16]}…"
    if not file.exists():
        return f"linked object {got.digest[:16]}… is absent"
    return None


_SYMLINK_PROBE = itertools.count()


def symlinks_supported(directory: Path | str) -> bool:
    """Whether ``directory``'s filesystem can hold the tree's symlinks."""

    parent = Path(directory)
    probe = parent / f".symlink-probe-{os.getpid()}-{next(_SYMLINK_PROBE)}"
    try:
        os.symlink("probe-target", probe)
    except (OSError, NotImplementedError):
        return False
    finally:
        try:
            os.unlink(probe)
        except OSError:
            pass
    return True


def projection_write_bytes(
    manifest: RepositoryManifest, *, symlinks: bool
) -> int:
    """Bytes projecting ``manifest`` will actually WRITE at the target path."""

    total = 0
    for entry in manifest.files:
        if is_tensor_container(entry.path):
            total += len(stub_bytes(entry.digest, entry.size_bytes))
        elif entry.size_bytes == 0:
            continue
        elif len(entry.chunks) > 1 or not symlinks:
            total += entry.size_bytes
    return total


__all__ = [
    "ProjectedSnapshot",
    "REF_PREFIX",
    "SNAPSHOTS_DIR",
    "UnresolvedProjection",
    "is_projection_artifact",
    "logical_size",
    "object_of_symlink",
    "projection_fault",
    "projection_write_bytes",
    "require_projection",
    "require_projection_for",
    "resolve_projection",
    "snapshot_root_of",
    "stub_at",
    "stub_at_any",
    "collected_entries",
    "collected_refusal",
    "symlinks_supported",
]
