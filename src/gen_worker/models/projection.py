"""What a projected snapshot tree is, and how this worker reads one.

A projected tree carries no tensor bytes. A non-tensor file (config, tokenizer,
dataset media, ``.so``) is a relative symlink into the CAS ``objects/``; a
tensor container is a ~128 B TFSSTUB1 pointer stub; the tensors themselves are
read straight out of the CAS objects through :func:`open_tensors`.

THE LESSON THIS MODULE EXISTS TO ENCODE (pgw#1308 finding 3, and it
generalises): the stub format's safety property is that a naive ``open()``
fails LOUDLY at the parse site. **That guarantee constrains nothing about what
the caller concludes from the failure.** Two callers on master read the exact
same correct loud failure and reached opposite wrong conclusions —
``store.py`` read it as CORRUPTION and deleted the model on every boot;
``detect_on_disk_dtype`` read it as ABSENCE and silently doubled VRAM. Neither
is a format bug and no change to the format can fix either. So every consumer
that can meet a projected tree needs its own stub-aware branch built on this
module, or an explicit :class:`UnresolvedProjection` refusal. Falling back to
a default is the defect.

Recovering ``(cas, manifest)`` from a bare directory path needs no sidecar:
:func:`gen_worker.models.cozy_snapshot.ensure_snapshot` publishes a tree at
``<base>/snapshots/<key>`` and pins that exact manifest in the same store
under the ref ``snapshot:<key>``, so the two are already addressable from the
tree's own location.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from gen_worker._vendor.tensorfs import (
    CASRef,
    FileEntry,
    LocalCAS,
    RepositoryManifest,
    Stub,
    TensorReader,
    open_tensors,
    read_stub,
)

_log = logging.getLogger("gen_worker.models.projection")

SNAPSHOTS_DIR = "snapshots"
REF_PREFIX = "snapshot:"


class UnresolvedProjection(RuntimeError):
    """A projected tree was met where its manifest could not be recovered.

    Raised INSTEAD of degrading to a default. Every site that raises this had
    a fall-through on master that silently produced a wrong answer.
    """


def stub_at(path: Path | str) -> Optional[Stub]:
    """The pointer stub at ``path``, or ``None`` when that file is not one.

    Bounded read: pointing this at a real 20 GiB shard costs one block.
    """

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
    """The ``(cas, manifest)`` backing a snapshot tree, or ``None``.

    ``None`` means "this path is not a snapshot tree this worker published" —
    an ordinary materialized directory, a bare HF download, a test fixture.
    It never means "the manifest is missing but the tree is projected"; that
    case is an :class:`UnresolvedProjection` at :func:`require_projection`.
    """

    tree = Path(root)
    if tree.parent.name != SNAPSHOTS_DIR:
        return None
    base = tree.parent.parent
    # Constructing a LocalCAS CREATES its directories. Only do that where one
    # already lives, so probing an ordinary path never leaves a store behind.
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


def snapshot_root_of(path: Path | str) -> Optional[Path]:
    """The snapshot tree a file lives in, or ``None``.

    A tree is the directory directly under ``snapshots/``, at any depth of
    component nesting below it.
    """

    try:
        resolved = Path(path).resolve()
    except OSError:
        return None
    for parent in resolved.parents:
        if parent.parent.name == SNAPSHOTS_DIR:
            return parent
    return None


def require_projection(root: Path | str, *, why: str) -> ProjectedSnapshot:
    """:func:`resolve_projection`, or a loud refusal naming what needed it.

    ``why`` is the consumer's own sentence about what it was about to get
    wrong. It is required because a bare "cannot resolve" traceback is what
    sends the investigation to the store instead of to the projection.
    """

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
    """The snapshot and manifest entry backing ONE stubbed file, or a refusal.

    The single seam every consumer holding a file path goes through. It
    refuses in three distinguishable ways — the file is not in a snapshot
    tree, the tree's manifest is unrecoverable, the manifest does not cover
    this file — because each sends the investigation somewhere different, and
    the defect this replaces sent it nowhere at all.
    """

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
    """How many bytes this file HOLDS, whether or not they are at this path.

    A stub's own ``st_size`` is ~128 B regardless of the model behind it, so
    any size comparison over a projected tree collapses to a tie and the
    choice becomes arbitrary sort order (pgw#1308's `utils/lora.py:309`
    finding: "discovery-only is not the same as unaffected"). The stub
    DECLARES the real size; read it.
    """

    stub = read_stub(path)
    if stub is not None:
        return stub.size
    try:
        return Path(path).stat().st_size
    except OSError:
        return 0


def object_of_symlink(path: Path | str) -> Optional[CASRef]:
    """The CAS object a projected symlink points at, or ``None``.

    Judged from the object path's own shape —
    ``…/objects/sha256/<aa>/<bb>/<digest>`` — so it holds for any store root
    and needs none passed in. ``None`` for a link that is not an object path;
    a projection never emits one, so a caller may treat that as a finding
    rather than as absence.
    """

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


__all__ = [
    "ProjectedSnapshot",
    "REF_PREFIX",
    "SNAPSHOTS_DIR",
    "UnresolvedProjection",
    "logical_size",
    "object_of_symlink",
    "require_projection",
    "require_projection_for",
    "resolve_projection",
    "snapshot_root_of",
    "stub_at",
]
