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


# pgw#1542: WHERE THE REPAIR DIED, carried to the one channel that survives.
#
# `ModelStore.ensure_pinned` has six named exits and every one of them is a LOG
# LINE. On a rented pod that is unreadable: five separate paths were checked and
# none carries worker stdout off the box (no worker-log table, `cozy logs` is
# local-only, privatedeploy has no logs verb, `bootstages` reads pgw#1355
# telemetry, and RunPod's REST API exposes no logs route). So "the repair ran
# and lost" and "the repair never ran" are the same observation to anyone
# reading after the pod dies — which is the exact ambiguity pgw#1536 set out to
# kill at boot, still live one layer down.
#
# The refusal message is the ONE channel proven to survive a pod, because it
# rides `error_message_safe`. So the outcome is recorded here, keyed by tree
# name, and read back at the raise site. A dict rather than a single slot: a
# pod serves many trees, and a global last-writer would attribute one tree's
# repair to another tree's refusal, which is worse than saying nothing.
_PIN_OUTCOMES: "dict[str, str]" = {}
_PIN_OUTCOMES_CAP = 64


def record_pin_outcome(tree_name: str, outcome: str) -> None:
    """Record which named exit `ensure_pinned` took for this tree."""
    if not tree_name:
        return
    if len(_PIN_OUTCOMES) >= _PIN_OUTCOMES_CAP and tree_name not in _PIN_OUTCOMES:
        # Bounded: a long-lived pod must not accumulate one entry per tree it
        # ever saw. Dropping the OLDEST keeps the most recent repairs, which
        # are the ones a live refusal is about.
        _PIN_OUTCOMES.pop(next(iter(_PIN_OUTCOMES)), None)
    _PIN_OUTCOMES[tree_name] = outcome


def pin_outcome(tree_name: str) -> str:
    """The recorded exit, or `not attempted` — NEVER an empty string.

    The empty string would render as "no repair", which is precisely the false
    reading this exists to prevent. `not attempted` is a claim about this
    process and is true: nothing called `ensure_pinned` for this tree here.
    """
    return _PIN_OUTCOMES.get(tree_name, "not attempted")
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


def stub_at_any(root: Path | str) -> bool:
    """Whether ANY file under ``root`` is a pointer stub.

    pgw#1513: the cheap "is this tree projected" question, asked by callers
    that must not hand a stubbed tree to a stock reader. It branches on
    :func:`read_stub` — the stub's own self-identifying header — and NEVER on
    a parse failure from something that tried to read one as weights, which is
    pgw#1308's mistake: two callers inferred a fact about the WORLD from a
    fact about a READER, and reached opposite wrong conclusions.

    Stops at the first stub: this is a predicate, not a census.
    """
    tree = Path(root)
    if not tree.is_dir():
        return False
    for path in tree.rglob("*"):
        if (path.is_file() or path.is_symlink()) and read_stub(path) is not None:
            return True
    return False


def collected_entries(root: Path | str) -> list[str]:
    """Projected entries under ``root`` whose CAS object is GONE.

    pgw#1513/pgw#1514, measured by the se#790 lane on a real 5.6 GB tree: a
    tree's manifest pin is the ONLY GC root its objects have, so a tree that
    outlives its pin and then meets a GC pass has every object deleted while
    the tree stands. Tensor containers remain stubs; everything else becomes a
    DANGLING SYMLINK — ``model_index.json`` among them.

    `Path.is_file()` FOLLOWS the link, which is what collapses "collected" and
    "absent" into one branch and makes a caller say "carries no
    model_index.json" about a tree that has one. Ask this instead, and ask it
    BEFORE any parse.
    """
    tree = Path(root)
    gone: list[str] = []
    if not tree.is_dir():
        return gone
    for path in sorted(tree.rglob("*")):
        if path.is_symlink() and is_projection_artifact(path) and not path.exists():
            gone.append(path.relative_to(tree).as_posix())
    # `model_index.json` FIRST, because it is the entry whose absence produces
    # the false "carries no model_index.json" — so it is the one a reader most
    # needs to see in a truncated list, and plain alphabetical order drops it
    # out of the window on any tree with early-alphabet components
    # (se#790 measured it landing 2nd of 3 by luck). Everything else keeps
    # sorted order, so the list stays stable and diffable.
    gone.sort(key=lambda rel: (rel != "model_index.json", rel))
    return gone


def collected_refusal(root: Path | str, entries: Sequence[str]) -> str:
    """THE wording for "this tree's bytes were collected", in one place.

    Both refusal sites call this — the eager bridge (pgw#1513) and
    ``skeleton.build`` (pgw#1514) — so a reader who greps either path lands in
    the same mental model. Two hand-written strings is how this shape reached
    four callers; the se#790 lane found a fifth instance in their OWN code
    while documenting the lesson in the same file's docstring.
    """
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


def is_projection_artifact(path: Path | str) -> bool:
    """Whether this path carries a POINTER instead of the bytes it names.

    A pointer stub or a symlink into the CAS. Not a judgement about whether
    the pointer is *correct* — that is :func:`projection_fault`.
    """

    file = Path(path)
    return file.is_symlink() or read_stub(file) is not None


def projection_fault(path: Path | str, *, digest: str, size: int) -> Optional[str]:
    """Why the projection artifact at ``path`` is not the one the manifest
    names, or ``None`` when it is exactly right.

    THE SINGLE STATEMENT of what a correct projected artifact is. The boot
    verifier (:func:`gen_worker.models.volume_verify.verify_projection`) and
    the snapshot chokepoint's convergence check both read it from here: a
    second copy of this rule is a second thing to get wrong, which is what the
    five duplicated fail-open header readers pgw#1330 collapsed already
    demonstrated in this repo.

    Complete rather than partial. A stub's whole content is a body identity
    and a logical size, and both are checked; a symlink's whole content is the
    object it names, and that object's presence and identity are checked. The
    bytes themselves were hashed at ADMISSION into a pod-local store, so there
    is nothing further at this path to check.

    Precondition: :func:`is_projection_artifact`. A path holding real bytes
    has no projection fault — it has a hash, and the caller must take it.
    """

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
    """Whether ``directory``'s filesystem can hold the tree's symlinks.

    Probed rather than assumed, and probed ONCE by the caller so the same
    answer sizes the disk check and drives ``project_snapshot(symlinks=…)``.
    Letting each side probe independently is how a headroom check comes to
    size a tree of symlinks while the projector writes a tree of copies.
    """

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
    """Bytes projecting ``manifest`` will actually WRITE at the target path.

    The dispatch mirrors ``tensorfs.project._project_entry`` arm for arm,
    because a disk check that sizes something other than what the writer
    writes is worse than no check: on master this sized a whole second copy
    of the model that a projection never makes, and before pgw#1263 it sized
    only the MISSING objects while the publish wrote a full copy with no
    check at all. ``tests/test_projection_headroom_pgw1308.py`` asserts this
    equals the real ``du`` of a real projection rather than trusting the
    mirror to stay in step.

    Three arms cost nothing: a tensor container is a ~128 B stub, an empty
    file is empty, and a single-object non-tensor file is a symlink. The one
    arm that costs its whole size is a CHUNKED non-tensor file, which has no
    single object to point at and no tensor reader to serve it — and, when
    the filesystem cannot hold symlinks, every non-tensor file.
    """

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
