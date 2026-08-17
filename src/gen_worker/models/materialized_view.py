"""A real directory of real files, for a loader that will not read our store.

**This is the pgw#1303 gate's mechanism, and it is the piece pgw#1308 step ⑥
could not ship without.** The chokepoint now publishes a PROJECTED tree, so a
consumer that hands a DIRECTORY to a third-party loader — `diffusers`
`from_pretrained`, `ComponentSpec.load`, `from_single_file`, `gguf.GGUFReader`,
`llama-server -m` — is handing it pointer stubs. Those loaders fail at their
own parse site, correctly and loudly, and there is nothing this repo can change
in them.

pgw#1330 cut every site pgw controls to native reads. What is left is the 23
sites that hand a path OUT, and pgw#1303 is Paul's ruling on whether they are
priced or deprecated. **Until that ruling they stay materialized** — and this
is where the price is paid, named, and made visible.

## What changes even before the ruling

Materialization used to be UNCONDITIONAL and WHOLE-TREE: every snapshot on
every pod carried a complete second copy, whether or not anything ever handed
a directory to a third party (pgw#1296(a) measured the 2.000x). Now:

*   nothing is copied until a gated site actually asks;
*   only the SUBTREE it asks for is copied — a `from_pretrained` on one
    component costs that component, not the model;
*   every copy goes through the single-file hatch with its §9 row on the line,
    so `scripts/lint_materialization_hatch.py` counts it;
*   the bytes are logged at INFO with the caller's own `why`, so a pod's 2x
    residency has a name and a call site instead of being the default.

That turns pgw#1303 from "should we allow this?" into a decision with a
measured price per site, which is what a ruling needs.

## Where the copy lives

``<base>/materialized/<snapshot-key>/<rel>`` — a sibling of ``snapshots/``,
keyed identically, so :func:`gen_worker.models.disk_gc.delete_ref_bytes` drops
it with the tree it belongs to and :func:`disk_gc.tree_bytes` counts it. A view
that outlived its snapshot would be disk nothing could name.
"""

from __future__ import annotations

import fcntl
import itertools
import logging
import os
import shutil
from pathlib import Path


from gen_worker._vendor.tensorfs import FileEntry, LocalCAS

from . import projection

_log = logging.getLogger("gen_worker.models.materialized_view")
_SCRATCH = itertools.count()

VIEWS_DIR = "materialized"

__all__ = ["VIEWS_DIR", "third_party_dir", "view_root_for"]


def view_root_for(snapshot_root: Path | str) -> Path:
    """Where a snapshot's materialized view lives, whether or not it exists."""

    tree = Path(snapshot_root)
    return tree.parent.parent / VIEWS_DIR / tree.name


def _materialize(cas: LocalCAS, entry: FileEntry, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # The single-file hatch. It streams, it is atomic, and it verifies the
    # reconstruction against the manifest's whole-file digest before the
    # rename — so a view is known-good bytes, not merely copied ones. The
    # pinned rev spells `extract()` as `materialize`; see VENDORED.toml.
    cas.materialize(entry, destination)  # mixed-cas-hatch: author-slot-directory


def third_party_dir(path: Path | str, *, why: str) -> Path:
    """``path``, made real, for a consumer that cannot read the CAS.

    Returns ``path`` UNCHANGED when it is not inside a projected snapshot —
    an ordinary materialized tree, a bare HF download, a test fixture — so a
    call site can be unconditional and stays correct on every tree shape. That
    is deliberate: a caller forced to branch on "is this projected?" is a
    caller that will get the branch wrong somewhere.

    ``why`` names the third party that needs real files. It is required and it
    is logged with the byte count, because the whole point of routing these
    through one seam is that the pgw#1303 ruling can read what it costs and
    where.
    """

    target = Path(path)
    # The tree ROOT is the common case and `snapshot_root_of` cannot see it:
    # it walks PARENTS looking for `snapshots/`, so it answers for a file
    # inside a tree and answers `None` for the tree itself. Ask directly
    # first, then walk. (Getting this backwards is silent — the seam returns
    # the projected path unchanged and the third party meets a stub.)
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
                # ONE file. The hatch is already atomic — it writes a temp
                # beside the destination, verifies the whole-file digest, and
                # renames — so there is no second staging step to add.
                out.parent.mkdir(parents=True, exist_ok=True)
                _materialize(snapshot.cas, wanted[0], out)
                written = wanted[0].size_bytes
            else:
                # A DIRECTORY. Built under a scratch name and renamed, so no
                # reader ever sees a half-filled view — the projector's rule.
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

    _log.info(
        "materialized_view snapshot=%s rel=%s bytes=%d files=%d why=%s "
        "(pgw#1303: this is the priced copy, not the default)",
        root.name, rel or "(whole tree)", written, len(wanted), why,
    )
    return out
