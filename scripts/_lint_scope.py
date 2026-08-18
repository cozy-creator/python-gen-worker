"""Which subtrees of `src/gen_worker` a repo guard may not judge, and why.

ONE HOME for that fact: a guard that spells its own exclusion (or forgets to)
makes a new such subtree a hunt through every scanner.

A directory belongs here only when the code in it is NOT OURS TO CHANGE:

* ``pb`` — generated from the .proto; the fix is in the .proto.
* ``_vendor`` — byte-identical snapshots of upstream repos, pinned by rev and
  held to a digest fence (pgw#1310). A guard finding here is fixed upstream and
  re-vendored; edited in place it would break the fence and diverge the
  snapshot. It is also not evidence about OUR architecture, which is what these
  guards measure.

This is not an allowlist for our own debt. Nothing else goes in it.
"""

from __future__ import annotations

from pathlib import Path

#: Directory names, matched anywhere in a path under `src/gen_worker`.
UNOWNED_DIRS = ("pb", "_vendor")


def is_unowned(path: Path, root: Path | None = None) -> bool:
    """True when `path` lies inside a subtree no guard here may judge."""
    parts = (path.relative_to(root) if root is not None else path).parts
    return any(name in parts for name in UNOWNED_DIRS)
