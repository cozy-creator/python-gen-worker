"""Where a ``cg-keyset-v1`` document lives, and who may write one.

Two kinds of location, and the difference is who derived the hashes:

* **Shipped roots** — read-only, produced by the MINT LANE. The baked image's
  ``/app/.tensorhub/`` (beside ``endpoint.lock``, the same place the decode-set
  and execution-lane blocks are stamped) and an explicit
  ``GEN_WORKER_CG_KEYSET`` path. This is what makes a FRESH pod — empty cache,
  no volume, first boot of the release — skip its traces, which the pod-local
  memo never could.
* **The pod's own cache** — written by a trace THIS machine ran (§4.28's
  compile-once-run-forever promise for cozy-local reads the same document
  through the same closure digest). Read after the shipped roots and the only
  root anything writes.

Both are the same document type. A key set is trusted because its closure digest
matches what this pod's code would trace, never because of where it was found —
so the two locations need no separate grammar, no separate parser and no separate
version ladder.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .document import (
    KEYSET_FILENAME, ClosureRow, GraphClassRow, KeySetDocument, decode, empty,
    encode, parse_closure)
from .document import ShippedClosure
from .fold import KeySource
from .identifiers import ClassHash, ClosureDigest, GraphClassName, KeySetError

logger = logging.getLogger(__name__)

__all__ = [
    "ENV_KEYSET_PATH",
    "IMAGE_KEYSET_DIR",
    "KeySetHit",
    "assert_honest",
    "class_hashes",
    "invalidate",
    "lookup",
    "read_document",
    "shipped_roots",
    "write_closure",
]

#: An explicit key set: a ``cg-keyset-v1.json`` file, or a directory holding one.
#: Carries a VALUE (where the document is), never a decision — a key set found
#: here is admitted by exactly the same closure match as any other.
ENV_KEYSET_PATH = "GEN_WORKER_CG_KEYSET"

#: The baked-image location, beside ``endpoint.lock``
#: (``gen_worker.entrypoint.MANIFEST_PATH``'s directory — restated rather than
#: imported so reading a key set never drags the entrypoint's import graph onto
#: a serve pod).
IMAGE_KEYSET_DIR = Path("/app/.tensorhub")


@dataclass(frozen=True)
class KeySetHit:
    """One closure's class set, and the honest statement of where it came from."""

    closure: ShippedClosure
    source: KeySource
    path: Path


def _candidate_files(root: Path) -> Tuple[Path, ...]:
    if root.is_dir():
        path = root / KEYSET_FILENAME
        return (path,) if path.is_file() else ()
    return (root,) if root.is_file() else ()


def shipped_roots(
    *, extra: Sequence[Path] = (),
) -> Tuple[Path, ...]:
    """Read-only roots, in precedence order. Mint-lane output only."""
    roots: List[Path] = []
    env = os.environ.get(ENV_KEYSET_PATH, "").strip()
    if env:
        roots.append(Path(env))
    roots.extend(Path(item) for item in extra)
    roots.append(IMAGE_KEYSET_DIR)
    return tuple(roots)


def read_document(path: Path) -> KeySetDocument:
    """Parse one document. Raises :class:`KeySetError` on anything unreadable."""
    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        raise KeySetError(
            "keyset_absent", f"key set {path} is unreadable: {exc}") from exc
    return decode(raw)


def lookup(
    digest: ClosureDigest,
    *,
    cache_dir: Optional[Path] = None,
    extra_roots: Sequence[Path] = (),
) -> Optional[KeySetHit]:
    """The class set for ``digest``, from the first root that holds it.

    A document that PARSES but does not carry this closure is not an error — it
    is a different release's key set, and the search moves on. A document that
    does not parse IS an error and it propagates: a malformed key set on a serve
    pod is a mint-lane bug, and reading past it is how a pod ends up silently
    tracing forever with a broken artifact sitting beside it.
    """
    for root in shipped_roots(extra=extra_roots):
        for path in _candidate_files(root):
            hit = _closure_from(path, digest, KeySource.SHIPPED)
            if hit is not None:
                return hit
    if cache_dir:
        for path in _candidate_files(Path(cache_dir)):
            hit = _closure_from(path, digest, KeySource.MEMO)
            if hit is not None:
                return hit
    return None


def _closure_from(
    path: Path, digest: ClosureDigest, source: KeySource,
) -> Optional[KeySetHit]:
    document = read_document(path)
    try:
        closure = parse_closure(document, digest)
    except KeySetError as exc:
        if exc.reason == "keyset_absent":
            return None
        raise
    return KeySetHit(closure=closure, source=source, path=path)


def class_hashes(
    digest: ClosureDigest, *, cache_dir: Optional[Path],
) -> Dict[GraphClassName, ClassHash]:
    """This machine's OWN cached graph axis for ``digest``, or ``{}``.

    The mint lane's honesty comparison reads through here, and it deliberately
    ignores shipped roots: what is being audited is what this pod's cache would
    have ANSWERED, and a shipped document is not this pod's claim.
    """
    if not cache_dir:
        return {}
    for path in _candidate_files(Path(cache_dir)):
        try:
            document = read_document(path)
            closure = parse_closure(document, digest)
        except KeySetError:
            return {}
        return closure.class_hashes
    return {}


def write_closure(
    cache_dir: Optional[Path], digest: ClosureDigest, row: ClosureRow,
) -> bool:
    """Record one closure in THIS machine's cache. Best effort, never fatal."""
    if not cache_dir:
        return False
    path = Path(cache_dir) / KEYSET_FILENAME
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            document = read_document(path)
        except KeySetError:
            document = empty()
        closures = dict(document.closures)
        closures[str(digest)] = row
        merged = KeySetDocument(
            schema=document.schema, version=document.version, closures=closures)
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(encode(merged))
        tmp.replace(path)
        return True
    except (OSError, KeySetError):
        logger.debug("keyset: cache write failed", exc_info=True)
        return False


def invalidate(cache_dir: Optional[Path], digest: ClosureDigest) -> bool:
    """Drop one closure from this machine's cache (a proven-dishonest row)."""
    if not cache_dir:
        return False
    path = Path(cache_dir) / KEYSET_FILENAME
    try:
        document = read_document(path)
    except KeySetError:
        return False
    if str(digest) not in document.closures:
        return False
    closures = {k: v for k, v in document.closures.items() if k != str(digest)}
    try:
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(encode(KeySetDocument(
            schema=document.schema, version=document.version,
            closures=closures)))
        tmp.replace(path)
        return True
    except (OSError, KeySetError):
        return False


def assert_honest(
    cache_dir: Optional[Path],
    digest: ClosureDigest,
    minted_entries: Mapping[str, Mapping[str, Any]],
) -> str:
    """THE honesty gate: what the cache answered must equal what the mint traced.

    Called from the publish path (``mint_supervisor.rule_on_boot_memo``) with the
    freshly minted artifact's own entry blocks. Returns ``''`` when the cache
    agreed (or held nothing for this closure), otherwise the reason — and the
    offending closure is dropped so the next boot re-derives rather than
    re-reading a hash proven wrong.

    The cache holds TCG class hashes, and ``graph_witness`` is one of the facts
    TCG folds into ``class_hash``, so comparing hashes also compares witnesses: a
    stale witness cannot survive a matching class hash.
    """
    memoized = class_hashes(digest, cache_dir=cache_dir)
    if not memoized:
        return ""
    disagreements: List[str] = []
    for name, block in sorted(minted_entries.items()):
        want = str((block or {}).get("class_hash") or "")
        had = memoized.get(GraphClassName(str(name)))
        if had and want and str(had) != want:
            disagreements.append(f"{name}: cached {had} != traced {want}")
    memo_names = {str(name) for name in memoized}
    extra = sorted(memo_names - set(minted_entries))
    missing = sorted(set(minted_entries) - memo_names)
    if extra or missing:
        disagreements.append(
            f"class set differs (cached-only {extra[:3]!r}, "
            f"traced-only {missing[:3]!r})")
    if not disagreements:
        return ""
    invalidate(cache_dir, digest)
    return (
        f"cg-keyset cache for closure {digest} was DISHONEST and has been "
        f"invalidated: " + "; ".join(disagreements[:4]))


def build_row(
    *,
    family: str,
    function: str,
    tcg_version: str,
    classes: Mapping[str, GraphClassRow],
    emitted_by: str = "",
) -> ClosureRow:
    """A closure row stamped with this moment. Thin, so emitters share one shape."""
    from .document import closure_row

    return closure_row(
        family=family,
        function=function,
        tcg_version=tcg_version,
        classes=classes,
        emitted_unix=int(time.time()),
        emitted_by=emitted_by,
    )
