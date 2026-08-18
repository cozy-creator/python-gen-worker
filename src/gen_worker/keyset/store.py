"""Where a ``cg-keyset-v1`` document lives, and who may write one.

Three kinds of location, and the difference is who derived the hashes:

* **Shipped roots** — read-only, produced by the MINT LANE. The baked image's
  ``/app/.tensorhub/`` (beside ``endpoint.lock``, the same place the decode-set
  and execution-lane blocks are stamped) and an explicit
  ``GEN_WORKER_CG_KEYSET`` path.
* **The DURABLE root** (pgw#1353) — ``$GEN_WORKER_LOCAL_CELLS_DIR/keysets``,
  written by a trace some pod of THIS endpoint ran. Read after the shipped
  roots and before the pod's own cache, and written whenever the cache is.
* **The pod's own cache** — written by a trace THIS machine ran (§4.28's
  compile-once-run-forever promise for cozy-local reads the same document
  through the same closure digest).

A FOURTH location lives one module over, in :mod:`gen_worker.keyset.hub`: the
platform's own store (pgw#1353 option (b) / th#2123), read after every root here
misses and written after a derive. It is not in this module because it is not a
FILESYSTEM location — it has no path, no ``mkdir``, and no atomic-rename story —
but it is the same document read through the same closure address, and
``closure_row`` below is how the row this module holds gets handed to it.

All of them are the same document type. A key set is trusted because its closure
digest matches what this pod's code would trace, never because of where it was
found — so the locations need no separate grammar, no separate parser and no
separate version ladder.

WHY THE DURABLE ROOT EXISTS, AND WHY IT IS NOT A FOURTH IDEA (pgw#1353)
----------------------------------------------------------------------
pgw#1327 offered a fresh pod exactly two ways to skip its traces: a document
baked into the image, or a document this container wrote earlier. **Neither
reaches a fleet pod.** The bake cannot be produced — the closure digest binds
the checkpoint ref (:mod:`gen_worker.keyset.closure`), which is a deploy-time
hub pick the image build cannot state. And the cache is
``tensorhub_cache_dir or tempfile.gettempdir()``, which the hub deliberately
leaves unset (th#850), so **the document a pod spends 805 s deriving is written
to /tmp and thrown away**. Measured on four sdxl pods: `keys_from=traced`,
778-833 s, every time, forever.

The platform already solved the identical problem for compiled graphs one layer
down. th#1813 has the hub place ``GEN_WORKER_LOCAL_CELLS_DIR`` on EVERY create,
pointing at the most durable storage the provider can back: the endpoint
NETWORK volume (survives the pod, shared by every pod of the endpoint) when one
is attached, else the pod's volume disk (survives a container restart). A key
set is the same KIND of artifact as the compiled graphs beside it — derived by a trace
this hardware ran, addressed by a digest that fails safe on drift, worthless to
anyone whose closure differs — so it belongs on the same root, in its own
subtree.

**It is an optimization, never a gate.** A pod with no root set, an unwritable
volume or a corrupt document still boots: the read falls through to the cache
and then to the deriver, exactly as before. That property is tested
(``test_keyset_durable_root_pgw1353``) rather than asserted, because the whole
value of the root is that the 805 s path is still THERE underneath it.
"""

from __future__ import annotations

import contextlib
import logging
import os
import secrets
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
    "DURABLE_KEYSET_DIRNAME",
    "ENV_KEYSET_PATH",
    "ENV_LOCAL_CELLS_DIR",
    "IMAGE_KEYSET_DIR",
    "KeySetHit",
    "assert_honest",
    "class_hashes",
    "closure_row",
    "durable_root",
    "invalidate",
    "lookup",
    "read_document",
    "shipped_roots",
    "write_closure",
    "writable_roots",
]

#: An explicit key set: a ``cg-keyset-v1.json`` file, or a directory holding one.
#: Carries a VALUE (where the document is), never a decision — a key set found
#: here is admitted by exactly the same closure match as any other.
ENV_KEYSET_PATH = "GEN_WORKER_CG_KEYSET"

#: th#1813's platform-placed durable store root. RESTATED, never imported from
#: :mod:`gen_worker.local_compiled_graph_store`, for the same reason
#: :data:`IMAGE_KEYSET_DIR` is restated: that module imports the vendored TCG
#: package, and reading a key set must not drag a tracer's import graph onto a
#: serve pod. ``test_the_durable_root_name_matches_the_compiled_graph_store`` pins the two
#: spellings together so the restatement cannot drift.
#:
#: A PATH, chosen by the party that knows the hardware (th#1813: *"where an
#: untrusted pod keeps its compiled graphs is a hardware fact the hub holds and
#: the code running on that pod does not"*) — never a behaviour switch. Its
#: absence means "forget sooner", never "behave differently".
ENV_LOCAL_CELLS_DIR = "GEN_WORKER_LOCAL_CELLS_DIR"

#: The key set's own subtree under the durable root. NAMESPACED rather than
#: sharing the root directly: ``local_compiled_graph_store`` owns ``aot-cells/`` there and
#: addresses it by ``ck1`` key, and two tiers writing sibling files into one
#: mount is how an operator ends up unable to say which tier a stray file
#: belongs to.
DURABLE_KEYSET_DIRNAME = "keysets"

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


def durable_root() -> Optional[Path]:
    """The platform-placed durable key-set subtree, or ``None`` when unplaced.

    ``None`` is an ordinary answer and not a degraded one: th#1813 places no
    root on trusted hardware (which publishes to the hub CAS instead) and none
    on a provider with no storage that outlives the container, and it writes the
    refusal at create time. A pod that gets no root derives, uses the document
    for its own boot, and forgets it — which is exactly what every pod does
    today.
    """
    env = os.environ.get(ENV_LOCAL_CELLS_DIR, "").strip()
    if not env:
        return None
    return Path(env).expanduser() / DURABLE_KEYSET_DIRNAME


def writable_roots(cache_dir: Optional[Path]) -> Tuple[Tuple[Path, KeySource], ...]:
    """Every root a derivation records into, in READ precedence order.

    THE single ordering. ``lookup``'s non-shipped leg, ``class_hashes``'
    honesty audit, ``write_closure`` and ``invalidate`` all walk this, so a
    root that can be read is by construction a root that can be written and
    invalidated. A row proven dishonest that survived in one of them because
    only three of the four functions knew it existed is precisely the failure
    this shape rules out.

    Durable before cache: on a fleet pod the cache is ``/tmp`` and the durable
    root is the endpoint's volume, so the shared answer wins over the
    container-local one. They can only disagree by holding different closures —
    the same closure digest is the same class set — so the order is about which
    is more likely to ANSWER, never about which is more likely to be right.
    """
    roots: List[Tuple[Path, KeySource]] = []
    durable = durable_root()
    if durable is not None:
        roots.append((durable, KeySource.DURABLE))
    if cache_dir:
        roots.append((Path(cache_dir), KeySource.MEMO))
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
    for root, source in writable_roots(cache_dir):
        for path in _candidate_files(root):
            hit = _closure_from(path, digest, source)
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
    """The graph axis a WRITABLE root would have answered for ``digest``, or ``{}``.

    The mint lane's honesty comparison reads through here, and it deliberately
    ignores shipped roots: what is being audited is what a root this pod may
    WRITE would have answered, and a shipped document is the mint lane's claim
    rather than this pod's.

    pgw#1353: that now includes the durable root, in ``lookup``'s own order, and
    it has to. The audit's whole job is to catch a stored row whose class hashes
    disagree with what a real mint traced; auditing only ``/tmp`` while the pod
    ANSWERED from the endpoint's volume would leave a proven-dishonest row on
    the shared volume, where every future pod of the endpoint would read it.
    """
    for root, _source in writable_roots(cache_dir):
        for path in _candidate_files(root):
            try:
                document = read_document(path)
                closure = parse_closure(document, digest)
            except KeySetError:
                break
            return closure.class_hashes
    return {}


def closure_row(
    cache_dir: Optional[Path], digest: ClosureDigest,
) -> Optional[ClosureRow]:
    """The RAW row a writable root holds for ``digest``, or ``None``.

    The producer half of th#2123's hub tier reads through here. Deliberately the
    raw :class:`ClosureRow` and not a :class:`ShippedClosure`: what travels to
    the hub is the document a pod recorded, byte for byte, and re-deriving it
    from a parsed view would mean this worker's writer and its reader could
    disagree about a field neither of them reads.

    Walks the SAME ``writable_roots`` ordering as everything else here, so the
    row that gets published is the row a re-read would answer with — not one
    from a root only this function knows about.
    """
    for root, _source in writable_roots(cache_dir):
        for path in _candidate_files(root):
            try:
                document = read_document(path)
            except KeySetError:
                break
            row = document.closures.get(str(digest))
            if row is not None:
                return row
    return None


def _publish(path: Path, document: KeySetDocument) -> None:
    """Replace ``path`` with ``document``, atomically and CONCURRENCY-SAFELY.

    The temp name carries a random suffix, and that is not decoration. The
    durable root can be a network volume MOUNTED BY SEVERAL PODS OF THE SAME
    ENDPOINT AT ONCE — which is the entire reason it is worth writing to — and
    the fixed ``cg-keyset-v1.tmp`` this replaces is a name two pods would write
    to simultaneously. ``os.replace`` is atomic, so the reader never sees a torn
    file either way; what a shared temp name loses is the WRITER's bytes, and it
    can lose them by interleaving two encodes into one file. A unique temp plus
    an atomic rename makes concurrent writers cost at most a lost MERGE (below),
    never a corrupt document.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{secrets.token_hex(8)}.tmp")
    try:
        tmp.write_bytes(encode(document))
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            tmp.unlink()
        raise


def _record(path: Path, digest: ClosureDigest, row: ClosureRow) -> bool:
    """Merge one closure into the document at ``path``. Best effort.

    The read-modify-write is NOT locked, deliberately. Two pods of one endpoint
    deriving different closures at the same moment can lose one of the two
    merges — and the cost of that loss is that one future boot re-derives, which
    is the cost of not having the row at all. A lock file on a network volume
    buys a correctness property nothing here needs and brings a stale-lock
    failure mode that would take out the read path too.
    """
    try:
        try:
            document = read_document(path)
        except KeySetError:
            document = empty()
        closures = dict(document.closures)
        closures[str(digest)] = row
        _publish(path, KeySetDocument(
            schema=document.schema, version=document.version, closures=closures))
        return True
    except (OSError, KeySetError):
        logger.debug("keyset: write to %s failed", path, exc_info=True)
        return False


def write_closure(
    cache_dir: Optional[Path], digest: ClosureDigest, row: ClosureRow,
) -> bool:
    """Record one closure in EVERY writable root. Best effort, never fatal.

    Returns whether any root took it. pgw#1353: this is the whole producer half
    of the durable root — ``boot_key.derive`` and ``emit.record_closure`` already
    funnel here, so a pod that pays 805 s of traces now leaves the answer
    somewhere the next pod of its endpoint can find it, with no new call site
    and no new argument.

    An unwritable root is a slower next boot, never a wrong key, so a failure on
    one root does not stop the other: a read-only volume must not cost this pod
    its own ``/tmp`` memo.
    """
    wrote = False
    for root, _source in writable_roots(cache_dir):
        if _record(root / KEYSET_FILENAME, digest, row):
            wrote = True
    return wrote


def invalidate(cache_dir: Optional[Path], digest: ClosureDigest) -> bool:
    """Drop one closure from EVERY writable root (a proven-dishonest row).

    Returns whether any root held it. Total across the roots on purpose: a row
    the mint lane just PROVED dishonest surviving on the endpoint's shared
    volume — because only the pod-local copy was dropped — would re-poison every
    pod of the endpoint on its next boot, and this pod would have no reason to
    look again.
    """
    dropped = False
    for root, _source in writable_roots(cache_dir):
        path = root / KEYSET_FILENAME
        try:
            document = read_document(path)
        except KeySetError:
            continue
        if str(digest) not in document.closures:
            continue
        closures = {
            key: value for key, value in document.closures.items()
            if key != str(digest)}
        try:
            _publish(path, KeySetDocument(
                schema=document.schema, version=document.version,
                closures=closures))
            dropped = True
        except (OSError, KeySetError):
            logger.debug("keyset: invalidate at %s failed", path, exc_info=True)
    return dropped


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
