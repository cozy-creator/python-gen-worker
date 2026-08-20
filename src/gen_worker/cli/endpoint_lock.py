"""``endpoint.lock`` — the discovery manifest PLUS the derive document.

pgw#1466. Before this module a lock carried only what discovery could read
statically (``entrypoints``/``execution_lanes``/``decode_set``) and the trace
lived nowhere: every process that wanted a graph set re-derived it, at ~165s a
go. Paul's ruling is that the trace is expensive and machine-independent, so it
is "worth saving locally for sure" — the derive DOCUMENT goes in the lock.

The split is the whole design:

* the DOCUMENT (small, declarative — specializations, ingress, digests) is
  embedded verbatim in ``[derive]``;
* the exported-program BLOBS stay in the tensorfs CAS, addressed by digests the
  document already carries.

Two rules this module exists to enforce, both from the 2026-08-19
"torchcg DOES TOO MUCH" ruling:

1. **Nothing derivable is restated.** The block stores the document's bytes and
   its digest. It does not lift ``specialization_hash``, dims, shapes or program
   digests up to the top level for convenience. Every such copy is a second
   place a producer can lie, and a reader that trusts the copy over the document
   is the bug. Callers that want a specialization read it back OUT of the
   document (:func:`graph_identities`), which is why that function parses rather
   than looks up.
2. **Hashes are opaque addresses.** ``specialization_hash`` folds
   ``specialization_dims`` into itself and is the graph axis of every cg-key. We
   never recompute one and never restate its inputs — a mismatch would not raise,
   it would read as a cache miss and silently re-mint the fleet.

The skip check is therefore NOT "does the document's digest match" (you cannot
know that without paying the 165s to produce it). It is an INPUTS digest: hash
everything the derive reads — author source, env closure, checkpoint identity,
trace device — and skip when that is unchanged AND every program blob the
document names is still in the CAS. Inputs are cheap to hash and are not derived
facts, so this adds no lie-seam.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

import msgspec

#: The file every verb reads and ``lock`` writes, in the endpoint directory.
LOCK_FILENAME = "endpoint.lock"

#: Top-level table holding the derive document. Named once, here.
DERIVE_BLOCK = "derive"

#: Bumped when the BLOCK's own shape changes. Part of the inputs digest, so a
#: format change invalidates every stored lock rather than being read wrongly.
DERIVE_BLOCK_V = 1


class LockError(RuntimeError):
    """The lock on disk cannot be used as it stands."""


@dataclass(frozen=True)
class DeriveBlock:
    """``[derive]`` — the trace, saved.

    ``document`` is the exact bytes :func:`gen_worker.release.derive_release`
    emitted. Verbatim matters: ``document_digest`` is a sha256 OF THOSE BYTES,
    so a re-encode (TOML round-trip, key reorder, pretty-print) would break the
    only thing that makes the document verifiable. It is carried as an ASCII
    string and hashed on the way back in.
    """

    #: Block format version (``DERIVE_BLOCK_V`` at write time).
    v: int
    #: torchcg's OWN graph-document interface version, read off the document
    #: rather than pinned here. The interface is v4 today and tcg#56 rekeys it
    #: again; a reader that parses an unfamiliar version is how a stale
    #: specialization gets adopted, so :func:`derive_is_reusable` treats any
    #: change as a MISS-that-re-derives instead of something to migrate.
    interface_v: int
    #: sha256 over the derive's INPUTS. The skip key; see module docstring.
    inputs_digest: str
    #: sha256 of ``document``'s bytes, as the producer computed it. OPAQUE.
    document_digest: str
    #: The derive document, verbatim ASCII JSON.
    document: str
    #: The device CLASS traced, `derive._trace_device`'s answer (``cuda``
    #: since tcg#64, on any host). A trace on another class is a DIFFERENT
    #: graph specialization, not a degraded one, so it is stated rather than
    #: assumed by every reader — which is also what retires a lock committed
    #: before the class stopped depending on the deriving box.
    trace_device: str
    #: The endpoint identity the derive stamped (``module:Class``).
    endpoint: str

    def decoded(self) -> dict[str, Any]:
        """The document as a dict. Verifies the digest before returning it."""
        raw = self.document.encode("ascii")
        actual = hashlib.sha256(raw).hexdigest()
        if actual != self.document_digest:
            raise LockError(
                f"[{DERIVE_BLOCK}] document does not match its own digest "
                f"(stored {self.document_digest}, actual {actual}). The lock "
                f"was edited by hand or truncated in transit; re-run "
                f"`gen-worker lock --force`."
            )
        return dict(json.loads(raw))

    def as_table(self) -> dict[str, Any]:
        return {
            "v": self.v,
            "interface_v": self.interface_v,
            "inputs_digest": self.inputs_digest,
            "document_digest": self.document_digest,
            "trace_device": self.trace_device,
            "endpoint": self.endpoint,
            "document": self.document,
        }

    @classmethod
    def from_table(cls, table: Mapping[str, Any]) -> "DeriveBlock":
        missing = [
            k
            for k in ("v", "interface_v", "inputs_digest", "document_digest",
                      "document", "trace_device", "endpoint")
            if k not in table
        ]
        if missing:
            raise LockError(
                f"[{DERIVE_BLOCK}] is missing {missing}; it was written by a "
                f"different version of `gen-worker lock`. Re-run "
                f"`gen-worker lock --force`."
            )
        return cls(
            v=int(table["v"]),
            interface_v=int(table["interface_v"]),
            inputs_digest=str(table["inputs_digest"]),
            document_digest=str(table["document_digest"]),
            document=str(table["document"]),
            trace_device=str(table["trace_device"]),
            endpoint=str(table["endpoint"]),
        )


def torchcg_format_versions() -> tuple[int, int]:
    """``(GRAPH_INTERFACE_FORMAT, DOCUMENT_FORMAT)`` — IMPORTED, never spelled.

    These are torchcg's numbers and this module must not carry a second copy of
    them: a hardcoded ``4`` here would keep answering 4 after torchcg rekeys, and
    a stale specialization would be adopted instead of re-derived. tcg#56 will
    bump the interface, which is exactly the case this must not sleep through.

    Both fold into :func:`inputs_digest`, so a bump invalidates every stored lock
    automatically — an old lock becomes a MISS that re-derives, never a document
    this code tries to migrate or parse under new rules.
    """
    from .._vendor.torchcg import DOCUMENT_FORMAT, GRAPH_INTERFACE_FORMAT

    return int(GRAPH_INTERFACE_FORMAT), int(DOCUMENT_FORMAT)


def tracer_digest() -> str:
    """A fingerprint of the CODE THAT TRACES: torchcg plus gen_worker's derive.

    The env closure covers both when both are INSTALLED — but the cozy-local
    shape Paul asked for runs them off ``PYTHONPATH`` from working trees, where
    no distribution names them. In that shape a tracer change is invisible to
    the closure, so a saved trace would be reused across a fix that changes
    what tracing MEANS.

    That is not hypothetical: tcg#57 changed the hollow session's default
    device and pgw#1465 deleted the meta demotion. Both alter the emitted
    graphs, and neither moved ``uv.lock``. Without this field the skip check
    would happily serve pre-fix graphs that no mint can compile.

    Best-effort by construction: a tracer we cannot locate contributes the empty
    string rather than raising, because a missing fingerprint must not break a
    lock — it only costs a re-derive.
    """
    h = hashlib.sha256()
    for module_name in ("torchcg", "gen_worker.release"):
        try:
            module = __import__(module_name, fromlist=["__file__"])
            origin = Path(getattr(module, "__file__", "") or "")
        except Exception:  # noqa: BLE001 - absence is an answer, not a failure
            continue
        if not origin.is_file():
            continue
        for path in sorted(origin.parent.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            h.update(path.name.encode("utf-8"))
            h.update(_file_digest(path).encode("ascii"))
    return h.hexdigest()


def graph_identities(document: Mapping[str, Any]) -> tuple[str, ...]:
    """Every graph IDENTITY the document names, deduped + sorted.

    Parsed out of the document on demand and deliberately NOT stored beside it:
    a stored copy is a fact that can disagree with its source. The walk is
    structural (any ``graph`` key under ``graphs``) rather than a fixed path,
    because the graph document's nesting is torchcg's to change and this reader
    must not encode a second copy of its schema.

    This used to collect ``program`` blob ADDRESSES. It cannot: a blob digest is
    machine-scoped (pgw#1462p2 measured 14/14 identities reproducing across
    boxes and 0/14 blob digests), so it is not in the document any more. The
    identity is, and it is what a local store is keyed by.
    """
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "graph" and isinstance(value, str) and value:
                    found.add(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(document.get("graphs", {}))
    return tuple(sorted(found))


def _file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_files(root: Path) -> tuple[Path, ...]:
    """The author source the derive actually imports, sorted and stable.

    ``src/**/*.py`` plus top-level ``*.py`` — the two layouts ``prime_sys_path``
    supports. Test trees and caches are excluded because changing a test cannot
    change a traced graph, and including them would make every test edit re-spend
    the trace.
    """
    seen: set[Path] = set()
    src = root / "src"
    if src.is_dir():
        seen.update(p for p in src.rglob("*.py"))
    seen.update(p for p in root.glob("*.py"))
    skip = ("__pycache__", ".venv", ".mypy_cache", ".pytest_cache")
    return tuple(
        sorted(
            p
            for p in seen
            if p.is_file() and not any(part in skip for part in p.parts)
        )
    )


def inputs_digest(
    *,
    root: Path,
    module_name: str,
    checkpoint_ref: str,
    trace_device: str,
    lockfile: Optional[Path] = None,
    extra: Iterable[str] = (),
) -> str:
    """Hash everything the derive READS. Unchanged inputs => skippable re-run.

    Deliberately excludes anything the derive PRODUCES. The point of an inputs
    digest is that it can be computed in milliseconds before deciding whether to
    spend ~165s; a digest over outputs could only be checked after paying.

    The env input is the endpoint's ``uv.lock``, by FILE DIGEST — the same file
    the document's compile stack is read from (pgw#1489). The hazard pgw#1472
    named here (a reuse carrying a saved ``[derive]`` block whose env this
    process no longer restates) is answered by construction rather than by
    hashing the process: the block's stack came from this lock, so a lock that
    moves re-derives, and a lock that has not moved states the same stack it
    stated before. What the venv happens to have installed is a diagnostic the
    verb prints, never a reuse key — an artifact is not invalidated by a docs
    extra.
    """
    h = hashlib.sha256()

    def field(label: str, value: str) -> None:
        # Length-prefixed so no two different field sets can collide by
        # concatenation (``a=1,b=23`` vs ``a=12,b=3``).
        h.update(f"{label}:{len(value)}:{value}\n".encode("utf-8"))

    interface_v, document_v = torchcg_format_versions()
    field("block_v", str(DERIVE_BLOCK_V))
    # A torchcg rekey changes what a saved trace MEANS, so it changes the skip
    # key. This is the whole mechanism by which tcg#56 forces a re-derive
    # instead of a silently-misread document.
    field("interface_v", str(interface_v))
    field("document_v", str(document_v))
    # The tracer's own source. See `tracer_digest`: in the cozy-local shape
    # neither torchcg nor gen_worker is an installed distribution, so this is
    # the only thing standing between a tracer fix and a silently-reused
    # pre-fix graph set.
    field("tracer", tracer_digest())
    field("module", module_name)
    field("checkpoint_ref", checkpoint_ref)
    field("trace_device", trace_device)
    field("lockfile", _file_digest(lockfile) if lockfile and lockfile.is_file() else "")
    for path in source_files(root):
        field(f"src:{path.relative_to(root).as_posix()}", _file_digest(path))
    for item in extra:
        field("extra", item)
    return h.hexdigest()


@dataclass(frozen=True)
class Reuse:
    """Whether a stored derive can be reused, and — always — WHY NOT.

    ``reason`` is populated on both branches. A skip decision that can only say
    "miss" turns every unexpected re-trace into a 165s mystery; naming the
    changed input is the difference between a cache and a black box.
    """

    ok: bool
    reason: str
    block: Optional[DeriveBlock] = None
    document: Optional[dict[str, Any]] = None


def derive_is_reusable(
    block: Optional[DeriveBlock],
    *,
    want_inputs_digest: str,
    want_trace_device: str,
    cas_has: Any = None,
) -> Reuse:
    """The skip check: can we serve this lock's trace instead of re-deriving?

    ``cas_has`` is a predicate over a GRAPH IDENTITY (typically
    ``LocalGraphStore.has_program``). It is checked because the document and the
    bytes are stored SEPARATELY by design — a garbage-collected CAS leaves a
    lock whose document is intact and whose programs are gone, and re-deriving
    is the only correct response. Pass ``None`` to skip the check when the
    caller does not need the programs (e.g. `run`, which never compiles).
    """
    if block is None:
        return Reuse(False, "no [derive] block — the trace was never saved")
    if block.v != DERIVE_BLOCK_V:
        return Reuse(
            False,
            f"[derive] block format v{block.v}, this gen-worker writes "
            f"v{DERIVE_BLOCK_V}",
        )
    # Device class is IDENTITY-BEARING: a cpu-traced graph and a cuda-traced
    # graph are different specializations, and a cuda mint of a cpu document
    # refuses by name (pgw#1458). Reusing across the boundary would produce a
    # lock that cannot be minted on the machine that reads it.
    if block.trace_device != want_trace_device:
        return Reuse(
            False,
            f"saved trace is {block.trace_device}-class, this host traces "
            f"{want_trace_device}-class — different specializations, not a "
            f"degraded one",
        )
    if block.inputs_digest != want_inputs_digest:
        return Reuse(
            False,
            "inputs changed (author source, uv.lock closure, checkpoint ref, "
            "trace device, a torchcg format version, or the tracer's own "
            "source)",
        )
    try:
        document = block.decoded()
    except LockError as exc:
        return Reuse(False, str(exc))
    if cas_has is not None:
        missing = [g for g in graph_identities(document) if not cas_has(g)]
        if missing:
            return Reuse(
                False,
                f"this box holds no serialized program for {len(missing)} of "
                f"the graphs the document names (first: {missing[0]}) — the "
                f"document survived a GC its programs did not, or it was "
                f"derived on another box",
            )
    return Reuse(True, "inputs unchanged and every program blob present",
                 block=block, document=document)


def read_lock(path: Path) -> dict[str, Any]:
    """The lock file as a dict, or a typed refusal naming the file."""
    try:
        raw = path.read_bytes()
    except FileNotFoundError as exc:
        raise LockError(f"no {path} — run `gen-worker lock` first") from exc
    try:
        decoded = msgspec.toml.decode(raw)
    except Exception as exc:  # noqa: BLE001 - any decode failure is the answer
        raise LockError(f"{path} is not valid TOML: {exc}") from exc
    if not isinstance(decoded, dict):
        raise LockError(f"{path} must decode to a TOML table")
    return dict(decoded)


def read_derive_block(path: Path) -> Optional[DeriveBlock]:
    """``[derive]`` from the lock at ``path``, or None when it carries none.

    None is the honest answer for a lock built by the BAKE path
    (``python -m gen_worker.discovery``), which emits discovery blocks only.
    Callers treat it as "trace not saved yet", never as an error — that is what
    lets `serve` overlap a trace with the weight download instead of refusing.
    """
    table = read_lock(path).get(DERIVE_BLOCK)
    if table is None:
        return None
    if not isinstance(table, dict):
        raise LockError(f"[{DERIVE_BLOCK}] in {path} must be a table")
    return DeriveBlock.from_table(table)


def write_lock(
    path: Path, manifest: Mapping[str, Any], derive: Optional[DeriveBlock]
) -> None:
    """Write discovery blocks + ``[derive]`` to ``path``, atomically.

    Atomic because every other verb reads this file: a `serve` that starts while
    `lock` is halfway through a write must see the old lock or the new one, never
    a truncated one. TOML has no way to express "incomplete", so the rename is
    what carries that guarantee.
    """
    # TOML has no null type, and the manifest carries optional fields nested
    # arbitrarily deep (an entrypoint's absent `execution_lane_exclusions`, a
    # slot's absent engine). Stripping only the top level encodes a wrong belief
    # about how deep None can be — the bake path strips recursively and so must
    # this one, or the two producers write different files from one manifest.
    from ..discovery.discover import _strip_none

    document: dict[str, Any] = dict(_strip_none(dict(manifest)))
    if derive is not None:
        document[DERIVE_BLOCK] = derive.as_table()
    encoded = msgspec.toml.encode(document)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_bytes(encoded)
    tmp.replace(path)


__all__ = [
    "DERIVE_BLOCK",
    "DERIVE_BLOCK_V",
    "LOCK_FILENAME",
    "DeriveBlock",
    "LockError",
    "Reuse",
    "derive_is_reusable",
    "inputs_digest",
    "graph_identities",
    "read_derive_block",
    "read_lock",
    "source_files",
    "torchcg_format_versions",
    "write_lock",
]
