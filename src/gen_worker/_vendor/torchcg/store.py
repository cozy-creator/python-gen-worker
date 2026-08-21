"""The abstract compiled-graph store, and its local-CAS implementation.

torchcg owns the compile LIFECYCLE; it knows nothing about hubs, JWTs,
releases or pods (tcg#41). Everything the lifecycle needs from persistence is
this interface: graph-set documents by name, artifacts positioned by
(graph, env), requirements manifests beside them, and hole listing. gen-worker
implements it hub-backed; cozy-local implements it over a bare local CAS;
``LocalGraphStore`` here is both the reference implementation and the
lifecycle test double -- same lifecycle code, any store.

Addresses live under ``torchcg/v2`` -- a NEW address space for the two-level
identity scheme. ``torchcg/v1`` (exact cg-key rows, ``storage.py``) is the
prior grammar; nothing reads both.

## A graph-set document this build cannot read is ABSENT, and it is discarded

tcg#69. Every byte in this store is DERIVED: a graph-set document is a
statement about programs that can be re-traced and artifacts that can be
re-minted from sources the store does not own. So a document written by an
older ``DOCUMENT_FORMAT`` is not a fault to report, it is a document this
build does not have -- ``get_graphs`` returns ``None``, the caller registers
holes and serves eager while a background mint refills, and the unreadable
bytes are dropped. That is the SAME path a store with no document at all
takes, which is what makes it safe: the miss branch is the one every caller
already exercises on a fresh box.

There is NO migration machinery here and there never will be (Paul,
2026-08-19: *"weights migrate/normalize once at ingest; graphs regenerate"*).
A format bump costs one re-mint, and code that translates old graph documents
would be a permanent tax paid to avoid a one-time cost that the store is
designed to absorb.

What this replaces: raising ``StoreError`` at ADOPT time. It was typed and
loud, so it was never silent -- but it landed on every local user's first
cold start after an upgrade, at serving time, on a machine that could simply
have rebuilt. The condition was diagnosed correctly and handled at the wrong
altitude.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..tensorfs import CASRef, DigestMismatch, LocalCAS, RefConflict

from .artifact import IO_BUFFER_BYTES
from .document import DocumentError, GraphSetDocument, LaneGraphs
from .graph_identity import EnvIdentity, is_graph_hash
from .lane import LaneError, require_pass_ref
from .requirements import RequirementsError, RequirementsManifest

logger = logging.getLogger(__name__)

_REF_PREFIX = "torchcg/v2"
#: tcg#86: `artifact.IO_BUFFER_BYTES` is the owner; this was the third of
#: three names for one buffer.
_READ_BUFFER = IO_BUFFER_BYTES


class StoreError(RuntimeError):
    """A compiled-graph store row is malformed, divergent, or unavailable."""


class PublishOutcome(StrEnum):
    PUBLISHED = "published"
    PRESENT = "present"


@runtime_checkable
class GraphStore(Protocol):
    """What the compile lifecycle requires of ANY store backend."""

    def get_graphs(self, name: str) -> GraphSetDocument | None:
        """Return the named graph-set document, or a clean miss."""
        ...

    def put_graphs(self, name: str, document: GraphSetDocument) -> None:
        """Stamp the named graph-set document, atomically."""
        ...

    def has_artifact(self, graph: str, env: EnvIdentity) -> bool:
        """Whether an artifact exists at (graph, env) -- the hole probe."""
        ...

    def fetch_artifact(self, graph: str, env: EnvIdentity, destination: str | Path) -> Path | None:
        """Verify and copy the (graph, env) artifact out, or a clean miss."""
        ...

    def publish_artifact(
        self,
        graph: str,
        env: EnvIdentity,
        artifact: str | Path,
        manifest: RequirementsManifest,
    ) -> PublishOutcome:
        """Publish one minted artifact with its mint-written manifest."""
        ...

    def get_manifest(self, graph: str, env: EnvIdentity) -> RequirementsManifest | None:
        """Return the manifest published beside (graph, env), or a miss."""
        ...


def holes(store: GraphStore, lane: LaneGraphs, env: EnvIdentity) -> tuple[str, ...]:
    """The graphs this lane still needs minted for this env.

    Partial-hit by construction: callers adopt what exists and mint ONLY
    these, publishing each as it completes -- never all-or-nothing.
    """

    return tuple(
        record.graph for record in lane.graphs if not store.has_artifact(record.graph, env)
    )


def _require_graph(graph: str) -> str:
    if not is_graph_hash(graph):
        raise StoreError(f"store rows are addressed by cg-graph-v1 hashes, got {graph!r}")
    return graph


def _document_ref(name: str) -> str:
    clean = str(name).strip()
    if not clean or "\\" in clean or any(part in ("", ".", "..") for part in clean.split("/")):
        raise StoreError(f"unsafe graph-set name {name!r}")
    return f"{_REF_PREFIX}/graphsets/{clean}"


def _artifact_ref(graph: str, env: EnvIdentity) -> str:
    return f"{_REF_PREFIX}/artifacts/{_require_graph(graph)}/{env.value}"


def _manifest_ref(graph: str, env: EnvIdentity) -> str:
    return f"{_REF_PREFIX}/manifests/{_require_graph(graph)}/{env.value}"


def _program_ref(graph: str) -> str:
    """This machine's serialized ExportedProgram for one graph specialization.

    KEYED BY THE GRAPH HASH, never by the blob's own digest, and that is the
    whole of the address-free ruling (Paul, 2026-08-19). `torch.export.save`
    emits different bytes on different machines for the same traced graph
    (pgw#1462p2: 14/14 identities, 0/14 blob digests), so the blob digest is a
    machine-scoped fact and cannot appear in a document every machine must
    agree on. The graph hash IS reproducible, so it is the key here and the
    identity in the document, and the bytes never leave the box that made them.
    """
    return f"{_REF_PREFIX}/programs/{_require_graph(graph)}"


def _side_table_ref(pass_name: str, key: str) -> str:
    """A transform pass's minted side table, addressed by (pass, domain key).

    Not positioned under a graph: a side table is produced BEFORE discovery
    (tcg#52's PASS -> DISCOVERY -> EXPORT), so no graph hash exists yet. It is
    also env-independent -- it is checkpoint bytes, not compiler output -- so
    a bucket minted on a mint pod is the same bucket a serving pod reads.
    """

    try:
        require_pass_ref(pass_name)
    except LaneError as exc:
        raise StoreError(str(exc)) from exc
    if not re.fullmatch(r"[0-9a-f]{8,64}", str(key)):
        raise StoreError(f"a side-table key is the canonical hex domain-key address, got {key!r}")
    return f"{_REF_PREFIX}/sidetables/{pass_name}/{key}"


def _digest_file(path: Path) -> CASRef:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_READ_BUFFER):
            digest.update(chunk)
    return CASRef(digest.hexdigest())


#: Sidecar written beside a verified copy-out, recording which CAS ref the
#: destination holds and the exact file identity the copy left behind.
_STAMP_SUFFIX = ".torchcg-src"


def _stamp_path(target: Path) -> Path:
    return target.with_name(target.name + _STAMP_SUFFIX)


def _write_stamp(target: Path, ref: CASRef) -> None:
    """Record that ``target`` IS ``ref``'s verified bytes, best-effort.

    A stamp that cannot be written costs the next boot one re-verify and
    nothing else, so failure here is swallowed on purpose.
    """

    try:
        stat = target.stat()
        _stamp_path(target).write_text(
            json.dumps(
                {
                    "format": 1,
                    "target": str(ref),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "ino": stat.st_ino,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            encoding="ascii",
        )
    except OSError:
        pass


def _stamped(target: Path, ref: CASRef) -> Path | None:
    """``target`` if a prior VERIFIED copy-out of exactly ``ref`` still sits
    there untouched, else ``None``.

    The warm-boot fast path (pgw#1546): re-hashing and re-copying bytes a
    previous boot already verified is per-boot work proportional to artifact
    size over a store nothing changed. The stamp is only honoured when the
    ref still names the same object AND the destination's size/mtime/inode
    identity matches what the verified copy recorded -- any rewrite, however
    small, changes that identity and falls back to the full verify+copy.
    """

    try:
        raw = json.loads(_stamp_path(target).read_bytes())
        stat = target.stat()
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict) or raw.get("target") != str(ref):
        return None
    if (
        raw.get("size") == stat.st_size
        and raw.get("mtime_ns") == stat.st_mtime_ns
        and raw.get("ino") == stat.st_ino
        and target.is_file()
        and not target.is_symlink()
    ):
        return target
    return None


class LocalGraphStore:
    """The reference ``GraphStore`` over one tensorfs ``LocalCAS``.

    First publish wins per (graph, env); a second publish of the same bytes
    is idempotent ``PRESENT``; different bytes under an occupied position are
    refused -- the mint is deterministic per env, so divergence is a bug to
    surface, not a race to arbitrate.
    """

    def __init__(self, cas: LocalCAS) -> None:
        self.cas = cas

    def _admit_file(self, path: Path) -> CASRef:
        ref = _digest_file(path)
        try:
            if self.cas.contains(ref):
                return ref
        except DigestMismatch:
            pass
        return self.cas.put_file(path, expected=ref)

    # -- graph-set documents ------------------------------------------------

    def get_graphs(self, name: str) -> GraphSetDocument | None:
        """The named document, or a clean miss -- including when it is stale.

        A document this build cannot decode is a MISS, not an error: see the
        module docstring. It is discarded on the way out so the next call is
        an ordinary miss against an empty position rather than a second
        discard, and so the box does not carry the dead bytes forever.
        """

        ref_name = _document_ref(name)
        ref = self.cas.read_ref(ref_name)
        if ref is None:
            return None
        try:
            blob = self.cas.verify_object(ref)
            return GraphSetDocument.decode(Path(blob).read_bytes())
        except (DigestMismatch, FileNotFoundError, ValueError, DocumentError) as exc:
            self._discard_graphs(ref_name, name, ref, exc)
            return None

    def _discard_graphs(self, ref_name: str, name: str, ref: CASRef, cause: Exception) -> None:
        """Drop an unreadable graph-set document and delete its bytes.

        ONE warning, and it names all three things a reader needs: which
        graph-set went away, which bytes were dropped, and why they could not
        be read. Silence here would turn a rebuild into an unexplained one.

        The ref goes first: it is the compare-and-swap, so a peer that
        replaced or discarded the same document a moment earlier simply wins,
        and this call does nothing but return the miss it already decided on.

        The bytes go second and ONLY if nothing else names them. This is a
        content-addressed store, so an object is shared the instant two
        positions hold identical bytes; deleting one position's object without
        that check would punch a hole under a live position. The scan is over
        ``refs`` -- small JSON records, one per position -- never over
        ``objects``, whose layout belongs to tensorfs and not to this module.
        """

        try:
            self.cas.compare_and_swap_ref(ref_name, None, expected=ref)
        except RefConflict:
            # A concurrent writer already replaced or dropped it. Whatever
            # stands now is theirs; this call's only job was to stop reading
            # the stale bytes, and returning a miss has done that.
            logger.warning(
                "torchcg: graph-set %r was unreadable by this build (%s) and a "
                "concurrent writer replaced it; serving as ABSENT this call",
                name,
                cause,
            )
            return
        reclaimed = self._discard_object(ref)
        logger.warning(
            "torchcg: DISCARDED graph-set %r (%s) -- unreadable by this build: "
            "%s. Compiled graphs are derived and disposable, so it is treated "
            "as ABSENT and will be re-minted; no migration is attempted. "
            "Reclaimed %d byte(s).",
            name,
            ref,
            cause,
            reclaimed,
        )

    def _discard_object(self, ref: CASRef) -> int:
        """Delete ``ref``'s bytes if no logical ref still names them.

        Returns the bytes reclaimed, 0 when the object is still shared or is
        already gone. Never raises: this runs on the way out of a read that
        has already decided its answer, and failing to reclaim a few kilobytes
        must not turn a clean miss back into an exception.
        """

        try:
            if any(self._targets(record) == ref for record in self._ref_records()):
                return 0
            path = self.cas.object_path(ref)
            size = path.stat().st_size
            path.unlink()
        except OSError:
            return 0
        return size

    def _ref_records(self) -> list[dict[str, object]]:
        records: list[dict[str, object]] = []
        for path in self.cas.refs.iterdir():
            if not path.is_file():
                continue
            try:
                raw = json.loads(path.read_bytes())
            except (OSError, ValueError):
                continue
            if isinstance(raw, dict):
                records.append(raw)
        return records

    @staticmethod
    def _targets(record: dict[str, object]) -> CASRef | None:
        try:
            return CASRef.parse(str(record.get("target", "")))
        except ValueError:
            return None

    def put_graphs(self, name: str, document: GraphSetDocument) -> None:
        ref_name = _document_ref(name)
        candidate = self.cas.put_bytes(document.encode())
        expected = self.cas.read_ref(ref_name)
        while True:
            if expected == candidate:
                return
            try:
                self.cas.compare_and_swap_ref(ref_name, candidate, expected=expected)
                return
            except RefConflict:
                expected = self.cas.read_ref(ref_name)

    # -- artifacts ----------------------------------------------------------

    def has_artifact(self, graph: str, env: EnvIdentity) -> bool:
        return self.cas.read_ref(_artifact_ref(graph, env)) is not None

    def artifact_skew(self, graph: str, env: EnvIdentity) -> str | None:
        """A sentence when the stored blob is a recognizable NON-envelope, else
        ``None`` — the census-grade shape probe (tcg#75/pgw#1561).

        Two magic bytes, not a decode: a per-boot census over every position
        must stay proportional to positions, not to artifact bytes. Absence is
        ``None`` here — ``has_artifact`` owns that question — and full
        verification stays where it always was, on the fetch path.
        """

        ref = self.cas.read_ref(_artifact_ref(graph, env))
        if ref is None:
            return None
        try:
            with self.cas.object_path(ref).open("rb") as handle:
                magic = handle.read(4)
        except (OSError, ValueError):
            return None  # unreadable-at-rest is the fetch path's verdict
        if magic[:2] == b"\x1f\x8b":
            return None
        if magic == b"PK\x03\x04":
            return (
                "stored blob is a bare AOTI .pt2 package (ZIP), not the "
                "compiled-graph envelope — published by a pre-envelope "
                "publisher; re-publish"
            )
        return "stored blob is not a compiled-graph envelope (unknown format)"

    def fetch_artifact(self, graph: str, env: EnvIdentity, destination: str | Path) -> Path | None:
        ref = self.cas.read_ref(_artifact_ref(graph, env))
        if ref is None:
            return None
        warm = _stamped(Path(destination), ref)
        if warm is not None:
            return warm
        try:
            blob = self.cas.verify_object(ref)
        except (DigestMismatch, FileNotFoundError, ValueError) as exc:
            raise StoreError(
                f"artifact ({graph}, {env.value}) failed CAS verification: {exc}"
            ) from exc
        fetched = self._copy_out(blob, destination)
        _write_stamp(fetched, ref)
        return fetched

    def _copy_out(self, blob: str | Path, destination: str | Path) -> Path:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            shutil.copyfile(blob, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return target

    def publish_artifact(
        self,
        graph: str,
        env: EnvIdentity,
        artifact: str | Path,
        manifest: RequirementsManifest,
    ) -> PublishOutcome:
        source = Path(artifact)
        if not source.is_file():
            raise StoreError(f"artifact {source} is not a file")
        if manifest.sm_compiled != env.sm:
            raise StoreError(
                f"manifest states sm {manifest.sm_compiled!r} but the publish "
                f"position states {env.sm!r}; the mint wrote one of them wrong"
            )
        candidate = self._admit_file(source)
        manifest_candidate = self.cas.put_bytes(manifest.encode())
        ref_name = _artifact_ref(graph, env)
        current = self.cas.read_ref(ref_name)
        if current is not None:
            if current == candidate:
                self._ensure_manifest(graph, env, manifest_candidate)
                return PublishOutcome.PRESENT
            if self._envelope_supersedes(current, source):
                # tcg#75 / pgw#1561: the occupied position holds bytes NO
                # reader of this store can load — the bare `.pt2` package a
                # pre-envelope publisher banked. That is not a diverged mint;
                # it is a shape this store's own loaders refuse by type
                # (`ArtifactFormatSkew`), so under tcg#69's doctrine the
                # incumbent is ABSENT-equivalent derived bytes and an envelope
                # replaces it. A real envelope-vs-envelope divergence still
                # refuses below.
                try:
                    self.cas.compare_and_swap_ref(ref_name, candidate, expected=current)
                except RefConflict:
                    if self.cas.read_ref(ref_name) != candidate:
                        raise StoreError(
                            f"artifact position ({graph}, {env.value}) moved "
                            f"during envelope repair"
                        ) from None
                reclaimed = self._discard_object(current)
                logger.warning(
                    "torchcg: REPLACED artifact (%s, %s) — the position held a "
                    "non-envelope blob (a bare package from a pre-envelope "
                    "publisher, unloadable by every reader of this store); the "
                    "envelope now stands. Reclaimed %d byte(s).",
                    graph, env.value, reclaimed,
                )
                self._ensure_manifest(graph, env, manifest_candidate)
                return PublishOutcome.PUBLISHED
            raise StoreError(
                f"artifact position ({graph}, {env.value}) already holds "
                f"different bytes; a deterministic mint diverged"
            )
        try:
            self.cas.compare_and_swap_ref(ref_name, candidate, expected=None)
        except RefConflict:
            if self.cas.read_ref(ref_name) == candidate:
                self._ensure_manifest(graph, env, manifest_candidate)
                return PublishOutcome.PRESENT
            raise StoreError(
                f"artifact position ({graph}, {env.value}) already holds "
                f"different bytes; a deterministic mint diverged"
            ) from None
        self._ensure_manifest(graph, env, manifest_candidate)
        return PublishOutcome.PUBLISHED

    def _envelope_supersedes(self, incumbent: CASRef, candidate_file: Path) -> bool:
        """Whether ``candidate_file`` is a gzip envelope while the incumbent
        object is NOT — the one migration this store performs (tcg#75).

        Sniffed off magic bytes, never decoded here: the candidate is fully
        verified on the way out by every reader, and the incumbent needs only
        to be recognized as unloadable-by-shape. An incumbent whose object is
        MISSING counts too — a live ref over absent bytes fails every fetch,
        so replacing it with verified bytes is a repair, not an arbitration.
        """

        try:
            with Path(candidate_file).open("rb") as handle:
                if handle.read(2) != b"\x1f\x8b":
                    return False
        except OSError:
            return False
        try:
            with self.cas.object_path(incumbent).open("rb") as handle:
                return handle.read(2) != b"\x1f\x8b"
        except (OSError, ValueError):
            return True

    def _ensure_manifest(self, graph: str, env: EnvIdentity, candidate: CASRef) -> None:
        """First manifest wins; an interrupted earlier publish is repaired."""

        try:
            self.cas.compare_and_swap_ref(_manifest_ref(graph, env), candidate, expected=None)
        except RefConflict:
            # An existing manifest beside the same artifact bytes stands; the
            # mint that wrote it stated what it linked, and restating differs
            # only if a minter lied about a deterministic env.
            pass

    def get_manifest(self, graph: str, env: EnvIdentity) -> RequirementsManifest | None:
        ref = self.cas.read_ref(_manifest_ref(graph, env))
        if ref is None:
            return None
        try:
            blob = self.cas.verify_object(ref)
            raw = json.loads(Path(blob).read_bytes().decode("ascii"))
            return RequirementsManifest.decode(raw)
        except (
            DigestMismatch,
            FileNotFoundError,
            ValueError,
            RequirementsError,
        ) as exc:
            raise StoreError(f"manifest ({graph}, {env.value}) is unreadable: {exc}") from exc

    # -- transform side tables (tcg#52) -------------------------------------

    def has_program(self, graph: str) -> bool:
        return self.cas.read_ref(_program_ref(graph)) is not None

    def fetch_program(self, graph: str, destination: str | Path) -> Path | None:
        """This machine's serialized program for ``graph``, or None.

        None is an ordinary answer: a box that has not traced this endpoint
        yet simply has no programs, and the caller's remedy is to derive
        rather than to fetch from anywhere.
        """
        ref = self.cas.read_ref(_program_ref(graph))
        if ref is None:
            return None
        warm = _stamped(Path(destination), ref)
        if warm is not None:
            return warm
        try:
            blob = self.cas.verify_object(ref)
        except (DigestMismatch, FileNotFoundError, ValueError) as exc:
            raise StoreError(f"program for {graph} failed CAS verification: {exc}") from exc
        fetched = self._copy_out(blob, destination)
        _write_stamp(fetched, ref)
        return fetched

    def put_program(self, graph: str, path: str | Path) -> str:
        """Bank this machine's serialized program for ``graph``.

        LAST WRITE WINS, and this is deliberately NOT the side table's
        "divergence refuses" rule. Two traces of one graph on one machine
        produce identical bytes, but a torch upgrade in place legitimately
        produces different ones for the same graph identity, and refusing
        would strand the box on its first re-derive after an upgrade. The
        bytes are local and cheap to recreate; the identity is what is
        protected, and it is protected by the key.
        """
        source = Path(path)
        if not source.is_file():
            raise StoreError(f"program {source} is not a file")
        candidate = self._admit_file(source)
        ref_name = _program_ref(graph)
        while True:
            current = self.cas.read_ref(ref_name)
            if current == candidate:
                return str(candidate)
            try:
                self.cas.compare_and_swap_ref(ref_name, candidate, expected=current)
            except RefConflict:
                continue
            return str(candidate)

    def has_side_table(self, pass_name: str, key: str) -> bool:
        return self.cas.read_ref(_side_table_ref(pass_name, key)) is not None

    def fetch_side_table(self, pass_name: str, key: str, destination: str | Path) -> Path | None:
        ref = self.cas.read_ref(_side_table_ref(pass_name, key))
        if ref is None:
            return None
        try:
            blob = self.cas.verify_object(ref)
        except (DigestMismatch, FileNotFoundError, ValueError) as exc:
            raise StoreError(
                f"side table ({pass_name}, {key}) failed CAS verification: {exc}"
            ) from exc
        return self._copy_out(blob, destination)

    def publish_side_table(self, pass_name: str, key: str, path: str | Path) -> str:
        """First bucket wins; identical bytes are idempotent, divergence refuses.

        A side table is a deterministic function of (checkpoint, pass, domain
        key), so two buckets under one address is a bug to surface -- the same
        rule the artifact position runs under.
        """

        source = Path(path)
        if not source.is_file():
            raise StoreError(f"side table {source} is not a file")
        candidate = self._admit_file(source)
        ref_name = _side_table_ref(pass_name, key)
        current = self.cas.read_ref(ref_name)
        if current is not None:
            if current == candidate:
                return str(candidate)
            raise StoreError(
                f"side-table position ({pass_name}, {key}) already holds different "
                f"bytes; a deterministic pass diverged"
            )
        try:
            self.cas.compare_and_swap_ref(ref_name, candidate, expected=None)
        except RefConflict:
            if self.cas.read_ref(ref_name) != candidate:
                raise StoreError(
                    f"side-table position ({pass_name}, {key}) already holds "
                    f"different bytes; a deterministic pass diverged"
                ) from None
        return str(candidate)


__all__ = [
    "GraphStore",
    "LocalGraphStore",
    "PublishOutcome",
    "StoreError",
    "holes",
]
