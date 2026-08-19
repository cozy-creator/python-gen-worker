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
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

from tensorfs import CASRef, DigestMismatch, LocalCAS, RefConflict

from .document import DocumentError, GraphSetDocument, LaneGraphs
from .graph_identity import EnvIdentity, is_graph_hash
from .lane import LaneError, require_pass_ref
from .requirements import RequirementsError, RequirementsManifest

_REF_PREFIX = "torchcg/v2"
_READ_BUFFER = 1 << 20


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

    def fetch_artifact(
        self, graph: str, env: EnvIdentity, destination: str | Path
    ) -> Path | None:
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
        record.graph
        for record in lane.graphs
        if not store.has_artifact(record.graph, env)
    )


def _require_graph(graph: str) -> str:
    if not is_graph_hash(graph):
        raise StoreError(f"store rows are addressed by cg-graph-v1 hashes, got {graph!r}")
    return graph


def _document_ref(name: str) -> str:
    clean = str(name).strip()
    if not clean or "\\" in clean or any(
        part in ("", ".", "..") for part in clean.split("/")
    ):
        raise StoreError(f"unsafe graph-set name {name!r}")
    return f"{_REF_PREFIX}/graphsets/{clean}"


def _artifact_ref(graph: str, env: EnvIdentity) -> str:
    return f"{_REF_PREFIX}/artifacts/{_require_graph(graph)}/{env.value}"


def _manifest_ref(graph: str, env: EnvIdentity) -> str:
    return f"{_REF_PREFIX}/manifests/{_require_graph(graph)}/{env.value}"


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
        raise StoreError(
            f"a side-table key is the canonical hex domain-key address, got {key!r}"
        )
    return f"{_REF_PREFIX}/sidetables/{pass_name}/{key}"


def _digest_file(path: Path) -> CASRef:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_READ_BUFFER):
            digest.update(chunk)
    return CASRef(digest.hexdigest())


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
        ref = self.cas.read_ref(_document_ref(name))
        if ref is None:
            return None
        try:
            blob = self.cas.verify_object(ref)
            return GraphSetDocument.decode(Path(blob).read_bytes())
        except (DigestMismatch, FileNotFoundError, ValueError, DocumentError) as exc:
            raise StoreError(f"graph-set document {name!r} is unreadable: {exc}") from exc

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

    def fetch_artifact(
        self, graph: str, env: EnvIdentity, destination: str | Path
    ) -> Path | None:
        ref = self.cas.read_ref(_artifact_ref(graph, env))
        if ref is None:
            return None
        try:
            blob = self.cas.verify_object(ref)
        except (DigestMismatch, FileNotFoundError, ValueError) as exc:
            raise StoreError(
                f"artifact ({graph}, {env.value}) failed CAS verification: {exc}"
            ) from exc
        return self._copy_out(blob, destination)

    def _copy_out(self, blob: str | Path, destination: str | Path) -> Path:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{target.name}.", dir=target.parent
        )
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

    def _ensure_manifest(self, graph: str, env: EnvIdentity, candidate: CASRef) -> None:
        """First manifest wins; an interrupted earlier publish is repaired."""

        try:
            self.cas.compare_and_swap_ref(
                _manifest_ref(graph, env), candidate, expected=None
            )
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
            raise StoreError(
                f"manifest ({graph}, {env.value}) is unreadable: {exc}"
            ) from exc

    # -- transform side tables (tcg#52) -------------------------------------

    def has_side_table(self, pass_name: str, key: str) -> bool:
        return self.cas.read_ref(_side_table_ref(pass_name, key)) is not None

    def fetch_side_table(
        self, pass_name: str, key: str, destination: str | Path
    ) -> Path | None:
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
