"""Materialize the ONE artifact the hub named (the delivery half).

Identity and delivery are split on purpose: ``aot_identity`` compares the
DECLARED identities (never bytes, §4.25/§4.26), and this module does the one
byte-level check that legitimately exists — the delivered bytes must hash to
the content digest the hub named. There is no listing, no ranking and no
sibling to fall back to: the grant either carries the named digest or the
attempt refuses typed.

Refusals are :class:`NamedArtifactUnavailable` (a ``RetryableError``): nothing
about the caller's input is wrong — the hub re-issues a grant or re-routes —
and the message ties the failure to the attempt and the spec digest so the
fleet event is actionable.
"""

from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Any, Optional

from gen_worker._vendor.tensorfs import (
    CASRef,
    Chunk,
    DigestMismatch,
    FileEntry,
)
from gen_worker.transfer.grants import TransferGrant, download
from gen_worker._vendor.torchcg import (
    ARTIFACT_KIND,
    StoreOutcome,
    is_compiled_graph_key,
)

from . import boot_phases, receipts
from .api.errors import RetryableError
from .compile_cache import parse_cell_ref
from .models.cache_paths import open_worker_cas, open_worker_engine

logger = logging.getLogger(__name__)

class NamedArtifactUnavailable(RetryableError):
    """The named cell's bytes could not be materialized AS NAMED."""

    def __init__(self, reason: str, detail: str) -> None:
        self.reason = reason
        super().__init__(f"{reason}: {detail}")


def _import_verified_artifact(
    artifact: Path,
    *,
    cell_ref: str,
    cache_dir: Optional[Path],
    what: str,
) -> None:
    """Bind verified delivery bytes to the one exact TCG key they declare."""

    family, named_key = parse_cell_ref(cell_ref)
    if not is_compiled_graph_key(named_key):
        raise NamedArtifactUnavailable(
            "artifact_unpinned",
            f"{what}: named ref {cell_ref!r} carries no compiled-graph key",
        )
    if not receipts.gate_delivered_artifact(artifact, family):
        raise NamedArtifactUnavailable(
            "artifact_receipt_refused",
            f"{what}: the hub receipt did not authorize {cell_ref!r}",
        )
    try:
        result = open_worker_engine(cache_dir).import_artifact(named_key, artifact)
    except Exception as exc:
        raise NamedArtifactUnavailable(
            "artifact_admission_failed",
            f"{what}: TCG refused {cell_ref!r}: {type(exc).__name__}: {exc}",
        ) from exc
    if result.outcome == StoreOutcome.DIVERGENT:
        raise NamedArtifactUnavailable(
            "artifact_divergent",
            f"{what}: local TCG already binds {named_key!r} to different bytes",
        )


def materialize_named_artifact(
    cell_ref: str,
    content_digest: str,
    presigned: Optional[Any],
    *,
    cache_dir: Optional[Path],
    what: str,
) -> Path:
    """Fetch (or reuse) the artifact tarball for exactly ``content_digest``.

    ``presigned`` is the grant's ``PresignedSnapshot`` for that digest, or
    ``None`` when the grant does not carry it. The cache is keyed by content
    digest, so a re-dispatch of the same spec is a verify-and-return.
    ``what`` names the attempt + spec digest for every refusal.
    """
    # THE cell-download phase. Without a producer here, an adopt leg cannot be
    # split into download vs admission — the first question anyone asks about
    # an adopt. This is the one place cell bytes move, so it is the one place
    # the span belongs.
    with boot_phases.span(
        boot_phases.PHASE_CELL_FETCH,
        ref=cell_ref,
        artifact_kind=ARTIFACT_KIND,
        artifact_key=str(content_digest or ""),
    ) if boot_phases.in_boot() else contextlib.nullcontext() as fetch_span:
        return _materialize_named_artifact(
            cell_ref, content_digest, presigned,
            cache_dir=cache_dir, what=what, span=fetch_span)


def _materialize_named_artifact(
    cell_ref: str,
    content_digest: str,
    presigned: Optional[Any],
    *,
    cache_dir: Optional[Path],
    what: str,
    span: Optional["boot_phases.BootSpan"] = None,
) -> Path:
    digest = str(content_digest or "").strip()
    if not digest:
        raise NamedArtifactUnavailable(
            "artifact_unpinned", f"{what}: named cell {cell_ref!r} carries no "
            "content digest")
    cas = open_worker_cas(cache_dir)
    # A bounded transfer staging path, never a second compiled-graph store.
    # The serve handoff removes it after TCG has resolved its CAS record.
    dest_dir = cas.root / "compiled-graph-transfer" / ".incoming"
    dest = dest_dir / f"{digest.split(':', 1)[-1]}.tar.gz"
    try:
        expected = CASRef.parse(digest)
    except ValueError as exc:
        raise NamedArtifactUnavailable("artifact_unpinned", f"{what}: {exc}") from exc
    if dest.is_file():
        try:
            cas.put_file(dest, expected=expected, size=dest.stat().st_size)
        except (DigestMismatch, OSError):
            dest.unlink(missing_ok=True)  # stale cache — refetch below
        else:
            # A cache hit is a real and countable outcome of this phase, not a
            # zero-byte fetch: the verify pass still reads the whole tarball.
            if span is not None:
                span.classify("cached", f"ref={cell_ref}")
                span.bytes_moved(dest.stat().st_size, boot_phases.SOURCE_LOCAL)
            _import_verified_artifact(
                dest, cell_ref=cell_ref, cache_dir=cache_dir, what=what
            )
            return dest

    if presigned is None or not list(getattr(presigned, "files", ())):
        raise NamedArtifactUnavailable(
            "missing_content",
            f"{what}: the grant carries no transport for the named cell "
            f"{cell_ref!r} (content digest {digest})")
    files = list(presigned.files)
    entry = next(
        (f for f in files if str(f.path or "").endswith(".tar.gz")),
        files[0] if len(files) == 1 else None)
    if entry is None:
        raise NamedArtifactUnavailable(
            "missing_content",
            f"{what}: the grant's transport for {cell_ref!r} names no "
            "artifact tarball")

    declared_size = int(entry.size_bytes or 0)
    remote_chunks = list(entry.chunks or ())
    if not remote_chunks and declared_size <= 0:
        # A cell artifact is compiled code fetched before serving; an entry
        # that cannot say how big it is cannot be sized against disk
        #  — refuse rather than an unbounded write.
        raise NamedArtifactUnavailable(
            "missing_content",
            f"{what}: transport for {cell_ref!r} declares no size_bytes")

    try:
        chunks = tuple(
            Chunk(CASRef.parse(f"sha256:{chunk.sha256}"), int(chunk.len))
            for chunk in remote_chunks
        )
        file_entry = FileEntry(dest.name, declared_size, expected, chunks)
        grants = (
            tuple(
                TransferGrant(chunk.digest, chunk.length, remote.url)
                for chunk, remote in zip(chunks, remote_chunks, strict=True)
            )
            if chunks
            else (TransferGrant(expected, declared_size, str(entry.url or "")),)
        )
    except ValueError as exc:
        raise NamedArtifactUnavailable(
            "artifact_fetch_failed",
            f"{what}: named cell {cell_ref!r} has invalid tensorfs transport: {exc}",
        ) from exc
    try:
        report = download(grants, cas)
        if not report.ok:
            failures = "; ".join(detail for _digest, detail in report.failures)
            raise RuntimeError(failures or "artifact grants expired")
        dest_dir.mkdir(parents=True, exist_ok=True)
        # A compiled-graph artifact leaving the tensorfs store for torchcg's
        # own cell store. Priced under §9's `tcg artifact export off-store`
        # row: the destination is a different store's file, not a path inside
        # a projected snapshot, so there is nothing here to point a stub at.
        # The pinned rev spells the single-file hatch `materialize`; §9 and
        # current upstream spell it `extract`.
        cas.materialize(file_entry, dest)  # mixed-cas-hatch: tcg-artifact-export
    except NamedArtifactUnavailable:
        raise
    except (ValueError, RuntimeError, OSError) as exc:
        dest.unlink(missing_ok=True)
        raise NamedArtifactUnavailable(
            "content_digest_mismatch" if isinstance(exc, DigestMismatch)
            else "artifact_fetch_failed",
            f"{what}: named cell {cell_ref!r} bytes refused against "
            f"{digest[:24]}…: {exc}") from exc
    if span is not None:
        span.classify("fetched", f"ref={cell_ref} chunks={len(remote_chunks)}")
        span.bytes_moved(declared_size or dest.stat().st_size,
                         boot_phases.SOURCE_CAS)
    _import_verified_artifact(
        dest, cell_ref=cell_ref, cache_dir=cache_dir, what=what
    )
    return dest


__all__ = ["NamedArtifactUnavailable", "materialize_named_artifact"]
