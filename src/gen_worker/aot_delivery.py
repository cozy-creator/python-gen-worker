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

import requests

from . import boot_phases
from .api.errors import RetryableError
from .bounded_stream import copy_bounded
from .models.chunk_cas import (
    CAS_CHUNK_SIZE_BYTES,
    ChunkSpec,
    DigestMismatch,
    download_chunked_file,
    verify_file_digest,
)

logger = logging.getLogger(__name__)

_DOWNLOAD_TIMEOUT_S = 120


class NamedArtifactUnavailable(RetryableError):
    """The named cell's bytes could not be materialized AS NAMED."""

    def __init__(self, reason: str, detail: str) -> None:
        self.reason = reason
        super().__init__(f"{reason}: {detail}")


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
        artifact_kind="aot-inductor",
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
    dest_dir = (
        Path(cache_dir) if cache_dir else Path.home() / ".cache" / "gen-worker"
    ) / "aot-cells"
    dest = dest_dir / f"{digest.split(':', 1)[-1]}.tar.gz"

    if dest.is_file():
        try:
            verify_file_digest(dest, digest)
        except DigestMismatch:
            dest.unlink(missing_ok=True)  # stale cache — refetch below
        except ValueError as exc:
            raise NamedArtifactUnavailable(
                "artifact_unpinned", f"{what}: {exc}") from exc
        else:
            # A cache hit is a real and countable outcome of this phase, not a
            # zero-byte fetch: the verify pass still reads the whole tarball.
            if span is not None:
                span.classify("cached", f"ref={cell_ref}")
                span.bytes_moved(dest.stat().st_size, boot_phases.SOURCE_LOCAL)
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
    chunks = list(entry.chunks or ())
    if not chunks and declared_size <= 0:
        # A cell artifact is compiled code fetched before serving; an entry
        # that cannot say how big it is cannot be sized against disk
        #  — refuse rather than an unbounded write.
        raise NamedArtifactUnavailable(
            "missing_content",
            f"{what}: transport for {cell_ref!r} declares no size_bytes")

    dest_dir.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    try:
        if chunks:
            # Every chunk's sha256 AND the whole-file digest are verified
            # inside the commit stream — one pass, no second read.
            download_chunked_file(
                [ChunkSpec(sha256=c.sha256, url=c.url, length=int(c.len))
                 for c in chunks],
                tmp,
                whole_digest=digest,
                total_size=declared_size,
                chunk_size_bytes=(
                    int(entry.chunk_size_bytes or 0) or CAS_CHUNK_SIZE_BYTES),
            )
        else:
            url = str(entry.url or "")
            if not url:
                raise ValueError("transport entry has no URL")
            with requests.get(url, timeout=_DOWNLOAD_TIMEOUT_S, stream=True) as dl:
                dl.raise_for_status()
                with open(tmp, "wb") as f:
                    got = copy_bounded(
                        dl.iter_content(1 << 20), f.write,
                        limit_bytes=declared_size,
                        what=f"named cell artifact {cell_ref}")
            if got != declared_size:
                raise ValueError(
                    f"truncated: transport declares {declared_size} bytes, "
                    f"got {got}")
            verify_file_digest(tmp, digest)
    except NamedArtifactUnavailable:
        tmp.unlink(missing_ok=True)
        raise
    except (ValueError, RuntimeError, requests.RequestException) as exc:
        tmp.unlink(missing_ok=True)
        raise NamedArtifactUnavailable(
            "content_digest_mismatch" if isinstance(exc, DigestMismatch)
            else "artifact_fetch_failed",
            f"{what}: named cell {cell_ref!r} bytes refused against "
            f"{digest[:24]}…: {exc}") from exc
    tmp.replace(dest)
    if span is not None:
        span.classify("fetched", f"ref={cell_ref} chunks={len(chunks)}")
        span.bytes_moved(declared_size or dest.stat().st_size,
                         boot_phases.SOURCE_CAS)
    return dest


__all__ = ["NamedArtifactUnavailable", "materialize_named_artifact"]
