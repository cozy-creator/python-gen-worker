"""th#1941 — the wire's snapshot map is keyed by COMPOSED MANIFEST DIGEST.

The hub composes each slot's manifest before it dispatches: `ModelBinding`
carries `manifest_digest`, and `DesiredResidency.snapshots` / `RunJob.snapshots`
are keyed by it. Two resolutions of one repo that differ in their component
sources are two entries with two digests and CANNOT interfere — which is the
whole point of the re-key, and the reason the worker no longer derives any
exclusion of its own.

The worker's residency, event and GC key is still the canonical ref, so this
module is the ONE place the two spellings meet: it resolves each binding's
`manifest_digest` to its snapshot and files it under that binding's ref.

A ref that two bindings claim at DIFFERENT digests is refused, not guessed —
that ambiguity is exactly th#1933's shape, and a worker that picked one of the
two would be re-deriving the mutation this deletes.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping

from .api.errors import ValidationError
from .models.refs import WireRef


class AmbiguousManifestError(ValidationError):
    """One ref arrived bound to two different composed manifests in a single
    message. Deterministic: no retry changes it."""


def resolved_repo_from_snapshot(snap: Any) -> Any:
    """``pb.Snapshot`` -> the typed ``WorkerResolvedRepo`` the download layer
    speaks: the ONE wire-boundary conversion. Everything downstream
    (``ensure_local``, ``ensure_snapshot_async``) is typed — no dict laundering.

    DIRECT FIELD ACCESS on ``digest``/``chunks``, deliberately — not
    ``getattr(f, "digest", "")``. These were read defensively once, and the
    default turned "the generated stub does not have this field" into "the hub
    sent an empty value": the vendored proto WAS stale, so every v2 snapshot
    arrived blank on the production gRPC path and nothing said why.
    """
    from .models.hub_client import (
        WorkerResolvedChunk,
        WorkerResolvedRepo,
        WorkerResolvedRepoFile,
    )

    return WorkerResolvedRepo(
        snapshot_digest=snap.digest,
        files=[
            WorkerResolvedRepoFile(
                path=f.path,
                size_bytes=int(f.size_bytes),
                url=f.url or None,
                digest=f.digest or "",
                chunks=tuple(
                    WorkerResolvedChunk(
                        sha256=(c.sha256 or "").strip().lower(),
                        url=c.url,
                        length=int(c.len),
                    )
                    for c in f.chunks
                ),
            )
            for f in snap.files
        ],
    )


def snapshot_from_resolved_repo(resolved: Any) -> Any:
    """``WorkerResolvedRepo`` -> ``pb.Snapshot``: the INVERSE, and it lives
    here for the reason this module exists — one place the two spellings meet.

    pgw#1491. The CLI resolves a checkpoint over HTTP
    (``hub_client.resolve_repo``, which answers ``WorkerResolvedRepo``) and
    then hands it to the SAME ``ModelStore.ensure_local`` a pod's boot uses,
    whose parameter is the gRPC spelling. Without this the CLI would need its
    own materializer, which is the two-paths-rot rule's exact shape: two
    downloaders that both produce trees and disagree about integrity.

    ``provenance`` is deliberately not synthesized: the proto calls it
    audit/display ONLY and nothing downstream reads it, so inventing one here
    would be manufacturing provenance the hub never stated.
    """
    from .pb import worker_scheduler_pb2 as pb

    return pb.Snapshot(
        digest=resolved.snapshot_digest,
        files=[
            pb.SnapshotFile(
                path=f.path,
                size_bytes=int(f.size_bytes),
                url=f.url or "",
                digest=f.digest or "",
                chunks=[
                    pb.ChunkRef(
                        sha256=c.sha256,
                        url=c.url,
                        len=int(c.length),
                    )
                    for c in (f.chunks or ())
                ],
            )
            for f in resolved.files
        ],
    )


def resolved_repos(
    wire: Mapping[str, Any],
    bindings: Iterable[Any] = (),
) -> Dict[str, Any]:
    """:func:`index_snapshots`, with every entry already converted to the
    typed ``WorkerResolvedRepo``. The shape a materializer wants.

    Keys are plain ``str`` — still canonical refs, but the serve layer holds
    no ``WireRef`` vocabulary and a NewType key would make an invariant
    ``Mapping`` refuse it there for no gain."""
    return {
        str(ref): resolved_repo_from_snapshot(snap)
        for ref, snap in index_snapshots(wire, bindings).items()
    }


def index_snapshots(
    wire: Mapping[str, Any],
    bindings: Iterable[Any] = (),
) -> Dict[WireRef, Any]:
    """Wire snapshot map -> the ref-keyed view the worker materializes from.

    Every binding's ref resolves to `wire[binding.manifest_digest]`. Entries no
    binding claimed (LoRA overlays, payload source refs — artifacts with no
    composition, which the hub keys by ref) pass through unchanged.
    """
    claimed: set[str] = set()
    out: Dict[WireRef, Any] = {}
    seen: Dict[str, str] = {}
    for binding in bindings or ():
        ref = str(getattr(binding, "ref", "") or "").strip()
        digest = str(getattr(binding, "manifest_digest", "") or "").strip()
        if not ref or not digest:
            continue
        prior = seen.get(ref)
        if prior is not None and prior != digest:
            raise AmbiguousManifestError(
                f"ref {ref!r} is bound to two composed manifests in one "
                f"message ({prior} and {digest}); the hub must send one "
                f"manifest per ref per message"
            )
        seen[ref] = digest
        claimed.add(digest)
        snapshot = wire.get(digest)
        if snapshot is not None:
            out[WireRef(ref)] = snapshot
    for key, snapshot in wire.items():
        if key not in claimed:
            out.setdefault(WireRef(key), snapshot)
    return out


__all__ = [
    "AmbiguousManifestError",
    "index_snapshots",
    "resolved_repo_from_snapshot",
    "resolved_repos",
]
