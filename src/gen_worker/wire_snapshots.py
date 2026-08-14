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


__all__ = ["AmbiguousManifestError", "index_snapshots"]
