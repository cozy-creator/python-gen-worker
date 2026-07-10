"""Publish ProducedFlavor outputs to Tensorhub — THE producer publish contract.

A conversion / dataset / training endpoint writes files locally, calls
``publish_flavors(ctx, flavors)``, and returns a result struct. Each flavor's
``path`` (file or directory) becomes ONE Tensorhub commit against the
destination repo (explicit ``destination_repo=`` or the job payload's
reserved ``destination.repo`` field). Nothing publishes implicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from ..models.ladder import (
    CLASS_BASE,
    Placement,
    classify_flavor_token,
    default_placement,
    placement_to_metadata,
)
from .hub import CommitFile, CommitResult, HubClient, files_from_tree
from .produced import ProducedFlavor

_PLACEMENT_ATTR_KEYS = ("placement_sm_allowed", "placement_sm_min", "placement_engines")


def _csv_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(v) for v in (s.strip() for s in raw.split(",")) if v)


def _csv_strs(raw: str) -> tuple[str, ...]:
    return tuple(v for v in (s.strip() for s in raw.split(",")) if v)


def _placement_block(attrs: Mapping[str, str], label: str) -> dict[str, Any] | None:
    """th#697: the placement stamp — arch requirements the SKU-aware
    precision ladder reads back at resolution. Derived from the flavor
    token's class defaults; producers may override via explicit
    ``precision_class`` / ``placement_*`` attrs. Base rows stay unstamped
    (bare = runs wherever it fits, by definition)."""
    explicit = any(attrs.get(k) for k in _PLACEMENT_ATTR_KEYS)
    cls = str(attrs.get("precision_class", "") or "").strip().lower()
    cls = cls or classify_flavor_token(label)
    if not cls or (cls == CLASS_BASE and not explicit):
        return None
    p = default_placement(cls) or Placement(cls)
    if explicit:
        p = Placement(
            cls,
            sm_allowed=_csv_ints(attrs.get("placement_sm_allowed", "")) or p.sm_allowed,
            sm_min=int(attrs.get("placement_sm_min", "") or 0) or p.sm_min,
            engines=_csv_strs(attrs.get("placement_engines", "")) or p.engines,
        )
    return placement_to_metadata(p)


def _flavor_files(flavor: ProducedFlavor) -> list[CommitFile]:
    path = Path(flavor.path)
    if path.is_dir():
        files = files_from_tree(path)
    elif path.is_file():
        files = [CommitFile(path=path.name, local_path=path)]
    else:
        raise FileNotFoundError(f"ProducedFlavor.path does not exist: {path}")
    for extra in flavor.extra_files or []:
        p = Path(extra)
        if p.is_file():
            files.append(CommitFile(path=p.name, local_path=p))
    return files


def publish_flavors(
    ctx: Any,
    flavors: Iterable[ProducedFlavor],
    *,
    destination_repo: str = "",
    tags: Iterable[str] | None = None,
    mode: str = "replace",
    metadata: Mapping[str, Any] | None = None,
) -> list[CommitResult]:
    """Publish each ProducedFlavor as one commit. ``destination_repo`` falls
    back to the reserved-name ``ctx.destination`` payload field.

    ``mode`` defaults to ``"replace"`` (th#597 C2): a producer's flavor
    export is a complete tree by definition — merging with the repo's prior
    :latest is how te#44 shipped an #fp8 checkpoint carrying 5.2GB of fp16
    base weights. Pass ``mode="merge"`` explicitly only for deliberate
    overlay publishes (e.g. a vae swap on top of an existing tree)."""
    dest = str(destination_repo or "").strip()
    if not dest:
        info = getattr(ctx, "destination", None) or {}
        dest = str((info.get("repo") if isinstance(info, dict) else "") or "").strip()
    if not dest:
        raise ValueError("destination_repo is required (payload.destination.repo)")

    client = HubClient.from_ctx(ctx)
    results: list[CommitResult] = []
    for flavor in flavors:
        attrs = {str(k): str(v) for k, v in (flavor.attributes or {}).items()}
        label = str(flavor.flavor or attrs.get("flavor") or attrs.get("dtype") or "").strip()
        # th#606: worker-addable provenance stamp fields. Producers declare
        # quant identity in the flavor attribute bag; it rides the commit's
        # `provenance` object onto the checkpoint's node stamp (parents /
        # derivation_op come from the orchestrator's token claim, never here).
        provenance = {
            k: attrs[k]
            for k in ("quantization_method", "quantization_library")
            if attrs.get(k)
        }
        placement = _placement_block(attrs, label)
        meta = {**(dict(metadata) if metadata else {}), **attrs}
        for k in _PLACEMENT_ATTR_KEYS:
            meta.pop(k, None)
        if placement:
            meta["placement"] = placement
        results.append(client.commit(
            destination_repo=dest,
            files=_flavor_files(flavor),
            tags=list(tags or []),
            mode=mode,
            flavor=label,
            flavors=list(flavor.flavors or []),
            dtype=attrs.get("dtype", ""),
            file_layout=attrs.get("file_layout", ""),
            file_type=attrs.get("file_type", ""),
            metadata=meta,
            provenance=provenance,
        ))
    return results


__all__ = ["publish_flavors"]
