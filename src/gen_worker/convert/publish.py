"""Publish ProducedFlavor outputs to Tensorhub — THE producer publish contract.

A conversion / dataset / training endpoint writes files locally, calls
``publish_flavors(ctx, flavors)``, and returns a result struct. Each flavor's
``path`` (file or directory) becomes ONE Tensorhub publish against the
destination repo (explicit ``destination_repo=`` or the job payload's
reserved ``destination.repo`` field). Nothing publishes implicitly.

Publishes over the chunked sha256 CAS (th#1303 ``HubClient.publish_v2``).
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Iterable, Mapping

from .. import activity as _activity
from ..models.ladder import (
    CLASS_BASE,
    Placement,
    classify_flavor_token,
    default_placement,
    placement_to_metadata,
)
from .hub import CommitFile, CommitResult, HubClient, files_from_tree
from .publish_journal import JOURNAL_NAME
from .produced import ProducedFlavor
from .writer import assert_one_file_per_component

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


def _source_stamps(ctx: Any, client: HubClient) -> tuple[str | None, bool | None]:
    """th#1411 restatement default: the pgw#654 stamps of the checkpoint being
    converted, read from the hub's resolve of ``ctx.source``. Best-effort —
    on any failure the publish proceeds unstamped and the hub's
    classification gate stays the enforcement."""
    info = getattr(ctx, "source", None) or {}
    ref = str((info.get("ref") if isinstance(info, dict) else "") or "").strip()
    if not ref:
        return None, None
    try:
        from ..models.hub_client import resolve_repo
        from ..models.refs import parse_model_ref

        th = parse_model_ref(ref).tensorhub
        if th is None:
            return None, None
        resolved = resolve_repo(th, base_url=client.base_url, token=client.token)
        # A new hub tells us whether false is evidence or merely the wire
        # default. Never turn unknown/inconclusive into an authored false on
        # the derived checkpoint. Empty status is an old hub, whose behavior
        # remains unchanged for rolling compatibility.
        distilled = (
            resolved.distilled
            if resolved.distilled_status in ("", "classified")
            else None
        )
        return resolved.objective, distilled
    except Exception as exc:
        log = getattr(ctx, "log", None)
        if callable(log):
            log(f"source-stamp read failed ({exc}); "
                "publishing without restated classification")
        return None, None


def _journal_beside(flavor: ProducedFlavor) -> Path:
    """The publish journal for one produced flavor (pgw#1003).

    NEXT TO the tree, never inside it: ``files_from_tree`` walks a flavor
    directory wholesale, and a journal written into it would publish itself as
    repo content on the next flavor. One journal per produced-output directory
    also means one file holding every flavor's in-flight session, which is
    exactly the set a successor should try to resume.
    """
    return Path(flavor.path).parent / JOURNAL_NAME


def _publish_leg(dest: str, label: str, stage: str, facts: Mapping[str, Any]) -> None:
    """One typed `convert_publish` event per LEG of the publish protocol.

    pgw#1004 B: `publish_flavors` is what every quantize / fuse / cast job
    calls — the 2h16m casts — and it used to pass no ``on_stage``, no
    ``progress`` and no ``part_progress``, so the highest-volume producer on
    the platform emitted ZERO `worker_activity_events` legs for its whole
    publish. "Declared 590 objects and is moving 37 GB" and "was refused
    before a byte left" were the same observation. Modelled on
    ``fleet_cells._publish_leg``, which already does this correctly.
    """
    detail = " ".join(f"{k}={v}" for k, v in sorted(dict(facts).items()))
    _activity.emit_event(
        "convert_publish", f"repo={dest} flavor={label}: {detail}", phase=stage)


def publish_flavors(
    ctx: Any,
    flavors: Iterable[ProducedFlavor],
    *,
    destination_repo: str = "",
    tags: Iterable[str] | None = None,
    mode: str = "replace",
    metadata: Mapping[str, Any] | None = None,
    objective: str | None = None,
    distilled: bool | None = None,
    journal_path: Path | None = None,
) -> list[CommitResult]:
    """Publish each ProducedFlavor as one commit. ``destination_repo`` falls
    back to the reserved-name ``ctx.destination`` payload field.

    ``mode`` defaults to ``"replace"`` (th#597 C2): a producer's flavor
    export is a complete tree by definition — merging with the repo's prior
    :latest is how te#44 shipped an #fp8 checkpoint carrying 5.2GB of fp16
    base weights. Pass ``mode="merge"`` explicitly only for deliberate
    overlay publishes (e.g. a vae swap on top of an existing tree).

    ``journal_path`` (pgw#1003) is where the in-flight ``publish_id`` is
    recorded so a retry on this pod re-uploads instead of re-casting. Pass the
    produced tree's own directory; omit it and the publish is exactly as
    unrecoverable as it was before."""
    dest = str(destination_repo or "").strip()
    if not dest:
        info = getattr(ctx, "destination", None) or {}
        dest = str((info.get("repo") if isinstance(info, dict) else "") or "").strip()
    if not dest:
        raise ValueError("destination_repo is required (payload.destination.repo)")

    client = HubClient.from_ctx(ctx)
    # th#1411: a v2 publish mints a new identity and inherits nothing, so a
    # publish into a classified repo must restate objective/distilled. Default:
    # restate the SOURCE checkpoint's hub stamps — this producer just derived
    # the flavors from exactly that source, and quantize/fuse/cast preserve
    # objective/distillation, so the restatement is a first-hand declaration,
    # not the silent inheritance th#1400 forbids. Explicit caller values win.
    if objective is None or distilled is None:
        src_objective, src_distilled = _source_stamps(ctx, client)
        if objective is None:
            objective = src_objective
        if distilled is None:
            distilled = src_distilled
    results: list[CommitResult] = []
    for flavor in flavors:
        # th#1362 item 4: OUR producers never emit shards, and this is the
        # last place a conversion / training-promote / cell-publish output can
        # be checked before it becomes somebody's checkpoint. It is NOT a
        # universal publish gate — a user's own sharded upload never reaches
        # this function; it goes to the hub's upload API and is accepted as
        # given. Checked rather than assumed because save_pretrained shards on
        # its own.
        assert_one_file_per_component(
            Path(flavor.path), producer=f"publish_flavors[{dest}]")
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
        # th#1303 phase 3, producer class 2 of N: the CONVERSION producer.
        # `publish_flavors` is what every quantize / fuse / cast / distil job
        # calls, so it is the highest-volume publisher after the mirror — and
        # it is in the SAME class the mirror flip already crossed (clone.py's
        # full-clone arm), which is why it goes second and not last.
        #
        # Safe here for the reason v2 requires: every file is a real local
        # file (`_flavor_files` walks the produced tree), so each digest is
        # PROVEN from bytes in hand rather than asserted. There is no
        # auto-select and no env knob — naming the protocol at the call site
        # is what makes "which producers are on v2 today?" answerable by
        # reading the code instead of by sampling traffic.
        #
        # A `merge` onto a prior blake3 manifest is a typed refusal
        # (`mixed_algorithm_manifest`), not a silent partial; `mode` here
        # defaults to "replace" (th#597 C2), so the common path is unaffected.
        results.append(client.publish_v2(
            destination_repo=dest,
            files=_flavor_files(flavor),
            tags=list(tags or []),
            mode=mode,
            # pgw#1004 B: the conversion producer now puts its own legs on the
            # wire. (The per-object liveness beat lives in the data plane
            # itself — `chunk_upload._beat` — so it cannot be lost by a caller
            # who forgets to pass a callback, which is exactly how it was
            # lost here.)
            on_stage=functools.partial(_publish_leg, dest, label),
            journal_path=journal_path or _journal_beside(flavor),
            flavor=label,
            flavors=list(flavor.flavors or []),
            dtype=attrs.get("dtype", ""),
            file_layout=attrs.get("file_layout", ""),
            file_type=attrs.get("file_type", ""),
            objective=str(objective or ""),
            distilled=distilled,
            metadata=meta,
            provenance=provenance,
        ))
    return results


__all__ = ["publish_flavors"]
