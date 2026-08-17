"""Publish ProducedFlavor outputs to Tensorhub — THE producer publish contract.

A conversion / dataset / training endpoint writes files locally, calls
``publish_flavors(ctx, flavors)``, and returns a result struct. Each flavor's
``path`` (file or directory) becomes ONE Tensorhub publish against the
destination repo (explicit ``destination_repo=`` or the job payload's
reserved ``destination.ref`` field). Nothing publishes implicitly.

Publishes over the chunked sha256 CAS (``HubClient.publish_v2``).
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Iterable, Mapping

from .. import activity as _activity
from ..api.errors import ValidationError
from ..hubio.client import CommitFile, CommitResult, HubClient, files_from_tree
from ..hubio.publish_state import JOURNAL_NAME
from ..models.ladder import CLASS_BASE, PRECISION_CLASSES
from .dtype_pins import dtype_bits, verify_produced_tree
from .produced import ProducedFlavor
from .writer import assert_one_file_per_component
from ..models.file_layout import validate_file_layout

# pgw#1300 deleted the ``placement_*`` OVERRIDE SURFACE, not just its defaults:
# the block no longer carries an SM allow-list, an SM floor or an engine list,
# so there is nothing left to override. Producers still set these — e.g.
# `training-endpoints/conversion/src/conversion/quant/modelopt.py:1263` writes
# `placement_sm_allowed` — and an attribute pgw does not read must not land in
# `checkpoints.metadata` as prose. They are dropped here until those producers
# stop writing them; then this tuple goes too.
_DEAD_PLACEMENT_ATTRS = ("placement_sm_allowed", "placement_sm_min", "placement_engines")


#: A base row is bf16/fp16/fp32 — 16 bits or wider. Anything narrower is on a
#: quantization lane, and only its producer knows which one (fp4 bytes are
#: svdq-fp4 or nvfp4-w4a4 depending on how they were made).
_BASE_STORAGE_BITS = 16


class PrecisionClassRefusal(ValidationError):
    """A publish whose precision class cannot be recorded as the hub reads it.

    A18 / pgw#1319 replaced ``classify_flavor_token`` — which guessed the class
    from a producer-local label — with a DECLARATION. A guess that missed
    published a row with no class at all, and the hub's fallback for an
    unstamped row is `ClassBase`: an fp8 checkpoint served as though it were
    bf16, invisibly, for as long as nobody looked (te#225 found two such rows
    live). So the failure mode is now a refusal at the call site, before a byte
    moves, naming the attribute to declare.
    """


def _precision_class_block(
    attrs: Mapping[str, str], produced_dtypes: Mapping[str, str],
) -> dict[str, Any] | None:
    """The ONE surviving key of ``checkpoints.metadata["placement"]``.

    pgw#1300 / th#2055: card admission is gone — pod purchase depends only on
    the endpoint owner's (GPU, lane) ladder — but `precision_class` survives,
    because tensorhub's `precision.StoredPrecisionOf` reads it as its strongest
    evidence for a stored class where no tensor-layout contract is proven.

    It is the producer's own ``precision_class`` attribute and nothing else.
    Base rows stay unstamped: the hub's own fallback for an unstamped row is
    `ClassBase`, so a block would restate it. Two refusals stand where the
    token classifier used to guess — a class outside the vocabulary, and narrow
    bytes nobody classified. ``produced_dtypes`` is read off the published
    tree's own safetensors headers, so the second one is decided by the bytes
    rather than by anything the producer says about them.
    """
    cls = str(attrs.get("precision_class", "") or "").strip().lower()
    if cls and cls not in PRECISION_CLASSES:
        raise PrecisionClassRefusal(
            f"precision_class={cls!r} is not a class tensorhub reads "
            f"({', '.join(sorted(PRECISION_CLASSES))}). Publishing it would "
            "record prose in the checkpoint metadata and the row would serve "
            "as base."
        )
    if not cls:
        narrow = sorted(
            f"{comp}={dt}" for comp, dt in produced_dtypes.items()
            if 0 < dtype_bits(dt) < _BASE_STORAGE_BITS
        )
        if narrow:
            raise PrecisionClassRefusal(
                f"this tree carries sub-{_BASE_STORAGE_BITS}-bit weights "
                f"({', '.join(narrow)}) and declares no precision class. "
                "Declare `precision_class` in the flavor's attributes (one of "
                f"{', '.join(sorted(PRECISION_CLASSES))}) — the bytes name the "
                "width but only the producer knows the lane, and an unstamped "
                "row is served as base."
            )
    if not cls or cls == CLASS_BASE:
        return None
    return {"precision_class": cls}


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
    """The restatement default: the classification stamps of the checkpoint
    being converted, read from the hub's resolve of ``ctx.source``. Best-effort —
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
        # `distilled_status` tells us whether false is evidence or merely the
        # wire default. Never turn unknown into an authored false on the
        # derived checkpoint. Only "classified" is evidence — the hub's own
        # rule (`modelfamily.StoredCheckpointFacts`), and an EMPTY status is
        # one of the unknowns: the resolve route omits the key whenever the
        # stored column is empty, so "" means nothing measured the axis.
        distilled = (
            resolved.distilled
            if resolved.distilled_status == "classified"
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
    """The publish journal for one produced flavor.

    NEXT TO the tree, never inside it: ``files_from_tree`` walks a flavor
    directory wholesale, and a journal written into it would publish itself as
    repo content on the next flavor. One journal per produced-output directory
    also means one file holding every flavor's in-flight session, which is
    exactly the set a successor should try to resume.
    """
    return Path(flavor.path).parent / JOURNAL_NAME


def _publish_leg(dest: str, artifact: str, stage: str, facts: Mapping[str, Any]) -> None:
    """One typed `convert_publish` event per LEG of the publish protocol.

    Without these, the highest-volume producer on the platform emits ZERO
    `worker_activity_events` legs for a multi-hour publish, and "declared 590
    objects and is moving 37 GB" is indistinguishable from "was refused before
    a byte left". Mirrors ``fleet_cells._publish_leg``.

    ``artifact`` is the produced path's own name — what tells one leg of an
    N-artifact export from another. It was the flavor token, which A18 deleted;
    a filename is the thing that actually exists, and it is always present.
    """
    detail = " ".join(f"{k}={v}" for k, v in sorted(dict(facts).items()))
    _activity.emit_event(
        "convert_publish", f"repo={dest} artifact={artifact}: {detail}",
        phase=stage)


def destination_release(ctx: Any, explicit: str = "") -> str:
    """THE release a producer's output attaches to: the explicit argument, else
    the invoking request's ``destination.release``.

    th#1987 made `release` mandatory at the hub's DECLARE, so a producer that
    can name none has a caller-side defect. Refusing here — before a byte
    moves — names the field the invoke must carry instead of costing the run a
    multi-GB upload and a 400.
    """
    rel = str(explicit or "").strip()
    if rel:
        return rel
    info = getattr(ctx, "destination", None) or {}
    if isinstance(info, dict):
        rel = str(info.get("release") or "").strip()
    if not rel:
        raise ValueError(
            "release is required (th#1987): the invoke named no "
            "`destination.release`, and publishing never cuts one. Cut a "
            "release on the destination repo and invoke with "
            "destination={ref, release}, or pass release= explicitly."
        )
    return rel


def destination_ref(ctx: Any, explicit: str = "") -> str:
    """THE bare ``owner/repo`` a producer publishes into: the explicit
    argument, else the invoking request's ``destination.ref``.

    ONE vocabulary with ``executor._producer_destination_repo``: the reserved
    struct's key is ``ref``, and tag/flavor/checkpoint selectors are stripped
    so a caller that passed ``owner/repo:tag`` still addresses the repo.

    pgw#1305: this used to read the retired ``destination.repo`` key and
    nothing else, so an invoke carrying the ``destination={ref, release}``
    that :func:`destination_release`'s own refusal asks for was told
    ``destination_repo is required`` — the two halves of one reserved struct
    disagreed about its key.
    """
    ref = str(explicit or "").strip()
    if not ref:
        info = getattr(ctx, "destination", None) or {}
        if isinstance(info, dict):
            ref = str(info.get("ref") or "").strip()
    for sep in (":", "@", "#"):
        ref = ref.split(sep, 1)[0]
    ref = ref.strip().strip("/")
    if not ref:
        raise ValueError(
            "destination_repo is required: the invoke named no "
            "`destination.ref`. Invoke with destination={ref, release}, or "
            "pass destination_repo= explicitly."
        )
    return ref


def publish_flavors(
    ctx: Any,
    flavors: Iterable[ProducedFlavor],
    *,
    destination_repo: str = "",
    release: str = "",
    mode: str = "replace",
    metadata: Mapping[str, Any] | None = None,
    objective: str | None = None,
    distilled: bool | None = None,
    journal_path: Path | None = None,
) -> list[CommitResult]:
    """Publish each ProducedFlavor as one commit. ``destination_repo`` falls
    back to the reserved-name ``ctx.destination`` payload field.

    ``mode`` defaults to ``"replace"``: a producer's flavor export is a
    complete tree by definition, and merging with the repo's prior :latest
    ships a quantized checkpoint carrying the base weights. Pass
    ``mode="merge"`` explicitly only for deliberate overlay publishes (e.g. a
    vae swap on top of an existing tree).

    ``release`` names the ALREADY-CUT release each flavor attaches to
    (th#1980), and it is MANDATORY (th#1987) — it falls back to the invoking
    request's ``destination.release`` and refuses when neither states one, so
    the producer is told at the call site instead of after the upload. Every
    flavor of one export lands in the same release and is told apart there by
    its contract. Publishing never cuts a release, so an unknown identifier is
    a typed ``HubReleaseNotFoundError`` — cut it and publish again, never
    re-upload. It is a first-class field on the declare request: stating it in
    ``metadata`` publishes inert prose the hub never reads.

    ``journal_path`` is where the in-flight ``publish_id`` is recorded so a
    retry on this pod re-uploads instead of re-casting. Pass the produced
    tree's own directory; omit it and the publish is unrecoverable."""
    # The hub-write declaration, checked before anything is read or uploaded
    # (pgw#1294). Undeclared code never had a grant minted for it, so this is
    # that refusal arriving at the call site instead of after the bytes moved.
    require = getattr(ctx, "_require_publish_declaration", None)
    if callable(require):
        require("publish_flavors")
    dest = destination_ref(ctx, destination_repo)
    release = destination_release(ctx, release)

    client = HubClient.from_ctx(ctx)
    # A v2 publish mints a new identity and inherits nothing, so a publish into
    # a classified repo must restate objective/distilled. Default: restate the
    # SOURCE checkpoint's hub stamps — this producer just derived the flavors
    # from exactly that source, and quantize/fuse/cast preserve
    # objective/distillation, so the restatement is a first-hand declaration
    # rather than silent inheritance. Explicit caller values win.
    if objective is None or distilled is None:
        src_objective, src_distilled = _source_stamps(ctx, client)
        if objective is None:
            objective = src_objective
        if distilled is None:
            distilled = src_distilled
    results: list[CommitResult] = []
    for flavor in flavors:
        # OUR producers never emit shards, and this is the last place a
        # conversion / training-promote / cell-publish output can be checked
        # before it becomes somebody's checkpoint. NOT a universal publish gate
        # — a user's own sharded upload never reaches this function; it goes to
        # the hub's upload API and is accepted as given. Checked rather than
        # assumed because save_pretrained shards on its own.
        assert_one_file_per_component(
            Path(flavor.path), producer=f"publish_flavors[{dest}]")
        # Same seam, same argument: our producers do not get to publish a
        # component NARROWER than its families.facts pin. Here there is no
        # source tree to compare against — this path is our own output, not a
        # mirror — so the pin is enforced outright, and the per-component
        # precision is published either way.
        produced_dtypes = verify_produced_tree(Path(flavor.path))
        attrs = {str(k): str(v) for k, v in (flavor.attributes or {}).items()}
        # Worker-addable provenance stamp fields. Producers declare quant
        # identity in the flavor attribute bag; it rides the commit's
        # `provenance` object onto the checkpoint's node stamp (parents /
        # derivation_op come from the orchestrator's token claim, never here).
        provenance = {
            k: attrs[k]
            for k in ("quantization_method", "quantization_library")
            if attrs.get(k)
        }
        placement = _precision_class_block(attrs, produced_dtypes)
        meta = {**(dict(metadata) if metadata else {}), **attrs}
        for k in _DEAD_PLACEMENT_ATTRS:
            meta.pop(k, None)
        # It rides its own typed field (and is PROVEN there); a metadata copy
        # would be an unproven second statement of the same thing.
        meta.pop("artifact_contract", None)
        if placement:
            meta["placement"] = placement
        if produced_dtypes:
            meta["component_dtypes"] = dict(produced_dtypes)
        # v2 is safe here for the reason v2 requires: every file is a real
        # local file (`_flavor_files` walks the produced tree), so each digest
        # is PROVEN from bytes in hand rather than asserted. There is no
        # auto-select and no env knob — naming the protocol at the call site is
        # what makes "which producers are on v2 today?" answerable by reading
        # the code instead of by sampling traffic.
        #
        # A `merge` onto a prior blake3 manifest is a typed refusal
        # (`mixed_algorithm_manifest`), not a silent partial; `mode` here
        # defaults to "replace", so the common path is unaffected.
        results.append(client.publish_v2(
            destination_repo=dest,
            files=_flavor_files(flavor),
            release=release,
            mode=mode,
            # The conversion producer's own legs. (The per-object liveness beat
            # lives in tensorfs's transfer progress callback, so it
            # cannot be lost by a caller who forgets to pass a callback.)
            on_stage=functools.partial(
                _publish_leg, dest, Path(flavor.path).name),
            journal_path=journal_path or _journal_beside(flavor),
            # When the producer declares one, the hub PROVES it against the
            # header before recording it.
            artifact_contract=attrs.get("artifact_contract", ""),
            dtype=attrs.get("dtype", ""),
            file_layout=validate_file_layout(attrs.get("file_layout", "")),
            file_type=attrs.get("file_type", ""),
            objective=str(objective or ""),
            distilled=distilled,
            metadata=meta,
            provenance=provenance,
        ))
    return results


__all__ = [
    "PrecisionClassRefusal",
    "destination_ref",
    "destination_release",
    "publish_flavors",
]
