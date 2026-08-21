"""Publish ProducedFlavor outputs to Tensorhub — THE producer publish contract."""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Any, Iterable, Mapping

from .. import activity as _activity
from .. import scratchrepo
from ..api.errors import ValidationError
from ..hubio.client import CommitFile, CommitResult, HubClient, files_from_tree
from ..hubio.publish_state import JOURNAL_NAME
from ..models.ladder import CLASS_BASE, PRECISION_CLASSES
from .dtype_pins import dtype_bits, verify_produced_tree
from .produced import ProducedFlavor
from .writer import assert_one_file_per_component
from ..models.file_layout import validate_file_layout

_DEAD_PLACEMENT_ATTRS = ("placement_sm_allowed", "placement_sm_min", "placement_engines")


_BASE_STORAGE_BITS = 16


class PrecisionClassRefusal(ValidationError):
    """A publish whose precision class cannot be recorded as the hub reads it."""


def _precision_class_block(
    attrs: Mapping[str, str], produced_dtypes: Mapping[str, str],
) -> dict[str, Any] | None:
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
    return Path(flavor.path).parent / JOURNAL_NAME


def _publish_leg(dest: str, artifact: str, stage: str, facts: Mapping[str, Any]) -> None:
    detail = " ".join(f"{k}={v}" for k, v in sorted(dict(facts).items()))
    _activity.emit_event(
        "convert_publish", f"repo={dest} artifact={artifact}: {detail}",
        phase=stage)


def destination_release(ctx: Any, explicit: str = "", dest: str = "") -> str:
    """THE release a producer's output attaches to: the explicit argument, else the invoking request's ``destination.release``."""
    rel = str(explicit or "").strip()
    if rel:
        return rel
    info = getattr(ctx, "destination", None) or {}
    if isinstance(info, dict):
        rel = str(info.get("release") or "").strip()
    if not rel and scratchrepo.derives_its_release(dest):
        return ""
    if not rel:
        raise ValueError(
            "release is required (th#1987): the invoke named no "
            "`destination.release`, and publishing never cuts one. Cut a "
            "release on the destination repo and invoke with "
            "destination={ref, release}, or pass release= explicitly."
        )
    return rel


def destination_ref(ctx: Any, explicit: str = "") -> str:
    """THE bare ``owner/repo`` a producer publishes into: the explicit argument, else the invoking request's ``destination.ref``."""
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
    """Publish each ProducedFlavor as one commit; destination_repo falls back to ctx.destination. mode defaults to "replace" — a flavor export is a complete tree, and merging with the repo's prior :latest ships a quantized checkpoint carrying the base weights. release is MANDATORY and names an ALREADY-CUT release (publishing never cuts one; unknown id is a typed HubReleaseNotFoundError). journal_path records the in-flight publish_id so a retry re-uploads instead of re-casting — omit it and the publish is unrecoverable."""
    require = getattr(ctx, "_require_publish_declaration", None)
    if callable(require):
        require("publish_flavors")
    dest = destination_ref(ctx, destination_repo)
    release = destination_release(ctx, release, dest)

    client = HubClient.from_ctx(ctx)
    if objective is None or distilled is None:
        src_objective, src_distilled = _source_stamps(ctx, client)
        if objective is None:
            objective = src_objective
        if distilled is None:
            distilled = src_distilled
    results: list[CommitResult] = []
    for flavor in flavors:
        assert_one_file_per_component(
            Path(flavor.path), producer=f"publish_flavors[{dest}]")
        produced_dtypes = verify_produced_tree(Path(flavor.path))
        attrs = {str(k): str(v) for k, v in (flavor.attributes or {}).items()}
        provenance = {
            k: attrs[k]
            for k in ("quantization_method", "quantization_library")
            if attrs.get(k)
        }
        placement = _precision_class_block(attrs, produced_dtypes)
        meta = {**(dict(metadata) if metadata else {}), **attrs}
        for k in _DEAD_PLACEMENT_ATTRS:
            meta.pop(k, None)
        meta.pop("artifact_contract", None)
        if placement:
            meta["placement"] = placement
        if produced_dtypes:
            meta["component_dtypes"] = dict(produced_dtypes)
        results.append(client.publish_v2(
            destination_repo=dest,
            files=_flavor_files(flavor),
            release=release,
            mode=mode,
            on_stage=functools.partial(
                _publish_leg, dest, Path(flavor.path).name),
            journal_path=journal_path or _journal_beside(flavor),
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
