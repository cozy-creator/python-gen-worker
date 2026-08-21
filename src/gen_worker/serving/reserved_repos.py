from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import msgspec

from .. import weight_position
from ..api.errors import ValidationError
from ..models.cache_paths import tensorhub_cas_dir, tensorhub_fill_source_dir
from ..models.download import ensure_local

from ..models.refs import normalize_model_ref

logger = logging.getLogger(__name__)

RESERVED_REPO_FIELDS: Tuple[str, ...] = (
    "source",
    "text_encoder",
    "candidate",
    "resume_from",
)

RESERVED_INFO_FIELDS: Tuple[str, ...] = RESERVED_REPO_FIELDS + ("destination",)

_SETTER_FOR: Dict[str, str] = {
    "source": "_set_source_path",
    "text_encoder": "_set_text_encoder_path",
    "candidate": "_set_candidate_path",
    "resume_from": "_set_resume_from_path",
}


def reserved_repo_info(payload: Any, field_name: str) -> Dict[str, Any]:
    """``payload.<field_name>`` as a plain dict ({} when absent)."""
    obj = getattr(payload, field_name, None)
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return dict(obj)
    try:
        out = msgspec.to_builtins(obj)
    except Exception:
        return {}
    return out if isinstance(out, dict) else {}


def reserved_context_kwargs(payload: Any) -> Dict[str, Any]:
    """The ``*_info`` constructor kwargs for this payload's reserved structs."""
    return {
        f"{name}_info": reserved_repo_info(payload, name)
        for name in RESERVED_INFO_FIELDS
    }


async def _materialize_one(
    ctx: Any,
    payload: Any,
    field_name: str,
    snapshots: Mapping[str, Any],
) -> None:
    info = reserved_repo_info(payload, field_name)
    if not info:
        return
    raw = str(info.get("ref") or "").strip()
    if not raw:
        raise ValidationError(
            f"payload.{field_name}.ref must be a non-empty repo ref"
        )
    try:
        ref = normalize_model_ref(raw)
    except ValueError as exc:
        raise ValidationError(
            f"payload.{field_name}.ref {raw!r} is not a valid repo ref: {exc}"
        ) from exc
    setter: Optional[Callable[[str], None]] = getattr(
        ctx, _SETTER_FOR[field_name], None
    )
    if not callable(setter):
        raise ValidationError(
            f"payload.{field_name} needs a producer context; this context has "
            f"no {_SETTER_FOR[field_name]}"
        )
    snapshot = snapshots.get(ref)
    if snapshot is None:
        path = await ensure_local(
            str(ref),
            cache_dir=tensorhub_cas_dir(),
            fill_source_dir=tensorhub_fill_source_dir(),
        )
        setter(str(path))
        return
    with weight_position.track(
        str(ref), weight_position.snapshot_bytes(snapshot),
    ) as position:
        path = await ensure_local(
            str(ref),
            snapshot=snapshot,
            cache_dir=tensorhub_cas_dir(),
            fill_source_dir=tensorhub_fill_source_dir(),
            progress=position.progress,
        )
    setter(str(path))
    logger.info(
        "reserved repo materialized field=%s ref=%s path=%s",
        field_name, ref, path,
    )


async def _materialize_datasets(ctx: Any, payload: Any) -> None:
    datasets = getattr(payload, "datasets", None)
    if not datasets:
        return
    resolve = getattr(ctx, "resolve_dataset", None)
    if not callable(resolve):
        raise ValidationError(
            "payload.datasets requires a producer context with resolve_dataset"
        )
    for entry in datasets:
        ref = str(getattr(entry, "ref", "") or "").strip()
        if not ref:
            raise ValidationError("payload.datasets entries need a non-empty ref")
        await asyncio.to_thread(resolve, ref)


async def materialize_reserved_inputs_async(
    ctx: Any,
    payload: Any,
    snapshots: Mapping[str, Any],
) -> None:
    """Fill every reserved path this payload declares, then its datasets."""
    for field_name in RESERVED_REPO_FIELDS:
        await _materialize_one(ctx, payload, field_name, snapshots)
        ctx.raise_if_cancelled("canceled")
    await _materialize_datasets(ctx, payload)
    ctx.raise_if_cancelled("canceled")


def materialize_reserved_inputs(
    ctx: Any,
    payload: Any,
    snapshots: Mapping[str, Any],
) -> None:
    """The SYNCHRONOUS entry the serve loop calls."""
    asyncio.run(materialize_reserved_inputs_async(ctx, payload, snapshots))


def declared_reserved_fields(payload: Any) -> Tuple[str, ...]:
    """Which reserved repo fields this payload actually names — for logs and for the tests that assert the population boundary."""
    return tuple(
        name for name in RESERVED_REPO_FIELDS
        if reserved_repo_info(payload, name)
    )


__all__ = [
    "RESERVED_INFO_FIELDS",
    "RESERVED_REPO_FIELDS",
    "declared_reserved_fields",
    "materialize_reserved_inputs",
    "materialize_reserved_inputs_async",
    "reserved_context_kwargs",
    "reserved_repo_info",
]
