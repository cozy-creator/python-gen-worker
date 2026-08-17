"""Catalog-stamped tuned values, decoded onto an instance.

Paul, 2026-08-17: tuned values *"stay catalog-stamped per release slot exactly
as th#1116 works today — only the delivery address changes."* This module is
the address change and nothing else.

The wire is unchanged: the hub stamps one resolved inference-defaults document
per model binding, and the LoRA overlays riding that binding carry their own
"no opinion unless stated" documents. What changes is where the decoded struct
lands — on the Model, as ``inst.tuned``, because a tuned value is a
checkpoint-level fact and a request is not the axis it varies on.

The composition rule is deliberately NOT reimplemented here: it is
``gen_worker.api.slot``'s, field by field, in overlay order, only over fields
the base struct declares. Two implementations of one merge is exactly how a
LoRA silently retunes a checkpoint differently depending on which path decoded
it.
"""

from __future__ import annotations

from collections.abc import Sequence

import msgspec

from ..families.base import GenerationDefaults
from .errors import ModelError, ModelRefusal
from .runtime import Model


def tuned_from_catalog(
    family: type[Model],
    stamped: str = "",
    lora_stamped: Sequence[str] = (),
) -> GenerationDefaults:
    """Decode one release slot's stamped values against a family's schema.

    ``stamped`` is the hub's ``ModelBinding.inference_defaults`` JSON for this
    checkpoint; empty means the hub stamped none, and the NEUTRAL schema
    defaults apply — the endpoint has no code-side recipe to fall back to, and
    reintroducing one would reverse SDK v2's deletion of ``default_config``.

    A malformed document RAISES. tensorhub schema-validates it at PUT time, so
    a decode failure here is real version skew, and serving a checkpoint with
    values nobody could parse is worse than refusing it.
    """

    from ..api.slot import _apply_lora_overrides

    schema = family.Tuned
    raw = (stamped or "").strip()
    try:
        base: GenerationDefaults = (
            msgspec.json.decode(raw.encode("utf-8"), type=schema) if raw else schema()
        )
    except (msgspec.ValidationError, msgspec.DecodeError) as exc:
        raise ModelError(
            ModelRefusal.TUNED_INVALID,
            f"family {family.FAMILY!r}: catalog tuned values failed "
            f"{schema.__name__} validation: {exc}",
        ) from exc
    try:
        return _apply_lora_overrides(family.FAMILY, base, family.FAMILY, lora_stamped)
    except ValueError as exc:
        raise ModelError(
            ModelRefusal.TUNED_INVALID,
            f"family {family.FAMILY!r}: a lora overlay's tuned values are invalid: {exc}",
        ) from exc


__all__ = ["tuned_from_catalog"]
