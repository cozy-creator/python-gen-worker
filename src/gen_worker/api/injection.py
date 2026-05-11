from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, get_args, get_origin, Annotated


class ModelRefSource(str, Enum):
    # FIXED means the model key is fixed by the function signature and does not
    # depend on the request payload.
    FIXED = "fixed"
    # PAYLOAD selects the model via a payload field key. The key value resolves
    # against a pre-declared keyspace (endpoint.toml [models]) at publish time.
    PAYLOAD = "payload"
    # PAYLOAD_REF accepts an arbitrary caller-supplied canonical ref at
    # invoke time. The orchestrator resolves the ref against tensorhub
    # with the caller's JWT (access check) + runs a compat validation chain
    # (file_layout / pipeline_class / architectures / components / lineage /
    # attributes) before dispatch. No pre-declared keyspace required.
    PAYLOAD_REF = "payload_ref"


@dataclass(frozen=True)
class ModelRef:
    """
    Metadata marker for signature-driven model selection/injection.

    This is intended to be used inside `typing.Annotated[..., ModelRef(...)]`.

    For PAYLOAD_REF, the common case requires zero scoping
    declarations — discovery auto-derives the expected pipeline class /
    architectures from the parameter's type annotation. Optional overrides
    below cover edge cases and stricter-than-default scoping.
    """

    source: ModelRefSource
    key: str
    # Optional explicit model ref string for FIXED source.
    # If set, discovery includes this in the baked manifest fixed keyspace.
    ref: Optional[str] = None
    dtypes: tuple[str, ...] = ()

    # PAYLOAD_REF scoping. All fields are optional; ignored for FIXED /
    # PAYLOAD sources.

    # Explicit pipeline-class / architectures allowlist. Overrides the
    # signature-derived gate. Use when the tenant's annotated type is
    # narrower than what the endpoint should actually accept (e.g. a custom
    # subclass that should also accept the standard parent class).
    allow_pipeline_classes: tuple[str, ...] = ()
    allow_architectures: tuple[str, ...] = ()

    # File-layout gate. "diffusers" | "singlefile" | "". Empty = no file-layout
    # restriction.
    required_file_layout: str = ""

    # Per-component class map for diffusers pipelines. Keys match
    # model_index.json top-level component names (unet / vae / text_encoder /
    # etc.); values are the expected class (UNet2DConditionModel /
    # AutoencoderKL / ...). Orchestrator matches against the ref's
    # `_diffusers_components` attribute. Empty = no per-component check.
    required_components: tuple = ()

    # Lineage scoping. When `require_lineage_descendant_of` is a non-empty
    # canonical ref (e.g. "stabilityai/stable-diffusion-xl-base-1.0"),
    # orchestrator walks the caller's ref's ancestor chain via
    # tensorhub.ListAncestors and rejects when the declared ancestor is not
    # in the walk. `require_lineage_verified=True` further restricts to
    # edges whose verification_status is "verified" (rejects user-asserted-
    # only edges).
    require_lineage_descendant_of: str = ""
    require_lineage_verified: bool = False


@dataclass(frozen=True)
class InjectionSpec:
    param_name: str
    param_type: Any
    model_ref: ModelRef


def parse_injection(annotation: Any) -> Optional[tuple[Any, ModelRef]]:
    """
    Returns (base_type, model_ref) if annotation is Annotated[base_type, ModelRef(...)],
    otherwise None.
    """

    origin = get_origin(annotation)
    if origin is not Annotated:
        return None
    args = get_args(annotation)
    if not args:
        return None
    base = args[0]
    meta = args[1:]
    for m in meta:
        if isinstance(m, ModelRef):
            return base, m
    return None


def type_qualname(t: Any) -> str:
    if hasattr(t, "__module__") and hasattr(t, "__qualname__"):
        return f"{t.__module__}.{t.__qualname__}"
    if hasattr(t, "__module__") and hasattr(t, "__name__"):
        return f"{t.__module__}.{t.__name__}"
    return repr(t)
