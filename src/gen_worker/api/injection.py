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

    Used inside `typing.Annotated[..., ModelRef(...)]`.

    Two forms for `Src.FIXED`:

    - Preferred (direct):
      ``ModelRef(Src.FIXED, ref="owner/repo", flavor="nf4")`` — function
      declares the exact repo and flavor it needs. No `endpoint.toml [models]`
      entry required; discovery synthesizes the entry into the manifest.
    - Legacy (local key):
      ``ModelRef(Src.FIXED, "local-key")`` — function references a key from
      `endpoint.toml [models]` which maps to a (ref, flavor). Still supported
      but adds an extra naming layer for no upside.

    For `Src.PAYLOAD`: ``ModelRef(Src.PAYLOAD, "field_name")`` where
    `field_name` is the payload field the caller fills with a key from
    `[models.<function_name>]`.

    For `Src.PAYLOAD_REF`: ``ModelRef(Src.PAYLOAD_REF, "field_name")`` where
    the caller supplies an arbitrary canonical ref at invoke time;
    orchestrator runs the compat validation chain before dispatch.
    """

    source: ModelRefSource
    # For PAYLOAD / PAYLOAD_REF: the payload field name (required).
    # For FIXED: the local endpoint.toml [models] key (legacy form). Empty
    # when the function uses the direct (ref, tag, flavor) form below.
    key: str = ""

    # Direct (ref, tag, flavor) — preferred for FIXED.
    # `ref` is the canonical repo ref (e.g. "owner/repo"). Setting `ref`
    # makes this a direct declaration; no endpoint.toml [models] entry is
    # required and discovery synthesizes one into the manifest.
    ref: Optional[str] = None
    # Repo tag to resolve against. Defaults to "prod" — the convention for
    # every endpoint we ship today. Override only when a function specifically
    # wants a non-prod tag (e.g. "canary" for a staged rollout).
    tag: str = "prod"
    # Flavor selector inside the repo's checkpoint group. e.g. "bf16", "nf4",
    # "fp8", "int8". The repo's flavors are tensorhub-side metadata; the
    # function picks one. Empty = default flavor for the repo.
    flavor: str = ""

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

    def __post_init__(self) -> None:
        # FIXED: either direct (ref, ...) form OR legacy local key, not both, not neither.
        if self.source == ModelRefSource.FIXED:
            if self.ref and self.key:
                raise ValueError(
                    "ModelRef(Src.FIXED, ...): pass EITHER `key=` (legacy local-key form) "
                    "OR `ref=` (direct form), not both."
                )
            if not self.ref and not self.key:
                raise ValueError(
                    "ModelRef(Src.FIXED, ...) requires either `ref=` (direct: "
                    'ModelRef(Src.FIXED, ref="owner/repo", flavor="nf4")) '
                    "or `key=` (legacy: ModelRef(Src.FIXED, \"local-key\"))."
                )
            if not self.ref and self.flavor:
                raise ValueError(
                    "ModelRef(Src.FIXED, ...) with `flavor=` requires `ref=` to be set. "
                    "Flavor without a ref is meaningless."
                )
        elif self.source in (ModelRefSource.PAYLOAD, ModelRefSource.PAYLOAD_REF):
            if not self.key:
                raise ValueError(
                    f"ModelRef({self.source.name}, ...) requires `key=<payload_field_name>`."
                )
            if self.ref:
                raise ValueError(
                    f"ModelRef({self.source.name}, ...): `ref=` is only valid for "
                    "Src.FIXED. PAYLOAD sources resolve refs at invoke time from the caller's payload."
                )


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
