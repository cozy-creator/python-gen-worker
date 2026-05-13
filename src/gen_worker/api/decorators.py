"""Decorator API for `@inference_function`.

Replaces the previous split between ``ResourceRequirements`` + ``ScalingHints``
with a single :class:`Resources` struct, and replaces the
``Annotated[T, ModelRef(...)]`` injection pattern with the ``models={...}``
kwarg carrying :class:`~gen_worker.api.binding.Repo` / :class:`Dispatch`
bindings.

See `progress.json` issue #9 (decorator-table-model-bindings).
"""

from __future__ import annotations

import typing
from typing import Any, Callable, Literal, Mapping, Optional, TypeVar, Union, overload

import msgspec

from .binding import Binding, Dispatch, Repo, _qualname

F = TypeVar("F", bound=Callable[..., Any])


def _force_setattr(obj: Any, name: str, value: Any) -> None:
    msgspec.structs.force_setattr(obj, name, value)


class Resources(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    """Hardware envelope + dynamic cost shape for one inference function.

    Replaces ``ResourceRequirements`` + ``ScalingHints``. Declared **per
    function** (not shared across an endpoint with a low floor) so the worker
    can self-advertise an accurate per-function availability map at boot and
    the orchestrator can route on accurate per-function hardware needs.

    Two field groups:

    **Static placement envelope** — hard gates. The worker compares these
    against host hardware at boot to mark functions unavailable; the
    orchestrator uses them as placement filters:

    - ``accelerator``: ``"cuda"`` / ``"none"``.
    - ``requires_gpu``: bool. Implies ``accelerator="cuda"`` for placement.
    - ``min_vram_gb``: float, GiB.
    - ``cuda_compute_min``: float, e.g. ``8.0`` for SM 8.0+.
    - ``required_libraries``: importable Python package names (``"flash_attn"``,
      ``"bitsandbytes"``, ``"torchao"``, etc.) the function needs.

    **Dynamic cost shape** — admission + scheduling. The orchestrator
    learns coefficients per ``(function, gpu_class, scaling_field)`` from
    observed runs after the tenant declares which payload fields drive cost:

    - ``vram_must_fit``: ``"full_model"`` or ``"largest_component"``. Selects
      which ``source.size_facts`` entry the orchestrator multiplies by
      ``vram_size_multiplier`` when computing required VRAM at submit.
    - ``vram_base``: constant VRAM overhead in bytes.
    - ``vram_size_multiplier``: multiplier on ``source.size_facts[vram_must_fit]``.
    - ``vram_scales_with``: payload fields that grow VRAM. Coefficients learned.
    - ``runtime_scales_with``: payload fields that grow runtime. Coefficients
      learned per gpu_class.

    Field names in ``vram_scales_with`` / ``runtime_scales_with`` must
    reference real fields on the function's payload struct;
    :func:`inference_function` validates this at decoration time.
    """

    # ----- Static placement envelope (hard gates) -------------------------
    accelerator: Literal["cuda", "none"] | None = None
    requires_gpu: bool | None = None
    min_vram_gb: float | None = None
    cuda_compute_min: float | None = None
    required_libraries: tuple[str, ...] = ()

    # ----- Dynamic cost shape (admission + scheduling) --------------------
    vram_must_fit: Literal["full_model", "largest_component"] | None = None
    vram_base: int = 0
    vram_size_multiplier: float = 0.0
    vram_scales_with: tuple[str, ...] = ()
    runtime_scales_with: tuple[str, ...] = ()

    # ----- Derived wire-shape field (set in __post_init__) ----------------
    compute_capability: dict[str, str] | None = None

    def __post_init__(self) -> None:
        # Normalize accelerator: "gpu" → "cuda", "cpu" → "none", "" → None.
        if self.accelerator is not None:
            accel = str(self.accelerator).strip().lower()
            if accel == "gpu":
                accel = "cuda"
            elif accel == "cpu":
                accel = "none"
            if accel == "":
                _force_setattr(self, "accelerator", None)
            elif accel in ("none", "cuda"):
                _force_setattr(self, "accelerator", accel)
                if accel == "cuda" and self.requires_gpu is None:
                    _force_setattr(self, "requires_gpu", True)
            else:
                raise ValueError(
                    f"accelerator must be 'none' or 'cuda', got {self.accelerator!r}"
                )

        # cuda_compute_min: validate, then derive compute_capability wire field.
        if self.cuda_compute_min is not None:
            val = float(self.cuda_compute_min)
            if val <= 0:
                raise ValueError(f"cuda_compute_min must be positive, got {val}")
            _force_setattr(self, "cuda_compute_min", val)
            _force_setattr(self, "compute_capability", {"min": f"{val:.1f}"})

        if self.min_vram_gb is not None:
            vram = float(self.min_vram_gb)
            if vram <= 0:
                raise ValueError(f"min_vram_gb must be positive, got {vram}")
            _force_setattr(self, "min_vram_gb", vram)

        if self.required_libraries:
            libs = tuple(str(x).strip() for x in self.required_libraries if str(x).strip())
            _force_setattr(self, "required_libraries", libs)

        if self.vram_must_fit not in (None, "full_model", "largest_component"):
            raise ValueError(
                f"vram_must_fit must be 'full_model', 'largest_component', or None; "
                f"got {self.vram_must_fit!r}"
            )
        if self.vram_base < 0:
            raise ValueError(f"vram_base must be >= 0, got {self.vram_base}")
        if self.vram_size_multiplier < 0:
            raise ValueError(
                f"vram_size_multiplier must be >= 0, got {self.vram_size_multiplier}"
            )


# Helpers used at decoration time --------------------------------------------


def _payload_field_names(payload_type: type) -> set[str]:
    """Return the field name set of a msgspec.Struct payload type."""
    try:
        return {f.name for f in msgspec.structs.fields(payload_type)}
    except Exception:
        return set()


def _payload_field_type(payload_type: type, field_name: str) -> Any:
    """Best-effort: read the static type annotation for a payload field."""
    hints: dict[str, Any] = {}
    try:
        hints = typing.get_type_hints(payload_type, include_extras=False)
    except Exception:
        hints = getattr(payload_type, "__annotations__", {}) or {}
    return hints.get(field_name)


def _literal_members(t: Any) -> Optional[tuple[Any, ...]]:
    """If ``t`` is a ``Literal[...]`` (possibly wrapped in ``Optional``),
    return its members tuple; else None.

    Handles ``Literal["a", "b"]``, ``Literal["a"] | None``, and
    ``Optional[Literal["a"]]``.
    """
    origin = typing.get_origin(t)
    if origin is Literal:
        return tuple(typing.get_args(t))
    if origin is Union:
        # type alias for `X | Y` and `Optional[X]`.
        members: list[Any] = []
        for arg in typing.get_args(t):
            if arg is type(None):
                continue
            sub = _literal_members(arg)
            if sub is None:
                return None
            members.extend(sub)
        if members:
            return tuple(members)
    return None


def _normalize_param_annotation_classes(ann: Any) -> tuple[str, ...]:
    """Pluck class FQNs from a parameter annotation.

    Used for the ``pipeline_classes`` emission in the manifest — we capture
    the function's annotated type(s) so consumers can see what classes the
    function actually accepts. This is informational/metadata; the
    *override* allowlist comes from ``.allow_override(*classes)``.
    """
    if ann is None:
        return ()
    # Strip Optional / Union.
    origin = typing.get_origin(ann)
    if origin is Union:
        names: list[str] = []
        for arg in typing.get_args(ann):
            if arg is type(None):
                continue
            if isinstance(arg, type):
                names.append(_qualname(arg))
        return tuple(names)
    if isinstance(ann, type):
        return (_qualname(ann),)
    return ()


# Decoration-time validation pipeline ----------------------------------------


def _validate_models(
    func_name: str,
    func: Callable[..., Any],
    payload_type: type,
    models: Mapping[str, Binding],
    param_types: Mapping[str, Any],
) -> dict[str, Binding]:
    """Validate the ``models=`` kwarg against the function signature.

    Returns a normalized mapping. Raises ``ValueError`` (with a typed-error
    fragment that mirrors the orchestrator codes) on any drift.
    """
    if not models:
        return {}

    payload_fields = _payload_field_names(payload_type)
    out: dict[str, Binding] = {}

    for param_name, binding in models.items():
        if not isinstance(binding, (Repo, Dispatch)):
            raise TypeError(
                f"@inference_function({func_name!r}): models[{param_name!r}] must be "
                f"a Repo or Dispatch instance; got {type(binding).__name__}"
            )

        # The param name must exist on the function signature (and not be the
        # payload itself).
        if param_name not in param_types:
            raise ValueError(
                f"@inference_function({func_name!r}): models[{param_name!r}] names a parameter "
                f"that does not exist on the function signature"
            )

        if isinstance(binding, Dispatch):
            # The discriminator field must exist on the payload struct.
            if binding.field not in payload_fields:
                raise ValueError(
                    f"@inference_function({func_name!r}): models[{param_name!r}] uses "
                    f"dispatch(field={binding.field!r}), but the payload struct "
                    f"{payload_type.__name__} has no such field"
                )
            # The field must be Literal[...]-typed.
            ftype = _payload_field_type(payload_type, binding.field)
            members = _literal_members(ftype)
            if members is None:
                raise ValueError(
                    f"@inference_function({func_name!r}): models[{param_name!r}] dispatches "
                    f"on payload field {binding.field!r}, which must be Literal[...]-typed; "
                    f"got {ftype!r}"
                )
            members_set = set(members)
            # Every table key must be a Literal member.
            for k in binding.table:
                if k not in members_set:
                    raise ValueError(
                        f"@inference_function({func_name!r}): models[{param_name!r}] dispatch "
                        f"table key {k!r} is not a member of "
                        f"Literal{list(members)} on payload field {binding.field!r}"
                    )

        out[param_name] = binding

    return out


def _validate_resources_payload_fields(
    func_name: str,
    payload_type: type,
    resources: Resources,
) -> None:
    """Reject scales_with entries that don't reference real payload fields."""
    payload_fields = _payload_field_names(payload_type)

    for axis_name, axis in (
        ("vram_scales_with", resources.vram_scales_with),
        ("runtime_scales_with", resources.runtime_scales_with),
    ):
        for fname in axis:
            # Allow dotted path traversal — only validate the first segment
            # (tenants reference nested payload fields via "specs[0].scheme").
            head = fname.split(".", 1)[0].split("[", 1)[0]
            if head and head not in payload_fields:
                raise ValueError(
                    f"@inference_function({func_name!r}): resources.{axis_name} entry {fname!r} "
                    f"references unknown_payload_field — {payload_type.__name__} has no "
                    f"field {head!r}"
                )


def _validate_payload_reserved_fields(func_name: str, payload_type: type) -> None:
    """Reject payload structs that use the reserved ``_models`` field name."""
    payload_fields = _payload_field_names(payload_type)
    if "_models" in payload_fields:
        raise ValueError(
            f"@inference_function({func_name!r}): payload struct {payload_type.__name__} "
            f"uses reserved field name '_models'. The orchestrator strips '_models' from "
            f"incoming payloads before dispatch, so the field cannot appear on the typed "
            f"payload. Rename the field or move its data into another key."
        )


def _find_payload_type(func: Callable[..., Any]) -> tuple[type, dict[str, Any]]:
    """Locate the msgspec.Struct payload type on the function signature.

    Returns (payload_type, param_annotations). The payload is the (one and
    only) parameter whose annotation is a ``msgspec.Struct`` subclass.
    Skips the first parameter (``ctx``).
    """
    import inspect

    hints = typing.get_type_hints(func, include_extras=False)
    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    param_types: dict[str, Any] = {}
    payload_type: Optional[type] = None
    for i, p in enumerate(params):
        ann = hints.get(p.name)
        param_types[p.name] = ann
        if i == 0:
            # Skip ctx.
            continue
        if isinstance(ann, type) and issubclass(ann, msgspec.Struct):
            if payload_type is not None:
                raise ValueError(
                    f"@inference_function({func.__name__!r}): more than one msgspec.Struct "
                    f"parameter — only one payload is allowed"
                )
            payload_type = ann

    if payload_type is None:
        raise ValueError(
            f"@inference_function({func.__name__!r}): missing msgspec.Struct payload parameter"
        )

    return payload_type, param_types


# Public decorator ------------------------------------------------------------


@overload
def inference_function(fn: F) -> F: ...
@overload
def inference_function(
    fn: None = None,
    *,
    label: Optional[str] = None,
    description: Optional[str] = None,
    resources: Optional[Resources] = None,
    models: Optional[Mapping[str, Binding]] = None,
) -> Callable[[F], F]: ...
def inference_function(
    fn: Optional[F] = None,
    *,
    label: Optional[str] = None,
    description: Optional[str] = None,
    resources: Optional[Resources] = None,
    models: Optional[Mapping[str, Binding]] = None,
) -> Any:
    """Mark a function as an inference endpoint.

    Usable as ``@inference_function`` (bare) or ``@inference_function(...)``.

    Args:
        label: Optional short label surfaced in the endpoint UI / search.
        description: Optional free-text description.
        resources: Per-function :class:`Resources` — hardware envelope +
            dynamic cost shape. The worker compares the envelope fields
            against host hardware at boot to mark this function unavailable
            on hosts that can't satisfy it. The orchestrator uses the same
            envelope for placement.
        models: Mapping ``{param_name: Repo | Dispatch}`` declaring the
            model binding for each injected parameter. Repo = fixed pick;
            Dispatch = payload-driven pick. Both support optional
            ``.allow_override(*classes)`` to permit caller substitution
            within an explicit pipeline-class allowlist.

    Decoration-time validation:

    - Every ``models[param_name]`` must be a Repo or Dispatch instance.
    - Dispatch.field must be a real payload field that is ``Literal[...]``-typed,
      and every table key must be a member of that Literal.
    - ``resources.vram_scales_with`` / ``runtime_scales_with`` must reference
      real payload fields (fails with ``unknown_payload_field`` otherwise).
    - The payload struct cannot use the reserved ``_models`` field name.
    """
    resources_value: Resources = resources if resources is not None else Resources()

    def apply(func: F) -> F:
        payload_type, param_types = _find_payload_type(func)
        _validate_payload_reserved_fields(func.__name__, payload_type)
        _validate_resources_payload_fields(func.__name__, payload_type, resources_value)
        validated_models = _validate_models(
            func.__name__, func, payload_type, models or {}, param_types
        )

        setattr(func, "_is_inference_function", True)
        setattr(func, "_function_label", (label or "").strip() or None)
        setattr(func, "_function_description", (description or "").strip() or None)
        # New public attributes (replacing _worker_resources / _scaling_hints).
        setattr(func, "__gen_worker_resources__", resources_value)
        setattr(func, "__gen_worker_bindings__", validated_models)
        # Back-compat shims that worker.py / discovery still read by the old
        # name. These point at the same merged object.
        setattr(func, "_worker_resources", resources_value)
        return func

    if fn is not None:
        return apply(fn)
    return apply


__all__ = [
    "Resources",
    "inference_function",
]
