"""Read-side typed decode of a checkpoint's hub row (model text + PARTIAL defaults JSONB) into a typed Defaults struct: matching model -> field-level overlay onto platform values; NULL model -> platform values + visible unclassified warning; wrong model -> typed ModelTypeMismatch; ill-typed field -> DefaultsDecodeError naming the field. Evolution rule: lenient decode, additive-only, a name never changes meaning — unknown fields ignored, missing fields take struct defaults, no version stamp. Knob fields merge to the NARROWEST [lo, hi] across the platform and checkpoint layers, so a platform envelope must be the WIDEST any lane declares or a checkpoint row's own recipe becomes unreachable (a silently clamped default, or an empty range)."""

from __future__ import annotations

import warnings
from typing import Mapping, Protocol, TypeVar

import msgspec

from .model_types import Knob

D = TypeVar("D", bound=msgspec.Struct)


class CarriesDefaults(Protocol[D]):
    """Structural face of a ModelType / LoraOverlay class: a wire ``name`` plus its ``Defaults`` struct."""

    name: str
    Defaults: type[D]


class DefaultsDecodeError(ValueError):
    """Typed refusal: a hub ``defaults`` row field is ill-typed."""

    def __init__(self, *, model: str, field: str, reason: str) -> None:
        self.model = model
        self.field = field
        self.reason = reason
        super().__init__(
            f"defaults row for model {model or '<unknown>'!r}: "
            f"field {field!r} is ill-typed: {reason}"
        )


class ModelTypeMismatch(ValueError):

    def __init__(self, *, expected: str, actual: str) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"checkpoint is classified {actual!r} but this deployment expects "
            f"{expected!r} — refusing to serve a mistyped checkpoint"
        )


class CheckpointDefaultsUnclassified(UserWarning):

    code = "checkpoint_defaults_unclassified"


class _IntKnobPatch(msgspec.Struct, frozen=True):
    default: int | None = None
    lo: int | None = None
    hi: int | None = None


class _FloatKnobPatch(msgspec.Struct, frozen=True):
    default: float | None = None
    lo: float | None = None
    hi: float | None = None


def _merge_int_knob(
    platform: Knob[int], patch: _IntKnobPatch, field: str
) -> Knob[int]:
    lo = platform.lo if patch.lo is None else (
        patch.lo if platform.lo is None else max(patch.lo, platform.lo)
    )
    hi = platform.hi if patch.hi is None else (
        patch.hi if platform.hi is None else min(patch.hi, platform.hi)
    )
    default = platform.default if patch.default is None else patch.default
    if lo is not None and default < lo:
        default = lo
    if hi is not None and default > hi:
        default = hi
    return Knob(default, lo=lo, hi=hi, name=platform.name or field)


def _merge_float_knob(
    platform: Knob[float], patch: _FloatKnobPatch, field: str
) -> Knob[float]:
    lo = platform.lo if patch.lo is None else (
        patch.lo if platform.lo is None else max(patch.lo, platform.lo)
    )
    hi = platform.hi if patch.hi is None else (
        patch.hi if platform.hi is None else min(patch.hi, platform.hi)
    )
    default = platform.default if patch.default is None else patch.default
    if lo is not None and default < lo:
        default = lo
    if hi is not None and default > hi:
        default = hi
    return Knob(default, lo=lo, hi=hi, name=platform.name or field)


def _merge_knob(
    platform: "Knob[int] | Knob[float]", raw: object, *, field: str, model: str
) -> "Knob[int] | Knob[float]":
    if isinstance(platform.default, int):
        try:
            int_patch = msgspec.convert(raw, type=_IntKnobPatch)
        except msgspec.ValidationError as e:
            raise DefaultsDecodeError(model=model, field=field, reason=str(e)) from e
        return _merge_int_knob(
            Knob(platform.default, lo=_as_int(platform.lo), hi=_as_int(platform.hi),
                 name=platform.name),
            int_patch,
            field,
        )
    try:
        float_patch = msgspec.convert(raw, type=_FloatKnobPatch)
    except msgspec.ValidationError as e:
        raise DefaultsDecodeError(model=model, field=field, reason=str(e)) from e
    return _merge_float_knob(
        Knob(float(platform.default), lo=_as_float(platform.lo),
             hi=_as_float(platform.hi), name=platform.name),
        float_patch,
        field,
    )


def _as_int(value: "int | float | None") -> int | None:
    return None if value is None else int(value)


def _as_float(value: "int | float | None") -> float | None:
    return None if value is None else float(value)


def decode_defaults(
    cls: type[D], row: Mapping[str, object] | None, *, model_name: str = ""
) -> D:
    """Decode one partial JSONB object over ``cls``'s platform values."""
    if not row:
        return cls()
    values: dict[str, object] = {}
    for f in msgspec.structs.fields(cls):
        key = f.encode_name
        if key not in row:
            continue
        raw = row[key]
        platform_default = f.default
        if isinstance(platform_default, Knob):
            values[f.name] = _merge_knob(
                platform_default, raw, field=key, model=model_name
            )
            continue
        try:
            values[f.name] = msgspec.convert(raw, type=f.type)
        except msgspec.ValidationError as e:
            raise DefaultsDecodeError(model=model_name, field=key, reason=str(e)) from e
    return cls(**values)


def decode_model_defaults(
    model_type: CarriesDefaults[D],
    *,
    model: str | None,
    defaults: Mapping[str, object] | None,
) -> D:
    """The read-side authority: hub row → typed ``Defaults`` (module docstring has the matrix)."""
    expected = model_type.name
    if model is None or model == "":
        warnings.warn(
            CheckpointDefaultsUnclassified(
                f"checkpoint is unclassified (no `model` row): serving "
                f"{expected!r} platform fallback defaults"
            ),
            stacklevel=2,
        )
        return model_type.Defaults()
    if model != expected:
        raise ModelTypeMismatch(expected=expected, actual=model)
    return decode_defaults(model_type.Defaults, defaults, model_name=expected)


__all__ = [
    "CarriesDefaults",
    "CheckpointDefaultsUnclassified",
    "DefaultsDecodeError",
    "ModelTypeMismatch",
    "decode_defaults",
    "decode_model_defaults",
]
