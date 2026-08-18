"""``@entrypoint`` — the stateless half of the pgw#1382 split.

An entrypoint is a MODULE-LEVEL function: pure request -> response logic
that holds no state and needs none (statelessness is STRUCTURAL — there is
no ``self`` to stash state on). The signature contract (Paul's ctx-first
ruling, 2026-08-17 — one parameter-order rule across the SDK, matching
``load(self, ctx)``; adapters-as-parameters ruling, same review)::

    @entrypoint
    def generate(ctx: RequestContext, payload: TextToImageInput,
                 model: SdxlModel) -> ImageOutput: ...

    @entrypoint
    def generate(ctx: RequestContext, payload: TextToImageInput,
                 model: SdxlModel, turbo: Adapter | None,
                 loras: list[Adapter]) -> ImageOutput: ...

* first: ``ctx`` annotated :class:`~gen_worker.serving.context.RequestContext`
* second: the payload, a ``msgspec.Struct`` — the wire schema
* remaining, in any order: model SLOTS (params annotated with a
  :class:`~gen_worker.serving.model.Model` subclass — at least one) and
  adapter SLOTS (``Adapter`` / ``Adapter | None`` single, ``list[Adapter]``
  the request's picks; the hub resolves what rides per deployment/request
  into them). The PARAMETER NAME is the slot name the request envelope
  keys per-slot picks on (``{"model": ref, "loras": [{"ref":…, "scale":…}]}``).
* return: a ``msgspec.Struct``

The decorator validates at import (typed refusal names the exact defect) and
stamps :data:`ENTRYPOINT_ATTR` with an :class:`EntrypointSpec` — the whole
publish-time extraction surface, readable without executing author code
beyond import.
"""

from __future__ import annotations

import inspect
import types
import typing
from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal, Tuple, Type, TypeVar, get_type_hints

import msgspec

from .context import Adapter
from .model import Model

F = TypeVar("F", bound=Callable[..., Any])

#: The one attribute @entrypoint stamps on the author's function.
ENTRYPOINT_ATTR = "__cozy_entrypoint__"


class EntrypointDeclarationError(TypeError):
    """An @entrypoint function does not satisfy the signature contract."""


@dataclass(frozen=True, slots=True)
class SlotSpec:
    """One declared slot parameter, in signature order after the payload.

    Kinds: ``model`` (a Model subclass — required), ``adapter``
    (``Adapter`` required / ``Adapter | None`` optional), ``adapters``
    (``list[Adapter]`` — the request's picks, empty when none ride)."""

    name: str
    kind: Literal["model", "adapter", "adapters"]
    #: The model class for a model slot; :class:`Adapter` for adapter slots.
    annotation: type
    #: ``Adapter | None`` and ``list[Adapter]`` slots are not required.
    required: bool = True


@dataclass(frozen=True, slots=True)
class EntrypointSpec:
    """One entrypoint, statically extracted: name, payload schema, ordered
    slots (param name = slot name), return type."""

    name: str
    fn: Callable[..., Any]
    payload_type: Type[msgspec.Struct]
    slots: Tuple[SlotSpec, ...]
    return_type: Type[msgspec.Struct]

    @property
    def model_params(self) -> Tuple[Tuple[str, type], ...]:
        return tuple(
            (slot.name, slot.annotation) for slot in self.slots if slot.kind == "model"
        )

    @property
    def adapter_params(self) -> Tuple[SlotSpec, ...]:
        return tuple(slot for slot in self.slots if slot.kind == "adapter")

    @property
    def model_classes(self) -> Tuple[type, ...]:
        """Referenced model classes, first-reference order, deduplicated."""
        seen: Dict[type, None] = {}
        for _, cls in self.model_params:
            seen.setdefault(cls)
        return tuple(seen)


def _refuse(fn: Callable[..., Any], message: str) -> EntrypointDeclarationError:
    return EntrypointDeclarationError(
        f"@entrypoint {fn.__module__}.{fn.__qualname__}: {message}"
    )


def _annotation_class(annotation: Any) -> type | None:
    """The concrete class of an annotation, seeing through Generic aliases
    (``RequestContext[D]`` -> ``RequestContext``)."""
    origin = typing.get_origin(annotation)
    candidate = origin if origin is not None else annotation
    return candidate if isinstance(candidate, type) else None


def _slot_of(fn: Callable[..., Any], name: str, annotation: Any) -> SlotSpec:
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        arms = [arm for arm in typing.get_args(annotation) if arm is not type(None)]
        if len(arms) == 1 and arms[0] is Adapter:
            return SlotSpec(name=name, kind="adapter", annotation=Adapter, required=False)
        raise _refuse(
            fn,
            f"parameter {name!r}: the only optional slot form is "
            f"`Adapter | None`, got {annotation!r}",
        )
    if origin is list:
        (item,) = typing.get_args(annotation) or (None,)
        if item is Adapter:
            return SlotSpec(
                name=name, kind="adapters", annotation=Adapter, required=False)
        raise _refuse(
            fn,
            f"parameter {name!r}: the only list slot form is "
            f"`list[Adapter]` (the request's adapter picks), got {annotation!r}",
        )
    concrete = _annotation_class(annotation)
    if concrete is Adapter:
        return SlotSpec(name=name, kind="adapter", annotation=Adapter, required=True)
    if concrete is Model:
        raise _refuse(
            fn,
            f"parameter {name!r} is annotated with the bare Model base; "
            "declare the author's model class",
        )
    if concrete is not None and issubclass(concrete, Model):
        return SlotSpec(name=name, kind="model", annotation=concrete, required=True)
    raise _refuse(
        fn,
        f"parameter {name!r} must be a model slot (annotated with a "
        "gen_worker.Model subclass) or an adapter slot (Adapter / "
        f"Adapter | None); the annotation IS the declaration and the param "
        f"name is the slot name — got {annotation!r}",
    )


def entrypoint(fn: F) -> F:
    """Mark a module-level function as an entrypoint (contract above)."""

    from .context import RequestContext

    if not inspect.isfunction(fn):
        raise EntrypointDeclarationError(
            f"@entrypoint marks module-level functions, got "
            f"{type(fn).__name__} — decorator kwargs died with @endpoint "
            "(pgw#1382); the Model class header carries lanes="
        )
    if "." in fn.__qualname__:
        raise _refuse(
            fn,
            "entrypoints are MODULE-LEVEL functions (statelessness is "
            "structural); methods and nested functions cannot be entrypoints",
        )
    try:
        hints = get_type_hints(fn)
    except Exception as exc:
        raise _refuse(fn, f"unresolvable type hints: {exc}") from exc

    parameters = list(inspect.signature(fn).parameters.values())
    for parameter in parameters:
        if parameter.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise _refuse(
                fn,
                f"parameter {parameter.name!r} is {parameter.kind.description}; "
                "the signature is plain positional: (ctx, payload, *slots)",
            )
    if len(parameters) < 3:
        raise _refuse(
            fn,
            f"takes {len(parameters)} parameters; the contract is "
            "(ctx: RequestContext, payload: msgspec.Struct, "
            "<slot>: <Model subclass or Adapter>, ...)",
        )

    ctx_parameter, payload_parameter, slot_parameters = (
        parameters[0], parameters[1], parameters[2:],
    )

    ctx_type = _annotation_class(hints.get(ctx_parameter.name))
    if ctx_type is None or not issubclass(ctx_type, RequestContext):
        raise _refuse(
            fn,
            f"first parameter {ctx_parameter.name!r} must be annotated "
            "RequestContext — ctx comes FIRST (Paul's parameter-order "
            f"ruling), got {hints.get(ctx_parameter.name)!r}",
        )

    payload_type = _annotation_class(hints.get(payload_parameter.name))
    if payload_type is None or not issubclass(payload_type, msgspec.Struct):
        raise _refuse(
            fn,
            f"second parameter {payload_parameter.name!r} must be the "
            "payload, annotated with a msgspec.Struct schema; got "
            f"{hints.get(payload_parameter.name)!r}",
        )

    slots = tuple(
        _slot_of(fn, parameter.name, hints.get(parameter.name))
        for parameter in slot_parameters
    )
    if not any(slot.kind == "model" for slot in slots):
        raise _refuse(
            fn,
            "declares no model slot; at least one parameter must be "
            "annotated with a gen_worker.Model subclass",
        )

    return_type = _annotation_class(hints.get("return"))
    if return_type is None or not issubclass(return_type, msgspec.Struct):
        raise _refuse(
            fn,
            f"return type must be a msgspec.Struct, got {hints.get('return')!r}",
        )

    spec = EntrypointSpec(
        name=fn.__name__,
        fn=fn,
        payload_type=payload_type,
        slots=slots,
        return_type=return_type,
    )
    setattr(fn, ENTRYPOINT_ATTR, spec)
    return fn


__all__ = [
    "ENTRYPOINT_ATTR",
    "EntrypointDeclarationError",
    "EntrypointSpec",
    "SlotSpec",
    "entrypoint",
]
