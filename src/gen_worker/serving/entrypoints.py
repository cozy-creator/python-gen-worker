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

    @entrypoint                       # WEIGHTLESS (pgw#1392): no slot at all
    def transform(ctx: RequestContext, payload: TransformInput
                  ) -> TransformOutput: ...

    @entrypoint(resources=Resources(  # the STAFFING ENVELOPE (pgw#1396)
        vcpus=16, max_gpu_count=4, max_gpus_per_execution_group=4,
        parallel=("sequence",),
        requires=LayoutRequirements(recommended="ram96g")))
    def generate(ctx: RequestContext, payload: GenerateInput,
                 video: H3Model) -> GenerateOutput: ...

* first: ``ctx`` annotated :class:`~gen_worker.serving.context.RequestContext`
* second: the payload, a ``msgspec.Struct`` — the wire schema
* remaining, in any order and **possibly none**: model SLOTS (params
  annotated with a :class:`~gen_worker.serving.model.Model` subclass) and
  adapter SLOTS (``Adapter`` / ``Adapter | None`` single, ``list[Adapter]``
  the request's picks; the hub resolves what rides per deployment/request
  into them). The PARAMETER NAME is the slot name the request envelope
  keys per-slot picks on (``{"model": ref, "loras": [{"ref":…, "scale":…}]}``).
  **Zero slots is a valid declaration (pgw#1392)** — a CPU workflow helper
  has no weights to type, so it declares none, its envelope carries no model
  field, and nothing is ever made resident for it. Zero slots is legal;
  junk slots are not — a declared parameter must still BE a valid slot.
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
from typing import (
    Any, Callable, Dict, Literal, Tuple, Type, TypeVar, get_type_hints, overload,
)

import msgspec

from .context import Adapter, DistillationAdapter
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
    (``Adapter``/``DistillationAdapter`` required, ``| None`` optional —
    the ANNOTATION records which adapter KIND the slot takes; the hub
    enforces it at pick time, the worker at resolution), ``adapters``
    (``list[Adapter]`` — the request's picks, empty when none ride)."""

    name: str
    kind: Literal["model", "adapter", "adapters"]
    #: The model class for a model slot; :class:`Adapter` or
    #: :class:`DistillationAdapter` for adapter slots (the slot's KIND).
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
    #: The declared :class:`~gen_worker.api.resources.Resources`, or ``None``.
    #: se#755/pgw#1396: the STAFFING ENVELOPE — the machine this function is
    #: placed on. Function scope because the hub's unit of placement is the
    #: function (pgw#1394) and because `music-analysis` declares three
    #: different vCPU floors across three functions of one endpoint.
    resources: Any = None

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
        if len(arms) == 1 and arms[0] in (Adapter, DistillationAdapter):
            return SlotSpec(
                name=name, kind="adapter", annotation=arms[0], required=False)
        raise _refuse(
            fn,
            f"parameter {name!r}: the only optional slot forms are "
            f"`Adapter | None` and `DistillationAdapter | None`, got "
            f"{annotation!r}",
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
    if concrete in (Adapter, DistillationAdapter):
        assert concrete is not None
        return SlotSpec(name=name, kind="adapter", annotation=concrete, required=True)
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


@overload
def entrypoint(fn: F) -> F: ...
@overload
def entrypoint(*, resources: Any) -> Callable[[F], F]: ...


def entrypoint(
    fn: F | None = None, *, resources: Any = None
) -> F | Callable[[F], F]:
    """Mark a module-level function as an entrypoint (contract above).

    ``resources=`` is the ONE kwarg this decorator takes, and it is the
    STAFFING ENVELOPE (se#755/pgw#1396): the machine this function is placed
    on — ``Resources(vcpus=…, max_gpu_count=…,
    max_gpus_per_execution_group=…, parallel=…, requires=…)``. It is FUNCTION
    scope for three reasons, none of them convenience:

    * the hub's unit of placement IS the function — pgw#1394 folded the
      per-lane VRAM floor to function scope for exactly this reason
      (*"a function is placed on a single machine before anything knows which
      lane it will serve"*), and every term here is placed by that same code;
    * one endpoint's functions legitimately differ: ``music-analysis``
      declares ``vcpus=2``, ``vcpus=8``, ``vcpus=8`` across three functions,
      and a single endpoint- or model-scoped number is a silent over- or
      under-buy of two of them;
    * a WEIGHTLESS entrypoint (pgw#1392) has no ``Model`` class to hang it on,
      and weightless endpoints are precisely the ones a CPU floor unblocks.

    This does NOT reopen pgw#1382. That hardcut moved the *weight and lane*
    declarations (``model=``, ``lanes=``) onto the ``Model`` class header
    because they are properties of the WEIGHTS. A machine ask is a property of
    the CODE THAT RUNS, which is this function.

    ``vcpus`` is NOT a ``requires=`` term and must never become one: the hub
    already reads ``resources.vcpus`` and normalizes it to ``min_vcpus``
    itself, so a second spelling would land in one payload key from two
    directions. The asymmetry with ``min_host_ram_gb`` (which IS a
    ``requires=`` term, at ``recommended`` only) is deliberate: vCPUs are
    selectable and filterable at both providers, host RAM on a RunPod GPU pod
    is neither, so a host-RAM MINIMUM stays forbidden (Paul, 2026-07-11).
    """

    if fn is None:
        def bind(inner: F) -> F:
            return entrypoint(inner, resources=resources)  # type: ignore[call-overload,no-any-return]
        return bind

    from .context import RequestContext

    if not inspect.isfunction(fn):
        raise EntrypointDeclarationError(
            f"@entrypoint marks module-level functions, got "
            f"{type(fn).__name__} — the only kwarg is resources= "
            "(pgw#1396); the Model class header carries lanes="
        )
    if resources is not None:
        from ..api.resources import Resources

        if not isinstance(resources, Resources):
            raise _refuse(
                fn,
                f"resources= takes a gen_worker.Resources, got "
                f"{type(resources).__name__} — the staffing envelope is a "
                "typed declaration so its refusals name your line, not a "
                "manifest key",
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
    if len(parameters) < 2:
        raise _refuse(
            fn,
            f"takes {len(parameters)} parameters; the contract is "
            "(ctx: RequestContext, payload: msgspec.Struct) followed by "
            "ZERO OR MORE slots (<Model subclass or Adapter>) — pgw#1392: a "
            "weightless entrypoint declares no slot at all",
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
    # pgw#1392: ZERO model slots is a valid declaration. A weightless
    # entrypoint (a CPU workflow helper — dj-utils, music-analysis) has no
    # weights to type, so it declares none; the envelope then has no model
    # field at all and nothing is ever resident for it. Zero slots is legal,
    # JUNK slots are not — `_slot_of` above still refuses every parameter
    # that is neither a Model subclass nor an adapter form.

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
        resources=resources,
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
