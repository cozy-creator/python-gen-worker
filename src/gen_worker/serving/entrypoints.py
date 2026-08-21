from __future__ import annotations

import collections.abc
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

ENTRYPOINT_ATTR = "__cozy_entrypoint__"


class EntrypointDeclarationError(TypeError):
    """An @entrypoint function does not satisfy the signature contract."""


@dataclass(frozen=True, slots=True)
class SlotSpec:
    """One declared slot parameter, in signature order after the payload."""

    name: str
    kind: Literal["model", "adapter", "adapters"]
    annotation: type
    required: bool = True


@dataclass(frozen=True, slots=True)
class EntrypointSpec:
    """One entrypoint, statically extracted: name, payload schema, ordered slots (param name = slot name), return type."""

    name: str
    fn: Callable[..., Any]
    payload_type: Type[msgspec.Struct]
    slots: Tuple[SlotSpec, ...]
    return_type: Type[msgspec.Struct]
    resources: Any = None
    publishes: bool = False
    env: Tuple[str, ...] = ()
    emits_media: bool | None = None
    kind: str = ""
    child_calls: bool = False
    handles: Tuple[str, ...] = ()
    ctx_type: type | None = None
    delta_type: Any = None
    delta_arms: Tuple[Type[msgspec.Struct], ...] = ()

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


_ENV_NAME_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")

KINDS = ("inference", "training", "conversion", "dataset", "eval")


def _validate_env_decl(fn: Callable[..., Any], env: Any) -> Tuple[str, ...]:
    if env is None:
        return ()
    if isinstance(env, str) or not isinstance(env, (list, tuple)):
        raise _refuse(
            fn,
            f"env= must be a list/tuple of environment-variable NAMES, got "
            f"{type(env).__name__} — a bare string would iterate into "
            "characters, so it is refused rather than accepted",
        )
    out: list[str] = []
    for name in env:
        candidate = str(name or "").strip()
        if (
            not candidate
            or len(candidate) > 64
            or not ("A" <= candidate[0] <= "Z")
            or any(c not in _ENV_NAME_CHARS for c in candidate)
        ):
            raise _refuse(
                fn,
                f"env= name {name!r} is not a valid environment-variable name "
                "(UPPER_SNAKE, starting with a letter, <=64 chars)",
            )
        if candidate in out:
            raise _refuse(fn, f"env= repeats {name!r}")
        out.append(candidate)
    return tuple(out)


def _delta_declaration(
    fn: Callable[..., Any], streams: Any
) -> Tuple[Any, Tuple[Type[msgspec.Struct], ...]]:
    if streams is None:
        return None, ()
    if isinstance(streams, (tuple, list)):
        candidates = tuple(streams)
    elif typing.get_origin(streams) in (typing.Union, types.UnionType):
        candidates = typing.get_args(streams)
    else:
        candidates = (streams,)
    arms: list[Type[msgspec.Struct]] = []
    for candidate in candidates:
        concrete = _annotation_class(candidate)
        if concrete is None or not issubclass(concrete, msgspec.Struct):
            raise _refuse(
                fn,
                f"streams= takes the msgspec.Struct chunk type(s) this "
                f"function emits (gen_worker.TokenDelta, gen_worker.ItemDelta, "
                f"your own Delta subclass, or a tuple/union of them), got "
                f"{candidate!r} — the type is what publish reports as "
                f"delta_output_schema",
            )
        arms.append(concrete)
    if not arms:
        raise _refuse(fn, "streams= declares no chunk type")
    if len(arms) == 1:
        return arms[0], (arms[0],)
    annotation: Any = typing.Union[tuple(arms)]
    untagged = [
        arm.__name__
        for arm in arms
        if getattr(arm, "__struct_config__", None) is None
        or getattr(arm.__struct_config__, "tag", None) is None
    ]
    if untagged:
        raise _refuse(
            fn,
            f"streams= declares several chunk types and {untagged} carry no "
            "msgspec tag, so the published delta_output_schema could not tell "
            "them apart. Subclass gen_worker.Delta (tagged on `type`), or "
            "declare tag=True on your struct",
        )
    return annotation, tuple(arms)


def _validate_handles_decl(fn: Callable[..., Any], handles: Any) -> Tuple[str, ...]:
    from ..models.execution_lanes import (
        known_execution_lane_bodies,
        valid_execution_lane_body,
    )

    if handles is None:
        return ()
    if isinstance(handles, str) or not isinstance(handles, (list, tuple)):
        raise _refuse(
            fn,
            f"handles= must be a list/tuple of lane BODY tokens, got "
            f"{type(handles).__name__} — a bare string would iterate into "
            "characters, so it is refused rather than accepted",
        )
    known = ", ".join(known_execution_lane_bodies())
    out: list[str] = []
    for raw in handles:
        token = str(raw or "").strip().lower()
        if not token:
            raise _refuse(fn, "handles= carries an empty token")
        if "+" in token:
            raise _refuse(
                fn,
                f"handles= token {raw!r} carries an execution axis; declare "
                "the lane BODY — eager/compiled is platform-managed and is "
                "not something a body can branch on",
            )
        if not valid_execution_lane_body(token):
            raise _refuse(
                fn,
                f"handles= token {raw!r} is not a known lane body "
                f"(known: {known}) — the hub validates this against the same "
                "table AFTER the image bake, so it is refused here instead",
            )
        if token in out:
            raise _refuse(fn, f"handles= repeats {raw!r}")
        out.append(token)
    return tuple(out)


@overload
def entrypoint(fn: F) -> F: ...
@overload
def entrypoint(
    *,
    resources: Any = ...,
    kind: str = ...,
    publishes: bool = ...,
    env: Any = ...,
    emits_media: bool | None = ...,
    streams: Any = ...,
    child_calls: bool = ...,
    handles: Any = ...,
) -> Callable[[F], F]: ...


def entrypoint(
    fn: F | None = None,
    *,
    resources: Any = None,
    kind: str = "",
    publishes: bool = False,
    env: Any = None,
    emits_media: bool | None = None,
    streams: Any = None,
    child_calls: bool = False,
    handles: Any = None,
) -> F | Callable[[F], F]:
    """Mark a module-level function as an entrypoint (contract above)."""

    if fn is None:
        def bind(inner: F) -> F:
            return entrypoint(  # type: ignore[call-overload,no-any-return]
                inner,
                resources=resources,
                kind=kind,
                publishes=publishes,
                env=env,
                emits_media=emits_media,
                streams=streams,
                child_calls=child_calls,
                handles=handles,
            )
        return bind

    from .context import RequestContext

    if not inspect.isfunction(fn):
        raise EntrypointDeclarationError(
            f"@entrypoint marks module-level functions, got "
            f"{type(fn).__name__} — the kwargs are resources= (pgw#1396), "
            "publishes=/env=/emits_media= (pgw#1406) and "
            "child_calls=/handles= (pgw#1579/pgw#1580); the Model class header "
            "carries lanes="
        )
    if not isinstance(publishes, bool):
        raise _refuse(
            fn,
            f"publishes= is a bool DECLARATION, got "
            f"{type(publishes).__name__} — it says this function MAY write to "
            "the hub; the request still says where",
        )
    if not isinstance(child_calls, bool):
        raise _refuse(
            fn,
            f"child_calls= is a bool DECLARATION, got "
            f"{type(child_calls).__name__} — it says this function calls OTHER "
            "endpoints, and it is what the hub mints the invoke_child grant "
            "against",
        )
    if emits_media is not None and not isinstance(emits_media, bool):
        raise _refuse(
            fn,
            f"emits_media= is a bool or None (undeclared), got "
            f"{type(emits_media).__name__}",
        )
    declared_kind = str(kind or "").strip().lower()
    if declared_kind and declared_kind not in KINDS:
        raise _refuse(
            fn,
            f"kind= must be one of {KINDS}, got {kind!r} — the vocabulary is "
            "the hub's (internal/functionkind), so a value it does not "
            "normalize would silently become 'inference'",
        )
    declared_env = _validate_env_decl(fn, env)
    delta_type, delta_arms = _delta_declaration(fn, streams)
    declared_handles = _validate_handles_decl(fn, handles)
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

    returns_iterator = _annotation_class(hints.get("return")) in (
        collections.abc.AsyncIterator, collections.abc.AsyncGenerator,
        collections.abc.Iterator, collections.abc.Generator,
    )
    if inspect.isasyncgenfunction(fn) or inspect.isgeneratorfunction(fn) or (
        returns_iterator
    ):
        raise _refuse(
            fn,
            "a generator entrypoint cannot produce the request's authoritative "
            "result — Python forbids `return <value>` inside a generator, and "
            "JobResult is a separate wire channel from the droppable "
            "JobProgress deltas. Declare the chunk type instead and return the "
            "terminal struct: `@entrypoint(streams=TokenDelta)` + "
            "`ctx.emit(TokenDelta(text=...))` in the loop + "
            "`return <YourOutput>(...)` at the end (pgw#1576)",
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
        resources=resources,
        kind=declared_kind,
        publishes=bool(publishes),
        env=declared_env,
        emits_media=emits_media,
        child_calls=bool(child_calls),
        handles=declared_handles,
        ctx_type=ctx_type,
        delta_type=delta_type,
        delta_arms=delta_arms,
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
