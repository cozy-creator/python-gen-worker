"""The publish-time instrumented derive (pgw#1370).

Per declared execution lane, inside the release env, on CPU: instantiate the
author's ``Model`` subclass, run its ``load`` AS-IS against a CONFIG-ONLY
checkpoint tree under ``torchcg.hollow_session``, drive the module's
``@entrypoint`` functions with AUTO-ENUMERATED trace payloads under
instrumented discovery, and stamp the observed graph set -- plus the lane
contracts and the model type's checkpoint-defaults schema -- as the static
release metadata document.

**pgw#1621 re-keyed every lane spelling in this document onto the tensor-layout
v2 STAMP PAIR** — ``"<topology>@N+<quant>@N"``, th#1809's ``LayoutId.render()``,
byte-shared with the hub's ``tensorfs.LayoutID.String``. Three fields carry it
and the hub's ``derive_document.go`` cross-checks all three; they have exactly
one producer here (:func:`lane_contract_handle`) so they cannot disagree. The
per-lane v1 layout ``document`` is DELETED with the v1 corpus: a v2 layout is
``quant(topology)``, computed by the Go engine and never stored, so there is
nothing left for a lane row to inline.

**EVERY model class declares REAL lanes** (Paul's ruling pair, 2026-08-20;
pgw#1597/pgw#1599). A lane answers checkpoint COMPATIBILITY and lane
SELECTION, not merely compilation, so there is no derived, borrowed or
implicit identity to trace under — a class that names no lane is REFUSED at
class definition, before this module ever sees it. **Compilation
participation is the MARK**: a model with
no ``ctx.compile`` call in ``load()`` traces, marks nothing, and is reported
as an unmarked lane — measured, not assumed. There is no eager keyword.

**Coverage is auto-enumerated -- inputs AND bindings** (Paul rulings,
2026-08-19/20; ``ctx.is_trace`` is DELETED from the author surface, so author
code is trace-oblivious and arm coverage is entirely the derive's job):

* payload schemas: one pass per enum-typed field value, every other field at
  its default, required non-defaulted fields synthesized minimally by type;
* adapter state: each injected ``Adapter | None`` parameter enumerates None
  AND a synthesized fake adapter carrying the model type's platform
  ``Lora.Defaults``; each ``list[Adapter]`` enumerates empty and one-fake
  (adapter I/O is neutralized at trace -- fake parameters hold no bytes);
* checkpoint-defaults variants: when the model type's Defaults schema
  carries ``cfg``, both the platform row and its cfg-flipped twin run (they
  change the executed arm and therefore the observed graphs), each under a
  fresh instance + ``load``.

An enumerated combination the author's code REFUSES with ``ValidationError``
is a legitimately impossible serving combination and is skipped (counted,
never silent); any other exception is a derive failure. Shapes/arms the
enumeration cannot express are discovered at the first live request (served
eager, minted in the background) -- enumeration is a pre-warming
completeness aid, never a correctness gate. A cross-product larger than the
cap warns and traces the deterministic prefix. Author code wrapped in
``torch.inference_mode()`` composes fine with the fake-tensor drive -- no
special handling; traced graphs are identical.

torchcg and tensorfs are imported from ``gen_worker._vendor`` -- the SAME
snapshots the serving miner compiles with (``serving/mint_child.py``,
``serving/host.py``, ``serving/hub_store.py`` all import
``gen_worker._vendor.torchcg``). The derive used to import a top-level
``torchcg`` on the theory that it is a release dependency whose pinned rev is
part of env identity; since pgw#1310 vendored both packages that theory is
inverted -- an endpoint-pinned torchcg would let the publish-time TRACE and
the serving-time MINT disagree about the compiler, which is the exact drift
the old refusal claimed to prevent (se#786, pgw#1462).
"""

from __future__ import annotations

import enum
import hashlib
import inspect
import itertools
import json
import types
import traceback
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, Optional

from ..serving.entrypoints import ENTRYPOINT_ATTR
from ..serving.model import (
    Model,
    ModelDeclarationError,
    model_marks_compile,
    model_declared_lanes,
    model_requires,
    model_shapes,
    model_structural,
)
from ..serving.lane_spec import DYNAMIC
from ..serving.model import model_type as _strict_model_type
from .trace_context import (
    StepBudgetReached,
    TraceLoadContext,
    TraceRequestContext,
)

_REAL_TENSOR_BYTES_CEILING = 64 * 1024 * 1024

DOCUMENT_KIND = "gen-worker.release-metadata@1"

ENUM_CAP = 64

TRACE_STEP_BUDGET: Optional[int] = 1

class DeriveError(RuntimeError):
    """The release derive cannot state this endpoint's graph set."""


def dynamic_dim_policy(shapes: Mapping[str, str]) -> Any:
    """Turn a model class's declared shape axes into torchcg's predicate."""

    aspect = shapes.get("aspect") == DYNAMIC
    if not aspect:
        return None

    def policy(_target: str, _name: str, axis: int) -> bool:
        if axis == 0:
            return False
        return axis >= 2

    return policy


class PayloadEnumerationRefused(DeriveError):

    def __init__(self, owner: str, field: str, annotation: Any) -> None:
        self.owner = owner
        self.field = field
        self.annotation = _render_annotation(annotation)
        super().__init__(
            f"{owner}: required payload field {field!r} of type "
            f"{annotation!r} cannot be auto-synthesized for the trace. Give "
            f"it a default, or reshape it so the schema states its axes "
            f"(enum fields enumerate)."
        )


def _render_annotation(annotation: Any) -> str:

    name = getattr(annotation, "__name__", None)
    if name and not typing.get_args(annotation):
        module = getattr(annotation, "__module__", "")
        return f"{module}.{name}" if module and module != "builtins" else str(name)
    return str(annotation).replace("typing.", "")


def model_model_type(cls: type) -> Optional[type]:
    """The class-header model type, or None for an undeclared base."""
    try:
        return _strict_model_type(cls)
    except ModelDeclarationError:
        return None


def lane_contract_handle(owner: str, lane: Any) -> str:
    """The lane's WIRE spelling — ``"<topology>@N+<quant>@N"``.

    pgw#1621: this used to unwrap a tensorfs v1 ``Contract`` object through
    four fallbacks. A lane is now a ``DeclaredLane`` that was fully READ at
    class definition, so the handle is a field read and the only failure left
    is being handed something that is not a lane at all.

    THE ONE PRODUCER of the three lane-spelling fields the hub's
    ``derive_document.go`` cross-checks — the ``lane_contracts`` map KEY, that
    entry's own ``stamp``, and ``graphs.lanes[].contract``. They re-key in
    LOCKSTEP because they are one call, not three transcriptions: the hub
    REFUSES the release (``release_compiled_graphs_invalid_lane``) when the key
    and the stamp disagree, and silently appends a phantom stamp-only lane when
    a ``graphs.lanes[].contract`` names something ``lane_contracts`` does not.
    """

    rendered = getattr(lane, "contract_id", None)
    if isinstance(rendered, str) and rendered:
        return rendered
    raise DeriveError(
        f"{owner}: {lane!r} is not a declared lane — it carries no "
        f"`contract_id`. Lanes reach the derive as `DeclaredLane` rows read at "
        f"class definition (`model_declared_lanes`); the tensorfs v1 Contract "
        f"OBJECT that used to be passed here is deleted (pgw#1621)."
    )


@dataclass(frozen=True)
class ReleaseDeriveResult:
    """The emitted document plus the summary a pipeline log wants."""

    document: bytes
    digest: str
    endpoint: str
    lane_graphs: dict[str, tuple[str, ...]]
    warnings: tuple[str, ...] = ()
    weightless: bool = False
    unmarked_lanes: tuple[str, ...] = ()
    unenumerable_entrypoints: tuple[tuple[str, str], ...] = ()
    unservable_payloads: tuple[str, ...] = ()

    @property
    def eager_permanent(self) -> bool:
        return not self.lane_graphs


def _torchcg() -> ModuleType:

    from .._vendor import torchcg

    return torchcg


def _hollow() -> ModuleType:

    import importlib

    return importlib.import_module("gen_worker._vendor.torchcg.hollow")


def _program_sink(cas_root: Optional[Path]) -> Optional[Any]:

    if cas_root is None:
        return None

    import tempfile

    import torch

    from .._vendor.tensorfs import LocalCAS
    from .._vendor.torchcg.store import LocalGraphStore

    store = LocalGraphStore(LocalCAS(Path(cas_root)))

    def sink(graph: str, program: Any) -> None:
        _assert_weights_free(torch, program)
        with tempfile.TemporaryDirectory() as scratch:
            staged = Path(scratch) / "program.pt2"
            torch.export.save(program, str(staged))
            store.put_program(graph, staged)

    return sink


def _trace_device() -> str:

    return "cuda"


def _assert_weights_free(torch: Any, program: Any) -> None:

    from torch._subclasses.fake_tensor import FakeTensor

    heavy = []
    for holder in ("state_dict", "constants"):
        for name, value in (getattr(program, holder, None) or {}).items():
            if not isinstance(value, torch.Tensor) or isinstance(value, FakeTensor):
                continue
            if value.device.type == "meta":
                continue
            if value.numel() * value.element_size() > _REAL_TENSOR_BYTES_CEILING:
                heavy.append(f"{holder}.{name} ({value.numel() * value.element_size()} bytes)")
    if heavy:
        raise DeriveError(
            f"the derive is about to serialize {len(heavy)} REAL tensor(s) far too "
            f"large to be config-derived buffers into a graph blob: {heavy[:6]!r}. "
            f"A graph artifact carries structure, never weights -- the miner binds "
            f"the checkpoint's real tensors before it compiles. Something was "
            f"loaded with weights instead of hollow."
        )


def _lane_model_class(module: ModuleType) -> tuple[Optional[type], str]:

    found: list[type] = []
    unmarked: list[type] = []
    for value in vars(module).values():
        if not (
            inspect.isclass(value)
            and issubclass(value, Model)
            and value is not Model
            and getattr(value, "__module__", None) == module.__name__
        ):
            continue
        try:
            # pgw#1621: the re-read loop that stood here is DELETED with its
            # reason. pgw#1391 added it because `model_lanes` handed back
            # UNREAD v1 contract objects whose `dtype` was a property that
            # could raise, so a dtype-less lane cleared declaration and died at
            # load on a rented pod. `model_declared_lanes` answers rows that
            # were fully read at class definition — dtype and sm floor off the
            # ratified quant rule — so there is nothing left here to re-read.
            model_declared_lanes(value)
            marked = model_marks_compile(value)
        except ModelDeclarationError as exc:
            raise DeriveError(str(exc)) from exc
        (found if marked else unmarked).append(value)
    if len(found) > 1:
        raise DeriveError(
            f"module {module.__name__!r} has more than one COMPILE-MARKING "
            f"model class ({[cls.__name__ for cls in found]!r}); a release "
            f"derives ONE. An auxiliary model that another slot drives simply "
            f"calls no `ctx.compile()` in its `load()` — that is the entire "
            f"eager declaration (Paul, 2026-08-20), and there is no keyword "
            f"for it."
        )
    if found:
        return (found[0], "")
    if len(unmarked) > 1:
        raise DeriveError(
            f"module {module.__name__!r} has more than one model class "
            f"({[cls.__name__ for cls in unmarked]!r}) and NONE of them marks "
            f"a compile target, so which one the release is ABOUT cannot be "
            f"read. Mark the compiled one via `ctx.compile()` in its `load()`; "
            f"an auxiliary model that another slot drives marks nothing and is "
            f"then unambiguous."
        )
    return (unmarked[0] if unmarked else None, "")


@dataclass(frozen=True)
class _Entrypoint:

    name: str
    fn: Any
    payload_param: str
    payload_type: type
    model_param: Optional[str]
    ctx_param: str
    injected: tuple[tuple[str, Any, Any], ...]
    model_slots: tuple[tuple[str, type], ...] = ()


def _injected_trace_value(name: str, parameter_name: str, annotation: Any) -> Any:

    if _optional_none(annotation):
        return None
    origin = typing.get_origin(annotation)
    if origin in (list, tuple) or (
        isinstance(origin, type) and origin.__name__ in ("Sequence", "list", "tuple")
    ):
        return [] if origin is not tuple else ()
    raise DeriveError(
        f"@entrypoint {name}: injected parameter {parameter_name!r} of type "
        f"{annotation!r} has no trace value (Optional facts inject None, "
        f"sequence facts inject empty)"
    )


_SLOT_ORDER = ("ctx", "payload", "model", "adapter")


def _entrypoints(
    module: ModuleType, model_cls: Optional[type]
) -> list[_Entrypoint]:
    import msgspec

    from ..request_context import RequestContext

    out: list[_Entrypoint] = []
    for name, fn in sorted(vars(module).items()):
        if not (inspect.isfunction(fn) and getattr(fn, ENTRYPOINT_ATTR, False)):
            continue
        hints = typing.get_type_hints(fn)
        parameters = list(inspect.signature(fn).parameters.values())
        payload_param = model_param = ctx_param = None
        payload_type: Any = None
        rest: list[tuple[str, Any]] = []
        model_slots: list[tuple[str, type]] = []
        roles: list[tuple[str, str]] = []
        for parameter in parameters:
            annotation = _strip_annotated(hints.get(parameter.name))
            if isinstance(annotation, type) and issubclass(annotation, msgspec.Struct):
                payload_param, payload_type = parameter.name, annotation
                roles.append((parameter.name, "payload"))
            elif isinstance(annotation, type) and issubclass(annotation, Model):
                model_slots.append((parameter.name, annotation))
                if annotation is model_cls and model_param is None:
                    model_param = parameter.name
                roles.append((parameter.name, "model"))
            elif typing.get_origin(annotation) is RequestContext or (
                isinstance(annotation, type) and issubclass(annotation, RequestContext)
            ):
                ctx_param = parameter.name
                roles.append((parameter.name, "ctx"))
            else:
                rest.append((parameter.name, hints.get(parameter.name)))
                roles.append((parameter.name, "adapter"))
        if model_cls is None:
            if model_slots:
                continue
        elif model_param is None:
            continue
        if payload_param is None:
            raise DeriveError(
                f"@entrypoint {name}: no parameter annotates a msgspec "
                f"payload struct"
            )
        if ctx_param is None:
            if len(rest) == 1:
                ctx_param = rest.pop(0)[0]
                roles = [
                    (item[0], "ctx" if item[0] == ctx_param else item[1])
                    for item in roles
                ]
            else:
                raise DeriveError(
                    f"@entrypoint {name}: cannot identify the ctx parameter "
                    f"among {[item[0] for item in rest]!r}; annotate it "
                    f"RequestContext"
                )
        _check_slot_order(name, roles)
        injected = tuple(
            (
                parameter_name,
                annotation,
                _injected_trace_value(name, parameter_name, _strip_annotated(annotation)),
            )
            for parameter_name, annotation in rest
        )
        out.append(
            _Entrypoint(
                name=name,
                fn=fn,
                payload_param=payload_param,
                payload_type=payload_type,
                model_param=model_param,
                ctx_param=ctx_param,
                injected=injected,
                model_slots=tuple(model_slots),
            )
        )
    if not out and model_cls is not None:
        raise DeriveError(
            f"no @entrypoint function binds model class {model_cls.__name__!r} "
            f"(the model parameter's annotation is the binding)"
        )
    return out


def _check_slot_order(name: str, roles: list[tuple[str, str]]) -> None:

    ranks = [_SLOT_ORDER.index(role) for _, role in roles]
    if ranks != sorted(ranks):
        spelled = ", ".join(f"{param}: {role}" for param, role in roles)
        raise DeriveError(
            f"@entrypoint {name}: parameters are out of the ruled order. "
            f"An entrypoint reads (ctx, payload, model(s), adapter(s)); got "
            f"({spelled})"
        )


def _strip_annotated(annotation: Any) -> Any:

    while typing.get_origin(annotation) is typing.Annotated:
        annotation = typing.get_args(annotation)[0]
    return annotation


def _optional_none(annotation: Any) -> bool:
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        return type(None) in typing.get_args(annotation)
    return False


def _synthesize_field(owner: str, name: str, annotation: Any) -> Any:

    annotation = _strip_annotated(annotation)
    if annotation is str:
        return "trace"
    if annotation is int:
        return 0
    if annotation is float:
        return 0.0
    if annotation is bool:
        return False
    if _optional_none(annotation):
        return None
    raise PayloadEnumerationRefused(owner, name, annotation)


def _literal_axis(annotation: Any) -> Optional[list[Any]]:

    stripped = _strip_annotated(annotation)
    if typing.get_origin(stripped) is typing.Literal:
        values = list(typing.get_args(stripped))
    else:
        origin = typing.get_origin(stripped)
        if origin is not typing.Union and origin is not types.UnionType:
            return None
        values = []
        for argument in typing.get_args(stripped):
            inner = _strip_annotated(argument)
            if typing.get_origin(inner) is typing.Literal:
                values.extend(typing.get_args(inner))
            elif inner is type(None):
                values.append(None)
            else:
                return None
    if not values:
        return None
    if not all(
        value is None or (isinstance(value, (int, float)) and not isinstance(value, bool))
        for value in values
    ):
        return None
    return values


def _payload_field_names(payload_type: type) -> tuple[str, ...]:

    import msgspec

    try:
        return tuple(field.name for field in msgspec.structs.fields(payload_type))
    except TypeError:
        return ()


def _auto_payloads(
    owner: str,
    payload_type: type,
    structural: Mapping[str, Any] = MappingProxyType({}),
) -> tuple[tuple[Any, ...], bool]:

    import msgspec

    try:
        struct_fields = msgspec.structs.fields(payload_type)
    except TypeError as exc:
        raise DeriveError(
            f"{owner}: payload type {payload_type!r} is not a msgspec struct: {exc}"
        ) from exc

    enum_axes: list[tuple[str, list[Any]]] = []
    base: dict[str, Any] = {}
    declared_axes: dict[str, list[Any]] = {}
    for axis, declaration in structural.items():
        variants = declaration.variants()
        if not any(field.name == declaration.field for field in struct_fields):
            continue
        declared_axes[declaration.field] = [value for _, value in variants]
    for field in struct_fields:
        annotation = _strip_annotated(field.type)
        declared_values = declared_axes.pop(field.name, None)
        if declared_values is not None:
            enum_axes.append((field.name, declared_values))
            continue
        if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
            values = list(annotation)
            if not values:
                raise DeriveError(
                    f"{owner}: payload field {field.name!r} enumerates an "
                    f"EMPTY enum"
                )
            enum_axes.append((field.name, values))
            continue
        literal_values = _literal_axis(annotation)
        if literal_values is not None:
            if not literal_values:
                raise DeriveError(
                    f"{owner}: payload field {field.name!r} enumerates an "
                    f"EMPTY Literal"
                )
            enum_axes.append((field.name, literal_values))
            continue
        if field.required:
            base[field.name] = _synthesize_field(owner, field.name, annotation)

    if not enum_axes:
        return (payload_type(**base),), False

    default_combo = tuple(
        field.default if field.default in values else values[0]
        for (name, values), field in zip(
            enum_axes,
            [
                next(f for f in struct_fields if f.name == name)
                for name, _ in enum_axes
            ],
        )
    )
    names = [name for name, _ in enum_axes]
    ordered_combos = itertools.chain(
        [default_combo],
        (
            combo
            for combo in itertools.product(*[values for _, values in enum_axes])
            if combo != default_combo
        ),
    )
    payloads: list[Any] = []
    capped = False
    for index, combo in enumerate(ordered_combos):
        if index >= ENUM_CAP:
            capped = True
            break
        payloads.append(payload_type(**base, **dict(zip(names, combo))))
    return tuple(payloads), capped


_ADAPTER_PICK: dict[str, Any] = {
    "type": "object",
    "properties": {
        "ref": {"type": "string", "description": "org/repo@release"},
        "scale": {"type": "number", "default": 1.0},
    },
    "required": ["ref"],
    "additionalProperties": False,
}


def _envelope_schema(plan: "_Entrypoint") -> dict[str, Any]:

    import msgspec

    models: dict[str, Any] = {}
    for slot_name, slot_cls in plan.model_slots:
        models[slot_name] = {
            "type": "string",
            "description": (
                f"pinned hub ref of the checkpoint bound to this "
                f"{slot_cls.__name__} slot"
            ),
        }

    adapters: dict[str, Any] = {}
    required_adapters: list[str] = []
    for parameter_name, annotation, _base in plan.injected:
        adapter_cls = _adapter_arm_class(annotation)
        stripped = _strip_annotated(annotation)
        if adapter_cls is None:
            continue
        origin = typing.get_origin(stripped)
        if origin in (list, tuple) or any(
            typing.get_origin(argument) in (list, tuple)
            for argument in typing.get_args(stripped)
        ):
            adapters[parameter_name] = {
                "type": "array",
                "items": _ADAPTER_PICK,
                "default": [],
                "x-adapter-type": adapter_cls.__name__,
            }
        else:
            entry = dict(_ADAPTER_PICK)
            entry["x-adapter-type"] = adapter_cls.__name__
            if _optional_none(stripped):
                adapters[parameter_name] = {"anyOf": [entry, {"type": "null"}],
                                            "default": None}
            else:
                adapters[parameter_name] = entry
                required_adapters.append(parameter_name)

    properties: dict[str, Any] = {"input": msgspec.json.schema(plan.payload_type)}
    required = ["input"]
    if models:
        properties["models"] = {
            "type": "object",
            "properties": models,
            "additionalProperties": False,
        }
    if adapters:
        properties["adapters"] = {
            "type": "object",
            "properties": adapters,
            "required": sorted(required_adapters),
            "additionalProperties": False,
        }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": plan.name,
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _fake_adapter(
    model_type: Optional[type], adapter_cls: Optional[type] = None
) -> Any:

    from ..serving.context import Adapter

    if adapter_cls is None:
        adapter_cls = Adapter
    lora = getattr(model_type, "Lora", None)
    defaults_type = getattr(lora, "Defaults", None)
    return adapter_cls(
        name="trace-adapter",
        path=Path("/nonexistent/trace-adapter"),
        defaults=defaults_type() if defaults_type is not None else None,
        ref="trace/adapter@0",
    )


def _adapter_arm_class(annotation: Any) -> Optional[type]:
    from ..serving.context import Adapter

    stack = [_strip_annotated(annotation)]
    while stack:
        current = stack.pop()
        if isinstance(current, type) and issubclass(current, Adapter):
            return current
        stack.extend(_strip_annotated(argument) for argument in typing.get_args(current))
    return None


def _injected_axes(
    plan: "_Entrypoint", model_type: Optional[type]
) -> list[list[tuple[str, Any]]]:

    axes: list[list[tuple[str, Any]]] = []
    for parameter_name, annotation, base_value in plan.injected:
        adapter_cls = _adapter_arm_class(annotation)
        if adapter_cls is not None:
            fake = _fake_adapter(model_type, adapter_cls)
            if _optional_none(_strip_annotated(annotation)):
                values: list[Any] = [None, fake]
            else:
                values = [base_value, [fake]]
            axes.append([(parameter_name, value) for value in values])
        else:
            axes.append([(parameter_name, base_value)])
    return axes


def _defaults_variants(model_type: Optional[type]) -> list[Any]:

    if model_type is None:
        return [None]
    defaults_type = getattr(model_type, "Defaults", model_type)
    try:
        instance = defaults_type()
    except Exception:
        return [None]
    variants: list[Any] = [instance]
    import msgspec

    try:
        field_names = {field.name for field in msgspec.structs.fields(defaults_type)}
    except TypeError:
        return variants
    if "cfg" in field_names:
        try:
            variants.append(msgspec.structs.replace(instance, cfg=not instance.cfg))
        except Exception:
            pass
    return variants


PROVENANCE_MAX_DEPTH = 4


def _components_mapping(value: Any) -> Optional[Mapping[str, Any]]:

    try:
        components = getattr(value, "components", None)
    except Exception:
        return None
    return components if isinstance(components, Mapping) else None


def _is_wrapper(value: Any) -> bool:

    import torch

    if isinstance(value, (torch.nn.Module, torch.Tensor)):
        return False
    if isinstance(value, (str, bytes, int, float, bool, Path)):
        return False
    return hasattr(value, "__dict__") and isinstance(getattr(value, "__dict__"), dict)


def _named_marked_modules(instance: Any, marked: list[Any]) -> dict[str, Any]:

    candidates: dict[int, str] = {}
    component_rows: list[tuple[str, str, Any]] = []
    truncated: list[str] = []

    def offer(module: Any, name: str) -> None:
        identity = id(module)
        if identity not in candidates or len(name) < len(candidates[identity]):
            candidates[identity] = name

    visited: set[int] = {id(instance)}

    def descend(node: Any, path: tuple[str, ...], depth: int) -> None:
        try:
            attributes = sorted(vars(node).items())
        except TypeError:
            return
        for attr, value in attributes:
            if value is None:
                continue
            here = (*path, attr)
            name = ".".join(here)
            offer(value, name)
            components = _components_mapping(value)
            if components is not None:
                for component_name, component in sorted(components.items()):
                    if component is None or not isinstance(component_name, str):
                        continue
                    component_rows.append((name, component_name, component))
                continue
            if not _is_wrapper(value):
                continue
            if id(value) in visited:
                continue
            if depth + 1 > PROVENANCE_MAX_DEPTH:
                truncated.append(name)
                continue
            visited.add(id(value))
            descend(value, here, depth + 1)

    descend(instance, (), 0)

    by_name: dict[str, list[Any]] = {}
    for _owner, component_name, component in component_rows:
        by_name.setdefault(component_name, []).append(component)
    for owner, component_name, component in component_rows:
        unique = all(other is component for other in by_name[component_name])
        offer(component, component_name if unique else f"{owner}.{component_name}")

    named: dict[str, Any] = {}
    for module in marked:
        name = candidates.get(id(module))
        if name is None:
            raise DeriveError(
                f"a module marked via ctx.compile() "
                f"({type(module).__name__}) is not reachable as a model "
                f"attribute or pipeline component; the release document "
                f"cannot name its provenance"
                + (
                    f". The provenance walk stopped at depth "
                    f"{PROVENANCE_MAX_DEPTH} on {sorted(set(truncated))!r}; if "
                    f"the module lives below one of those, it is nested "
                    f"deeper than the derive will look."
                    if truncated
                    else ""
                )
            )
        if name in named and named[name] is not module:
            raise DeriveError(
                f"two marked modules both resolve to provenance name "
                f"{name!r}; the release document names a graph's provenance "
                f"and cannot pick between them. Give one of them a distinct "
                f"attribute or component name."
            )
        named[name] = module
    return named


def _compile_stack_from_lockfile(lockfile: Path) -> tuple[tuple[str, str], ...]:

    from ..env_identity import EnvIdentityError, compile_stack_from_lockfile

    try:
        return compile_stack_from_lockfile(lockfile)
    except EnvIdentityError as exc:
        raise DeriveError(str(exc)) from exc


def _defaults_schema(model_type: Optional[type]) -> Optional[dict[str, Any]]:

    if model_type is None:
        return None
    defaults_type = getattr(model_type, "Defaults", model_type)
    import msgspec

    return msgspec.json.schema(defaults_type)


_SAFETENSORS_DTYPES = {
    "BF16": "bfloat16",
    "F16": "float16",
    "F32": "float32",
    "F64": "float64",
    "F8_E4M3": "float8_e4m3fn",
    "F8_E5M2": "float8_e5m2",
}


def _torch_dtype(value: Any) -> Any:
    import torch

    if value is None or isinstance(value, torch.dtype):
        return value
    if isinstance(value, str):
        spelled = _SAFETENSORS_DTYPES.get(value.upper(), value.lower())
        candidate = getattr(torch, spelled, None)
        if isinstance(candidate, torch.dtype):
            return candidate
    raise DeriveError(f"contract dtype {value!r} names no torch dtype")


# ── The per-lane v1 layout DOCUMENT is DELETED (pgw#1621) ───────────────────
#
# `_contract_document` and `_contract_digest` stood here and emitted
# `lane_contracts[<lane>].document` — the lane's whole tensorfs v1 contract
# document, inlined — plus the producer's digest of it. Both are gone, because
# the thing they carried no longer exists.
#
# There IS no per-lane v1 document under v2. A layout is `quant(topology)`,
# COMPUTED by the Go engine and never stored (tensorfs `spec/v2/README.md`), so
# there is nothing for a lane row to inline: N topologies and M rules replace
# N x M documents. What identifies the lane is the STAMP PAIR, which the row
# already carries and which the hub's own corpus resolves.
#
# The hub side is already there: `release_compiled_graphs.lane_contract.document`
# is `omitempty`, and th#2250 narrowed the ingest gate to read the STAMP and
# nothing else. Omitting the field is what the gate now wants — the SPELLING is
# what it cross-checks, and it cross-checks it three ways (see
# :func:`lane_contract_handle`).
#
# Nothing in this repo READ the field. The three near-misses, checked:
# `discovery/entrypoints_v2.py` and `discovery/validation.py` mention
# `lane_contracts` in PROSE only; `serving/hub_store.py` transcribes the hub's
# th#2133 answer shape in a docstring and consumes `lane`/`graphs` from it, not
# `lane_contract.document`. The rest of the hits are test fixtures.


def _resolve_lane(torchcg: ModuleType, lane: Any) -> Any:
    """The resolved ``ctx.lane``: the pair render plus a REAL torch dtype.

    pgw#1621: both halves are now FIELD READS off a ``DeclaredLane``, so the
    refusals that stood here are gone with what made them possible. The dtype
    used to be a tensorfs ``Contract`` property that could RAISE (a v1 document
    could simply omit its top-level dtype); a v2 quant rule's
    ``declared_dtype`` is required by the rule schema and read at class
    definition, so a dtype-less lane is inexpressible rather than guarded.
    """

    return torchcg.LaneRef(lane.contract_id, dtype=_torch_dtype(lane.dtype))


def endpoint_source_root(module: ModuleType) -> Optional[Path]:
    """The SOURCE ROOT the author's modules live under, or None if unknowable."""

    import sys

    name = (getattr(module, "__package__", "") or module.__name__).split(".")[0]
    top = sys.modules.get(name) or module
    path = getattr(top, "__file__", None) or getattr(module, "__file__", None)
    if not path:
        return None
    try:
        here = Path(path).resolve()
    except OSError:
        return None
    if here.name == "__init__.py":
        return here.parent.parent
    return here.parent


def _third_party_root(where: Path) -> bool:

    return any(
        part in ("site-packages", "dist-packages", "__pypackages__")
        for part in where.parts
    )


def _sdk_root() -> Path:
    import gen_worker

    return Path(gen_worker.__file__).resolve().parent


def deepest_endpoint_frame(
    exc: BaseException, endpoint_root: Optional[Path]
) -> Optional[traceback.FrameSummary]:
    """The DEEPEST frame of ``exc``, but only if the author's code raised it."""

    if endpoint_root is None:
        return None
    frames = traceback.extract_tb(exc.__traceback__)
    if not frames:
        return None
    deepest = frames[-1]
    try:
        where = Path(deepest.filename).resolve()
    except OSError:
        return None
    if where.is_relative_to(_sdk_root()):
        return None
    if _third_party_root(where):
        return None
    if not where.is_relative_to(endpoint_root):
        return None
    return deepest


def _derive_lane(
    torchcg: ModuleType,
    cls: type,
    lane: Any,
    plans: list[tuple[_Entrypoint, tuple[Any, ...]]],
    checkpoint_dir: Path,
    warnings: list[str],
    program_sink: Optional[Any] = None,
    slot_checkpoints: Mapping[str, Path] = MappingProxyType({}),
    endpoint_root: Optional[Path] = None,
    unservable: Optional[list[dict[str, Any]]] = None,
    dynamic_dims: Any = None,
) -> Optional[Any]:

    from ..api.errors import ValidationError

    hollow = _hollow()
    handle = lane_contract_handle(f"class {cls.__name__!r}", lane)
    resolved = _resolve_lane(torchcg, lane)
    model_type = model_model_type(cls)
    merged: dict[str, Any] = {}
    all_targets: set[str] = set()
    observed_targets: set[str] = set()
    refused = 0
    total_combos = 0

    for defaults_instance in _defaults_variants(model_type):
        model = cls()
        load_ctx = TraceLoadContext(
            lane=resolved,
            checkpoint_dir=checkpoint_dir,
            model_type=model_type,
            defaults_instance=defaults_instance,
        )
        request_ctx = TraceRequestContext(
            lane=resolved,
            checkpoint_ref=f"trace:{checkpoint_dir.name}",
            step_budget=TRACE_STEP_BUDGET,
        )
        with hollow.hollow_session(
            _trace_device(), dtype_for=load_ctx.component_dtype
        ) as session:
            try:
                model.load(load_ctx)
            except hollow.HollowError as exc:
                raise DeriveError(f"lane {handle!r}: {exc}") from exc
            except Exception as exc:
                raise DeriveError(
                    f"lane {handle!r}: load() failed under the trace "
                    f"session: {type(exc).__name__}: {exc}"
                ) from exc
            if not load_ctx.marked_modules:
                return None
            modules = _named_marked_modules(model, load_ctx.marked_modules)

            aides: dict[str, Any] = {}
            for plan, _payloads in plans:
                for slot_name, slot_cls in plan.model_slots:
                    if slot_cls is cls or slot_name in aides:
                        continue
                    slot_tree = slot_checkpoints.get(slot_name, checkpoint_dir)
                    aide = slot_cls()
                    try:
                        aide.load(
                            TraceLoadContext(
                                lane=resolved,
                                checkpoint_dir=slot_tree,
                                model_type=model_model_type(slot_cls),
                                defaults_instance=None,
                            )
                        )
                    except Exception as exc:
                        shared = (
                            " (the PRIMARY checkpoint — this slot has no "
                            "--checkpoint-ref of its own; an auxiliary model "
                            "with a separate checkpoint needs "
                            f"`--checkpoint-ref {slot_name}=<ref>`)"
                            if slot_name not in slot_checkpoints
                            else ""
                        )
                        raise DeriveError(
                            f"lane {handle!r}: entrypoint {plan.name!r} slot "
                            f"{slot_name!r} ({slot_cls.__name__}) failed to "
                            f"load from {slot_tree}{shared} under the trace "
                            f"session: "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc
                    aides[slot_name] = aide

            def drive() -> None:
                nonlocal refused, total_combos
                for plan, payloads in plans:
                    axes = _injected_axes(plan, model_type)
                    slots = {
                        slot_name: (
                            model if slot_cls is cls else aides[slot_name]
                        )
                        for slot_name, slot_cls in plan.model_slots
                    }
                    for binding in itertools.product(*axes) if axes else [()]:
                        for index, payload in enumerate(payloads):
                            total_combos += 1
                            try:
                                plan.fn(**{
                                    plan.payload_param: payload,
                                    plan.ctx_param: request_ctx,
                                    **slots,
                                    **dict(binding),
                                })
                            except StepBudgetReached:
                                pass
                            except ValidationError:
                                refused += 1
                            except Exception as exc:
                                frame = deepest_endpoint_frame(exc, endpoint_root)
                                if frame is None or unservable is None:
                                    raise DeriveError(
                                        f"lane {handle!r}: entrypoint "
                                        f"{plan.name!r} failed on auto-enumerated "
                                        f"payload {index} ({payload!r}) with "
                                        f"binding {dict(binding)!r} under the "
                                        f"trace session: "
                                        f"{type(exc).__name__}: {exc}"
                                    ) from exc
                                row = {
                                    "entrypoint": plan.name,
                                    "payload": index,
                                    "binding": {
                                        str(k): str(v)
                                        for k, v in dict(binding).items()
                                    },
                                    "frame": (
                                        f"{Path(frame.filename).name}:"
                                        f"{frame.lineno} in {frame.name}"
                                    ),
                                    "error": f"{type(exc).__name__}: {exc}",
                                }
                                if row not in unservable:
                                    unservable.append(row)

            try:
                lane_graphs = torchcg.discover_modules(
                    handle, modules, drive, program_sink=program_sink,
                    session=session, dynamic_dims=dynamic_dims,
                )
                if set(lane_graphs.targets) - {
                    record.target for record in lane_graphs.graphs
                }:
                    request_ctx.step_budget = None
                    lane_graphs = torchcg.discover_modules(
                        handle, modules, drive, program_sink=program_sink,
                        session=session, dynamic_dims=dynamic_dims,
                    )
            except DeriveError:
                raise
            except torchcg.DiscoveryError as exc:
                raise DeriveError(f"lane {handle!r}: {exc}") from exc
        all_targets.update(lane_graphs.targets)
        for record in lane_graphs.graphs:
            merged.setdefault(record.graph, record)
            observed_targets.add(record.target)

    if refused:
        warnings.append(
            f"lane {handle}: {refused}/{total_combos} enumerated "
            f"combination(s) refused by the author's own validation "
            f"(impossible servings; skipped)"
        )
    unobserved = tuple(sorted(all_targets - observed_targets))
    if unobserved:
        raise DeriveError(
            f"lane {handle!r}: marked module(s) {list(unobserved)!r} were "
            f"never CALLED while driving {total_combos} auto-enumerated "
            f"combination(s). ctx.compile must mark the module the code "
            f"actually CALLS (e.g. the vae's .decoder, not the vae, when "
            f"only .decode() runs) -- silent zero-graph discovery is not an "
            f"outcome."
        )
    return torchcg.LaneGraphs(
        contract=handle,
        targets=tuple(sorted(all_targets)),
        graphs=tuple(merged.values()),
        unobserved_targets=(),
    )


def derive_release(
    module: ModuleType,
    *,
    checkpoint_dir: Path,
    lockfile: Optional[Path] = None,
    graph_cas: Optional[Path] = None,
    slot_checkpoints: Mapping[str, Path] = MappingProxyType({}),
) -> ReleaseDeriveResult:
    """Derive the release metadata document for one endpoint module."""

    torchcg = _torchcg()
    program_sink = _program_sink(graph_cas)

    if lockfile is None:
        raise DeriveError(
            "a derive states the compile stack it traced under, and that is "
            "read from the endpoint's uv.lock: pass `lockfile=`. Restating "
            "the installed set instead is what pgw#1489 deleted — it is a "
            "second representation of the environment the lock already pins"
        )
    stack = _compile_stack_from_lockfile(lockfile)

    cls, _ = _lane_model_class(module)
    endpoint_name = f"{module.__name__}:{cls.__name__ if cls else ''}".rstrip(":")

    lanes: list[Any] = []
    unmarked_lanes: list[str] = []
    lane_contracts: dict[str, Any] = {}
    entrypoints: dict[str, Any] = {}
    warnings: list[str] = []
    plans: list[tuple[_Entrypoint, tuple[Any, ...]]] = []
    unenumerable: list[tuple[str, str]] = []
    unservable_payloads: list[dict[str, Any]] = []
    for plan in _entrypoints(module, cls):
        owner = f"@entrypoint {plan.name}"
        refusal: Optional[PayloadEnumerationRefused] = None
        if cls is None:
            payloads: tuple[Any, ...] = ()
        else:
            try:
                payloads, capped = _auto_payloads(
                    owner, plan.payload_type, model_structural(cls)
                )
            except PayloadEnumerationRefused as exc:
                refusal = exc
                payloads = ()
                capped = False
            if capped:
                warnings.append(
                    f"{owner}: enum cross-product exceeds the cap "
                    f"({ENUM_CAP}); tracing the deterministic prefix -- the "
                    f"rest is first-encounter discovery (eager + background "
                    f"mint)"
                )
        if refusal is None:
            plans.append((plan, payloads))
        entrypoints[plan.name] = {
            "envelope_schema": _envelope_schema(plan),
            "model_slots": {
                slot_name: slot_cls.__name__
                for slot_name, slot_cls in plan.model_slots
            },
            "traced_passes": len(payloads),
        }
        if refusal is not None:
            entrypoints[plan.name]["unenumerable"] = {
                "field": refusal.field,
                "type": refusal.annotation,
                "reason": "payload_field_not_synthesizable",
            }
            unenumerable.append((plan.name, str(refusal)))
            warnings.append(
                f"{owner}: NOT enumerated — {refusal.field!r} "
                f"({refusal.annotation}) cannot be synthesized. This "
                f"entrypoint has no traced coverage; it serves eager and "
                f"mints on first encounter. Every other entrypoint is "
                f"unaffected."
            )

    if cls is not None:
        reachable = {
            field_name
            for plan, _ in plans
            for field_name in _payload_field_names(plan.payload_type)
        }
        for axis, declaration in model_structural(cls).items():
            if declaration.field not in reachable:
                warnings.append(
                    f"class {cls.__name__!r}: structural axis {axis!r} names "
                    f"payload field {declaration.field!r}, which NO derived "
                    f"entrypoint carries — it enumerated nothing. Either the "
                    f"field was renamed, or the axis belongs on a different "
                    f"model class."
                )
        requires = model_requires(cls)
        for lane in model_declared_lanes(cls):
            lane_graphs = _derive_lane(
                torchcg, cls, lane, plans, checkpoint_dir, warnings,
                program_sink=program_sink,
                slot_checkpoints=slot_checkpoints,
                endpoint_root=endpoint_source_root(module),
                unservable=unservable_payloads,
                dynamic_dims=dynamic_dim_policy(model_shapes(cls)),
            )
            if lane_graphs is None:
                unmarked_lanes.append(
                    lane_contract_handle(f"class {cls.__name__!r}", lane)
                )
                continue
            lanes.append(lane_graphs)
            # THE SECOND of the three lane-spelling fields the hub cross-checks.
            # `lane_graphs.contract` is `lane_contract_handle`'s answer for THIS
            # lane, carried through `_derive_lane`, and it is the same object
            # that keys the map below and that `graphs.lanes[].contract`
            # carries — one producer, so the three cannot drift into
            # `release_compiled_graphs_invalid_lane`.
            #
            # `document` and `digest` are GONE (pgw#1621): a v2 layout is
            # computed, not stored, so there is no per-lane document to inline.
            # See the block above `_resolve_lane`.
            entry: dict[str, Any] = {"stamp": lane_graphs.contract}
            floor = requires.get(lane_graphs.contract)
            if floor is not None:
                entry["requires"] = floor.render()
            # THE FIRST field: the map KEY. Key != stamp is the hub's hard
            # refusal `release_compiled_graphs_invalid_lane`, which is why this
            # is the same expression and not a second spelling of it.
            lane_contracts[lane_graphs.contract] = entry

    for row in unservable_payloads:
        row_entry = entrypoints.get(str(row["entrypoint"]))
        if row_entry is None:
            continue
        skipped: list[Any] = row_entry.setdefault("unservable", [])
        skipped.append({k: v for k, v in row.items() if k != "entrypoint"})
    for row in unservable_payloads:
        warnings.append(
            f"@entrypoint {row['entrypoint']}: payload {row['payload']} is "
            f"UNSERVABLE and was skipped — {row['error']} (at {row['frame']}). "
            f"Its graphs are not in this document; every other payload is "
            f"unaffected."
        )

    graphs_document = torchcg.GraphSetDocument(stack=stack, lanes=tuple(lanes))
    payload_dict: dict[str, Any] = {
        "v": 1,
        "kind": DOCUMENT_KIND,
        "endpoint": endpoint_name,
        # THE THIRD field lives in here: `graphs.lanes[i].contract`, which is
        # the `contract=` each `LaneGraphs` was built with — again
        # `lane_contract_handle`'s one answer. A `graphs.lanes[].contract` the
        # `lane_contracts` map does not carry does NOT refuse at the hub: it
        # silently appends a phantom stamp-only lane, which is the failure mode
        # that makes the lockstep matter more than the refusal does.
        "graphs": graphs_document.as_dict(),
        "lane_contracts": lane_contracts,
        "entrypoints": entrypoints,
        "fork_axes": {
            "structural": [
                declaration.as_document(axis)
                for axis, declaration in (
                    model_structural(cls) if cls is not None else {}
                ).items()
            ],
            "shapes": model_shapes(cls) if cls is not None else {},
        },
        "checkpoint_defaults_schema": _defaults_schema(
            model_model_type(cls) if cls is not None else None
        ),
        "model_type": (
            getattr(model_model_type(cls), "__name__", None)
            if cls is not None
            else None
        ),
    }
    document = json.dumps(
        payload_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return ReleaseDeriveResult(
        document=document,
        digest=hashlib.sha256(document).hexdigest(),
        endpoint=endpoint_name,
        lane_graphs={
            lane.contract: tuple(record.graph for record in lane.graphs)
            for lane in graphs_document.lanes
        },
        warnings=tuple(warnings),
        weightless=cls is None and bool(plans),
        unmarked_lanes=tuple(unmarked_lanes),
        unenumerable_entrypoints=tuple(unenumerable),
        unservable_payloads=tuple(
            f"{r['entrypoint']}[{r['payload']}]: {r['error']} (at {r['frame']})"
            for r in unservable_payloads
        ),
    )


__all__ = [
    "DOCUMENT_KIND",
    "ENUM_CAP",
    "DeriveError",
    "PayloadEnumerationRefused",
    "ReleaseDeriveResult",
    "derive_release",
]
