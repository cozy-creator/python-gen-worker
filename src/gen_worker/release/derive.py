"""The publish-time instrumented derive (pgw#1370).

Per declared execution lane, inside the release env, on CPU: instantiate the
author's ``Model`` subclass, run its ``load`` AS-IS against a CONFIG-ONLY
checkpoint tree under ``torchcg.hollow_session``, drive the module's
``@entrypoint`` functions with AUTO-ENUMERATED trace payloads under
instrumented discovery, and stamp the observed graph set -- plus the lane
contracts and the model type's checkpoint-defaults schema -- as the static
release metadata document.

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

torchcg is imported TOP-LEVEL and lazily: the derive runs inside the release
env, torchcg is a release dependency there, and the env's pinned rev is the
one whose version already sits in the lockfile-closure env identity. A
missing torchcg is a typed refusal, never a silent fallback.
"""

from __future__ import annotations

import enum
import hashlib
import inspect
import itertools
import json
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

from ..api.entrypoint import ENTRYPOINT_ATTR
from ..api.model_base import (
    Model,
    lane_contract_handle,
    model_lanes,
    model_model_type,
)
from .trace_context import TraceLoadContext, TraceRequestContext

DOCUMENT_KIND = "gen-worker.release-metadata@1"

#: Auto-enumeration cross-product cap. Overflow warns and traces the
#: deterministic prefix (field declaration order x enum declaration order);
#: the rest is first-encounter discovery.
ENUM_CAP = 64


class DeriveError(RuntimeError):
    """The release derive cannot state this endpoint's graph set."""


@dataclass(frozen=True)
class ReleaseDeriveResult:
    """The emitted document plus the summary a pipeline log wants."""

    document: bytes
    digest: str
    endpoint: str
    lane_graphs: dict[str, tuple[str, ...]]  # lane contract -> graph hashes
    warnings: tuple[str, ...] = ()

    @property
    def eager_permanent(self) -> bool:
        return not self.lane_graphs


def _torchcg() -> ModuleType:
    try:
        import torchcg
    except ImportError as exc:
        raise DeriveError(
            "gen-worker release derive runs INSIDE the release env, and "
            "torchcg is a release dependency there (the env's pinned rev is "
            "part of env identity). Install/pin torchcg in the endpoint's "
            "environment; gen-worker deliberately does not bundle it."
        ) from exc
    return torchcg


def _lane_model_class(module: ModuleType) -> Optional[type]:
    """The ONE lanes-declaring Model subclass in the module, or None."""

    found: list[type] = []
    for value in vars(module).values():
        if (
            inspect.isclass(value)
            and issubclass(value, Model)
            and value is not Model
            and getattr(value, "__module__", None) == module.__name__
            and model_lanes(value)
        ):
            found.append(value)
    if len(found) > 1:
        raise DeriveError(
            f"module {module.__name__!r} declares lanes on "
            f"{[cls.__name__ for cls in found]!r}; a release derives ONE "
            f"model class"
        )
    return found[0] if found else None


@dataclass(frozen=True)
class _Entrypoint:
    """One entrypoint, bound BY ANNOTATION ROLE, order-agnostic.

    The payload parameter is the msgspec Struct; the model parameter is the
    ``Model`` subclass (its annotation IS the model declaration, its NAME is
    the slot name); ctx is the RequestContext-annotated (or sole remaining)
    parameter; every OTHER parameter is a platform-injected FACT
    (``turbo: Adapter | None``, ``loras: list[Adapter]``) and takes its
    trace value from its annotation shape -- Optional injects None, a
    sequence injects empty. The derive calls by KEYWORD, so the author owns
    the order.
    """

    name: str
    fn: Any
    payload_param: str
    payload_type: type
    model_param: str
    ctx_param: str
    #: (param name, annotation, base trace value) per platform-injected fact.
    injected: tuple[tuple[str, Any, Any], ...]


def _injected_trace_value(name: str, parameter_name: str, annotation: Any) -> Any:
    """The trace-time value of a platform-injected fact, by annotation shape.

    No adapter is ever bound at trace (the derive stamps a RELEASE, not a
    deployment), so Optional facts inject None and sequence facts inject
    empty. A fact the shape cannot state is refused by name.
    """

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


def _entrypoints(module: ModuleType, model_cls: type) -> list[_Entrypoint]:
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
        for parameter in parameters:
            annotation = _strip_annotated(hints.get(parameter.name))
            if isinstance(annotation, type) and issubclass(annotation, msgspec.Struct):
                payload_param, payload_type = parameter.name, annotation
            elif annotation is model_cls:
                model_param = parameter.name
            elif typing.get_origin(annotation) is RequestContext or (
                isinstance(annotation, type) and issubclass(annotation, RequestContext)
            ):
                ctx_param = parameter.name
            else:
                rest.append((parameter.name, hints.get(parameter.name)))
        if model_param is None:
            continue
        if payload_param is None:
            raise DeriveError(
                f"@entrypoint {name}: no parameter annotates a msgspec "
                f"payload struct"
            )
        if ctx_param is None:
            # No RequestContext annotation anywhere: the sole remaining
            # parameter is ctx (the minimal (payload, model, ctx) shape).
            if len(rest) == 1:
                ctx_param = rest.pop(0)[0]
            else:
                raise DeriveError(
                    f"@entrypoint {name}: cannot identify the ctx parameter "
                    f"among {[item[0] for item in rest]!r}; annotate it "
                    f"RequestContext"
                )
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
            )
        )
    if not out:
        raise DeriveError(
            f"no @entrypoint function binds model class {model_cls.__name__!r} "
            f"(the model parameter's annotation is the binding)"
        )
    return out


def _strip_annotated(annotation: Any) -> Any:
    """``Annotated[T, ...]`` carries wire metadata; the trace wants T."""

    while typing.get_origin(annotation) is typing.Annotated:
        annotation = typing.get_args(annotation)[0]
    return annotation


def _optional_none(annotation: Any) -> bool:
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        return type(None) in typing.get_args(annotation)
    return False


def _synthesize_field(owner: str, name: str, annotation: Any) -> Any:
    """A minimal trace value for a REQUIRED non-enum field, by type."""

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
    raise DeriveError(
        f"{owner}: required payload field {name!r} of type {annotation!r} "
        f"cannot be auto-synthesized for the trace. Give it a default, or "
        f"reshape it so the schema states its axes (enum fields enumerate)."
    )


def _auto_payloads(owner: str, payload_type: type) -> tuple[tuple[Any, ...], bool]:
    """Auto-enumerated trace payloads for one entrypoint, plus the capped flag.

    One payload per cross-product entry over the struct's ENUM-typed fields
    (field declaration order x enum declaration order -- deterministic);
    every other field at its default; required non-defaulted fields
    synthesized minimally by type.
    """

    import msgspec

    try:
        struct_fields = msgspec.structs.fields(payload_type)
    except TypeError as exc:
        raise DeriveError(
            f"{owner}: payload type {payload_type!r} is not a msgspec struct: {exc}"
        ) from exc

    enum_axes: list[tuple[str, list[Any]]] = []
    base: dict[str, Any] = {}
    for field in struct_fields:
        annotation = _strip_annotated(field.type)
        if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
            values = list(annotation)
            if not values:
                raise DeriveError(
                    f"{owner}: payload field {field.name!r} enumerates an "
                    f"EMPTY enum"
                )
            enum_axes.append((field.name, values))
            continue
        if field.required:
            base[field.name] = _synthesize_field(owner, field.name, annotation)

    if not enum_axes:
        return (payload_type(**base),), False

    names = [name for name, _ in enum_axes]
    combos = itertools.product(*[values for _, values in enum_axes])
    payloads: list[Any] = []
    capped = False
    for index, combo in enumerate(combos):
        if index >= ENUM_CAP:
            capped = True
            break
        payloads.append(payload_type(**base, **dict(zip(names, combo))))
    return tuple(payloads), capped


def _fake_adapter(model_type: Optional[type]) -> Any:
    """A synthesized adapter for the enumeration's adapter-riding arms.

    Carries the model type's platform ``Lora.Defaults`` (what
    ``adapter.defaults`` reads as); its path points nowhere -- adapter I/O is
    neutralized at trace by the load context.
    """

    from ..api.model_base import Adapter

    lora = getattr(model_type, "Lora", None)
    defaults_type = getattr(lora, "Defaults", None)
    return Adapter(
        name="trace-adapter",
        path=Path("/nonexistent/trace-adapter"),
        defaults=defaults_type() if defaults_type is not None else None,
        ref="trace/adapter@0",
    )


def _adapter_annotation(annotation: Any) -> bool:
    from ..api.model_base import Adapter

    stripped = _strip_annotated(annotation)
    if isinstance(stripped, type) and issubclass(stripped, Adapter):
        return True
    return any(
        isinstance(argument, type) and issubclass(argument, Adapter)
        for argument in typing.get_args(stripped)
    )


def _injected_axes(
    plan: "_Entrypoint", model_type: Optional[type]
) -> list[list[tuple[str, Any]]]:
    """Per injected parameter, its enumerated trace values.

    Adapter-shaped facts enumerate BOTH states (absent and riding); other
    facts keep their single trace value.
    """

    axes: list[list[tuple[str, Any]]] = []
    for parameter_name, annotation, base_value in plan.injected:
        if _adapter_annotation(annotation):
            if _optional_none(_strip_annotated(annotation)):
                values: list[Any] = [None, _fake_adapter(model_type)]
            else:
                values = [base_value, [_fake_adapter(model_type)]]
            axes.append([(parameter_name, value) for value in values])
        else:
            axes.append([(parameter_name, base_value)])
    return axes


def _defaults_variants(model_type: Optional[type]) -> list[Any]:
    """The recipe-relevant checkpoint-defaults variants, platform values.

    The platform row always runs; when the schema carries ``cfg``, its
    flipped twin runs too -- cfg selects the executed arm (batch-2 guidance
    vs guidance-free) and therefore the graph set.
    """

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


def _named_marked_modules(instance: Any, marked: list[Any]) -> dict[str, Any]:
    """Deterministic provenance names for the author's ctx.compile marks.

    The author marks REAL objects; the document needs stable names. Names
    come from where the module actually lives on the model instance:
    component names of any ``.components``-bearing attribute (the diffusers
    convention), then bare attribute names, then dotted
    ``attribute.component`` as the disambiguated spelling. A marked module
    that cannot be named on the instance is refused -- provenance is part of
    the release row.
    """

    candidates: dict[int, str] = {}

    def offer(module: Any, name: str) -> None:
        identity = id(module)
        if identity not in candidates or len(name) < len(candidates[identity]):
            candidates[identity] = name

    for attr, value in sorted(vars(instance).items()):
        if value is None:
            continue
        offer(value, attr)
        components = getattr(value, "components", None)
        if isinstance(components, Mapping):
            for name, component in sorted(components.items()):
                if component is None or not isinstance(name, str):
                    continue
                unique = _unique_component(instance, name, component)
                offer(component, name if unique else f"{attr}.{name}")

    named: dict[str, Any] = {}
    for module in marked:
        name = candidates.get(id(module))
        if name is None:
            raise DeriveError(
                f"a module marked via ctx.compile() "
                f"({type(module).__name__}) is not reachable as a model "
                f"attribute or pipeline component; the release document "
                f"cannot name its provenance"
            )
        if name in named and named[name] is not module:
            raise DeriveError(
                f"two marked modules both resolve to provenance name {name!r}"
            )
        named[name] = module
    return named


def _unique_component(instance: Any, name: str, component: Any) -> bool:
    seen = 0
    for value in vars(instance).values():
        components = getattr(value, "components", None)
        if isinstance(components, Mapping) and components.get(name) is not None:
            if components.get(name) is not component:
                return False
            seen += 1
    return seen <= 1


def _closure_entries_from_lockfile(lockfile: Path) -> dict[str, str]:
    import tomllib

    try:
        parsed = tomllib.loads(lockfile.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise DeriveError(f"cannot read lockfile {lockfile}: {exc}") from exc
    entries: dict[str, str] = {}
    for package in parsed.get("package", ()):
        name = package.get("name")
        version = package.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            continue
        known = entries.get(name)
        if known is not None and known != version:
            raise DeriveError(
                f"lockfile {lockfile} resolves {name!r} to both {known!r} and "
                f"{version!r}; a release env has ONE resolved closure"
            )
        entries[name] = version
    if not entries:
        raise DeriveError(f"lockfile {lockfile} states no resolved packages")
    return entries


def _defaults_schema(model_type: Optional[type]) -> Optional[dict[str, Any]]:
    """msgspec JSON Schema of the class-header model type's Defaults.

    ``Model[SDXL]`` is the single, statically-extractable source of the
    endpoint's model type; the schema of ``SDXL.Defaults`` is what the hub
    validates per-checkpoint deploy rows against. This replaces the
    hub-embedded per-family defaults registry (storage half: th#2133).
    """

    if model_type is None:
        return None
    defaults_type = getattr(model_type, "Defaults", model_type)
    import msgspec

    return msgspec.json.schema(defaults_type)


#: safetensors dtype spellings -> torch dtypes (tensorfs#113 carries the
#: contract's load dtype in the document's additive top-level `dtype` field,
#: safetensors spelling; torch never appears in tensorfs).
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


def _contract_document(lane: Any) -> Optional[dict[str, Any]]:
    """The lane contract's canonical document, when the object carries one.

    Duck-typed against tensorfs#111's Contract surface (in design): the full
    document travels in the release metadata so the platform needs no prior
    knowledge of the layout. A bare handle string carries no document.
    """

    for attribute in ("document", "as_dict", "to_dict"):
        value = getattr(lane, attribute, None)
        if callable(value):
            value = value()
        if isinstance(value, dict):
            return value
    return None


def _resolve_lane(torchcg: ModuleType, cls: type, lane: Any) -> Any:
    """The resolved ``ctx.lane``: always a LaneRef with a REAL torch dtype.

    A contract OBJECT (tensorfs registry entry / inline Contract) carries its
    own dtype (tensorfs#113's top-level field, safetensors spelling). A bare
    handle string resolves through the model-type pointer table
    (``gen_worker.models.model_types.CONTRACT_DTYPES`` -- the interim home
    until the tensorfs#111/#113 surface lands).
    """

    handle = lane_contract_handle(f"class {cls.__name__!r}", lane)
    dtype = getattr(lane, "dtype", None) if not isinstance(lane, str) else None
    if dtype is None:
        from ..models.model_types import CONTRACT_DTYPES

        dtype = CONTRACT_DTYPES.get(handle)
    if dtype is None:
        raise DeriveError(
            f"lane {handle!r}: no dtype resolution -- the contract object "
            f"carries none and the model-type pointer table "
            f"(gen_worker.models.model_types.CONTRACT_DTYPES) does not know "
            f"the handle. Import a dtype-bearing contract object, or "
            f"register the handle."
        )
    return torchcg.LaneRef(handle, dtype=_torch_dtype(dtype))


def _derive_lane(
    torchcg: ModuleType,
    cls: type,
    lane: Any,
    plans: list[tuple[_Entrypoint, tuple[Any, ...]]],
    checkpoint_dir: Path,
    warnings: list[str],
) -> Any:
    """One lane's instrumented runs, merged across defaults variants.

    Per variant: fresh model, ``load`` (defaults variants change what
    ``ctx.defaults()`` answers, so the instance is rebuilt), then every
    (entrypoint x payload x adapter-state) combination drives under
    instrumented discovery. Combinations the author REFUSES with
    ``ValidationError`` are legitimately impossible servings and are
    skipped, counted in the warnings.
    """

    from ..api.errors import ValidationError

    handle = lane_contract_handle(f"class {cls.__name__!r}", lane)
    resolved = _resolve_lane(torchcg, cls, lane)
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
            lane=resolved, checkpoint_ref=f"trace:{checkpoint_dir.name}"
        )
        with torchcg.hollow_session():
            try:
                model.load(load_ctx)
            except torchcg.HollowError as exc:
                raise DeriveError(f"lane {handle!r}: {exc}") from exc
            except Exception as exc:
                raise DeriveError(
                    f"lane {handle!r}: load() failed under the trace "
                    f"session: {type(exc).__name__}: {exc}"
                ) from exc
            if not load_ctx.marked_modules:
                raise DeriveError(
                    f"lane {handle!r}: load() marked nothing via ctx.compile(). "
                    f"A lane-declaring model compiles SOMETHING; a model that "
                    f"wants eager-forever declares no lanes instead."
                )
            modules = _named_marked_modules(model, load_ctx.marked_modules)

            def drive() -> None:
                nonlocal refused, total_combos
                for plan, payloads in plans:
                    axes = _injected_axes(plan, model_type)
                    for binding in itertools.product(*axes) if axes else [()]:
                        for index, payload in enumerate(payloads):
                            total_combos += 1
                            try:
                                plan.fn(**{
                                    plan.payload_param: payload,
                                    plan.model_param: model,
                                    plan.ctx_param: request_ctx,
                                    **dict(binding),
                                })
                            except ValidationError:
                                # The author refusing an impossible serving
                                # combination is correct behavior, not a
                                # derive failure.
                                refused += 1
                            except Exception as exc:
                                raise DeriveError(
                                    f"lane {handle!r}: entrypoint "
                                    f"{plan.name!r} failed on auto-enumerated "
                                    f"payload {index} ({payload!r}) with "
                                    f"binding {dict(binding)!r} under the "
                                    f"trace session: "
                                    f"{type(exc).__name__}: {exc}"
                                ) from exc

            try:
                lane_graphs = torchcg.discover_modules(handle, modules, drive)
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
) -> ReleaseDeriveResult:
    """Derive the release metadata document for one endpoint module."""

    torchcg = _torchcg()

    if lockfile is not None:
        closure = torchcg.closure_hash(_closure_entries_from_lockfile(lockfile))
    else:
        closure = torchcg.closure_hash(torchcg.installed_closure())

    cls = _lane_model_class(module)
    endpoint_name = f"{module.__name__}:{cls.__name__ if cls else ''}".rstrip(":")

    lanes: list[Any] = []
    lane_contracts: dict[str, Any] = {}
    warnings: list[str] = []
    if cls is not None:
        plans: list[tuple[_Entrypoint, tuple[Any, ...]]] = []
        for plan in _entrypoints(module, cls):
            owner = f"@entrypoint {plan.name}"
            payloads, capped = _auto_payloads(owner, plan.payload_type)
            if capped:
                warnings.append(
                    f"{owner}: enum cross-product exceeds the cap "
                    f"({ENUM_CAP}); tracing the deterministic prefix -- the "
                    f"rest is first-encounter discovery (eager + background "
                    f"mint)"
                )
            plans.append((plan, payloads))

        for lane in model_lanes(cls):
            lane_graphs = _derive_lane(
                torchcg, cls, lane, plans, checkpoint_dir, warnings
            )
            lanes.append(lane_graphs)
            lane_contracts[lane_graphs.contract] = {
                "stamp": lane_graphs.contract,
                "document": _contract_document(lane),
            }

    graphs_document = torchcg.GraphSetDocument(closure=closure, lanes=tuple(lanes))
    payload_dict: dict[str, Any] = {
        "v": 1,
        "kind": DOCUMENT_KIND,
        "endpoint": endpoint_name,
        "graphs": graphs_document.as_dict(),
        "lane_contracts": lane_contracts,
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
    )


__all__ = [
    "DOCUMENT_KIND",
    "ENUM_CAP",
    "DeriveError",
    "ReleaseDeriveResult",
    "derive_release",
]
