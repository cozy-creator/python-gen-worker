"""The publish-time instrumented derive (pgw#1370).

Per declared execution lane, inside the release env, on CPU: instantiate the
endpoint class, run the author's ``setup`` and handlers AS-IS against a
CONFIG-ONLY checkpoint tree under ``torchcg.hollow_session``, drive them with
AUTO-ENUMERATED trace payloads under instrumented discovery, and stamp the
observed graph set -- plus the endpoint's checkpoint-defaults schema -- as
the static release metadata document.

**Coverage is auto-enumerated from the payload schemas** (Paul ruling,
2026-08-18): one pass per handler per enum-typed field value, every other
field at its default, required non-defaulted fields synthesized minimally by
type. The author surface is exactly CODE + LANES -- there is no samples
surface. Shapes a schema cannot express are discovered at the first live
request (served eager, minted in the background), so enumeration is a
pre-warming completeness aid, never a correctness gate. A cross-product
larger than the cap warns and traces the deterministic prefix.

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

from ..api.decorators import ATTR, EndpointDecl, lane_contract_handle
from ..api.endpoint_base import endpoint_model_type
from .trace_context import TraceRequestContext

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
    lane_graphs: dict[str, tuple[str, ...]]  # lane name -> graph hashes
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


def _lane_endpoint_class(module: ModuleType) -> Optional[type]:
    """The ONE lanes-declaring endpoint class in the module, or None."""

    found: list[type] = []
    for value in vars(module).values():
        if inspect.isclass(value) and getattr(value, "__module__", None) == module.__name__:
            decl = getattr(value, ATTR, None)
            if isinstance(decl, EndpointDecl) and decl.lanes:
                found.append(value)
    if len(found) > 1:
        raise DeriveError(
            f"module {module.__name__!r} declares lanes on "
            f"{[cls.__name__ for cls in found]!r}; a release derives ONE "
            f"endpoint class"
        )
    return found[0] if found else None


def _handler_payload_types(cls: type) -> list[tuple[str, type]]:
    """(handler attr, payload struct type) in deterministic handler order."""

    out: list[tuple[str, type]] = []
    for attr, fn in getattr(cls, "__gen_worker_handlers__", ()):
        parameters = [
            parameter
            for parameter in inspect.signature(fn).parameters.values()
            if parameter.name != "self"
        ]
        if len(parameters) < 2:
            continue
        hints = typing.get_type_hints(fn)
        payload_type = hints.get(parameters[1].name)
        if payload_type is None:
            raise DeriveError(
                f"{cls.__name__}.{attr}: the payload parameter "
                f"{parameters[1].name!r} carries no resolvable type annotation"
            )
        out.append((attr, payload_type))
    if not out:
        raise DeriveError(f"{cls.__name__}: no routable (self, ctx, payload) handlers")
    return out


def _optional_none(annotation: Any) -> bool:
    origin = typing.get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        return type(None) in typing.get_args(annotation)
    return False


def _synthesize_field(owner: str, name: str, annotation: Any) -> Any:
    """A minimal trace value for a REQUIRED non-enum field, by type."""

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
    """Auto-enumerated trace payloads for one handler, plus the capped flag.

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
        annotation = field.type
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


def _named_marked_modules(
    instance: Any, marked: list[Any]
) -> dict[str, Any]:
    """Deterministic provenance names for the author's ctx.compile marks.

    The author marks REAL objects; the document needs stable names. Names
    come from where the module actually lives on the instance: component
    names of any ``.components``-bearing attribute (the diffusers
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
                offer(component, name if _unique_component(instance, name, component) else f"{attr}.{name}")

    named: dict[str, Any] = {}
    for module in marked:
        name = candidates.get(id(module))
        if name is None:
            raise DeriveError(
                f"a module marked via ctx.compile() "
                f"({type(module).__name__}) is not reachable as an instance "
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

    ``Endpoint[SDXL]`` is the single, statically-extractable source of the
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
    plans: list[tuple[str, tuple[Any, ...]]],
    checkpoint_dir: Path,
) -> Any:
    """One lane's instrumented run: fresh instance, setup, drive, discover."""

    handle = lane_contract_handle(f"class {cls.__name__!r}", lane)
    instance = cls()
    ctx = TraceRequestContext(
        lane=_resolve_lane(torchcg, cls, lane),
        checkpoint_dir=checkpoint_dir,
        model_type=endpoint_model_type(cls),
    )
    with torchcg.hollow_session():
        try:
            instance.setup(ctx)
        except torchcg.HollowError as exc:
            raise DeriveError(f"lane {handle!r}: {exc}") from exc
        except Exception as exc:
            raise DeriveError(
                f"lane {handle!r}: setup() failed under the trace "
                f"session: {type(exc).__name__}: {exc}"
            ) from exc
        if not ctx.marked_modules:
            raise DeriveError(
                f"lane {handle!r}: setup() marked nothing via ctx.compile(). "
                f"A lane-declaring endpoint compiles SOMETHING; an endpoint "
                f"that wants eager-forever declares no lanes instead."
            )
        modules = _named_marked_modules(instance, ctx.marked_modules)

        def drive() -> None:
            for attr, payloads in plans:
                for index, payload in enumerate(payloads):
                    try:
                        getattr(instance, attr)(ctx, payload)
                    except Exception as exc:
                        raise DeriveError(
                            f"lane {handle!r}: handler {attr!r} failed on "
                            f"auto-enumerated payload {index} "
                            f"({payload!r}) under the trace session: "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc

        try:
            lane_graphs = torchcg.discover_modules(handle, modules, drive)
        except DeriveError:
            raise
        except torchcg.DiscoveryError as exc:
            raise DeriveError(f"lane {handle!r}: {exc}") from exc
    if lane_graphs.unobserved_targets:
        total = sum(len(payloads) for _, payloads in plans)
        raise DeriveError(
            f"lane {handle!r}: marked module(s) "
            f"{list(lane_graphs.unobserved_targets)!r} were never CALLED "
            f"while driving {total} auto-enumerated payload(s). ctx.compile "
            f"must mark the module the code actually CALLS (e.g. the vae's "
            f".decoder, not the vae, when only .decode() runs) -- silent "
            f"zero-graph discovery is not an outcome."
        )
    return lane_graphs


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

    cls = _lane_endpoint_class(module)
    endpoint_name = f"{module.__name__}:{cls.__name__ if cls else ''}".rstrip(":")

    lanes: list[Any] = []
    lane_contracts: dict[str, Any] = {}
    warnings: list[str] = []
    if cls is not None:
        plans: list[tuple[str, tuple[Any, ...]]] = []
        for attr, payload_type in _handler_payload_types(cls):
            owner = f"{cls.__name__}.{attr}"
            payloads, capped = _auto_payloads(owner, payload_type)
            if capped:
                warnings.append(
                    f"{owner}: enum cross-product exceeds the cap "
                    f"({ENUM_CAP}); tracing the deterministic prefix -- the "
                    f"rest is first-encounter discovery (eager + background "
                    f"mint)"
                )
            plans.append((attr, payloads))

        decl: EndpointDecl = getattr(cls, ATTR)
        for lane in decl.lanes:
            lane_graphs = _derive_lane(torchcg, cls, lane, plans, checkpoint_dir)
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
            endpoint_model_type(cls) if cls is not None else None
        ),
        "model_type": (
            getattr(endpoint_model_type(cls), "__name__", None)
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
