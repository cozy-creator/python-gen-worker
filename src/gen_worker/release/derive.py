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

from ..api.decorators import ATTR, EndpointDecl
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


def _roots_of(instance: Any) -> dict[str, object]:
    """The author namespace discovery resolves compile paths against.

    Instance attributes first (``self.pipe``), then each attribute's
    ``.components`` mapping (the diffusers convention) so the contract file's
    short spelling (``"unet"``) resolves. A component name shared by two
    different objects is dropped from the bare namespace -- the qualified
    path (``"pipe.unet"``) stays unambiguous.
    """

    roots: dict[str, object] = {}
    ambiguous: set[str] = set()
    for attr, value in sorted(vars(instance).items()):
        if value is None:
            continue
        roots[attr] = value
        components = getattr(value, "components", None)
        if isinstance(components, Mapping):
            for name, component in sorted(components.items()):
                if component is None or not isinstance(name, str):
                    continue
                if name in roots and roots[name] is not component:
                    ambiguous.add(name)
                else:
                    roots.setdefault(name, component)
    for name in ambiguous:
        roots.pop(name, None)
    return roots


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


def _defaults_schema(requested: list[type]) -> Optional[dict[str, Any]]:
    """msgspec JSON Schema of the endpoint's ONE defaults struct, if any.

    This replaces the hub-embedded per-family defaults registry: the
    endpoint release EXPORTS the schema its code reads, and the hub
    validates deploy-state rows against it (storage half: th#2133).
    """

    if not requested:
        return None
    types_seen = {getattr(t, "Defaults", t) for t in requested}
    if len(types_seen) > 1:
        raise DeriveError(
            f"endpoint requested checkpoint defaults of "
            f"{sorted(t.__name__ for t in types_seen)!r}; a release exports "
            f"ONE defaults schema"
        )
    import msgspec

    return msgspec.json.schema(types_seen.pop())


def _derive_lane(
    torchcg: ModuleType,
    cls: type,
    lane: Any,
    plans: list[tuple[str, tuple[Any, ...]]],
    checkpoint_dir: Path,
) -> tuple[Any, list[type]]:
    """One lane's instrumented run: fresh instance, setup, drive, discover."""

    instance = cls()
    ctx = TraceRequestContext(lane=lane, checkpoint_dir=checkpoint_dir)
    with torchcg.hollow_session():
        try:
            setup = getattr(instance, "setup", None)
            if setup is not None:
                setup(ctx)
        except torchcg.HollowError as exc:
            raise DeriveError(f"lane {lane.name!r}: {exc}") from exc
        except Exception as exc:
            raise DeriveError(
                f"lane {lane.name!r}: setup() failed under the trace "
                f"session: {type(exc).__name__}: {exc}"
            ) from exc

        def drive() -> None:
            for attr, payloads in plans:
                for index, payload in enumerate(payloads):
                    try:
                        getattr(instance, attr)(ctx, payload)
                    except Exception as exc:
                        raise DeriveError(
                            f"lane {lane.name!r}: handler {attr!r} failed on "
                            f"auto-enumerated payload {index} "
                            f"({payload!r}) under the trace session: "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc

        roots = _roots_of(instance)
        try:
            lane_graphs = torchcg.discover_lane(lane, roots, drive)
        except DeriveError:
            raise
        except torchcg.DiscoveryError as exc:
            raise DeriveError(f"lane {lane.name!r}: {exc}") from exc
    if lane_graphs.unobserved_targets:
        total = sum(len(payloads) for _, payloads in plans)
        raise DeriveError(
            f"lane {lane.name!r}: compile target(s) "
            f"{list(lane_graphs.unobserved_targets)!r} were never CALLED "
            f"while driving {total} auto-enumerated payload(s). A lane path "
            f"must name the module the code actually calls (e.g. "
            f"'vae.decoder', not 'vae') -- silent zero-graph discovery is "
            f"not an outcome."
        )
    return lane_graphs, list(ctx.requested_defaults_types)


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
    requested_defaults: list[type] = []
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
            lane_graphs, lane_defaults = _derive_lane(
                torchcg, cls, lane, plans, checkpoint_dir
            )
            lanes.append(lane_graphs)
            requested_defaults.extend(lane_defaults)

    graphs_document = torchcg.GraphSetDocument(closure=closure, lanes=tuple(lanes))
    payload_dict: dict[str, Any] = {
        "v": 1,
        "kind": DOCUMENT_KIND,
        "endpoint": endpoint_name,
        "graphs": graphs_document.as_dict(),
        "checkpoint_defaults_schema": _defaults_schema(requested_defaults),
    }
    document = json.dumps(
        payload_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return ReleaseDeriveResult(
        document=document,
        digest=hashlib.sha256(document).hexdigest(),
        endpoint=endpoint_name,
        lane_graphs={
            lane.name: tuple(record.graph for record in lane.graphs)
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
