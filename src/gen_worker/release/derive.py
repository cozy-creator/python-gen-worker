"""The publish-time instrumented derive (pgw#1370).

Per declared execution lane, inside the release env, on CPU: instantiate the
author's ``Model`` subclass, run its ``load`` AS-IS against a CONFIG-ONLY
checkpoint tree under ``torchcg.hollow_session``, drive the module's
``@entrypoint`` functions with AUTO-ENUMERATED trace payloads under
instrumented discovery, and stamp the observed graph set -- plus the lane
contracts and the model type's checkpoint-defaults schema -- as the static
release metadata document.

**A RELEASE DERIVES EVERY COMPILE-MARKING MODEL CLASS** (Paul's ruling,
2026-08-21; pgw#1650): *"Of course both qwen image and qwen image edit can
exist in the same endpoint. Why wouldn't they be able to? Just compile each
component and swap them in and out of the pipeline."* Each subject class
traces its OWN entrypoints against its OWN checkpoint tree
(``checkpoint_trees``, keyed by class name) and states its own graph set; the
document carries them in ``classes[]``, keyed by (class x lane), and
``graphs``/``lane_contracts`` are the release-wide UNION over those rows (every
merge rule is stated in :func:`_merge_lanes`). Two classes CAN declare the same
lane — the two qwen arms do, because their checkpoints are byte-layout
identical — which is why the union is a merge and not a concatenation. The
serving side already routes per class: ``serving/serve_loop.py`` keys backends
``(model_cls, checkpoint_ref, lane)`` and resolves a ``DeployBinding`` per
class, so swapping the arms in and out is existing machinery, not new.

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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, Optional

from ..demand_envelope import advertised_envelope, demand_document
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
from .drive_hygiene import (
    accelerator_is_alive,
    dead_accelerator_sentence,
    eager_only_compile,
)
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
    """Turn a model class's declared shape axes into torchcg's predicate.

    pgw#1603: a declared shape axis is ALWAYS traced symbolically — the
    author's STATIC/DYNAMIC choice controls what is MINTED (N static
    binaries vs 1 range binary), never how many traces run. STATIC keeps its
    per-bucket records, stamped by binding concrete shapes to the shared
    symbolic parent (:func:`static_bind_declared`); DYNAMIC serves the one
    range record. Batch (axis 0) stays a structural fork in both cases.
    """

    if not shapes:
        return None

    def policy(_target: str, _name: str, axis: int) -> bool:
        if axis == 0:
            return False
        return axis >= 2

    return policy


def static_bind_declared(shapes: Mapping[str, str]) -> bool:
    """Whether declared buckets are STAMPED from the symbolic parent.

    True exactly when the class declares shape axes and none is DYNAMIC. A
    mixed declaration is refused rather than guessed: the dim policy above
    cannot yet aim a symbol at ONE named axis, so serving would turn a
    STATIC axis into a range the author never declared.
    """

    if not shapes:
        return False
    choices = set(shapes.values())
    if DYNAMIC in choices and len(choices) > 1:
        raise DeriveError(
            "shapes= mixes STATIC and DYNAMIC axes; the derive's axis policy "
            "cannot yet aim a symbol at one NAMED axis, so a mixed "
            "declaration would serve a STATIC axis as a range. Declare the "
            "shape axes uniformly for now (pgw#1603)."
        )
    return DYNAMIC not in choices


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
    #: Every subject class this release derived (pgw#1650) — a release derives
    #: EVERY compile-marking class, not one.
    classes: tuple[str, ...] = ()
    #: ``(class, lane stamp, graph identities)`` — the per-class graph sets
    #: BEFORE the release-wide union, which is what a pipeline log wants to
    #: read when two classes share a lane.
    class_lane_graphs: tuple[tuple[str, str, tuple[str, ...]], ...] = ()
    #: The component names the construction census states, sorted.
    census_components: tuple[str, ...] = ()
    #: Why there is NO census — ``NO_PIPELINE_INDEX`` for a tree that is not a
    #: diffusers pipeline and is therefore never streaming-served. Empty means
    #: a census was emitted. There is no third value: a census that could not
    #: be COMPUTED fails the build.
    census_absent: str = ""

    @property
    def eager_permanent(self) -> bool:
        return not self.lane_graphs


def _torchcg() -> ModuleType:

    # tcg#90/pgw#1656: the discovery, lane and document surface is pgw's now.
    # torchcg is `program -> keyed artifact` and knows nothing of lanes.
    from .. import graphs

    return graphs


def _hollow() -> ModuleType:

    import importlib

    return importlib.import_module("gen_worker.graphs.hollow")


def _program_sink(cas_root: Optional[Path]) -> Optional[Any]:

    if cas_root is None:
        return None

    import tempfile

    import torch

    from .._vendor.tensorfs import LocalCAS
    from .._vendor.torchcg.store import LocalGraphStore

    store = LocalGraphStore(LocalCAS(Path(cas_root)))

    from .._vendor.torchcg.mint import strip_diagnostics

    def sink(graph: str, program: Any) -> None:
        _assert_weights_free(torch, program)
        # pgw#1603: per-node stack traces and nn_module_stack strings were
        # ~60% of every serialized program (measured: 1.8 MB of a 3.1 MB
        # sd15 graph JSON) and nothing on the mint path reads them.
        strip_diagnostics(program)
        with tempfile.TemporaryDirectory() as scratch:
            staged = Path(scratch) / "program.pt2"
            torch.export.save(program, str(staged))
            store.put_program(graph, staged)

    return sink


def _trace_device() -> str:

    return "cuda"


def _refuse_a_dead_accelerator(
    said: str, session: Any, cause: Optional[BaseException] = None
) -> None:
    """A drive that killed the card is refused BY THE DRIVE (pgw#1659).

    Asked right after discovery, so the refusal names what happened instead of
    whatever touched the device next. ``cause`` is the message discovery
    produced; it stays as the chained exception because it is real evidence,
    just not a diagnosis.
    """

    if session is None or accelerator_is_alive(session.drive_device):
        return
    raise DeriveError(
        f"{said}: {dead_accelerator_sentence(session.drive_device_type)}"
    ) from cause


def _assert_weights_free(torch: Any, program: Any) -> None:
    """The last thing between a hollow derive and a graph blob.

    Asked of the STORAGE, never of the TYPE (pgw#1198, re-broken here and fixed
    in pgw#1661). `isinstance(..., FakeTensor)` cannot see a wrapper subclass
    over fake data — what a `setup()`-time quantizer leaves on a hollow denoiser
    — and such a tensor prices itself off its OUTER metadata, so h3's 300
    virtual weights read as ~23 GB of weights on a card holding 0.0 GiB.
    `is_virtual` recurses through `__tensor_flatten__` and answers meta too,
    which is why the separate meta arm that stood here is gone.
    """

    from ..meta_instantiation import is_virtual

    heavy = []
    for holder in ("state_dict", "constants"):
        for name, value in (getattr(program, holder, None) or {}).items():
            if not isinstance(value, torch.Tensor) or is_virtual(value):
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


def _module_model_classes(module: ModuleType) -> tuple[list[type], list[type]]:

    marked: list[type] = []
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
            marks = model_marks_compile(value)
        except ModelDeclarationError as exc:
            raise DeriveError(str(exc)) from exc
        (marked if marks else unmarked).append(value)
    return (
        sorted(marked, key=lambda cls: cls.__name__),
        sorted(unmarked, key=lambda cls: cls.__name__),
    )


def _subject_classes(module: ModuleType) -> tuple[type, ...]:
    """The classes this release DERIVES — every compile-marking one (pgw#1650).

    Paul, 2026-08-21: *"Of course both qwen image and qwen image edit can exist
    in the same endpoint. Why wouldn't they be able to? Just compile each
    component and swap them in and out of the pipeline."* Each subject traces
    its own entrypoints against its own checkpoint tree and states its own
    graph set; the document carries them keyed by (class x lane).

    THE ONE REFUSAL LEFT is the case that is still genuinely unreadable:
    several classes and NOT ONE of them marks a compile target. Subjecthood is
    read off the MARK (pgw#1597/#1599: "compilation participation is the
    MARK"), so with no mark anywhere there is nothing to distinguish a release
    subject from an auxiliary model another slot drives.

    That refusal is only sound because the mark is read TOTALLY (pgw#1655).
    ``model_marks_compile`` used to answer ``False`` for "does not compile"
    and for "cannot see it" alike, so a DELEGATED mark
    (``engine.compile_dit(ctx.compile)``) read as auxiliary — minimax-h3 fell
    into the refusal below while both its lanes declared correctly, and
    wan-2.2's two MoE classes were dropped from the subject set without a
    word. An unreadable mark is now a refusal at the class DECLARATION, so
    every class that reaches this gate has an answer that was stated.
    """

    marked, unmarked = _module_model_classes(module)
    if marked:
        return tuple(marked)
    if len(unmarked) > 1:
        raise DeriveError(
            f"module {module.__name__!r} has more than one model class "
            f"({[cls.__name__ for cls in unmarked]!r}) and NONE of them marks "
            f"a compile target, so which one the release is ABOUT cannot be "
            f"read. A release derives every COMPILE-MARKING class (pgw#1650), "
            f"and there are none here. The mark is read TOTALLY (pgw#1655) — "
            f"a mark HANDED ON, `engine.compile_dit(ctx.compile)`, counts as "
            f"much as one called in place, and a `load()` that hides the "
            f"context is refused at its own declaration — so this is a real "
            f"absence, not an unreadable one. Mark each compiled class via "
            f"`ctx.compile` in its `load()`; an auxiliary model that another "
            f"slot drives marks nothing and is then unambiguous."
        )
    return tuple(unmarked)


def _checkpoint_tree(
    trees: Mapping[str, Path],
    primary: Path,
    cls: Optional[type],
    slot_name: str = "",
) -> Path:
    """The tree ONE model class loads from: its class name, then its slot name.

    pgw#1650: a release has several subject classes, and two of them can hold
    the same slot NAME in their own entrypoints (both qwen arms take
    ``model:``) while binding different checkpoints. The class is the thing
    that owns a checkpoint, so the class name is the key; the slot name stays
    readable for an auxiliary model, which is how se#794's second tree is
    spelled today.
    """

    if cls is not None and cls.__name__ in trees:
        return trees[cls.__name__]
    if slot_name and slot_name in trees:
        return trees[slot_name]
    return primary


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
    pinned: Mapping[str, str] = MappingProxyType({}),
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
        # pgw#1603: a pinned axis contributes exactly its item's class — the
        # cross product over structural classes is the ITEM enumeration, run
        # in parallel, never a per-item payload fan.
        pin = pinned.get(axis)
        if pin is not None:
            declared_axes[declaration.field] = [
                value for name, value in variants if name == pin
            ]
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


#: Why a lane carries no census. There is no third value and no soft row: a
#: tree that HAS a `model_index.json` and cannot be censused fails the build.
NO_PIPELINE_INDEX = "no_model_index_json"


def _construction_census(
    checkpoint_dir: Path, lanes: tuple[tuple[str, Any], ...]
) -> dict[str, Any]:
    """The CONSTRUCTION CENSUS of this run's tree (pgw#1647, th#2281/th#2287).

    Computed here because here is where the tree and the IMAGE meet. What a
    module IS — its tie groups, the classes its config's quantizer swaps in, the
    buffers its ``__init__`` computes — is a code x config fact decided by THIS
    image's transformers and diffusers. The tensorfs stamp stays the BYTES'
    identity; this is the MODULE's.

    **ITS KEY IS IMAGE x CONFIG TREE** (th#2287's adjudication, 2026-08-21), not
    source x image like the rest of the release contract — and this function is
    already at that key, because it is the BIND-TIME
    ``release derive --checkpoint`` run that censuses, and that run has exactly
    one primary tree. The hub re-keys the emitted document onto the bind
    (release x image digest x config digest) and never asks this side to.

    **ONE census for the whole release, and the invariance is PROVEN here.**
    A lane's only effect on construction is the dtype it casts wide floats to,
    and the census records those as ``census.LANE_DTYPE`` because the lane
    contract and ``engine._assert_lane_dtype`` already state that fact exactly.
    So every declared lane must produce the SAME census — and rather than
    assume it, this builds one per lane and refuses if two disagree. A
    disagreement means a lane changes what the module IS, which is a fact worth
    a refusal rather than a silently-picked winner.

    Config-only and allocation-free: parameters come up on meta and buffers are
    computed from config, so a 66 GiB DiT costs what a tiny one costs. It goes
    through :func:`~gen_worker.serving.streaming.skeleton.build_modules`, which
    is the reader production serves with — a census taken by a second parser
    would be a statement about the second parser.

    **REFUSE ON TRACEBACK** (Paul's derive ruling). A tree that carries a
    ``model_index.json`` and cannot be censused FAILS the build, named. There is
    no soft row-marking and no green release with the reason in a log, because
    the entire point of moving this question to publish time is that a release
    which cannot say what module it builds must never reach a card.
    """
    from ..serving.streaming import census as _census
    from ..serving.streaming.skeleton import MODEL_INDEX

    if not (Path(checkpoint_dir) / MODEL_INDEX).is_file():
        # Not a refusal: a tree with no component index is not a diffusers
        # pipeline and the streaming loader never binds to one. The hub's door
        # (th#2281) is what decides whether a STREAMING-served release may ship
        # without a census; this side states the fact and never guesses.
        return {"absent": NO_PIPELINE_INDEX}

    agreed: Optional[Any] = None
    agreed_handle = ""
    for handle, dtype in lanes:
        try:
            taken = _census.for_tree(checkpoint_dir, compute_dtype=dtype)
        except Exception as exc:
            raise DeriveError(
                f"lane {handle!r}: the CONSTRUCTION CENSUS could not be "
                f"computed from {checkpoint_dir} — {type(exc).__name__}: {exc}. "
                f"A release states what module it builds or it is not a "
                f"release: this is the meta-skeleton family "
                f"(pgw#1626/#1638/#1644), and every one of those four walls was "
                f"a construction question answered on a rented card because "
                f"nothing asked it here first"
            ) from exc
        if agreed is None:
            agreed, agreed_handle = taken, handle
            continue
        if taken != agreed:
            raise DeriveError(
                f"lanes {agreed_handle!r} and {handle!r} build DIFFERENT "
                f"modules from {checkpoint_dir}: their construction censuses "
                f"disagree ({agreed.digest} vs {taken.digest}). A lane declares "
                f"a dtype and a layout, not a different model — the census "
                f"records lane-governed dtypes as {_census.LANE_DTYPE!r} "
                f"precisely so that a lane cannot move it. Publishing one of "
                f"the two would make the release document describe a module "
                f"half of its lanes do not build"
            )
    if agreed is None:  # pragma: no cover — a class with no declared lane
        return {"absent": NO_PIPELINE_INDEX}
    document: dict[str, Any] = agreed.as_document()
    return document


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


@dataclass(frozen=True)
class DeriveItem:
    """One trace unit: (class × lane × defaults-variant × structural combo).

    pgw#1603: the item enumeration IS Paul's trace-count directive. Items
    are the structural variants — each runs ONE author-code drive over its
    shape fan and pays ONE symbolic export per observed group — and they are
    independent, so they run in parallel processes. Shape buckets never
    become items; they are covered inside an item's drive and stamped by
    static bind when the serving declaration is STATIC. Subject classes
    (pgw#1650) are the outermost axis, so a multi-class release parallelizes
    across its classes too.
    """

    index: int
    class_index: int
    lane_index: int
    defaults_index: int
    pinned: tuple[tuple[str, str], ...]


def _structural_combos(
    structural: Mapping[str, Any],
) -> list[tuple[tuple[str, str], ...]]:

    axes = list(structural.items())
    if not axes:
        return [()]
    choices = [
        [(axis, name) for name, _value in declaration.variants()]
        for axis, declaration in axes
    ]
    return [tuple(combo) for combo in itertools.product(*choices)]


def derive_items(module: ModuleType) -> list[DeriveItem]:
    """The item enumeration, deterministic and shared by parent and workers."""

    items: list[DeriveItem] = []
    for class_index, cls in enumerate(_subject_classes(module)):
        lanes = model_declared_lanes(cls)
        defaults_count = len(_defaults_variants(model_model_type(cls)))
        combos = _structural_combos(model_structural(cls))
        for lane_index in range(len(lanes)):
            for defaults_index in range(defaults_count):
                for combo in combos:
                    items.append(
                        DeriveItem(
                            len(items), class_index, lane_index,
                            defaults_index, combo,
                        )
                    )
    return items


def _item_plans(
    module: ModuleType, cls: type, item: DeriveItem
) -> list[tuple[_Entrypoint, tuple[Any, ...]]]:
    """This item's (plan, payloads): structural axes pinned, shape fan whole."""

    structural = model_structural(cls)
    pinned = dict(item.pinned)
    first = {
        axis: declaration.variants()[0][0]
        for axis, declaration in structural.items()
    }
    out: list[tuple[_Entrypoint, tuple[Any, ...]]] = []
    for plan in _entrypoints(module, cls):
        fields = set(_payload_field_names(plan.payload_type))
        absent = [
            axis for axis in pinned if structural[axis].field not in fields
        ]
        if any(pinned[axis] != first[axis] for axis in absent):
            # A plan that cannot spell a pinned axis rides that axis's FIRST
            # class only — once per (class × lane × defaults), never per pin.
            continue
        try:
            payloads, _capped = _auto_payloads(
                f"@entrypoint {plan.name}", plan.payload_type, structural,
                pinned=pinned,
            )
        except PayloadEnumerationRefused:
            continue  # recorded parent-side from the full enumeration
        out.append((plan, payloads))
    return out


def _derive_lane_item(
    torchcg: ModuleType,
    cls: type,
    lane: Any,
    plans: list[tuple[_Entrypoint, tuple[Any, ...]]],
    checkpoint_dir: Path,
    warnings: list[str],
    defaults_instance: Any,
    program_sink: Optional[Any] = None,
    slot_checkpoints: Mapping[str, Path] = MappingProxyType({}),
    endpoint_root: Optional[Path] = None,
    unservable: Optional[list[dict[str, Any]]] = None,
    dynamic_dims: Any = None,
    static_bind: bool = False,
) -> tuple[Optional[Any], int, int]:
    """One item's drive + exports: ``(LaneGraphs | None, refused, total)``."""

    from ..api.errors import ValidationError

    hollow = _hollow()
    handle = lane_contract_handle(f"class {cls.__name__!r}", lane)
    said = f"class {cls.__name__} lane {handle!r}"
    resolved = _resolve_lane(torchcg, lane)
    model_type = model_model_type(cls)
    refused = 0
    total_combos = 0

    if True:
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
        # pgw#1659: `torch.compile` is IDENTITY for the whole drive. An author
        # arming a compiled module in `load()` would otherwise have inductor
        # generate a real kernel and launch it on a FAKE data pointer, which
        # kills this process's accelerator for everything after it.
        with eager_only_compile(), hollow.hollow_session(
            _trace_device(), dtype_for=load_ctx.component_dtype
        ) as session:
            try:
                model.load(load_ctx)
            except hollow.HollowError as exc:
                raise DeriveError(f"{said}: {exc}") from exc
            except Exception as exc:
                raise DeriveError(
                    f"{said}: load() failed under the trace "
                    f"session: {type(exc).__name__}: {exc}"
                ) from exc
            if not load_ctx.marked_modules:
                return None, 0, 0
            modules = _named_marked_modules(model, load_ctx.marked_modules)

            aides: dict[str, Any] = {}
            for plan, _payloads in plans:
                for slot_name, slot_cls in plan.model_slots:
                    if slot_cls is cls or slot_name in aides:
                        continue
                    slot_tree = _checkpoint_tree(
                        slot_checkpoints, checkpoint_dir, slot_cls, slot_name
                    )
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
                        named = (
                            slot_cls.__name__ in slot_checkpoints
                            or slot_name in slot_checkpoints
                        )
                        shared = (
                            " (the PRIMARY checkpoint — this class has no "
                            "--checkpoint-ref of its own; a model with a "
                            "separate checkpoint needs "
                            f"`--checkpoint-ref {slot_cls.__name__}=<ref>`)"
                            if not named
                            else ""
                        )
                        raise DeriveError(
                            f"{said}: entrypoint {plan.name!r} slot "
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
                                        f"{said}: entrypoint "
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

            notes: list[str] = []
            try:
                lane_graphs = torchcg.discover_modules(
                    handle, modules, drive, program_sink=program_sink,
                    session=session, dynamic_dims=dynamic_dims,
                    static_bind=static_bind, notes=notes,
                )
                if set(lane_graphs.targets) - {
                    record.target for record in lane_graphs.graphs
                }:
                    request_ctx.step_budget = None
                    lane_graphs = torchcg.discover_modules(
                        handle, modules, drive, program_sink=program_sink,
                        session=session, dynamic_dims=dynamic_dims,
                        static_bind=static_bind, notes=notes,
                    )
                _refuse_a_dead_accelerator(said, session)
            except DeriveError:
                raise
            except torchcg.DiscoveryError as exc:
                # pgw#1659: a sticky accelerator fault raised mid-drive and
                # swallowed by author code surfaces as whatever discovery
                # touched the card next — a literal constant, most of the time.
                # Ask the machine before relaying the message.
                _refuse_a_dead_accelerator(said, session, exc)
                raise DeriveError(f"{said}: {exc}") from exc
            # pgw#1603 acceptance (c): an axis that dropped to per-bucket
            # tracing is said in the LOCK, never only in a logger.
            warnings.extend(dict.fromkeys(notes))

    return lane_graphs, refused, total_combos


def _merge_lane_items(
    torchcg: ModuleType,
    cls: type,
    lane: Any,
    outcomes: list[tuple[Optional[Any], int, int]],
    warnings: list[str],
) -> Optional[Any]:
    """Merge one (class × lane)'s item outcomes into its final ``LaneGraphs``."""

    handle = lane_contract_handle(f"class {cls.__name__!r}", lane)
    said = f"class {cls.__name__} lane {handle!r}"
    if all(outcome[0] is None for outcome in outcomes):
        return None
    merged: dict[str, Any] = {}
    all_targets: set[str] = set()
    observed_targets: set[str] = set()
    refused = 0
    total_combos = 0
    for lane_graphs, item_refused, item_total in outcomes:
        refused += item_refused
        total_combos += item_total
        if lane_graphs is None:
            continue
        all_targets.update(lane_graphs.targets)
        for record in lane_graphs.graphs:
            merged.setdefault(record.graph, record)
            observed_targets.add(record.target)

    if refused:
        warnings.append(
            f"{said}: {refused}/{total_combos} enumerated "
            f"combination(s) refused by the author's own validation "
            f"(impossible servings; skipped)"
        )
    unobserved = tuple(sorted(all_targets - observed_targets))
    if unobserved and total_combos:
        raise DeriveError(
            f"{said}: marked module(s) {list(unobserved)!r} were "
            f"never CALLED while driving {total_combos} auto-enumerated "
            f"combination(s). ctx.compile must mark the module the code "
            f"actually CALLS (e.g. the vae's .decoder, not the vae, when "
            f"only .decode() runs) -- silent zero-graph discovery is not an "
            f"outcome."
        )
    if unobserved:
        # ZERO combinations is a DIFFERENT fact from a mark that survived real
        # driving, and reading them as one refusal is what pgw#1650 found:
        # every entrypoint this class owns refused enumeration (a required
        # payload field the derive cannot synthesize — an input image, say), so
        # nothing was ever called and the mark could not be observed. That is
        # the outcome this module already states for an unenumerable
        # entrypoint: no traced coverage, eager serving, mint on first
        # encounter. The lane is DECLARED with its targets stated UNOBSERVED,
        # which is exactly what `LaneGraphs.unobserved_targets` is for; killing
        # the whole release's derive over it would take the OTHER classes'
        # graphs down with it.
        warnings.append(
            f"{said}: marked module(s) {list(unobserved)!r} were never driven "
            f"— this class's entrypoint(s) enumerate NOTHING, so the lane is "
            f"declared with its targets UNOBSERVED. They serve eager and mint "
            f"on first encounter; every other class is unaffected."
        )
    return torchcg.LaneGraphs(
        contract=handle,
        targets=tuple(sorted(all_targets)),
        graphs=tuple(merged.values()),
        unobserved_targets=unobserved,
    )


@dataclass(frozen=True)
class _ItemOutcome:
    """One item's picklable report."""

    lane_graphs: Optional[Any]
    refused: int
    total: int
    warnings: tuple[str, ...]
    unservable: tuple[dict[str, Any], ...]


def _trace_worker_count(requested: Optional[int], items: int) -> int:
    """The parallel degree, DERIVED: the item count capped by this host.

    Config with a derivation, never a magic number (pgw#1603): the natural
    width is one process per derive item, a builder pays no more than its
    core count, and an explicit ``--trace-workers`` (or the
    ``GEN_WORKER_TRACE_WORKERS`` config) caps it below that.
    """

    if items <= 1:
        return 1
    import os

    if requested is None:
        configured = os.environ.get("GEN_WORKER_TRACE_WORKERS", "").strip()
        if configured:
            requested = int(configured)
    if requested is not None:
        return max(1, min(int(requested), items))
    cores = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    return max(1, min(items, cores))


def _run_item_in_process(
    module: ModuleType,
    item: DeriveItem,
    *,
    checkpoint_dir: Path,
    checkpoint_trees: Mapping[str, Path],
    program_sink: Optional[Any],
) -> _ItemOutcome:

    cls = _subject_classes(module)[item.class_index]
    lane = model_declared_lanes(cls)[item.lane_index]
    defaults_instance = _defaults_variants(model_model_type(cls))[
        item.defaults_index
    ]
    shapes = model_shapes(cls)
    warnings: list[str] = []
    unservable: list[dict[str, Any]] = []
    lane_graphs, refused, total = _derive_lane_item(
        _torchcg(), cls, lane, _item_plans(module, cls, item),
        _checkpoint_tree(checkpoint_trees, checkpoint_dir, cls),
        warnings, defaults_instance,
        program_sink=program_sink,
        slot_checkpoints=checkpoint_trees,
        endpoint_root=endpoint_source_root(module),
        unservable=unservable,
        dynamic_dims=dynamic_dim_policy(shapes),
        static_bind=static_bind_declared(shapes),
    )
    return _ItemOutcome(
        lane_graphs=lane_graphs,
        refused=refused,
        total=total,
        warnings=tuple(warnings),
        unservable=tuple(unservable),
    )


def _derive_item_task(spec: Mapping[str, Any]) -> _ItemOutcome:
    """One derive item, in a spawned worker: re-import, run, report picklable."""

    import importlib
    import sys

    for entry in reversed(list(spec["sys_path"])):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    global TRACE_STEP_BUDGET
    TRACE_STEP_BUDGET = spec["step_budget"]
    module = importlib.import_module(str(spec["module"]))
    item = derive_items(module)[int(spec["item_index"])]
    raw_cas = str(spec["graph_cas"])
    return _run_item_in_process(
        module, item,
        checkpoint_dir=Path(str(spec["checkpoint_dir"])),
        checkpoint_trees={
            name: Path(value)
            for name, value in dict(spec["checkpoint_trees"]).items()
        },
        program_sink=_program_sink(Path(raw_cas) if raw_cas else None),
    )


def _run_items(
    module: ModuleType,
    items: list[DeriveItem],
    *,
    checkpoint_dir: Path,
    graph_cas: Optional[Path],
    program_sink: Optional[Any],
    checkpoint_trees: Mapping[str, Path],
    trace_workers: Optional[int],
) -> list[_ItemOutcome]:
    """All derive items, in parallel processes when the host has the width.

    Item order is the merge order either way, so the document does not
    depend on the parallel degree. Program bytes land in the shared
    content-addressed store from whichever process derived them — its
    compare-and-swap refs make concurrent writers safe.
    """

    workers = _trace_worker_count(trace_workers, len(items))
    if workers <= 1:
        return [
            _run_item_in_process(
                module, item,
                checkpoint_dir=checkpoint_dir,
                checkpoint_trees=checkpoint_trees,
                program_sink=program_sink,
            )
            for item in items
        ]

    import concurrent.futures
    import multiprocessing
    import sys

    specs = [
        {
            "module": module.__name__,
            "sys_path": list(sys.path),
            "item_index": item.index,
            "checkpoint_dir": str(checkpoint_dir),
            "graph_cas": str(graph_cas) if graph_cas is not None else "",
            "checkpoint_trees": {
                name: str(value) for name, value in checkpoint_trees.items()
            },
            "step_budget": TRACE_STEP_BUDGET,
        }
        for item in items
    ]
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers, mp_context=context
    ) as pool:
        return list(pool.map(_derive_item_task, specs))


@dataclass(frozen=True)
class _ClassDerivation:
    """One subject class's whole view of the release (pgw#1650)."""

    name: str
    model_type: Optional[str]
    defaults_schema: Optional[dict[str, Any]]
    lanes: tuple[Any, ...]
    lane_contracts: dict[str, Any]
    entrypoints: dict[str, Any]
    fork_axes: dict[str, Any]
    unmarked_lanes: tuple[str, ...]
    unenumerable: tuple[tuple[str, str], ...]
    unservable: tuple[dict[str, Any], ...]
    traced_entrypoints: int

    def as_document(self, stack: tuple[tuple[str, str], ...]) -> dict[str, Any]:
        return {
            "class": self.name,
            "model_type": self.model_type,
            "checkpoint_defaults_schema": self.defaults_schema,
            "graphs": _torchcg().GraphSetDocument(
                stack=stack, lanes=self.lanes
            ).as_dict(),
            "lane_contracts": self.lane_contracts,
            "entrypoints": self.entrypoints,
            "fork_axes": self.fork_axes,
        }


def _derive_class(
    torchcg: ModuleType,
    module: ModuleType,
    cls: Optional[type],
    *,
    warnings: list[str],
    lane_outcomes: Mapping[int, list[_ItemOutcome]],
) -> _ClassDerivation:
    """Assemble ONE subject class's derivation from its items' outcomes.

    pgw#1603: the TRACING happened in `_run_items` — per (class × lane ×
    defaults × structural combo), in parallel; this consumes the outcomes in
    item order and states the class's document rows exactly as the serial
    derive did.
    """

    said = f"class {cls.__name__!r}: " if cls is not None else ""
    lanes: list[Any] = []
    unmarked_lanes: list[str] = []
    lane_contracts: dict[str, Any] = {}
    entrypoints: dict[str, Any] = {}
    plans: list[tuple[_Entrypoint, tuple[Any, ...]]] = []
    unenumerable: list[tuple[str, str]] = []
    unservable_payloads: list[dict[str, Any]] = []

    for plan in _entrypoints(module, cls):
        owner = f"@entrypoint {plan.name}"
        refusal: Optional[PayloadEnumerationRefused] = None
        capped = False
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
            if capped:
                warnings.append(
                    f"{said}{owner}: enum cross-product exceeds the cap "
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
                f"{said}{owner}: NOT enumerated — {refusal.field!r} "
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
        for lane_index, lane in enumerate(model_declared_lanes(cls)):
            outcomes = lane_outcomes.get(lane_index, [])
            for outcome in outcomes:
                warnings.extend(outcome.warnings)
                for raw in outcome.unservable:
                    row = dict(raw)
                    if row not in unservable_payloads:
                        unservable_payloads.append(row)
            lane_graphs = _merge_lane_items(
                torchcg, cls, lane,
                [
                    (outcome.lane_graphs, outcome.refused, outcome.total)
                    for outcome in outcomes
                ],
                warnings,
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
            # pgw#1600. THE DEMAND FORMULA, SERIALIZED. Data, not Python: a
            # term list plus the closed vocabulary it is written against, so
            # tensorhub (Go) evaluates `worst_case = manifest weight bytes +
            # demand(advertised envelope)` at pod-buy time without running any
            # of ours. The envelope is taken over EVERY entrypoint THIS CLASS
            # serves, because the pod must hold the worst case of the whole
            # advertised surface, not of one function.
            entry["demand"] = demand_document(
                lane.request,
                advertised_envelope(*(plan.payload_type for plan, _ in plans)),
            )
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
            f"{said}@entrypoint {row['entrypoint']}: payload {row['payload']} "
            f"is UNSERVABLE and was skipped — {row['error']} (at "
            f"{row['frame']}). Its graphs are not in this document; every "
            f"other payload is unaffected."
        )

    model_type_cls = model_model_type(cls) if cls is not None else None
    return _ClassDerivation(
        name=cls.__name__ if cls is not None else "",
        model_type=getattr(model_type_cls, "__name__", None),
        defaults_schema=_defaults_schema(model_type_cls),
        lanes=tuple(lanes),
        lane_contracts=lane_contracts,
        entrypoints=entrypoints,
        fork_axes={
            "structural": [
                declaration.as_document(axis)
                for axis, declaration in (
                    model_structural(cls) if cls is not None else {}
                ).items()
            ],
            "shapes": model_shapes(cls) if cls is not None else {},
        },
        unmarked_lanes=tuple(unmarked_lanes),
        unenumerable=tuple(unenumerable),
        unservable=tuple(unservable_payloads),
        traced_entrypoints=len(plans),
    )


def _merge_lanes(
    torchcg: ModuleType,
    derivations: Sequence[_ClassDerivation],
    warnings: list[str],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """The RELEASE-WIDE view of a per-class graph set (pgw#1650).

    Two subject classes legitimately declare the SAME lane — both qwen arms
    are ``qwen-image.diffusers@1+plain.bf16@1``, because the two checkpoints
    are byte-layout identical. The hub keys
    ``release_compiled_graph_documents.lanes`` by the STAMP ALONE and refuses a
    ``lane_contracts`` key that is not its own entry's stamp
    (``release_compiled_graphs_invalid_lane``), and ``GraphSetDocument`` itself
    refuses a repeated lane contract — so one row per stamp is not a choice.
    Every merge rule is stated here rather than implied:

    * ``targets`` and ``graphs`` UNION (a graph identity is the whole trace, so
      two classes that produce the same identity produced the same graph);
    * ``requires`` must AGREE — it is derived from the contract's own dtype,
      never hand-written, so a disagreement is a producer bug and refuses;
    * ``demand`` keeps the block with the largest ``worst_case_request_bytes``
      and NAMES the class it came from (``worst_case_class``). A pod running
      this lane must hold the worst case of what it serves, which is exactly
      the rule the block already applies across entrypoints.

    Per-class fidelity is never lost: ``classes[]`` carries each class's own
    lanes, contracts and demand verbatim. The union exists so today's hub
    decodes an accurate superset with no hub change (th#2277 owns the
    release-doc shape; a per-class lane row is theirs to add).
    """

    targets: dict[str, set[str]] = {}
    unobserved: dict[str, set[str]] = {}
    graphs: dict[str, dict[str, Any]] = {}
    passes: dict[str, tuple[str, ...]] = {}
    entries: dict[str, dict[str, Any]] = {}
    owners: dict[str, list[str]] = {}
    for derivation in derivations:
        for lane in derivation.lanes:
            stamp = lane.contract
            owners.setdefault(stamp, []).append(derivation.name)
            targets.setdefault(stamp, set()).update(lane.targets)
            unobserved.setdefault(stamp, set()).update(lane.unobserved_targets)
            for record in lane.graphs:
                graphs.setdefault(stamp, {}).setdefault(record.graph, record)
            seen_passes = passes.setdefault(stamp, tuple(lane.passes))
            if tuple(lane.passes) != seen_passes:
                raise DeriveError(
                    f"lane {stamp!r} is declared by "
                    f"{owners[stamp]!r} with DIFFERENT transform passes "
                    f"({list(seen_passes)!r} vs {list(lane.passes)!r}). A "
                    f"serving boot adopts one lane document, and a graph "
                    f"derived under a different pass set is a graph for a "
                    f"module the boot does not have."
                )
            entry = dict(derivation.lane_contracts.get(stamp) or {})
            if not entry:
                continue
            merged = entries.get(stamp)
            if merged is None:
                entries[stamp] = entry
                continue
            if merged.get("requires") != entry.get("requires"):
                raise DeriveError(
                    f"lane {stamp!r} is declared by {owners[stamp]!r} with "
                    f"DIFFERENT capability floors ({merged.get('requires')!r} "
                    f"vs {entry.get('requires')!r}). A floor is derived from "
                    f"the contract's own dtype and is never written by hand, "
                    f"so this cannot be an authoring difference."
                )
            if _worst_case(entry) > _worst_case(merged):
                entries[stamp] = entry
                merged = entry
    for stamp, names in owners.items():
        if len(names) < 2 or stamp not in entries:
            continue
        winner = max(
            names,
            key=lambda name: _worst_case(
                next(
                    (
                        d.lane_contracts.get(stamp) or {}
                        for d in derivations
                        if d.name == name
                    ),
                    {},
                )
            ),
        )
        entries[stamp]["worst_case_class"] = winner
        warnings.append(
            f"lane {stamp}: declared by {len(names)} model classes "
            f"({', '.join(sorted(names))}); the release-wide `demand` row is "
            f"{winner}'s (the largest worst case). Each class's own row is in "
            f"`classes[]`."
        )
    lanes = tuple(
        torchcg.LaneGraphs(
            contract=stamp,
            targets=tuple(sorted(targets[stamp])),
            graphs=tuple(graphs.get(stamp, {}).values()),
            # A target another class OBSERVED is observed for the lane: the
            # release-wide row states unobserved only what NO class reached.
            unobserved_targets=tuple(sorted(
                unobserved[stamp]
                - {record.target for record in graphs.get(stamp, {}).values()}
            )),
            passes=passes[stamp],
        )
        for stamp in sorted(owners)
    )
    return lanes, entries


def _worst_case(entry: Mapping[str, Any]) -> int:

    demand = entry.get("demand")
    if not isinstance(demand, Mapping):
        return -1
    value = demand.get("worst_case_request_bytes")
    return int(value) if isinstance(value, int) else -1


def _agreed(values: Sequence[Any]) -> Optional[Any]:
    """The one value every subject class states, or None when they differ."""

    distinct = [value for index, value in enumerate(values)
                if value not in values[:index]]
    return distinct[0] if len(distinct) == 1 else None


def derive_release(
    module: ModuleType,
    *,
    checkpoint_dir: Path,
    lockfile: Optional[Path] = None,
    graph_cas: Optional[Path] = None,
    checkpoint_trees: Mapping[str, Path] = MappingProxyType({}),
    trace_workers: Optional[int] = None,
) -> ReleaseDeriveResult:
    """Derive the release metadata document for one endpoint module.

    ``checkpoint_trees`` names the tree ONE model needs when it is not the
    primary one — keyed by MODEL CLASS, or by entrypoint slot name.

    ``trace_workers`` (pgw#1603) is the parallel degree over derive ITEMS —
    the (class × lane × defaults-variant × structural-class) trace units.
    ``None`` derives it from this host: ``min(item count, cores)``, with
    ``GEN_WORKER_TRACE_WORKERS`` as the config override. ``1`` runs the
    items sequentially in-process.
    """

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

    subjects = _subject_classes(module)
    marked, unmarked = _module_model_classes(module)
    known = {cls.__name__ for cls in marked + unmarked}
    slots = {
        slot_name
        for cls in (subjects or (None,))
        for plan in _entrypoints(module, cls)
        for slot_name, _slot_cls in plan.model_slots
    }
    unknown = sorted(set(checkpoint_trees) - known - slots)
    if unknown:
        raise DeriveError(
            f"--checkpoint/--checkpoint-ref names {unknown!r}, which is "
            f"neither a model class of {module.__name__!r} "
            f"({sorted(known)!r}) nor an entrypoint model slot "
            f"({sorted(slots)!r}). A tree nothing loads from is a typo, not a "
            f"spare."
        )

    endpoint_name = (
        f"{module.__name__}:{'+'.join(cls.__name__ for cls in subjects)}"
    ).rstrip(":")

    warnings: list[str] = []

    # THE CONSTRUCTION CENSUS (pgw#1647), before any tracing — it is the cheaper
    # question, and a tree that cannot be built from its own configs must not
    # spend a trace first. ONE per release, over the PRIMARY tree, under the
    # union of every declared lane of every subject class: the census is
    # lane-invariant by construction and that invariance is checked here.
    construction_census = _construction_census(
        checkpoint_dir,
        tuple(
            (
                lane_contract_handle(f"class {cls.__name__!r}", lane),
                _torch_dtype(lane.dtype),
            )
            for cls in subjects
            for lane in model_declared_lanes(cls)
        ),
    )
    for cls in subjects:
        owner = _checkpoint_tree(checkpoint_trees, checkpoint_dir, cls)
        if owner != checkpoint_dir:
            # LOUD, not silent, and not a refusal. The census describes the tree
            # it was taken from, and th#2281 keys its storage by CONFIG DIGEST —
            # so a class loading its own tree needs its own census row, which
            # this document has no field for yet. Saying so is the honest
            # answer; emitting the primary tree's census as if it described this
            # class's would be the "second carrier" defect one level up.
            warnings.append(
                f"class {cls.__name__!r} loads its own checkpoint tree "
                f"({owner}), and the construction census in this document is "
                f"the PRIMARY tree's ({checkpoint_dir}). Nothing is MIS-KEYED: "
                f"the census's key is image x config tree (th#2287), this run "
                f"binds the primary tree, and the hub files the document under "
                f"that bind. It is INCOMPLETE — no published census describes "
                f"the auxiliary tree, so its serve-time fence replays only the "
                f"census it builds itself. A per-tree census belongs to the "
                f"BIND CONTRACT (th#2287 slice 2b), not to this document"
            )

    items = derive_items(module)
    outcomes = _run_items(
        module, items,
        checkpoint_dir=checkpoint_dir,
        graph_cas=graph_cas,
        program_sink=program_sink,
        checkpoint_trees=checkpoint_trees,
        trace_workers=trace_workers,
    )
    by_class: dict[int, dict[int, list[_ItemOutcome]]] = {}
    for item, outcome in zip(items, outcomes, strict=True):
        by_class.setdefault(item.class_index, {}).setdefault(
            item.lane_index, []
        ).append(outcome)

    derivations = [
        _derive_class(
            torchcg, module, cls,
            warnings=warnings,
            lane_outcomes=by_class.get(class_index, {}),
        )
        for class_index, cls in enumerate(subjects or (None,))
    ]

    lanes, lane_contracts = _merge_lanes(torchcg, derivations, warnings)
    graphs_document = torchcg.GraphSetDocument(stack=stack, lanes=lanes)

    entrypoints: dict[str, Any] = {}
    for derivation in derivations:
        for name, block in derivation.entrypoints.items():
            entrypoints.setdefault(name, block)

    defaults_schema = _agreed([d.defaults_schema for d in derivations])
    model_type_name = _agreed([d.model_type for d in derivations])
    if len(derivations) > 1 and model_type_name is None:
        # NOT silently overloaded: the release is about several model TYPES
        # (wan-2.2 is the live case), the hub's release document holds ONE
        # defaults schema, and `classes[]` holds each class's own. th#2277
        # owns the release-doc shape and the per-class field is theirs.
        warnings.append(
            "this release's subject classes declare DIFFERENT model types "
            f"({sorted(str(d.model_type) for d in derivations)}), so the "
            "release-wide `model_type` and `checkpoint_defaults_schema` are "
            "null — one release document holds one of each. Every class's own "
            "type and schema are in `classes[]` (th#2277 owns the per-class "
            "hub field)."
        )

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
        # pgw#1647 / th#2281. What the module IS, per declared lane — the
        # complete tensor set incl. computed non-persistent buffers, the tied
        # alias groups, the quantizer's swapped classes, eval mode. The hub
        # stores it and forwards it; it interprets no torch semantics. The
        # serve-time fence REPLAYS this instead of re-deriving trust.
        "construction_census": construction_census,
        "entrypoints": entrypoints,
        # pgw#1650: THE PER-CLASS BREAKDOWN, and the authoritative one. A
        # release derives EVERY compile-marking class (Paul, 2026-08-21), each
        # against its own checkpoint tree, so `graphs`/`lane_contracts` above
        # are the UNION over these rows — see `_merge_lanes` for every merge
        # rule. `fork_axes`, `model_type` and the defaults schema are class
        # facts and live here undiluted.
        "classes": [
            derivation.as_document(stack)
            for derivation in derivations
            if derivation.name
        ],
        "fork_axes": {
            "structural": [
                row
                for derivation in derivations
                for row in derivation.fork_axes["structural"]
            ],
            "shapes": {
                axis: value
                for derivation in derivations
                for axis, value in derivation.fork_axes["shapes"].items()
            },
        },
        "checkpoint_defaults_schema": defaults_schema,
        "model_type": model_type_name,
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
        weightless=not subjects and any(d.traced_entrypoints for d in derivations),
        unmarked_lanes=tuple(
            name for d in derivations for name in d.unmarked_lanes
        ),
        unenumerable_entrypoints=tuple(
            row for d in derivations for row in d.unenumerable
        ),
        unservable_payloads=tuple(
            f"{r['entrypoint']}[{r['payload']}]: {r['error']} (at {r['frame']})"
            for d in derivations for r in d.unservable
        ),
        classes=tuple(d.name for d in derivations if d.name),
        class_lane_graphs=tuple(
            (d.name, lane.contract, tuple(record.graph for record in lane.graphs))
            for d in derivations for lane in d.lanes
        ),
        census_components=tuple(
            sorted(construction_census.get("components", {}))
        ),
        census_absent=str(construction_census.get("absent", "")),
    )


__all__ = [
    "DOCUMENT_KIND",
    "ENUM_CAP",
    "DeriveError",
    "DeriveItem",
    "PayloadEnumerationRefused",
    "ReleaseDeriveResult",
    "derive_items",
    "derive_release",
    "static_bind_declared",
]
