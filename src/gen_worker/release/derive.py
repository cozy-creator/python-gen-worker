"""The publish-time instrumented derive (pgw#1370).

Per declared execution lane, inside the release env, on CPU: instantiate the
author's ``Model`` subclass, run its ``load`` AS-IS against a CONFIG-ONLY
checkpoint tree under ``torchcg.hollow_session``, drive the module's
``@entrypoint`` functions with AUTO-ENUMERATED trace payloads under
instrumented discovery, and stamp the observed graph set -- plus the lane
contracts and the model type's checkpoint-defaults schema -- as the static
release metadata document.

**A contract is METADATA, never a gate** (Paul, 2026-08-19, "A NORMAL TRACE
MUST JUST WORK"; pgw#1488). A model class that names no tensorfs contract is
traced under a DERIVED lane identity — ``derived.<model type>@1``, computed
identically here and at serve — and its load dtype comes from the checkpoint
when no contract states one. Nothing rekeys: ``cg-graph-v1`` hashes the
canonical trace plus ingress plus passes, so the lane's NAME never entered
graph identity, and a lane that declares a contract publishes exactly the
bytes it always did. Eager-forever is the class header's ``eager_only=``
declaration, with a reason, and never an inference from an empty lane tuple.

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
import logging
import itertools
import json
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Optional

from ..serving.entrypoints import ENTRYPOINT_ATTR
from ..serving.model import (
    Model,
    ModelDeclarationError,
    eager_only_reason,
    is_derived_lane,
    lane_handle,
    model_lanes,
    model_requires,
)
from ..serving.model import model_type as _strict_model_type
from .trace_context import (
    StepBudgetReached,
    TraceLoadContext,
    TraceRequestContext,
)

_LOG = logging.getLogger("gen_worker.release.derive")

#: A hollow derive's REAL tensors are config-computed buffers and lifted
#: constants -- KB to MB. Anything past this is a checkpoint weight that
#: escaped hollow instantiation, and it must never reach a graph blob.
_REAL_TENSOR_BYTES_CEILING = 64 * 1024 * 1024

DOCUMENT_KIND = "gen-worker.release-metadata@1"

#: Auto-enumeration cross-product cap. Overflow warns and traces the
#: deterministic prefix (field declaration order x enum declaration order);
#: the rest is first-encounter discovery.
ENUM_CAP = 64

#: Denoise steps per enumerated pass. Every step of a diffusion loop runs the
#: SAME shapes, so one is the whole observation; the derive re-drives
#: unbudgeted when a marked module has still not been reached (post-loop
#: modules like a marked VAE decoder). None = the author's own step count.
TRACE_STEP_BUDGET: Optional[int] = 1


class DeriveError(RuntimeError):
    """The release derive cannot state this endpoint's graph set."""


def model_model_type(cls: type) -> Optional[type]:
    """The class-header model type, or None for an undeclared base."""
    try:
        return _strict_model_type(cls)
    except ModelDeclarationError:
        return None


def lane_contract_handle(owner: str, lane: Any) -> str:
    try:
        return lane_handle(lane)
    except ModelDeclarationError as exc:
        raise DeriveError(f"{owner}: {exc}") from exc


@dataclass(frozen=True)
class ReleaseDeriveResult:
    """The emitted document plus the summary a pipeline log wants."""

    document: bytes
    digest: str
    endpoint: str
    lane_graphs: dict[str, tuple[str, ...]]  # lane contract -> graph hashes
    warnings: tuple[str, ...] = ()
    #: pgw#1392: NO model class anywhere, so there is nothing to hold at all.
    #: Distinct from eager-permanent, which holds a model and compiles none —
    #: both land on "no lanes", and a log that conflates them lies.
    weightless: bool = False
    #: pgw#1488: the class's DECLARED eager-forever reason (``eager_only=``).
    #: Non-empty means no trace was attempted and the author said why.
    eager_only: str = ""
    #: Lanes whose identity was DERIVED (no contract declared). Their handles
    #: read ``derived.*``; contract metadata attaches to the artifacts later.
    derived_lanes: tuple[str, ...] = ()
    #: Lanes that TRACED and found nothing marked via ``ctx.compile``. Zero
    #: graphs because the author marked zero modules — measured, not assumed.
    unmarked_lanes: tuple[str, ...] = ()

    @property
    def eager_permanent(self) -> bool:
        return not self.lane_graphs


def _torchcg() -> ModuleType:
    """The VENDORED torchcg -- the one the miner compiles with.

    Never a top-level ``torchcg``: if an endpoint pinned one, the trace and
    the mint would run different compilers and their graph identities could
    disagree silently. The vendored rev is recorded in
    ``gen_worker/_vendor/VENDORED.toml``.
    """

    from .._vendor import torchcg

    return torchcg


def _hollow() -> ModuleType:
    """The publish-time hollow session module, imported BY ITS OWN NAME.

    torchcg deliberately does not re-export ``hollow`` from the package root
    (it names diffusers/transformers loaders; a root export would put them on
    every serve-role import closure).
    """

    import importlib

    return importlib.import_module("gen_worker._vendor.torchcg.hollow")


def _program_sink(cas_root: Optional[Path]) -> Optional[Any]:
    """Store each discovered graph's SERIALIZED ExportedProgram in the CAS.

    Paul's ruling (2026-08-20): the derive keeps THE WHOLE TRACED GRAPH, not
    just its hash -- "we only ever need to run trace() once" now holds
    literally. The runtime miner downloads this blob and runs inductor on
    it; it never re-traces and never executes author code at mint time.

    Bytes-at-rest is tensorfs's charter (LIBRARY-BOUNDARIES), so the blob
    goes into a tensorfs ``LocalCAS`` and only its digest travels in the
    release document, beside the cg-graph-v1 hash and the ingress spec.
    Portability needs no new fence: an ExportedProgram is torch-coupled and
    the document's own env closure pins torch -- the same validity rule
    compiled artifacts already live under.
    """

    if cas_root is None:
        return None

    import io

    import torch

    from .._vendor.tensorfs import LocalCAS

    cas = LocalCAS(Path(cas_root))

    def sink(graph: str, program: Any) -> str:
        _assert_weights_free(torch, program)
        buffer = io.BytesIO()
        torch.export.save(program, buffer)
        del graph
        return str(cas.put_bytes(buffer.getvalue()))

    return sink


def _trace_device() -> str:
    """The DEVICE CLASS this derive traces on -- and it is a real choice.

    pgw#1458: a graph's device is established at TRACE time and cannot be
    re-homed downstream, so a cpu-traced graph cannot be cuda-minted. torchcg
    records the class in the declaration and refuses the mismatch by name
    (`RuntimeCompatibility.key`), which is the whole point -- but the refusal
    is only useful if the derive states the class DELIBERATELY rather than
    inheriting a default that happens to be wrong for the host.

    A fake-cuda trace needs no silicon in principle, and torchcg proves that
    for plain modules. It does NOT yet hold for a full diffusers pipeline on a
    GPU-less box: `encode_prompt` moves real token ids to the execution device
    and the fake-tensor path runs a real cuda kernel for them
    (`No CUDA GPUs are available`). Until that is closed, this states the
    truth about the host instead of failing at the first pipeline call.

    So: cuda when there IS a device, cpu otherwise, and the fallback SAYS what
    it costs. A cpu-derived document is not a degraded cuda one -- it is a
    different graph specialization, and a cuda mint of it refuses by name rather than
    serving something wrong.
    """

    try:
        import torch
    except ImportError:  # pragma: no cover - torch is the derive's premise
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    _LOG.warning(
        "derive: no CUDA device is visible, so this derive traces on CPU and "
        "produces CPU-CLASS graphs. They are a different graph specialization from the "
        "cuda graphs a GPU pod serves — a cuda mint of this document refuses "
        "by name (pgw#1458), it does not silently serve. Derive on a "
        "CUDA-bearing host to publish servable graphs."
    )
    return "cpu"


def _assert_weights_free(torch: Any, program: Any) -> None:
    """Prove the blob carries no weights -- WITHOUT rewriting anyone's device.

    This used to be ``_demote_fakes_to_meta``, which replaced every fake tensor
    with a META one. The rationale was real -- ``torch.export.save`` must not
    write a phantom storage that makes the archive claim bytes it does not have
    -- but the cure destroyed the device, and pgw#1458 made the device
    load-bearing. The result was one blob with TWO device stories: 1,922 graph
    node metas on ``cuda:0`` against 686 state-dict entries on ``meta``, and
    AOTI reads BOTH, so every sd1.5 class died on
    ``FakeTensorDeviceMismatchError cuda:0 and meta`` -- the mirror image of
    the failure the device work had just fixed. ``meta`` has no device
    sub-type, so it cannot express "no bytes, on cuda"; a fake tensor already
    does.

    Measured, which is why the rewrite is gone rather than adjusted: saving a
    fake-parameter program writes **0 bytes** of weight payload and records the
    fake tensor's OWN device in ``model_weights_config.json`` (cpu trace ->
    cpu, cuda trace -> cuda), and the cpu blob reloads with its state dict
    intact. Shape + dtype + device + no bytes is exactly the property the
    demotion was reaching for, and the fake tensor has it already.

    What survives is the INVARIANT, asserted instead of enforced by rewriting:
    a real tensor with real storage in the state dict would put weights in a
    graph artifact. Buffers hollow_session computed for REAL are legitimate and
    small (their values are what a literal-bearing trace digests), so the
    refusal names a size floor rather than realness.
    """

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
    """``(the ONE traced Model subclass or None, the eager_only reason)``.

    pgw#1488: every model class has a lane unless it declares
    ``eager_only="<reason>"``, so "which class do we trace" is now the same
    question as "which class is not declared eager". A module whose only model
    class IS eager-declared answers ``(None, reason)`` — the same shape the
    eager-permanent path always had, plus the author's own words for it.
    """

    found: list[type] = []
    eager: list[tuple[type, str]] = []
    for value in vars(module).values():
        if not (
            inspect.isclass(value)
            and issubclass(value, Model)
            and value is not Model
            and getattr(value, "__module__", None) == module.__name__
        ):
            continue
        reason = eager_only_reason(value)
        if reason:
            eager.append((value, reason))
            continue
        try:
            lanes = model_lanes(value)
            # pgw#1391: `model_lanes` hands back the lane OBJECTS without
            # reading them, so a class whose lane is a CONTRACT still has to
            # have that contract read here. A derived lane has nothing to
            # read — it carries no document by construction, which is the
            # whole point — so it is not put through the contract check.
            from ..serving.model import lane_dtype

            for lane in lanes:
                if is_derived_lane(lane):
                    continue
                lane_dtype(lane, where=f"class {value.__qualname__!r}")
        except ModelDeclarationError as exc:
            raise DeriveError(str(exc)) from exc
        if lanes:
            found.append(value)
    if len(found) > 1:
        raise DeriveError(
            f"module {module.__name__!r} has more than one compilable model "
            f"class ({[cls.__name__ for cls in found]!r}); a release derives "
            f"ONE. An auxiliary model that another slot drives and that "
            f"compiles nothing declares "
            f"eager_only=\"<why>\" on its class header — that is the "
            f"declaration, and it is not an empty lanes tuple."
        )
    return (found[0] if found else None, eager[0][1] if eager and not found else "")


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
    #: ``None`` for a WEIGHTLESS entrypoint (pgw#1392) — no model, no lane,
    #: nothing traced; it is never a trace subject.
    model_param: Optional[str]
    ctx_param: str
    #: (param name, annotation, base trace value) per platform-injected fact.
    injected: tuple[tuple[str, Any, Any], ...]
    #: Every model slot in signature order: (param name = SLOT NAME, class).
    #: The lane-declaring class may fill more than one slot (h3's `video` and
    #: `video_ref` are two checkpoints of one model type).
    model_slots: tuple[tuple[str, type], ...] = ()


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


#: The RULED signature order (Paul, 2026-08-19 line review; pgw#1382): one
#: rule with ``load(self, ctx)``. Malformed order is a typed refusal HERE, at
#: derive/publish, not a runtime surprise.
_SLOT_ORDER = ("ctx", "payload", "model", "adapter")


def _entrypoints(
    module: ModuleType, model_cls: Optional[type]
) -> list[_Entrypoint]:
    """The module's entrypoints bound to ``model_cls``.

    ``model_cls`` is ``None`` for a module that declares no lane-bearing
    Model class. Two disjoint cases live behind that: an eager-permanent
    (``lanes=()``) module — whose entrypoints DO carry model slots and
    derive no lane, exactly as before — and a WEIGHTLESS module (pgw#1392),
    whose entrypoints carry ZERO model slots. Only the latter derive here,
    so an eager-permanent release is byte-identical to what it was.
    """
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
                # Every Model-annotated parameter is a SLOT; its NAME is the
                # slot name in the request envelope (th#2140 5c).
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
            # No lane-bearing class: only a WEIGHTLESS entrypoint (pgw#1392)
            # derives. One that declares slots belongs to a lane this module
            # does not have (eager-permanent) — unchanged, skipped.
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
            # No RequestContext annotation anywhere: the sole remaining
            # parameter is ctx (the minimal (payload, model, ctx) shape).
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
    """ctx-FIRST: ``(ctx, payload, model(s), adapter(s))`` -- a typed refusal.

    One rule with ``load(self, ctx)`` (Paul, 2026-08-19). The derive still
    calls by KEYWORD; the order is a READABILITY contract the publish gate
    enforces so every endpoint in the fleet reads the same way.
    """

    ranks = [_SLOT_ORDER.index(role) for _, role in roles]
    if ranks != sorted(ranks):
        spelled = ", ".join(f"{param}: {role}" for param, role in roles)
        raise DeriveError(
            f"@entrypoint {name}: parameters are out of the ruled order. "
            f"An entrypoint reads (ctx, payload, model(s), adapter(s)); got "
            f"({spelled})"
        )


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


def _literal_axis(annotation: Any) -> Optional[list[Any]]:
    """A ``Literal[...]`` field's values -- a PRESET axis, same as an enum.

    Shape-rich packed models state their reachable QUANTITIES as numeric
    Literals rather than enums (h3's ``StepPreset = Literal[20, 30, 50]``,
    ``DurationS = Literal[5, 10]``, ``Fps = Literal[24, 48, 60]``): the API
    boundary refuses everything else, so the Literal IS the preset-reachable
    set, and enumerating it is exactly the lazy-coverage rule -- enumerate
    what presets reach, discover the rest on first encounter.

    NUMERIC literals only, deliberately. A numeric preset is a quantity the
    model RUNS at (it reaches the graph's ingress: sequence length, frame
    count, step ladder); a STRING literal names a host-side policy
    (``SdxlScheduler``, ``ImageFormat``) that never changes a marked
    module's shapes, and cross-producting it would explode the enumeration
    for zero graph specializations. A string axis that DOES bear shape is spelled as
    a StrEnum, which enumerates on the branch above.
    """

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

    # pgw#1384: the DEFAULT-parameter combination comes FIRST -- the serving
    # hole list inherits document order and the miner mints in that order,
    # so the class an all-defaults payload exercises must lead. Then enum
    # declaration order for the rest.
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


#: One adapter pick on the wire: the fully-pinned hub ref plus its strength.
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
    """The request envelope's JSON Schema, DERIVED FROM THE SIGNATURE.

    Parameter name IS slot name (th#2140 5c): an author naming a parameter
    ``turbo`` publishes the envelope key ``adapters.turbo``; a model
    parameter named ``video_ref`` publishes ``models.video_ref``. Renaming a
    parameter is therefore a VISIBLE API BREAK and publish flags it exactly
    as it flags a disappearing lane. The hub publishes this schema as the
    entrypoint's auto-generated docs.
    """

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
    """A synthesized adapter for the enumeration's adapter-riding arms.

    Carries the model type's platform ``Lora.Defaults`` (what
    ``adapter.defaults`` reads as); its path points nowhere -- adapter I/O is
    neutralized at trace by the load context. ``adapter_cls`` is the slot's
    declared KIND (``DistillationAdapter`` for a distillation slot — the
    annotation is the declaration, pgw#1382's typed-takeover guard).
    """

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
    """The Adapter subclass an adapter-shaped fact declares, else None.

    ``DistillationAdapter | None`` and ``list[Adapter]`` both answer their
    ELEMENT type, so the synthesized fake is the slot's own kind.
    """
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
    """Per injected parameter, its enumerated trace values.

    Adapter-shaped facts enumerate BOTH states (absent and riding); other
    facts keep their single trace value.
    """

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


def _contract_document(
    owner: str, lane: Any, warnings: list[str]
) -> Optional[dict[str, Any]]:
    """The lane contract's canonical layout document.

    The whole point of contract OBJECTS (Paul, 2026-08-18) is that the full
    layout TRAVELS in the release metadata, so the platform needs no prior
    knowledge of it. tensorfs#111 spells it as ``Contract.document``, a
    canonical JSON **STRING** — an earlier duck-typed reader here accepted
    only a ``dict`` and therefore shipped ``"document": null`` on every lane
    while the stamp looked correct.

    A lane that CLAIMS a document must produce a readable one (typed
    refusal: a stamp with an unreadable layout behind it is worse than no
    row). A lane object that exposes none at all — a resolved ``LaneRef``
    stand-in — travels stamp-only with a WARNING naming it, never silently.
    """

    if is_derived_lane(lane):
        # pgw#1488: a DERIVED lane has no contract and never claimed one. The
        # honest row is no document — the artifacts exist under the derived
        # identity and a contract, when someone authors one, ATTACHES to them
        # as fleet metadata. That is a later step, not a precondition.
        return None

    claimed = False
    for attribute in ("document", "as_dict", "to_dict"):
        value = getattr(lane, attribute, None)
        if value is None:
            continue
        claimed = True
        if callable(value):
            value = value()
        if isinstance(value, dict):
            return value
        if isinstance(value, (str, bytes)):
            try:
                parsed = json.loads(value)
            except ValueError:
                continue
            if isinstance(parsed, dict):
                return parsed
    if claimed:
        raise DeriveError(
            f"{owner}: lane {lane!r} exposes a layout document that cannot "
            f"be read as JSON. The document travels in the release metadata; "
            f"a stamp with an unreadable layout behind it is worse than no "
            f"row."
        )
    # pgw#1391: this was a WARNING, and it fired on EVERY lane — including the
    # live `sdxl.diffusers-bf16@1`, whose document tensorfs really does
    # publish. Every release therefore carried `"document": null` and an empty
    # digest, which is se#756's "the release proves NO lane". A stamp with no
    # layout behind it is the bug, so it refuses. Under the contract library
    # every real lane HAS a document.
    raise DeriveError(
        f"{owner}: lane {lane_handle(lane)} carries NO layout document. The "
        f"whole point of contract OBJECTS is that the layout TRAVELS in the "
        f"release metadata, so the platform needs no prior knowledge of it — "
        f"a stamp alone is a claim the hub cannot intern. Declare the lane as "
        f"an imported tensorfs Contract "
        f"(`gen_worker.models.SDXL_DIFFUSERS_BF16`-style), never a handle "
        f"string or a stand-in."
    )


def _contract_digest(lane: Any) -> str:
    """The contract object's OWN digest of its layout document.

    The hub interns the document and computes its own digest (th#2146); the
    producer's digest travels beside it so a mismatch is an assertion rather
    than a silent divergence in canonical serialization. Always the
    ``sha256:``-prefixed spelling, whatever the object stores.
    """

    digest = getattr(lane, "digest", None)
    if not isinstance(digest, str) or not digest:
        return ""
    return digest if digest.startswith("sha256:") else f"sha256:{digest}"


def _resolve_lane(torchcg: ModuleType, cls: type, lane: Any) -> Any:
    """The resolved ``ctx.lane``: always a LaneRef with a REAL torch dtype.

    A contract OBJECT (tensorfs registry entry / inline Contract) carries its
    own dtype (tensorfs#113's top-level field, safetensors spelling); the
    Model class header refuses dtype-less lanes at declaration, so this
    refusal is the derive-side restatement, never a fallback path.
    """

    handle = lane_contract_handle(f"class {cls.__name__!r}", lane)
    if is_derived_lane(lane):
        # pgw#1488: no contract, so no contract dtype. The precision comes
        # from the CHECKPOINT, resolved inside the load context (which is the
        # one place that holds the tree) so the trace and the serve read the
        # same source. `LaneRef` carries the handle; dtype stays open here.
        return torchcg.LaneRef(handle, dtype=None)
    try:
        # tensorfs#113's `dtype` is a PROPERTY that RAISES on a contract
        # declaring none (minimax.h3-dit-diffusers today), so this is a try,
        # never a getattr default -- otherwise the refusal below is bypassed
        # by an un-caught MissingDtype traceback.
        dtype = getattr(lane, "dtype", None)
    except Exception:
        dtype = None
    if dtype is None:
        raise DeriveError(
            f"lane {handle!r}: the contract object carries no dtype. A lane "
            f"IS a tensorfs layout contract (an imported object; its load "
            f"dtype is the contract's) — never a bare string."
        )
    return torchcg.LaneRef(handle, dtype=_torch_dtype(dtype))


def _derive_lane(
    torchcg: ModuleType,
    cls: type,
    lane: Any,
    plans: list[tuple[_Entrypoint, tuple[Any, ...]]],
    checkpoint_dir: Path,
    warnings: list[str],
    program_sink: Optional[Any] = None,
) -> Optional[Any]:
    """One lane's instrumented runs, merged across defaults variants.

    Per variant: fresh model, ``load`` (defaults variants change what
    ``ctx.defaults()`` answers, so the instance is rebuilt), then every
    (entrypoint x payload x adapter-state) combination drives under
    instrumented discovery. Combinations the author REFUSES with
    ``ValidationError`` are legitimately impossible servings and are
    skipped, counted in the warnings.
    """

    from ..api.errors import ValidationError

    hollow = _hollow()
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
        # TRACE_STEP_BUDGET: every step of a denoise loop runs the same
        # shapes, so one step per enumerated pass observes the whole set.
        # Modules that run after the loop are caught by the unbudgeted
        # re-drive below.
        request_ctx = TraceRequestContext(
            lane=resolved,
            checkpoint_ref=f"trace:{checkpoint_dir.name}",
            step_budget=TRACE_STEP_BUDGET,
        )
        # pgw#1458: the session is CAPTURED and handed to discovery. A
        # GPU-less cuda trace fakes the lifted constants; `discover_modules`
        # restores their real values FROM THE SESSION before it hashes and
        # before `program_sink` serializes. Omitting `session=` does not
        # degrade quietly — torchcg raises `DiscoveryError` naming the faked
        # constants — but a digest taken over faked values would be a lie,
        # so the wiring is the point, not the refusal.
        with hollow.hollow_session(_trace_device()) as session:
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
                if is_derived_lane(lane):
                    # pgw#1488. The trace RAN — the model loaded under the
                    # hollow session and the author marked no module, which is
                    # an observation, not a failure. Zero graphs is the honest
                    # answer and the caller prints it as one. The refusal below
                    # stays for a lane the author DECLARED: declaring a lane is
                    # saying "graphs key here", and then compiling nothing is a
                    # contradiction only the author can resolve.
                    return None
                raise DeriveError(
                    f"lane {handle!r}: load() marked nothing via ctx.compile(). "
                    f"A lane-declaring model compiles SOMETHING; a model that "
                    f"wants eager-forever declares "
                    f"eager_only=\"<why>\" instead."
                )
            modules = _named_marked_modules(model, load_ctx.marked_modules)

            # Secondary model slots (an auxiliary model with its own
            # checkpoint, e.g. h3's RIFE interpolator): a fresh instance per
            # slot, loaded under the SAME hollow session. Every slot named in
            # a signature must be fillable at trace or the release cannot
            # state its graph set.
            aides: dict[str, Any] = {}
            for plan, _payloads in plans:
                for slot_name, slot_cls in plan.model_slots:
                    if slot_cls is cls or slot_name in aides:
                        continue
                    aide = slot_cls()
                    try:
                        aide.load(
                            TraceLoadContext(
                                lane=resolved,
                                checkpoint_dir=checkpoint_dir,
                                model_type=model_model_type(slot_cls),
                                defaults_instance=None,
                            )
                        )
                    except Exception as exc:
                        raise DeriveError(
                            f"lane {handle!r}: entrypoint {plan.name!r} slot "
                            f"{slot_name!r} ({slot_cls.__name__}) failed to "
                            f"load under the trace session: "
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
                                # The pass observed its shapes; the remaining
                                # denoise steps repeat them.
                                pass
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
                lane_graphs = torchcg.discover_modules(
                    handle, modules, drive, program_sink=program_sink,
                    session=session,
                )
                if set(lane_graphs.targets) - {
                    record.target for record in lane_graphs.graphs
                }:
                    # A marked module the budgeted drive never reached (it
                    # runs AFTER the denoise loop). Re-drive unbudgeted
                    # before calling it unobserved.
                    request_ctx.step_budget = None
                    lane_graphs = torchcg.discover_modules(
                        handle, modules, drive, program_sink=program_sink,
                        session=session,
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
    graph_cas: Optional[Path] = None,
) -> ReleaseDeriveResult:
    """Derive the release metadata document for one endpoint module.

    The document's `closure` is THIS process's installed set (pgw#1472,
    :func:`gen_worker.env_identity.env_closure_hash`) — there is no second
    source and no flag that picks one. The derive runs inside the release
    image, so the mint child and the serving pod restate the same value from
    the same image; a stamp that any of the three cannot restate is an
    unadoptable release.
    """

    torchcg = _torchcg()
    program_sink = _program_sink(graph_cas)

    from ..env_identity import EnvIdentityError, env_closure_hash

    try:
        closure = env_closure_hash()
    except EnvIdentityError as exc:
        raise DeriveError(str(exc)) from exc

    cls, eager_only = _lane_model_class(module)
    endpoint_name = f"{module.__name__}:{cls.__name__ if cls else ''}".rstrip(":")

    lanes: list[Any] = []
    derived_lanes: list[str] = []
    unmarked_lanes: list[str] = []
    lane_contracts: dict[str, Any] = {}
    entrypoints: dict[str, Any] = {}
    warnings: list[str] = []
    plans: list[tuple[_Entrypoint, tuple[Any, ...]]] = []
    for plan in _entrypoints(module, cls):
        owner = f"@entrypoint {plan.name}"
        if cls is None:
            # pgw#1392: a WEIGHTLESS entrypoint has no lane, so there is no
            # trace subject and no pass is ever run. `traced_passes` says 0
            # because 0 is the truth, not because the enumeration was
            # skipped. It still publishes its envelope schema — the hub's
            # auto-generated API docs are the point of this block.
            payloads: tuple[Any, ...] = ()
        else:
            payloads, capped = _auto_payloads(owner, plan.payload_type)
            if capped:
                warnings.append(
                    f"{owner}: enum cross-product exceeds the cap "
                    f"({ENUM_CAP}); tracing the deterministic prefix -- the "
                    f"rest is first-encounter discovery (eager + background "
                    f"mint)"
                )
        plans.append((plan, payloads))
        entrypoints[plan.name] = {
            "envelope_schema": _envelope_schema(plan),
            # Renders EMPTY for a weightless entrypoint — an honest {}, never
            # a fabricated slot.
            "model_slots": {
                slot_name: slot_cls.__name__
                for slot_name, slot_cls in plan.model_slots
            },
            "traced_passes": len(payloads),
        }

    if cls is not None:
        requires = model_requires(cls)
        for lane in model_lanes(cls):
            lane_graphs = _derive_lane(
                torchcg, cls, lane, plans, checkpoint_dir, warnings,
                program_sink=program_sink,
            )
            if lane_graphs is None:
                # Traced, nothing marked (pgw#1488). No lane row: an empty one
                # would claim a keyed graph set that does not exist.
                unmarked_lanes.append(
                    lane_contract_handle(f"class {cls.__name__!r}", lane)
                )
                continue
            if is_derived_lane(lane):
                derived_lanes.append(lane_graphs.contract)
            lanes.append(lane_graphs)
            entry: dict[str, Any] = {
                "stamp": lane_graphs.contract,
                "document": _contract_document(
                    f"class {cls.__name__!r}", lane, warnings
                ),
                # The PRODUCER's own digest of that document. The hub interns
                # the layout and derives its own digest (th#2146); carrying
                # this lets it ASSERT the two agree instead of trusting a
                # re-serialization round-trip. Empty when the lane object
                # states none.
                "digest": _contract_digest(lane),
            }
            if is_derived_lane(lane):
                # pgw#1488: `document: null` is a BUG on a declared contract
                # (pgw#1391) and the HONEST state on a derived one. The reader
                # cannot tell those apart from a null, so the producer says
                # which it is. Written only on the derived branch, so every
                # contract-declaring release keeps its bytes exactly.
                entry["derived"] = True
            # ie#740 placement floor for THIS lane, read off the class header
            # (`requires=`). Absent = undeclared, and the platform default is
            # what the deployment gets — the honest state, never an invented
            # floor.
            floor = requires.get(lane_graphs.contract)
            if floor is not None:
                entry["requires"] = floor.render()
            lane_contracts[lane_graphs.contract] = entry

    graphs_document = torchcg.GraphSetDocument(closure=closure, lanes=tuple(lanes))
    payload_dict: dict[str, Any] = {
        "v": 1,
        "kind": DOCUMENT_KIND,
        "endpoint": endpoint_name,
        "graphs": graphs_document.as_dict(),
        "lane_contracts": lane_contracts,
        # The per-entrypoint request envelope, DERIVED FROM THE SIGNATURE
        # (parameter name = slot name). The hub publishes these as the
        # release's auto-generated API docs; a renamed parameter is a
        # visible API break, flagged like a disappearing lane.
        "entrypoints": entrypoints,
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
        # No lane class AND entrypoints still derived => every one of them
        # declared zero model slots. An eager-permanent module also has no
        # lane class, but its entrypoints carry slots and derive no plan.
        weightless=cls is None and bool(plans),
        eager_only=eager_only,
        derived_lanes=tuple(derived_lanes),
        unmarked_lanes=tuple(unmarked_lanes),
    )


__all__ = [
    "DOCUMENT_KIND",
    "ENUM_CAP",
    "DeriveError",
    "ReleaseDeriveResult",
    "derive_release",
]
