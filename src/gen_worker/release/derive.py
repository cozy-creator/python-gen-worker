"""The publish-time instrumented derive (pgw#1370).

Per declared execution lane, inside the release env, on CPU: instantiate the
author's ``Model`` subclass, run its ``load`` AS-IS against a CONFIG-ONLY
checkpoint tree under ``torchcg.hollow_session``, drive the module's
``@entrypoint`` functions with AUTO-ENUMERATED trace payloads under
instrumented discovery, and stamp the observed graph set -- plus the lane
contracts and the model type's checkpoint-defaults schema -- as the static
release metadata document.

**EVERY model class declares REAL lanes** (Paul's ruling pair, 2026-08-20;
pgw#1597/pgw#1599). A lane answers checkpoint COMPATIBILITY and lane
SELECTION, not merely compilation, so there is no derived, borrowed or
implicit identity to trace under — a class that names no tensorfs contract is
REFUSED at class definition, before this module ever sees it. Contracts are
made CHEAP rather than optional (tensorfs#130 generates a candidate from a
safetensors header). **Compilation participation is the MARK**: a model with
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
    lane_handle,
    model_marks_compile,
    model_lanes,
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

#: pgw#1599: the global ``DYNAMIC_AXES`` derive FLAG is DELETED. Which axis is
#: worth collapsing is a MEASURED, PER-MODEL question (pgw#1548), so it is
#: declared on the model class that measured it (``shapes={"aspect": DYNAMIC}``)
#: — never passed on a command line, where one word silently re-keyed every
#: graph in the fleet at once. torchcg still takes a
#: ``(target, input name, axis) -> bool`` predicate and holds no opinion about
#: what an axis MEANS; naming them stays the endpoint layer's job, and
#: :func:`dynamic_dim_policy` is now built FROM the declaration.


class DeriveError(RuntimeError):
    """The release derive cannot state this endpoint's graph set."""


def dynamic_dim_policy(shapes: Mapping[str, str]) -> Any:
    """Turn a model class's declared shape axes into torchcg's predicate.

    ``shapes`` is :func:`gen_worker.serving.model.model_shapes`'s answer —
    ``{axis: "static" | "dynamic"}``, written by the model's author.

    * ``aspect`` — axes 2.. of a rank-4+ feed, i.e. a latent's spatial sides.
      DYNAMIC collapses the whole aspect fan into one record (measured cold
      mint: sd15 100 s, SERVABLE — pgw#1548).
    * ``batch`` — axis 0. NEVER offered, and not declarable: CFG/batch is a
      PERMANENTLY STATIC shape fork (Paul, 2026-08-20), on two measured
      grounds — batch-dynamic removed zero specializations on the real
      endpoint, and batch-dynamic records fail to mint (tcg#78).
    """

    aspect = shapes.get("aspect") == DYNAMIC
    if not aspect:
        return None

    def policy(_target: str, _name: str, axis: int) -> bool:
        if axis == 0:
            return False  # batch: permanently static, never offered
        # Rank is not handed to the predicate, so "axis 2 or beyond" is the
        # spelling of spatial here. Axis 1 is a channel or a sequence length
        # and is never offered: neither varies across an aspect fan, so
        # admitting it would widen a graph over an axis no observation
        # supports.
        return axis >= 2

    return policy


class PayloadEnumerationRefused(DeriveError):
    """THIS ENTRYPOINT's payload cannot be auto-enumerated (pgw#1449).

    A property of one signature, never of the module -- so it is caught per
    entrypoint, stated in the document, and the entrypoints that CAN be
    enumerated are derived anyway. Deliberately a NARROW subclass with
    exactly one raise site: catching ``DeriveError`` instead would swallow
    slot-order violations, empty enums and non-msgspec payloads, which are
    author defects that must still take the module down.

    The distinction is the one the derive already draws for a combination
    the author refuses with ``ValidationError``: an enumeration the derive
    cannot reach is counted and named, not treated as a broken endpoint.
    """

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
    """A stable spelling of a type for the document (never ``repr`` noise)."""

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
    #: Lanes that TRACED and found nothing marked via ``ctx.compile``. Zero
    #: graphs because the author marked zero modules — measured, not assumed.
    #: pgw#1599: this is now the ONLY eager statement there is. `eager_only=`
    #: and the derived-lane machinery are deleted; a model with no marks
    #: declares real lanes like every other and simply mints no graph.
    unmarked_lanes: tuple[str, ...] = ()
    #: pgw#1449: entrypoints the enumerator could not build a trace payload
    #: for, name -> the typed reason. They are STATED, not silently dropped,
    #: and they no longer take the module's other entrypoints down with them.
    unenumerable_entrypoints: tuple[tuple[str, str], ...] = ()
    #: pgw#1527: enumerated payloads the ENDPOINT's own code refused to serve.
    #: One line each, naming the author frame — a skipped payload has to be
    #: louder than a missing one.
    unservable_payloads: tuple[str, ...] = ()

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

    Paul, 2026-08-19 (address-free): the bytes are local and their digest is
    machine-scoped, so nothing about them travels. What the document carries is
    the cg-graph-v1 identity; what this stores is one machine's bytes for that
    identity, under it. A mint on any box asks its own store for "the program
    for graph X" and never for a digest somebody else computed.

    Bytes-at-rest is tensorfs's charter (LIBRARY-BOUNDARIES), so the blob goes
    into a tensorfs ``LocalCAS`` through torchcg's ``LocalGraphStore``, which
    owns the graph->bytes ref.

    ⚠️ THIS BODY WAS SILENTLY REVERTED ONCE (pgw#1512 `c2edae08`, a conflict
    resolution in a commit about per-component dtype that says nothing about
    the sink). The revert restored digest-keyed banking while the document
    already carried NO address, so producer and consumer keyed on different
    things and no program could ever be resolved — a total break with no
    error anywhere, because a miss is silent. `test_derive_runs_in_a_release_
    env_with_no_top_level_torchcg_or_tensorfs` now asserts `has_program` per
    identity, which is the fence: this cannot be reverted green again.
    """

    if cas_root is None:
        return None

    import tempfile

    import torch

    from .._vendor.tensorfs import LocalCAS
    from .._vendor.torchcg.store import LocalGraphStore

    store = LocalGraphStore(LocalCAS(Path(cas_root)))

    def sink(graph: str, program: Any) -> None:
        _assert_weights_free(torch, program)
        # torch.export.save to a FILE, because the store admits files: it
        # hashes and links them in, so a large program never has to exist
        # twice in memory the way a BytesIO round-trip forces.
        with tempfile.TemporaryDirectory() as scratch:
            staged = Path(scratch) / "program.pt2"
            torch.export.save(program, str(staged))
            store.put_program(graph, staged)

    return sink


def _trace_device() -> str:
    """The DEVICE CLASS this derive traces on. Always ``cuda``, GPU or not.

    pgw#1458 stands: a graph's device is established at TRACE time and cannot
    be re-homed downstream, so a cpu-traced graph cannot be cuda-minted, and
    torchcg refuses the mismatch by name (`RuntimeCompatibility.key`). What
    is gone (Paul, 2026-08-19 "A NORMAL TRACE MUST JUST WORK"; tcg#64) is the
    HOST answering the question. It never should have: the fleet compiles
    cuda graphs, so a derive that emits cpu-class ones because the box it ran
    on had no GPU produces a document nothing can serve, and hands the author
    a device taxonomy to reason about for a device the trace never uses.

    A trace needs no silicon. torchcg's session drives the author's code on a
    device that EXISTS -- real sigmas, real token ids, real `encode_prompt` --
    and restates each exported program onto the stated device before it is
    hashed. Measured: a CPU-only sd1.5 derive reproduces the graph keys of the
    GPU-traced one, key for key.

    This is what makes `gen-worker lock` an AUTHOR-time command (2026-08-19
    "the full artifact axis and who runs what"): endpoint.lock is committed to
    git, produced once per version on whatever machine the author has.
    """

    return "cuda"


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
    """``(the ONE traced Model subclass or None, "")``.

    pgw#1599: EVERY model class declares real lanes — `lanes=` is required,
    `lanes=()` and `eager_only=` are deleted — so "which class do we trace"
    is now "which class MARKS a compile target". A module whose model classes
    mark nothing answers ``(None, "")``: nothing to trace, and the absent
    mark IS the author's statement (Paul's ruling pair, 2026-08-20).
    """

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
            lanes = model_lanes(value)
            # pgw#1391: `model_lanes` hands back the lane OBJECTS without
            # reading them, so a class whose lane is a CONTRACT still has to
            # have that contract read here.
            from ..serving.model import lane_dtype

            for lane in lanes:
                lane_dtype(lane, where=f"class {value.__qualname__!r}")
            marked = model_marks_compile(value)
        except ModelDeclarationError as exc:
            raise DeriveError(str(exc)) from exc
        # pgw#1599: the MARK selects the trace subject, not the lane. Every
        # class declares real lanes now (an auxiliary RIFE interpolator names
        # the `rife.*` document exactly as the DiT names its own), so "has a
        # lane" no longer distinguishes the compiled half of an endpoint from
        # the eager half. The `ctx.compile` mark does, and it is the author's
        # only statement about it.
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
    # NOBODY MARKS. The module is still not weightless — it holds a model
    # class with a real lane, a model type and a defaults schema, and all of
    # those belong in the document. It derives as the subject and reports
    # ZERO graphs (`unmarked_lanes`), which is the measured answer. Dropping
    # to `None` here would publish it as WEIGHTLESS — no model_type, no lane
    # row, no defaults schema — which is a different and false statement.
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
    raise PayloadEnumerationRefused(owner, name, annotation)


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


def _payload_field_names(payload_type: type) -> tuple[str, ...]:
    """Field names of one entrypoint's payload struct, or ``()``."""

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
    """Auto-enumerated trace payloads for one entrypoint, plus the capped flag.

    One payload per cross-product entry over the struct's ENUM-typed fields
    (field declaration order x enum declaration order -- deterministic);
    every other field at its default; required non-defaulted fields
    synthesized minimally by type.

    ``structural`` is the model class's declared STRUCTURAL fork axes
    (pgw#1599). It contributes ONE REPRESENTATIVE PER VARIANT CLASS for the
    payload field it names — not the field's full value set. That is the
    whole economy of the declaration: sdxl serves 8 schedulers that produce
    exactly 2 timestep dtypes, so 2 traces cover 8/8 where the blind
    cross-product would have cost 8 and `_literal_axis` (string literals
    excluded as host-side policy) enumerated 0 — which is why 5 of sdxl's 8
    scheduler configs fell to loud eager with nothing able to say why.
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
    #: payload field -> the declared representatives, one per variant class.
    declared_axes: dict[str, list[Any]] = {}
    for axis, declaration in structural.items():
        variants = declaration.variants()
        if not any(field.name == declaration.field for field in struct_fields):
            continue  # this axis forks a DIFFERENT entrypoint's payload
        declared_axes[declaration.field] = [value for _, value in variants]
    for field in struct_fields:
        annotation = _strip_annotated(field.type)
        declared_values = declared_axes.pop(field.name, None)
        if declared_values is not None:
            # An AUTHOR-DECLARED axis wins over whatever the annotation would
            # have enumerated: the author measured which values fork the
            # program, and the platform never invents an axis (pgw#1597).
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


#: How far the provenance walk descends from the model instance (pgw#1506).
#: `model.engine.pipeline.components[...]` is depth 2; the headroom is for a
#: second wrapper, and the bound exists so a cyclic or pathological object
#: graph refuses with a sentence instead of running forever.
PROVENANCE_MAX_DEPTH = 4


def _components_mapping(value: Any) -> Optional[Mapping[str, Any]]:
    """``value.components`` when it is a Mapping -- the diffusers convention.

    Read through ``try`` because ``components`` is a PROPERTY on every
    diffusers pipeline and a half-built one can raise; a wrapper that cannot
    answer is simply not a component holder.
    """

    try:
        components = getattr(value, "components", None)
    except Exception:
        return None
    return components if isinstance(components, Mapping) else None


def _is_wrapper(value: Any) -> bool:
    """A plain object that may HOLD a pipeline -- worth descending into.

    Deliberately narrow. An ``nn.Module`` is never descended: its ``vars()``
    is ``_parameters``/``_modules``/``_buffers``, so walking one would offer
    provenance names for the marked module's own internals and turn a 10 GiB
    denoiser into a traversal. Everything without an instance ``__dict__``
    (msgspec Structs, primitives, containers) has no attributes to walk.
    """

    import torch

    if isinstance(value, (torch.nn.Module, torch.Tensor)):
        return False
    if isinstance(value, (str, bytes, int, float, bool, Path)):
        return False
    return hasattr(value, "__dict__") and isinstance(getattr(value, "__dict__"), dict)


def _named_marked_modules(instance: Any, marked: list[Any]) -> dict[str, Any]:
    """Deterministic provenance names for the author's ctx.compile marks.

    The author marks REAL objects; the document needs stable names. Names
    come from where the module actually lives on the model instance:
    component names of any ``.components``-bearing attribute (the diffusers
    convention), then bare attribute names, then the dotted PATH as the
    disambiguated spelling. A marked module that cannot be named on the
    instance is refused -- provenance is part of the release row.

    **The walk is RECURSIVE (pgw#1506).** It used to look exactly one level
    deep -- ``vars(instance)`` plus ``.components`` on each value -- which
    assumed the model holds its pipeline directly. That is sdxl's and sd15's
    shape, not a rule: an endpoint with a runtime engine owns the engine and
    the ENGINE owns the pipeline, so minimax-h3's DiT sits at
    ``model.engine.pipeline.components['transformer']`` and the resolver
    could not see it. The engine wrapper carries real state (an AdaLN cache,
    conditioner buffers, the serve recipe), so it is a legitimate authoring
    shape and the tracer is what had to learn to traverse it. Adding a
    second reference to the pipeline on the model would have satisfied
    depth 1 and left every other engine-wrapper endpoint broken.

    Depth-1 endpoints are unaffected by construction: the candidate set at
    depth 1 is offered exactly as before, and the recursion only ADDS names
    that were previously unreachable.
    """

    candidates: dict[int, str] = {}
    #: (owner path, component name, component) for every components mapping
    #: found anywhere in the walk -- uniqueness is decided across ALL of them,
    #: because "is this bare name unique?" is a question about the instance,
    #: not about one attribute.
    component_rows: list[tuple[str, str, Any]] = []
    #: Paths not descended because the depth bound was reached. Named only if
    #: a marked module turns out to be unnameable, where it is the likely why.
    truncated: list[str] = []

    def offer(module: Any, name: str) -> None:
        identity = id(module)
        if identity not in candidates or len(name) < len(candidates[identity]):
            candidates[identity] = name

    # Cycle safety by IDENTITY: an engine that back-references its model
    # (`self.engine.model is self`) is an ordinary shape and must not loop.
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
                # A pipeline's components ARE its provenance; there is nothing
                # below them worth a name.
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
        # The bare name when it means ONE object across the whole instance,
        # the full dotted path otherwise -- the same preference order as
        # before, asked across every mapping the walk found instead of one.
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
    """The endpoint's compile stack, from the ONE definition (pgw#1489).

    The body lives in :mod:`gen_worker.env_identity`: a serving process reads
    the SAME rows off the SAME file to decide whether these artifacts are
    hers, and a second implementation of "what compiles this" is precisely
    how the two ends came to disagree.
    """

    from ..env_identity import EnvIdentityError, compile_stack_from_lockfile

    try:
        return compile_stack_from_lockfile(lockfile)
    except EnvIdentityError as exc:
        raise DeriveError(str(exc)) from exc


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


def endpoint_source_root(module: ModuleType) -> Optional[Path]:
    """The SOURCE ROOT the author's modules live under, or None if unknowable.

    The sys.path entry that holds the endpoint's code — `minimax_h3.main` ->
    `<repo>/src`, a bare fixture module -> the directory holding it. None when
    the module has no file (namespace/frozen), and None is treated as "cannot
    prove endpoint-owned", which keeps the conservative default.

    **pgw#1533: this used to return the top-level PACKAGE directory**
    (`<repo>/src/minimax_h3`), which silently uncovered every author module
    that is not inside that one package. h3's fps refusal raises in
    `src/cozy_rife.py` — a SIBLING top-level module at the same source root —
    so `deepest_endpoint_frame` could not prove it was the author's and the
    whole derive died on a product bug that pgw#1527 exists to skip. A shared
    helper module beside the main package is the common case, not an exotic
    one, so the roster was wrong for most endpoints rather than a few.

    Widening the root is safe only because :func:`deepest_endpoint_frame`
    SUBTRACTS: the SDK and any third-party install root are excluded from it
    explicitly, so a wider root can add author files and never adds a library.
    """

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
    # A PACKAGE's `__init__.py` sits one level below the source root; a bare
    # module sits directly in it.
    if here.name == "__init__.py":
        return here.parent.parent
    return here.parent


def _third_party_root(where: Path) -> bool:
    """Is this path inside an installed-dependency tree?

    Named by the install layout rather than by a list of libraries, so torch,
    diffusers and anything else an endpoint pulls in are covered without being
    enumerated.
    """

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
    """The DEEPEST frame of ``exc``, but only if the author's code raised it.

    pgw#1527, and the narrowness IS the ruling. A payload the endpoint cannot
    serve is a product fact and should cost ONE payload; an SDK defect is the
    derive's whole point and must still kill it loudly. Walls 1-8 were every
    one of them an SDK-frame exception (torchcg's hollow session, the trace
    context, the provenance walk, the output floor) — a blanket catch would
    have swallowed all eight and turned each into a quiet coverage gap, which
    is exactly the counter-argument the filer raised against themselves.

    So the test is positive and it is on the DEEPEST frame only: the innermost
    thing that actually raised must be a file under the endpoint's own source
    root. Author code that calls into torch and trips a shape error there
    reports a TORCH frame, and stays fatal — deliberately, because this cannot
    tell that apart from an SDK-induced one without guessing.

    Never endpoint-owned: anything under the SDK, even if the two roots
    somehow overlap. Returns None whenever it cannot PROVE the frame is the
    author's, so "unsure" always means "fatal".
    """

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
    # THE SUBTRACTION, and it is what makes a wide source root safe (pgw#1533).
    # The root is now the sys.path entry holding the author's modules, so a
    # sibling helper beside the main package is covered — but a root can only
    # ADD author files if everything that is not the author's is removed from
    # it first, and removed by construction rather than by a list:
    #
    #   * the SDK, even if it somehow sits under the same root;
    #   * any installed-dependency tree (site-packages / dist-packages), which
    #     is where torch, diffusers and everything else an endpoint pulls in
    #     actually live.
    #
    # The degenerate case is the safe one: an endpoint pip-INSTALLED into
    # site-packages has a source root that is entirely subtracted, so nothing
    # is claimed and every failure stays fatal. "Unsure" still resolves to
    # fatal, which is the property pgw#1527 was built on.
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
        # pgw#1512: the session asks the derive for EACH component's
        # precision instead of being handed one dtype for the tree. The
        # resolver is total over every tree in this session — the primary's
        # and each secondary slot's (pgw#1508) — because it decides from the
        # tree and subfolder it is given, so there is nothing to swap when an
        # aide loads from its own checkpoint.
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
                # pgw#1599: the trace RAN — the model loaded under the hollow
                # session and the author marked no module, which is an
                # OBSERVATION, not a failure. Under the ruling pair a declared
                # lane no longer means "graphs key here": a lane answers
                # checkpoint compatibility and lane selection, and the
                # `ctx.compile` mark — absent here — is the only compilation
                # statement there is. Zero graphs is the honest answer and the
                # caller prints it as one.
                return None
            modules = _named_marked_modules(model, load_ctx.marked_modules)

            # Secondary model slots (an auxiliary model with its own
            # checkpoint, e.g. h3's RIFE interpolator): a fresh instance per
            # slot, loaded under the SAME hollow session. Every slot named in
            # a signature must be fillable at trace or the release cannot
            # state its graph set.
            #
            # pgw#1508: EACH SLOT GETS ITS OWN TREE. This comment said "an
            # auxiliary model with its own checkpoint" and then handed every
            # aide the PRIMARY's `checkpoint_dir`, so h3's `generate`
            # (video -> minimax-h3, rife -> rife-4.25) tried to build a RIFE
            # interpolator out of the DiT's tree and refused on a missing
            # `flownet`. The binding table has been per-slot since 0.9.0; the
            # derive's world-model now matches it. A slot with no entry falls
            # back to the primary tree, which is every single-slot endpoint
            # and is why their documents do not move.
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
                        # Name the SLOT and the TREE IT WAS GIVEN. The failure
                        # that produced pgw#1508 read as a broken aide class
                        # when it was a wrong checkpoint, and one line saying
                        # which tree was used is the difference between a
                        # one-read diagnosis and an afternoon.
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
                                # The pass observed its shapes; the remaining
                                # denoise steps repeat them.
                                pass
                            except ValidationError:
                                # The author refusing an impossible serving
                                # combination is correct behavior, not a
                                # derive failure.
                                refused += 1
                            except Exception as exc:
                                # pgw#1527: ONE unservable payload costs one
                                # payload, not the document — but only when the
                                # AUTHOR's code is what raised. Anything deeper
                                # in the SDK, torch or diffusers is still fatal:
                                # the derive exists to find those, and walls 1-8
                                # were every one of them.
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
                                    # WHERE the author's code raised, so a
                                    # skipped payload points at the line that
                                    # has to change.
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
                    # A marked module the budgeted drive never reached (it
                    # runs AFTER the denoise loop). Re-drive unbudgeted
                    # before calling it unobserved.
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
    """Derive the release metadata document for one endpoint module.

    ``checkpoint_dir`` is the PRIMARY model's tree. ``slot_checkpoints`` maps a
    secondary model slot to its OWN tree (pgw#1508) -- an auxiliary model is a
    different model with a different checkpoint, which the serving binding
    table has said since 0.9.0 and the derive used to contradict. A slot with
    no entry falls back to the primary tree, so every single-slot endpoint
    derives exactly the bytes it did before.
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

    cls, _ = _lane_model_class(module)
    endpoint_name = f"{module.__name__}:{cls.__name__ if cls else ''}".rstrip(":")

    lanes: list[Any] = []
    unmarked_lanes: list[str] = []
    lane_contracts: dict[str, Any] = {}
    entrypoints: dict[str, Any] = {}
    warnings: list[str] = []
    plans: list[tuple[_Entrypoint, tuple[Any, ...]]] = []
    unenumerable: list[tuple[str, str]] = []
    #: pgw#1527: payloads the ENDPOINT could not serve, one row each.
    unservable_payloads: list[dict[str, Any]] = []
    for plan in _entrypoints(module, cls):
        owner = f"@entrypoint {plan.name}"
        refusal: Optional[PayloadEnumerationRefused] = None
        if cls is None:
            # pgw#1392: a WEIGHTLESS entrypoint has no lane, so there is no
            # trace subject and no pass is ever run. `traced_passes` says 0
            # because 0 is the truth, not because the enumeration was
            # skipped. It still publishes its envelope schema — the hub's
            # auto-generated API docs are the point of this block.
            payloads: tuple[Any, ...] = ()
        else:
            try:
                payloads, capped = _auto_payloads(
                    owner, plan.payload_type, model_structural(cls)
                )
            except PayloadEnumerationRefused as exc:
                # pgw#1449: ONE unenumerable signature used to cost the whole
                # module — `gen-worker lock` died here and wrote NO lock, so
                # the entrypoints the enumerator CAN reach never got written
                # either. The derive is a pre-warming completeness aid, never
                # a correctness gate: an endpoint that derives 2 of 3
                # entrypoints is strictly more useful than one that derives
                # none, and the third is STATED rather than dropped.
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
            # Renders EMPTY for a weightless entrypoint — an honest {}, never
            # a fabricated slot.
            "model_slots": {
                slot_name: slot_cls.__name__
                for slot_name, slot_cls in plan.model_slots
            },
            "traced_passes": len(payloads),
        }
        if refusal is not None:
            # A TYPED row, not a warning string: the hub and the miner read
            # this document, and "this entrypoint has no traced coverage, for
            # this reason" is a fact about the release, not a log line. The
            # key is absent on every enumerable entrypoint, so a document that
            # has no refusals is byte-identical to one derived before this.
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
        # A declared axis that reaches NO entrypoint payload field enumerates
        # nothing and would be a silent no-op — the exact silence the
        # declaration exists to end. Say so; do not refuse (an axis may
        # legitimately name a field only one of several entrypoints carries,
        # and `_auto_payloads` already skips it per entrypoint).
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
        for lane in model_lanes(cls):
            lane_graphs = _derive_lane(
                torchcg, cls, lane, plans, checkpoint_dir, warnings,
                program_sink=program_sink,
                slot_checkpoints=slot_checkpoints,
                endpoint_root=endpoint_source_root(module),
                unservable=unservable_payloads,
                # pgw#1599: read off the MODEL CLASS, never a CLI flag.
                dynamic_dims=dynamic_dim_policy(model_shapes(cls)),
            )
            if lane_graphs is None:
                # Traced, nothing marked (pgw#1488). No lane row: an empty one
                # would claim a keyed graph set that does not exist.
                unmarked_lanes.append(
                    lane_contract_handle(f"class {cls.__name__!r}", lane)
                )
                continue
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
            # ie#740 placement floor for THIS lane, read off the class header
            # (`requires=`). Absent = undeclared, and the platform default is
            # what the deployment gets — the honest state, never an invented
            # floor.
            floor = requires.get(lane_graphs.contract)
            if floor is not None:
                entry["requires"] = floor.render()
            lane_contracts[lane_graphs.contract] = entry

    # pgw#1527: a skipped payload is STATED, per entrypoint, and is absent
    # entirely when nothing was skipped — so a document with no unservable
    # payload is byte-identical to one derived before this existed.
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
        "graphs": graphs_document.as_dict(),
        "lane_contracts": lane_contracts,
        # The per-entrypoint request envelope, DERIVED FROM THE SIGNATURE
        # (parameter name = slot name). The hub publishes these as the
        # release's auto-generated API docs; a renamed parameter is a
        # visible API break, flagged like a disappearing lane.
        "entrypoints": entrypoints,
        # pgw#1599: the author's declared FORK AXES travel in the document,
        # so the mint scheduler and the hub read the CLOSED key set instead of
        # inferring it. `structural` names the axes that fork the PROGRAM (and
        # what measurement said so); `shapes` names the per-axis static/dynamic
        # choice that decides whether a lane mints N bucket artifacts or one
        # symbolic one. Both empty for a weightless module.
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
        # No lane class AND entrypoints still derived => every one of them
        # declared zero model slots. An eager-permanent module also has no
        # lane class, but its entrypoints carry slots and derive no plan.
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
