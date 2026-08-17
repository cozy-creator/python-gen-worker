"""The ONE decorator walker: ``@endpoint`` objects -> EndpointSpec table.

Shared by the worker runtime (dispatch), build-time discovery (endpoint.lock),
and the local CLI. Signature inspection lives here and nowhere else.
"""

from __future__ import annotations

import collections.abc as cabc
import inspect
import logging
import types as py_types
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import msgspec

from .api.binding import Binding, ModelRef
from .api.compile_axis import PayloadAxis, extract_payload_axes
from .api.decorators import (
    ATTR, VARIANT_ATTR, WF_ATTR, Compile, ConfigParam, DynamicDim, EndpointDecl,
    Resources, VariantDecl, WorkerFunctionDecl,
)
from .api.jobs import JOB_ATTR, JobDecl
from .api.formula import RuntimeFormula
from .api.slot import Slot
from .api.tree import derive_components, validate_no_sibling_parts
from .discovery.names import slugify_name
from .discovery.walk import find_endpoints, find_jobs
from .families.base import GenerationDefaults
from .models.tensor_layout_contract import LAYOUT_KEY_ANY_COMPONENT
from .warmup import validate_class_warmup
import dataclasses
from .api.compile_axis import warm_guidance_values
from .graph_facts import facts_digest
from .serving_facts import SlotEvidence
from .api.export_contract import (
    declares_export_contract, register_export_declaration, registered_entry,
)

logger = logging.getLogger(__name__)

_ITER_ORIGINS = (
    typing.Iterator, typing.Iterable, typing.AsyncIterator, typing.AsyncIterable,
    cabc.Iterator, cabc.Iterable, cabc.AsyncIterator, cabc.AsyncIterable,
)


def _is_struct(t: Any) -> bool:
    try:
        return isinstance(t, type) and issubclass(t, msgspec.Struct)
    except Exception:
        return False


@dataclass(frozen=True)
class CompileCell:
    """The complete compile-cell configuration one endpoint function
    declares: the ``Compile`` block enriched with the spec-level
    facts that shape the traced graph family — the decorator-level
    ``lora_bucket`` and the warm guidance representatives derived from the
    payload's ``CompileAxis`` classes. This is the object the compile
    machinery (compile_cache / fleet_cells / local_cells / aot_serve)
    consumes; raw ``Compile`` never travels past the registry."""

    shapes: Tuple[Tuple[int, ...], ...]
    targets: Tuple[str, ...]
    family: str
    regional: bool
    # This FUNCTION's effective text pin (per-lane: a
    # @worker_function(text_len=) override wins over the class Compile's).
    text_len: Optional[int]
    dynamic: Tuple[DynamicDim, ...]
    lora_bucket: int
    # CLASS-scoped unions: every sibling function on one class shares one cell
    # family, so the contract facts digest the UNION across the class's
    # functions — never one function's own view (per-function digests split the
    # cell: turbo fails closed on w8a8 lanes). Union is per CLASS only; sibling
    # @endpoint classes keep their own contracts (two checkpoints = two
    # instances = two cells).
    guidance_scales: Tuple[float, ...]
    text_lens: Tuple[int, ...] = ()
    # The family's DECLARED numerics band, carried through to the gate — it is
    # read from this object, so a band that stops here is a band nothing
    # applies. Deliberately NOT in `contract_facts()` below: a numerics band is
    # not a graph axis and must never move a cell key.
    numerics_floor: Optional[float] = None
    numerics_warn: Optional[float] = None

    @classmethod
    def from_declaration(
        cls, cfg: Any, *, lora_bucket: int = 0, text_len: Optional[int] = None,
        guidance_scales: Tuple[float, ...] = (), text_lens: Tuple[int, ...] = (),
    ) -> "CompileCell":
        """THE ``Compile`` -> ``CompileCell`` map, in one place.

        Two call sites build this object — the registry's per-spec
        :meth:`EndpointSpec.compile_cell` and the local CLI's desktop arm — and
        they differ only in the enrichments above, which are spec-scoped facts
        the raw declaration cannot know. Everything else is a straight carry, so
        a field ADDED to ``Compile`` that one site copied and the other forgot
        is a whole path silently judged at a default nobody chose. One
        constructor is the fence against that.
        """
        return cls(
            shapes=tuple(cfg.shapes),
            targets=tuple(cfg.targets),
            family=str(cfg.family or ""),
            regional=bool(cfg.regional),
            text_len=text_len if text_len is not None else cfg.text_len,
            dynamic=tuple(cfg.dynamic),
            lora_bucket=int(lora_bucket or 0),
            guidance_scales=guidance_scales,
            text_lens=text_lens,
            numerics_floor=cfg.numerics_floor,
            numerics_warn=cfg.numerics_warn,
        )

    def contract_text_lens(self) -> Tuple[int, ...]:
        if self.text_lens:
            return tuple(sorted({int(v) for v in self.text_lens}))
        return (int(self.text_len),) if self.text_len is not None else ()

    def contract_facts(self) -> Dict[str, Any]:
        """Canonical DECLARED compile-contract facts. NOT a key-axis input —
        the exported-cell key reads recorded artifact blocks only. Its one
        serialized consumer is the manifest's opaque
        ``shape_contract_digest``."""
        return {
            "v": 1,
            "shapes": sorted([int(v) for v in row] for row in self.shapes),
            "targets": [str(t) for t in self.targets],
            "text_lens": [int(v) for v in self.contract_text_lens()],
            "dynamic": [
                {"dim": d.dim, "min": d.min, "max": d.max} for d in self.dynamic
            ],
            "regional": bool(self.regional),
            "lora_bucket": int(self.lora_bucket or 0),
            "guidance": sorted(float(v) for v in self.guidance_scales),
        }

    def contract_digest(self) -> str:
        return facts_digest(self.contract_facts())


@dataclass(frozen=True)
class EndpointSpec:
    """One externally-routable function: a handler plus wire shape.

    ``cls`` is None for function-shaped (stateless) endpoints — the worker
    calls ``method`` directly with no instance/setup.
    """

    name: str                     # routable function name (canonical slug)
    method: Callable[..., Any]    # unbound function object
    kind: str                     # inference | training | dataset | conversion
    payload_type: type            # msgspec.Struct
    output_mode: str              # "single" | "stream"
    cls: Optional[type] = None
    attr_name: str = ""           # python attribute name on the class
    output_type: Optional[type] = None   # struct returned (single mode)
    delta_type: Optional[type] = None    # struct yielded (stream mode; None if union)
    is_async: bool = False        # coroutine or async generator
    is_async_gen: bool = False
    ctx_param: str = "ctx"
    payload_param: str = "payload"
    resources: Resources = field(default_factory=Resources)
    models: Dict[str, Binding] = field(default_factory=dict)  # slot -> binding
    # Slot-declared entries in `models`: slot -> Slot metadata
    # (selected_by/default_checkpoint). A subset of `models`'s keys — bare
    # bindings have no entry here.
    slots: Dict[str, Slot[Any]] = field(default_factory=dict)
    # Resolved family name per Slot (the endpoint's Compile(family=...), or
    # the handler's derived config schema's @family registration) —
    # precomputed once here so ctx.slots doesn't need EndpointDecl.compile.
    slot_family: Dict[str, str] = field(default_factory=dict)
    # pgw#1333: per-slot SERVING FACTS — what the catalog says the resolved
    # checkpoint IS. Never authored and never discovered: `_dispatched_spec`
    # folds them in from the neutral orders exactly as it folds in `models`,
    # and `child_preflight.bind_slots` reinstalls them in a child that re-ran
    # discovery. A slot absent from this map has no EVIDENCE, which is not the
    # same as a checkpoint the catalog classified as nothing — the two states
    # were one state before this field, and every governed mint refused.
    slot_facts: Dict[str, SlotEvidence] = field(default_factory=dict)
    # pgw#1332: handler parameter -> the generated family TYPE bound to it.
    # A handler parameter of this name receives a fully RESOLVED instance
    # (graph + bound weights + catalog-stamped tuned values), so two parameters
    # of one family type are two checkpoints with independent tuning. Declared
    # statically so placement can prefetch the weights and verify the VRAM fit
    # before a request lands.
    #
    # Beside `slot_facts` and NOT folded into it, deliberately: a slot fact is
    # what the catalog says a resolved checkpoint IS, carried per REQUEST; a
    # family binding is what a handler PARAMETER is, declared once and read by
    # placement. Same struct, different axes.
    families: Dict[str, Any] = field(default_factory=dict)
    # The handler's DERIVED config schema — the D in `ctx: RequestContext[D]`.
    # None when the handler annotates a bare context. Catalog recipe metadata
    # decodes against this type; code owns the schema, the catalog owns the
    # values.
    defaults_type: Optional[type] = None
    # DERIVED component tree per introspectable pipeline slot —
    # {slot: {part_name: "weights"|"config"}}. Published into the release
    # manifest at build time (path vocabulary for per-path policy and
    # component-level routing).
    slot_components: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # Compile-graph equivalence classes declared on payload fields
    # (CompileAxis annotations).
    payload_axes: Tuple[PayloadAxis, ...] = ()
    # Resident traced LoRA rank bucket (decorator-level; 0 = branchless).
    lora_bucket: int = 0
    runtime: Optional[str] = None
    compile: Optional[Compile] = None  # opt-in torch.compile spec
    # The function makes endpoint-to-endpoint child calls; emitted into the
    # discovery manifest so the hub mints the invoke_child grant.
    child_calls: bool = False
    # Handlers on one live instance run single-flight unless the class
    # explicitly declared itself re-entrant (mutates no instance state).
    reentrant: bool = False
    # @variant_of: this function is the variant_kind variant of the sibling
    # function variant_of (both slugs). Empty = not a variant.
    variant_of: str = ""
    variant_kind: str = ""
    # This handler's declared objective contract (from @worker_function).
    # None = unrestricted / serves either.
    objectives: Optional[Tuple[str, ...]] = None
    # This handler's declared serving tasks (@worker_function).
    # None = undeclared, which resolves NO quant approval hub-side.
    tasks: Optional[Tuple[str, ...]] = None
    distilled: Optional[bool] = None
    # This handler's opt-in reference contract (@worker_function).
    # None = it never participates in the platform reference layer.
    accepts_references: Optional[Any] = None
    # This handler's effective text pin
    # (@worker_function(text_len=) else the class Compile.text_len).
    text_len: Optional[int] = None
    # Per-function non-axis warm overrides
    # (validated at walk time; require a recorded warm_reason).
    warm_overrides: Dict[str, Any] = field(default_factory=dict)
    warm_reason: str = ""
    # Which EXECUTION GROUP this dispatch runs on. Never authored and never
    # discovered — the executor derives it from the delivered topology and the
    # job's rank-0 device, then clones the spec so the group rides the identity
    # every downstream consumer already threads. Two groups are two cards, so
    # they are two resident instances; folding the ordinal into
    # ``instance_key`` is what makes the whole per-group registry fall out of
    # machinery that already keys one record per (class, resolved binding set).
    # Omitted from the key at 0 so every single-group pod's keys stay
    # byte-identical.
    device_group_ordinal: int = 0
    # CLASS-scoped unions (set by extract_specs over the class's sibling specs;
    # a function-shaped endpoint unions with itself).
    guidance_union: Tuple[float, ...] = ()
    text_lens: Tuple[int, ...] = ()
    # Declared lane bodies this endpoint's code branches on (ctx.lane).
    # Empty = platform-managed only.
    handles: Tuple[str, ...] = ()
    # Declared compute-time formula for this handler; None = undeclared
    # (scalar EWMA fall-through hub-side).
    runtime_formula: Optional["RuntimeFormula"] = None
    # Declared config parameters (ctx.config) + declared env names.
    config: Tuple[ConfigParam, ...] = ()
    env: Tuple[str, ...] = ()
    # The hub-write declaration (th#2049/pgw#1294). Rides the manifest row; the
    # hub mints the write grant off it, and the SDK refuses the publisher
    # surface without it.
    publishes: bool = False
    module: str = ""              # declaring module
    walked_module: str = ""       # top-level package the object was found under

    @property
    def needs_gpu(self) -> bool:
        return bool(self.resources.gpu)

    def compile_cell(self) -> Optional[CompileCell]:
        """The enriched compile-cell configuration, or None for uncompiled
        functions. This — never the raw ``Compile`` — is what the
        executor hands the compile machinery."""
        cfg = self.compile
        if cfg is None:
            return None

        return CompileCell.from_declaration(
            cfg,
            lora_bucket=int(self.lora_bucket or 0),
            text_len=self.text_len,
            guidance_scales=(
                self.guidance_union
                if self.guidance_union
                else warm_guidance_values(self.payload_axes)
            ),
            text_lens=self.text_lens,
        )

    @property
    def instance_key(self) -> Any:
        """Specs sharing this key share one class instance (same class + same
        resolved binding set + same execution group). The group ordinal is
        elided at 0, so a single-group worker's keys are byte-identical to
        every key this worker has ever computed."""
        base = (self.cls, tuple(sorted(self.models.items())))
        return base if not self.device_group_ordinal else (
            base + (("device_group", int(self.device_group_ordinal)),)
        )


@dataclass(frozen=True)
class JobSpec:
    """One submittable ``@job``: a run-once function plus its wire shape.

    Deliberately the same SHAPE as :class:`EndpointSpec` where the two overlap
    (``name``/``method``/``payload_type``/``output_type``/``resources``/
    ``env``/``publishes``), because portability between the decorators is a
    tested requirement, not a coincidence. What a job does NOT carry is as
    load-bearing as what it does: no ``cls`` and no setup (the run-once
    lifecycle forks a fresh child per job), no execution lanes, no compile
    cell, no slots or bindings — a job that wants a hub-resolved model NAMES
    it in its payload and fetches through the snapshot machinery.
    """

    name: str                     # submittable job name (canonical slug)
    method: Callable[..., Any]
    payload_type: type            # msgspec.Struct
    output_type: type             # msgspec.Struct
    resources: Resources = field(default_factory=Resources)
    env: Tuple[str, ...] = ()
    resumable: bool = False
    visibility: str = "private"
    publishes: bool = False
    emits_media: bool = False
    # pgw#1332: the families this job binds, same shape and same meaning as
    # EndpointSpec's — portability between the decorators is a tested
    # requirement, so a body that takes a family instance promotes unchanged.
    families: Dict[str, Any] = field(default_factory=dict)
    is_async: bool = False
    ctx_param: str = "ctx"
    payload_param: str = "payload"
    python_name: str = ""
    module: str = ""
    walked_module: str = ""

    @property
    def needs_gpu(self) -> bool:
        return bool(self.resources.gpu)


def extract_job_spec(obj: Any, *, walked_module: str = "") -> Optional[JobSpec]:
    """The JobSpec one ``@job``-decorated function declares, or None."""
    decl: Optional[JobDecl] = getattr(obj, JOB_ATTR, None)
    if decl is None:
        return None
    hints = typing.get_type_hints(obj, include_extras=False)
    params = list(inspect.signature(obj).parameters.values())
    slug = slugify_name(decl.name)
    if not slug:
        raise ValueError(f"@job name {decl.name!r} cannot be normalized")
    return JobSpec(
        name=slug,
        method=obj,
        payload_type=hints[params[1].name],
        output_type=hints["return"],
        resources=decl.resources,
        env=tuple(decl.env),
        resumable=bool(decl.resumable),
        visibility=decl.visibility,
        publishes=bool(decl.publishes),
        emits_media=bool(decl.emits_media),
        families=dict(getattr(decl, "families", {}) or {}),
        is_async=inspect.iscoroutinefunction(obj),
        ctx_param=params[0].name,
        payload_param=params[1].name,
        python_name=obj.__name__,
        module=getattr(obj, "__module__", "") or "",
        walked_module=walked_module or (getattr(obj, "__module__", "") or ""),
    )


def collect_jobs(module_names: List[str]) -> List[JobSpec]:
    """Walk top-level modules (+ submodules) and return every JobSpec."""
    out: List[JobSpec] = []
    for found in find_jobs(list(module_names)):
        spec = extract_job_spec(found.obj, walked_module=found.walked_module)
        if spec is not None:
            out.append(spec)
    _assert_unique_job_names(out)
    return out


def _assert_unique_job_names(specs: List[JobSpec]) -> None:
    seen: Dict[str, JobSpec] = {}
    for s in specs:
        prior = seen.get(s.name)
        if prior is not None and prior.method is not s.method:
            raise ValueError(
                f"duplicate job name {s.name!r} "
                f"({prior.module} and {s.module})"
            )
        seen[s.name] = s


def _derive_defaults_type(
    owner: str, method: Callable[..., Any], ctx_param: str
) -> Optional[type]:
    """The D in ``ctx: RequestContext[D]`` — the handler's derived per-model
    config schema (``GenerationDefaults`` subclass). ``None`` for a
    bare/unannotated context parameter. The annotation is the ONLY declaration
    site; the catalog owns values, code owns this schema."""
    try:
        hints = typing.get_type_hints(method, include_extras=False)
    except Exception:
        return None
    ann = hints.get(ctx_param)
    if ann is None:
        return None
    origin = typing.get_origin(ann)
    if origin is None:
        return None
    args = typing.get_args(ann)
    if not args or len(args) != 1:
        return None
    candidate = args[0]
    if isinstance(candidate, typing.TypeVar):
        return None

    if not (isinstance(candidate, type) and issubclass(candidate, GenerationDefaults)):
        raise ValueError(
            f"{owner}: context annotation {ann!r} parameterizes "
            f"RequestContext with {candidate!r}, which is not a "
            "GenerationDefaults subclass — the type argument is the "
            "handler's per-model config schema"
        )
    return candidate


def _inspect_return(owner: str, ret: Any) -> tuple[str, Optional[type], Optional[type]]:
    """-> (output_mode, output_type, delta_type)."""
    if _is_struct(ret):
        return "single", ret, None
    origin = typing.get_origin(ret)
    if origin in _ITER_ORIGINS:
        args = typing.get_args(ret) or ()
        item = args[0] if args else None
        if _is_struct(item):
            return "stream", item, item
        # Union of delta structs: still a stream.
        if typing.get_origin(item) in (typing.Union, py_types.UnionType):
            return "stream", None, None
        raise ValueError(
            f"{owner}: streaming return must be "
            f"(Async)Iterator[msgspec.Struct], got {ret!r}"
        )
    raise ValueError(
        f"{owner}: return type must be msgspec.Struct or "
        f"(Async)Iterator[msgspec.Struct], got {ret!r}"
    )


def _is_selected_by_annotation(ann: Any) -> bool:
    """A ``selected_by`` payload field must type as plain
    ``str``, or as ``str | ModelRef`` — the wire also accepts a structured
    :class:`~gen_worker.api.binding.ModelRef` object (a client-supplied
    BYOM pick), which the hub resolves to a concrete ref before the worker
    ever sees the request; the SDK never bakes the curated-value enum into
    either shape."""
    if ann is str:
        return True
    origin = typing.get_origin(ann)
    if origin in (typing.Union, py_types.UnionType):
        return frozenset(typing.get_args(ann)) == frozenset((str, ModelRef))
    return False


def _validate_slot_selected_by(
    owner: str, slots: Dict[str, Slot[Any]], payload_type: type
) -> None:
    """A Slot's ``selected_by`` must name a plain-``str`` (or
    ``str | ModelRef``) field on THIS handler's payload — validated
    per-handler (not per-class) because one ``models=`` decl can be shared by
    methods with different payload types. ``selected_by`` slots may omit
    ``default_checkpoint`` (deploy-time bindings seed the hub mapping). Fails
    at spec-construction time (discovery walk / CLI collection / executor
    boot), never at first invoke."""
    if not slots:
        return
    try:
        hints = typing.get_type_hints(payload_type, include_extras=False)
    except Exception:
        hints = getattr(payload_type, "__annotations__", {}) or {}
    for slot_name, slot in slots.items():
        if not slot.selected_by:
            continue
        if slot.selected_by not in hints:
            raise ValueError(
                f"{owner}: Slot({slot_name!r}).selected_by={slot.selected_by!r} "
                f"names no field on payload {payload_type.__name__!r}"
            )
        ann = hints[slot.selected_by]
        if not _is_selected_by_annotation(ann):
            raise ValueError(
                f"{owner}: Slot({slot_name!r}).selected_by="
                f"{slot.selected_by!r} must be a plain str field (or "
                f"str | ModelRef) on {payload_type.__name__!r} (got {ann!r}); "
                "the hub overlays the live allowed-value enum onto this "
                "field, it is never baked into the SDK type"
            )


def _validate_compile_arms(
    owner: str,
    cls: Optional[type],
    compile: Optional[Compile],
    models: Dict[str, Binding],
) -> None:
    """``compile=`` only ever arms automatically on a setup() slot
    the WORKER loads itself (a pipeline-class annotation exposing
    ``from_pretrained`` — mirrors the executor's annotation branch in
    ``_injection_kwargs``). An endpoint whose setup() model slots are ALL
    self-loading (str/Path/other non-pipeline annotations — the endpoint
    builds its own pipeline) never reaches that arming path: the declared
    ``compile=`` seeds the manifest/shape contract but never runs, silently.
    Fail loudly at discovery time unless the endpoint opts into the
    ``arm_compile()`` seam itself (best-effort source scan — a missing
    source is never a build blocker, only a missed opt-in detection)."""
    if compile is None or cls is None or not models:
        return
    setup = getattr(cls, "setup", None)
    if not callable(setup):
        return
    try:
        hints = typing.get_type_hints(setup)
    except Exception:
        return
    worker_loaded = any(
        isinstance(ann, type) and callable(getattr(ann, "from_pretrained", None))
        for ann in (hints.get(name) for name in models)
    )
    if worker_loaded:
        return
    try:
        source = inspect.getsource(setup)
    except (OSError, TypeError):
        return  # can't prove either way; don't block the build on it
    if "arm_compile" in source:
        return
    raise ValueError(
        f"{owner}: compile=Compile(...) is declared but no setup() model "
        "slot is annotated with a pipeline class (all slots are self-loading "
        "str/Path annotations) -- the worker only arms compile on slots it "
        "loads itself (a class annotation exposing from_pretrained), so "
        "this compile= block seeds the manifest but never runs at request "
        "time. Fix one of: (1) annotate the slot with the pipeline class "
        "(e.g. `pipeline: WanPipeline` instead of `pipeline: str`) so the "
        "worker loads it and arms compile automatically, or (2) keep the "
        "self-loading slot and call gen_worker.arm_compile(pipe) yourself "
        "at the end of setup(), after placement -- same cache-artifact-"
        "gated policy, eager otherwise."
    )


def _validate_slot_layout_keys(
    owner: str,
    slots: Dict[str, Slot[Any]],
    slot_components: Dict[str, Dict[str, str]],
) -> None:
    """Every `Slot(layouts=...)` key is `"*"` or a real component path.

    §1.33 pins the DEMAND per (slot, component path), and the path vocabulary
    is the DERIVED tree the manifest already publishes — so a key that matches
    no component is not a stricter declaration, it is a silently absent one.
    The handles themselves were validated at the Slot constructor; only the
    keys need the tree, and the tree only exists here.
    """
    for name, slot in slots.items():
        declared = getattr(slot, "layouts", None)
        if not declared:
            continue
        tree = slot_components.get(name) or {}
        for key in declared:
            if key == LAYOUT_KEY_ANY_COMPONENT:
                continue
            if key in tree:
                continue
            if not tree:
                raise ValueError(
                    f"{owner}: slot {name!r} declares layouts[{key!r}] but "
                    f"{slot.pipeline_cls.__name__} is not introspectable, so "
                    "no component tree is derived — a self-loading slot can "
                    f"only declare {LAYOUT_KEY_ANY_COMPONENT!r}."
                )
            raise ValueError(
                f"{owner}: slot {name!r} declares layouts[{key!r}], which is "
                f"not a component of {slot.pipeline_cls.__name__}. Its derived "
                f"tree is: {', '.join(sorted(tree))} (or "
                f"{LAYOUT_KEY_ANY_COMPONENT!r} for the whole tree)."
            )


def _slot_is_family_agnostic(
    name: str, slot: Slot[Any], slots: Dict[str, Slot[Any]],
) -> bool:
    """Has this slot opted into the family vocabulary at all?

    An AUXILIARY slot declared as a bare type — ``Slot(RifeInterpolatorPipeline)``
    with no ``Hub(...)``, no ref — says nothing about architecture. A frame
    interpolator and an upscaler consume decoded RGB frames and know nothing
    about the model that produced them; ONE mirror is meant to serve every
    consumer. Stamping such a slot with the FUNCTION's family makes an
    assertion the endpoint never made, and the hub's compatibility gate then
    fails closed against it because a family-agnostic artifact classifies as
    nothing. No catalog-side stamp fixes it honestly: compatibility compares
    architecture ROOTS, so stamping `rife-4.25` as `wan22` would be FALSE and
    would break the moment ltx shares the same mirror, which is the design.
    The gate no-ops on an empty slot family, so emitting "" is the fix.

    TWO conditions, not one. "No ref" alone would also empty the family of a
    deliberately DEFAULTLESS root slot — a real model slot bound at deploy time
    through `?bindings=` — silently dropping the LoRA-overlay policing. The
    root slot is the function's own model and keeps the function's family
    whether or not it names a ref; only a non-root, ref-less slot is
    family-agnostic.
    """
    if slot.default_checkpoint is not None:
        return False
    if getattr(slot, "root", False):
        return False
    if len(slots) == 1:
        return False  # a single slot IS the root (executor._spec_root_slot)
    if name == "pipeline" and not any(
            getattr(s, "root", False) for s in slots.values()):
        return False  # the implicit root
    return True


def _spec_for_handler(
    *,
    fn_name: str,
    method: Callable[..., Any],
    decl: EndpointDecl,
    cls: Optional[type],
    attr_name: str,
    models: Dict[str, Binding],
    slots: Dict[str, Slot[Any]],
    resources: Resources,
    walked_module: str,
) -> EndpointSpec:
    owner = f"{cls.__name__}.{attr_name}" if cls is not None else method.__name__
    hints = typing.get_type_hints(method, include_extras=False)
    sig = inspect.signature(method)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    if len(params) < 2:
        raise ValueError(
            f"{owner}: must accept (ctx, payload); got params "
            f"{[p.name for p in params]}"
        )
    ctx_param, payload_param = params[0].name, params[1].name

    payload_type = hints.get(payload_param)
    if not isinstance(payload_type, type) or not _is_struct(payload_type):
        raise ValueError(
            f"{owner}: payload param {payload_param!r} must be annotated "
            f"with a msgspec.Struct (got {payload_type!r})"
        )
    _validate_slot_selected_by(owner, slots, payload_type)
    _validate_compile_arms(owner, cls, decl.compile, models)
    # Sibling-as-part declarations are refused, not migrated.
    validate_no_sibling_parts(owner, slots, models)
    # The config schema derives from the handler's context annotation
    # (`ctx: RequestContext[D]`), never from a Slot kwarg.
    defaults_type = _derive_defaults_type(owner, method, ctx_param)
    defaults_family = str(
        getattr(defaults_type, "__gen_worker_family__", "") or ""
    ) if defaults_type is not None else ""
    compile_family = decl.compile.family if decl.compile is not None else ""
    # Compile(family=...) is the explicit, functionally-load-bearing
    # declaration (compile-cache keying) — it wins over the derived config
    # schema's own @family registration when both are present.
    function_family = compile_family or defaults_family
    slot_family = {
        name: (
            slot.family
            if slot.family is not None
            else ("" if _slot_is_family_agnostic(name, slot, slots)
                  else function_family)
        )
        for name, slot in slots.items()
    }
    # Derive each introspectable pipeline slot's component tree.
    slot_components = {
        name: tree for name, tree in (
            (name, derive_components(slot.pipeline_cls))
            for name, slot in slots.items()
        ) if tree
    }
    # The DEMAND's component keys are checked against the tree just derived. A
    # key matching nothing is the failure mode to prevent — it reads as a
    # declaration and gates nothing.
    _validate_slot_layout_keys(owner, slots, slot_components)
    payload_axes = extract_payload_axes(owner, payload_type)
    ret = hints.get("return")
    if ret is None:
        raise ValueError(f"{owner}: missing return type annotation")
    output_mode, output_type, delta_type = _inspect_return(owner, ret)

    # The wire/dispatch name is the SLUG (matches the discovery manifest and
    # tensorhub's canonical function names — `_` -> `-`): the orchestrator's
    # RunJob.function_name and the worker's advertised available_functions
    # must agree, and the platform normalizes to slugs everywhere.
    slug = slugify_name(fn_name)
    if not slug:
        raise ValueError(f"{owner}: function name {fn_name!r} cannot be normalized")

    variant: Optional[VariantDecl] = getattr(method, VARIANT_ATTR, None)
    variant_of_slug = ""
    variant_kind = ""
    if variant is not None:
        variant_of_slug = slugify_name(variant.of)
        variant_kind = slugify_name(variant.kind)
        if not variant_of_slug or not variant_kind:
            raise ValueError(
                f"{owner}: @variant_of({variant.of!r}, kind={variant.kind!r}) "
                "cannot be normalized to slugs"
            )
        if variant_of_slug == slug:
            raise ValueError(f"{owner}: @variant_of cannot target itself")

    # Resolve + validate this handler's declared compute-time formula now that
    # the payload type is known. Fields may resolve against the derived
    # defaults schema (catalog recipe), not only numeric wire defaults.
    runtime_formula = decl.runtime_formula.get(attr_name)
    if runtime_formula is not None:
        runtime_formula.validate_for_payload(
            payload_type, owner, defaults_type=defaults_type)

    # Per-function facts declared at the definition site.
    wf: Optional[WorkerFunctionDecl] = getattr(method, WF_ATTR, None)
    warm_overrides: Dict[str, Any] = {}
    if wf is not None and wf.warm:
        try:
            payload_fields = {f.name for f in msgspec.structs.fields(payload_type)}
        except Exception:
            payload_fields = set()
        axis_fields = {a.field for a in payload_axes}
        for wf_field in wf.warm:
            if wf_field not in payload_fields:
                raise ValueError(
                    f"{owner}: @worker_function warm= names {wf_field!r}, "
                    f"which is not a field of {payload_type.__name__!r}"
                )
            if wf_field in axis_fields:
                raise ValueError(
                    f"{owner}: @worker_function warm= overrides AXIS field "
                    f"{wf_field!r} — axis classes already carry their warm= "
                    "representatives; per-function warm overrides are for "
                    "non-axis fields that change tracing only"
                )
        warm_overrides = dict(wf.warm)
    wf_text_len = wf.text_len if wf is not None else None
    if wf_text_len is not None and decl.compile is None:
        raise ValueError(
            f"{owner}: @worker_function(text_len=...) requires the class to "
            "declare compile= (the pin is a compile shape-contract fact)"
        )

    return EndpointSpec(
        name=slug,
        method=method,
        kind=decl.kind,
        payload_type=payload_type,
        output_mode=output_mode,
        cls=cls,
        attr_name=attr_name,
        output_type=output_type,
        delta_type=delta_type,
        is_async=bool(
            inspect.iscoroutinefunction(method) or inspect.isasyncgenfunction(method)
        ),
        is_async_gen=inspect.isasyncgenfunction(method),
        ctx_param=ctx_param,
        payload_param=payload_param,
        resources=resources,
        models=dict(models),
        slots=dict(slots),
        slot_family=slot_family,
        families=dict(getattr(decl, "families", {}) or {}),
        defaults_type=defaults_type,
        slot_components=slot_components,
        payload_axes=payload_axes,
        lora_bucket=int(getattr(decl, "lora_bucket", 0) or 0),
        runtime=decl.runtime,
        compile=decl.compile,
        child_calls=decl.child_calls,
        reentrant=decl.reentrant,
        variant_of=variant_of_slug,
        variant_kind=variant_kind,
        objectives=(tuple(wf.objectives) if wf is not None and wf.objectives is not None else None),
        tasks=(tuple(wf.tasks) if wf is not None and wf.tasks is not None else None),
        distilled=(wf.distilled if wf is not None else None),
        accepts_references=(wf.accepts_references if wf is not None else None),
        text_len=(wf_text_len if wf_text_len is not None
                  else (decl.compile.text_len if decl.compile is not None else None)),
        warm_overrides=warm_overrides,
        warm_reason=(wf.warm_reason if wf is not None else ""),
        handles=tuple(decl.handles),
        runtime_formula=runtime_formula,
        config=tuple(decl.config),
        env=tuple(decl.env),
        publishes=bool(decl.publishes),
        module=getattr(cls or method, "__module__", "") or "",
        walked_module=walked_module,
    )


def extract_specs(obj: Any, *, walked_module: str = "") -> List[EndpointSpec]:
    """All EndpointSpecs declared by one decorated object (one per handler)."""
    decl: Optional[EndpointDecl] = getattr(obj, ATTR, None)
    if decl is None:
        return []
    walked = walked_module or (getattr(obj, "__module__", "") or "")

    if decl.is_function:
        return _apply_class_unions([_spec_for_handler(
            fn_name=decl.name or obj.__name__,
            method=obj,
            decl=decl,
            cls=None,
            attr_name="",
            models=dict(decl.models),
            slots=dict(decl.slots),
            resources=decl.resources,
            walked_module=walked,
        )])

    cls = obj
    handlers: list[tuple[str, Callable[..., Any]]] = list(
        getattr(cls, "__gen_worker_handlers__", []) or []
    )
    out: List[EndpointSpec] = []

    for attr_name, method in handlers:
        out.append(_spec_for_handler(
            fn_name=attr_name, method=method, decl=decl, cls=cls,
            attr_name=attr_name, models=dict(decl.models),
            slots=dict(decl.slots),
            resources=decl.resources, walked_module=walked,
        ))
    out = _apply_class_unions(out)
    # Boot warmup is default-on for GPU inference classes — fail at walk time
    # (discovery/CI/boot), never at first request.

    validate_class_warmup(cls, decl, out)
    return out


def _apply_class_unions(specs: List[EndpointSpec]) -> List[EndpointSpec]:
    """Sibling functions of ONE class share one cell family, so the class's
    compile-contract warm facts are the UNION across its functions — the
    guidance warm set (a distilled sibling with no guidance field contributes
    nothing yet consumes the same cell) and the per-lane text pins (qwen t2i
    512 / edit 1024 digest as one dual-pin contract). Scope is exactly one
    class — sibling @endpoint
    CLASSES keep divergent contracts by design (ernie's base/turbo are two
    checkpoints, two instances, two cells). Compile-less classes pass
    through untouched."""

    compiled = [s for s in specs if s.compile is not None]
    if not compiled:
        return specs
    guidance_union = tuple(sorted({
        float(v)
        for s in compiled
        for v in warm_guidance_values(s.payload_axes)
    }))
    text_lens = tuple(sorted({
        int(s.text_len) for s in compiled if s.text_len is not None
    }))
    out: List[EndpointSpec] = []
    for s in specs:
        if s.compile is None:
            out.append(s)
        else:
            out.append(dataclasses.replace(
                s, guidance_union=guidance_union, text_lens=text_lens))
    return out


def register_declared_exports(specs: Sequence[EndpointSpec]) -> Tuple[str, ...]:
    """A ``compile=`` block that declares an EXPORT CONTRACT *is* its family's
    export declaration — register it at collection time.

    Without this, ``export_declaration(family)`` resolves only when somebody
    imports a separate declaration module by name, which is a MINT-REQUEST
    concept (``aot_declaration.load_declaration``). A serving pod loads the
    endpoint and nothing else, so `aot_mint.mint` would refuse every family on
    "no registered export declaration" before exporting a single graph. The
    declaration travels with the endpoint that owns it.

    Never raises: a conflicting registration is a real defect, but it is the
    AOT lane's defect and must not take down endpoint collection (which every
    boot, every CLI walk and every discovery pass runs).
    """

    registered: List[str] = []
    seen: set[int] = set()
    for spec in specs:
        compile_decl = spec.compile
        if compile_decl is None or id(compile_decl) in seen:
            continue
        seen.add(id(compile_decl))
        family = str(getattr(compile_decl, "family", "") or "").strip()
        # The same INTENT question the decorator asks — a dynamo-only
        # `compile=` block declares no export contract and is registered
        # nowhere; anything reaching for the export vocabulary is held to
        # carrying classes by `register_export_declaration`.
        if not family or not declares_export_contract(compile_decl):
            continue
        if registered_entry(family) is compile_decl:
            continue
        try:
            register_export_declaration(compile_decl)
        except Exception as exc:  # noqa: BLE001 — never fail collection
            logger.warning(
                "export declaration for family %r not registered from its "
                "endpoint: %s", family, exc)
            continue
        registered.append(family)
    return tuple(registered)


def collect_endpoints(module_names: List[str]) -> List[EndpointSpec]:
    """Walk top-level modules (+ submodules) and return every EndpointSpec."""
    out: List[EndpointSpec] = []
    for found in find_endpoints(list(module_names)):
        out.extend(extract_specs(found.obj, walked_module=found.walked_module))
    _assert_unique_names(out)
    register_declared_exports(out)
    return out


def collect_from_namespace(module: Any) -> List[EndpointSpec]:
    """Extract EndpointSpecs from a single already-imported module's namespace."""
    out: List[EndpointSpec] = []
    seen: set[int] = set()
    for obj in list(vars(module).values()):
        if not (inspect.isclass(obj) or inspect.isfunction(obj)) or id(obj) in seen:
            continue
        seen.add(id(obj))
        out.extend(extract_specs(obj))
    _assert_unique_names(out)
    register_declared_exports(out)
    return out


def _assert_unique_names(specs: List[EndpointSpec]) -> None:
    seen: Dict[str, EndpointSpec] = {}
    for s in specs:
        prior = seen.get(s.name)
        if prior is not None and prior.method is not s.method:
            raise ValueError(
                f"duplicate routable function name {s.name!r} "
                f"({prior.module} and {s.module})"
            )
        seen[s.name] = s
