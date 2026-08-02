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
from .api.compile_axis import extract_payload_axes
from .api.decorators import (
    ATTR, VARIANT_ATTR, WF_ATTR, Compile, EndpointDecl, Resources,
    VariantDecl, WorkerFunctionDecl,
)
from .api.formula import RuntimeFormula
from .api.slot import Slot
from .api.tree import derive_components, validate_no_sibling_parts
from .discovery.names import slugify_name
from .discovery.walk import find_endpoints
from .families.base import GenerationDefaults
from .warmup import validate_class_warmup
import dataclasses
from .api.compile_axis import warm_guidance_values
from .cell_key import contract_digest

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
    declares (SDK v2): the ``Compile`` block enriched with the spec-level
    facts that shape the traced graph family — the decorator-level
    ``lora_bucket`` and the warm guidance representatives derived from the
    payload's ``CompileAxis`` classes. This is the object the compile
    machinery (compile_cache / fleet_cells / local_cells / trt_engine)
    consumes; raw ``Compile`` never travels past the registry."""

    shapes: tuple
    targets: tuple
    family: str
    regional: bool
    # This FUNCTION's effective text pin (per-lane, gap #6: a
    # @worker_function(text_len=) override wins over the class Compile's).
    text_len: Optional[int]
    dynamic: tuple
    lora_bucket: int
    # CLASS-scoped unions (pgw#654 / pgw#647 gap #1): every sibling
    # function on one class shares one cell family, so the contract facts
    # digest the UNION across the class's functions — never one function's
    # own view (per-function digests split the cell: turbo fails closed on
    # w8a8 lanes). Union is per CLASS only; sibling @endpoint classes keep
    # their own contracts (two checkpoints = two instances = two cells).
    guidance_scales: tuple
    text_lens: tuple = ()

    def contract_text_lens(self) -> tuple:
        if self.text_lens:
            return tuple(sorted({int(v) for v in self.text_lens}))
        return (int(self.text_len),) if self.text_len is not None else ()

    def contract_facts(self) -> Dict[str, Any]:
        """Canonical declared-shape-contract facts — the ck2 ``contract``
        axis digests exactly this (pgw#647: a worker on a newer contract
        must never consume an older cell)."""
        return {
            "v": 3,
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

        return contract_digest(self.contract_facts())


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
    # Slot-declared entries in `models` (pgw#520): slot -> Slot metadata
    # (selected_by/default_checkpoint). A subset of `models`'s keys — bare
    # bindings have no entry here.
    slots: Dict[str, Slot] = field(default_factory=dict)
    # Resolved family name per Slot (the endpoint's Compile(family=...), or
    # the handler's derived config schema's @family registration) —
    # precomputed once here so ctx.slots doesn't need EndpointDecl.compile.
    slot_family: Dict[str, str] = field(default_factory=dict)
    # SDK v2 (pgw#647): the handler's DERIVED config schema — the D in
    # `ctx: RequestContext[D]`. None when the handler annotates a bare
    # context. Catalog recipe metadata decodes against this type; code owns
    # the schema, the catalog owns the values (th#1116).
    defaults_type: Optional[type] = None
    # SDK v2: DERIVED component tree per introspectable pipeline slot —
    # {slot: {part_name: "weights"|"config"}}. Published into the release
    # manifest at build time (path vocabulary for per-path policy and
    # component-level routing).
    slot_components: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # SDK v2: compile-graph equivalence classes declared on payload fields
    # (CompileAxis annotations) — replaces Compile(guidance_scales=...).
    payload_axes: tuple = ()
    # SDK v2: resident traced LoRA rank bucket (decorator-level; 0 = branchless).
    lora_bucket: int = 0
    timeout_ms: Optional[int] = None
    runtime: Optional[str] = None
    compile: Optional[Compile] = None  # opt-in torch.compile spec (#384)
    # th#826: the function makes endpoint-to-endpoint child calls; emitted
    # into the discovery manifest so the hub mints the invoke_child grant.
    child_calls: bool = False
    # pgw#647: handlers on one live instance run single-flight unless the
    # class explicitly declared itself re-entrant (mutates no instance state).
    reentrant: bool = False
    # th#1004 @variant_of: this function is the variant_kind variant of the
    # sibling function variant_of (both slugs). Empty = not a variant.
    variant_of: str = ""
    variant_kind: str = ""
    # pgw#654: this handler's declared objective contract (from
    # @worker_function). None = unrestricted / serves either.
    objectives: Optional[tuple] = None
    # th#1257: this handler's declared serving tasks (@worker_function).
    # None = undeclared, which resolves NO quant approval hub-side.
    tasks: Optional[tuple] = None
    distilled: Optional[bool] = None
    # pgw#654 gap #6: this handler's effective text pin
    # (@worker_function(text_len=) else the class Compile.text_len).
    text_len: Optional[int] = None
    # pgw#654 derived warm plan: per-function non-axis warm overrides
    # (validated at walk time; require a recorded warm_reason).
    warm_overrides: Dict[str, Any] = field(default_factory=dict)
    warm_reason: str = ""
    # pgw#748 phase 1: which EXECUTION GROUP this dispatch runs on (th#1285
    # `G×D` packing). Never authored and never discovered — the executor
    # derives it from the delivered topology and the job's rank-0 device, then
    # clones the spec so the group rides the identity every downstream
    # consumer already threads. Two groups are two cards, so they are two
    # resident instances; folding the ordinal into ``instance_key`` is what
    # makes the whole per-group registry fall out of machinery that already
    # keys one record per (class, resolved binding set). Omitted from the key
    # at 0 so every single-group pod's keys stay byte-identical.
    device_group_ordinal: int = 0
    # pgw#654 CLASS-scoped unions (set by extract_specs over the class's
    # sibling specs; a function-shaped endpoint unions with itself).
    guidance_union: tuple = ()
    text_lens: tuple = ()
    # th#1050: declared lane bodies this endpoint's code branches on
    # (ctx.lane). Empty = platform-managed only.
    handles: tuple = ()
    # th#1051: declared compute-time formula for this handler; None =
    # undeclared (scalar EWMA fall-through hub-side).
    runtime_formula: Optional["RuntimeFormula"] = None
    # th#1087: declared config parameters (ctx.config) + declared env names.
    config: tuple = ()
    env: tuple = ()
    module: str = ""              # declaring module
    walked_module: str = ""       # top-level package the object was found under

    @property
    def needs_gpu(self) -> bool:
        return bool(self.resources.gpu)

    def compile_cell(self) -> Optional[CompileCell]:
        """The enriched compile-cell configuration (SDK v2), or None for
        uncompiled functions. This — never the raw ``Compile`` — is what the
        executor hands the compile machinery."""
        cfg = self.compile
        if cfg is None:
            return None

        effective_text_len = self.text_len if self.text_len is not None else cfg.text_len
        return CompileCell(
            shapes=tuple(cfg.shapes),
            targets=tuple(cfg.targets),
            family=str(cfg.family or ""),
            regional=bool(cfg.regional),
            text_len=effective_text_len,
            dynamic=tuple(cfg.dynamic),
            lora_bucket=int(self.lora_bucket or 0),
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
        every key this worker has ever computed (pgw#748)."""
        base = (self.cls, tuple(sorted(self.models.items())))
        return base if not self.device_group_ordinal else (
            base + (("device_group", int(self.device_group_ordinal)),)
        )


def _derive_defaults_type(
    owner: str, method: Callable[..., Any], ctx_param: str
) -> Optional[type]:
    """SDK v2 (pgw#647): the D in ``ctx: RequestContext[D]`` — the handler's
    derived per-model config schema (``GenerationDefaults`` subclass).
    ``None`` for a bare/unannotated context parameter. The annotation is the
    ONLY declaration site: ``Slot(default_config=...)`` is deleted; the
    catalog owns values (th#1116), code owns this schema."""
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
    """pgw#524 item 5: a ``selected_by`` payload field must type as plain
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
    owner: str, slots: Dict[str, Slot], payload_type: type
) -> None:
    """pgw#520: a Slot's ``selected_by`` must name a plain-``str`` (or
    ``str | ModelRef``, pgw#524 item 5) field on THIS handler's payload —
    validated per-handler (not per-class) because one ``models=`` decl can
    be shared by methods with different payload types. ``selected_by``
    slots may omit ``default_checkpoint`` (pgw#617: deploy-time bindings
    seed the hub mapping — tensorhub relaxed the mirrored registration
    rule in th#980). Fails at spec-construction time (discovery walk /
    CLI collection / executor boot), never at first invoke."""
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
    """pgw#517: ``compile=`` only ever arms automatically on a setup() slot
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


def _slot_is_family_agnostic(
    name: str, slot: Slot, slots: Dict[str, Slot],
) -> bool:
    """pgw#747: has this slot opted into the family vocabulary at all?

    An AUXILIARY slot declared as a bare type — ``Slot(RifeInterpolatorPipeline)``
    with no ``Hub(...)``, no ref — says nothing about architecture. A frame
    interpolator and an upscaler consume decoded RGB frames and know nothing
    about the model that produced them; ONE mirror is meant to serve every
    consumer. Stamping such a slot with the FUNCTION's family (which is what
    happened, unconditionally, to every slot) makes an assertion the endpoint
    never made — and the hub's th#586 gate then fails closed against it,
    because a family-agnostic artifact classifies as nothing:

        binding_incompatible: image-to-video/interpolator: slot declares family
        "wan-2.2-i2v-a14b" but the artifact's family is undeterminable

    That blocked every wan-2.2 and ltx-video-2.3 release carrying the `fps` or
    `resolution` preset — manifest-wide, so it also blocked the functions that
    do not use them. No catalog-side stamp fixes it honestly: `Compatible`
    compares architecture ROOTS, so stamping `rife-4.25` as `wan22` would be
    FALSE and would break the moment ltx shares the same mirror, which is the
    design. The gate already no-ops on an empty slot family, so emitting "" is
    the whole fix.

    TWO conditions, not one. "No ref" alone would also empty the family of a
    deliberately DEFAULTLESS root slot — a real model slot bound at deploy time
    through `?bindings=` (krea-2 ships exactly that shape) — silently dropping
    the LoRA-overlay policing pgw#523 added it for. The root slot is the
    function's own model and keeps the function's family whether or not it
    names a ref; only a non-root, ref-less slot is family-agnostic.
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
    slots: Dict[str, Slot],
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
    # SDK v2 lint: sibling-as-part declarations are deleted, not migrated.
    validate_no_sibling_parts(owner, slots, models)
    # SDK v2: the config schema derives from the handler's context
    # annotation (`ctx: RequestContext[D]`), never from a Slot kwarg.
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
        name: ("" if _slot_is_family_agnostic(name, slot, slots)
               else function_family)
        for name, slot in slots.items()
    }
    # SDK v2: derive each introspectable pipeline slot's component tree.
    slot_components = {
        name: tree for name, tree in (
            (name, derive_components(slot.pipeline_cls))
            for name, slot in slots.items()
        ) if tree
    }
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

    # th#1051: resolve + validate this handler's declared compute-time
    # formula now that the payload type is known. pgw#654 gap #4: fields
    # may resolve against the derived defaults schema (catalog recipe),
    # not only numeric wire defaults.
    runtime_formula = decl.runtime_formula.get(attr_name)
    if runtime_formula is not None:
        runtime_formula.validate_for_payload(
            payload_type, owner, defaults_type=defaults_type)

    # pgw#654: per-function facts declared at the definition site.
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
        text_len=(wf_text_len if wf_text_len is not None
                  else (decl.compile.text_len if decl.compile is not None else None)),
        warm_overrides=warm_overrides,
        warm_reason=(wf.warm_reason if wf is not None else ""),
        handles=tuple(decl.handles),
        runtime_formula=runtime_formula,
        config=tuple(decl.config),
        env=tuple(decl.env),
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
    # gw#470: boot warmup is default-on for GPU inference classes — fail at
    # walk time (discovery/CI/boot), never at first request.

    validate_class_warmup(cls, decl, out)
    return out


def _apply_class_unions(specs: List[EndpointSpec]) -> List[EndpointSpec]:
    """pgw#654 / pgw#647 gap #1: sibling functions of ONE class share one
    cell family, so the class's compile-contract warm facts are the UNION
    across its functions — the guidance warm set (a distilled sibling with
    no guidance field contributes nothing yet consumes the same cell) and
    the per-lane text pins (gap #6: qwen t2i 512 / edit 1024 digest as one
    dual-pin contract). Scope is exactly one class — sibling @endpoint
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
    """pgw#805: a ``compile=`` block that carries graph CLASSES *is* its
    family's export declaration — register it at collection time.

    Without this, ``export_declaration(family)`` resolves only when somebody
    imports a separate declaration module by name, which is a MINT-REQUEST
    concept (``aot_declaration.load_declaration``). A serving pod loads the
    endpoint and nothing else, so `aot_mint.mint` refused every family on
    "no registered export declaration" before it could export a single graph
    — one half of why no serving pod has ever produced an AOT cell. The
    declaration now travels with the endpoint that owns it, which is the
    end state `sdxl/aot/declaration.py`'s own docstring names ("the
    endpoint's SDK-bump lane folds these fields into the ``@endpoint
    compile=`` block and deletes this file").

    Never raises: a conflicting registration is a real defect, but it is the
    AOT lane's defect and must not take down endpoint collection (which every
    boot, every CLI walk and every discovery pass runs).
    """
    from .api.export_contract import (
        register_export_declaration, registered_entry,
    )

    registered: List[str] = []
    seen: set[int] = set()
    for spec in specs:
        compile_decl = spec.compile
        if compile_decl is None or id(compile_decl) in seen:
            continue
        seen.add(id(compile_decl))
        family = str(getattr(compile_decl, "family", "") or "").strip()
        if not family or not getattr(compile_decl, "classes", ()):
            continue
        # pgw#853: `registered_entry`, NOT `export_declaration` — reading the
        # registry back through the evaluating accessor would detonate a
        # blocked family's thunk inside endpoint COLLECTION, which is the
        # exact blast radius this issue exists to remove.
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
