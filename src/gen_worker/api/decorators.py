"""``@endpoint`` — the one decorator.

* Plain function -> stateless endpoint::

      @endpoint
      def hello(ctx, payload: In) -> Out: ...

* Class (+ optional ``setup()``) -> stateful endpoint. Every public method
  (not ``setup``/``warmup``/``shutdown``, not ``_``-prefixed) is a routable
  function; helpers must be underscore-prefixed::

      @endpoint(model=HF("org/repo", dtype="bf16"), resources=Resources(vram_gb=24))
      class Generate:
          def setup(self, model: FluxPipeline): self.model = model
          def generate(self, ctx, p: Input) -> Output: ...

* ``kind="conversion" | "training" | "dataset"`` selects the context subclass.
  Producer kinds publish explicitly — write files locally, call
  ``gen_worker.convert.publish_flavors(ctx, flavors)`` (one Tensorhub commit per
  ``ProducedFlavor``), and return a result struct. Generator handlers are
  rejected for producer kinds: nothing is ever published by yielding.
* An async-generator handler streams (inference only); there is no separate
  streaming decorator.
* ``runtime="vllm"`` boots an engine-hosting server subprocess before setup.

Checkpoint SELECTION is a runtime payload argument, not a build-time fan-out:
a handler whose payload declares a field typed with a ``ModelChoice`` subclass
picks, per request, which curated checkpoint runs against the resident base
(pgw#509 — ``gen_worker.api.model``). Divergent WIRE contracts are separate
methods; only weight-sharing forces one class.
"""

from __future__ import annotations

import inspect
import math
import re
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, TypeVar, Union, overload

import msgspec

from .binding import BINDING_TYPES, Binding
from .formula import RuntimeFormula
from .slot import DEFAULT_REGIMES, REGIMES, Slot

T = TypeVar("T")
SlotLike = Union[Binding, Slot]

KINDS = ("inference", "training", "dataset", "conversion")
RESERVED_METHODS = frozenset({"setup", "warmup", "shutdown"})


class Resources(msgspec.Struct, frozen=True, omit_defaults=True):
    """Hardware envelope for one function: ``Resources(gpu, vram_gb,
    compute_capability, libraries, ram_gb, vcpus)``.

    ``vram_gb`` is the recommended minimum CARD size: the total VRAM (GB) of
    the smallest card the function targets — ``vram_gb=24`` means "runs on a
    24 GB card". It is an optional placement hint (the orchestrator may use
    it to pick a GPU SKU), not a free-memory requirement: the platform
    reserves ~1 GB (``GPU_VRAM_OVERHEAD_GB``) for driver/framebuffer/CUDA
    context, so ``vram_gb=24`` serves on a 24 GB card even though only
    ~23.6 GB is free.

    ``Resources(vram_gb=12)`` implies ``gpu=True`` — declaring VRAM without a
    GPU is a contradiction, not an under-declaration.

    ``vram_gb`` is never a hard gate by itself: a smaller card still serves
    the function through the runtime fit ladder (fp8 storage / emergency
    4-bit / CPU offload / CPU-only), degraded but running. Set
    ``strict_vram=True`` only for bindings that cannot tolerate CPU-resident
    weights (a compiled fixed-shape graph, a TensorRT engine) — the worker
    then refuses the CPU-touching rungs (offload / cpu) outright instead of
    serving slowly. The on-GPU rungs (fp8 storage, emergency 4-bit) remain
    available under ``strict_vram``.

    ``ram_gb`` / ``vcpus`` (gw#490) declare the HOST-side ask: minimum host
    RAM (GB) and vCPU count the pod must be created with. Video-class
    endpoints need both (pinned TE park + CPU-heavy encode); the hub maps
    them to provider pod-creation minimums (th#740) and destroys
    under-allocated pods at create. Host asks do not imply ``gpu=True``.
    """

    gpu: bool = False
    vram_gb: float | None = None
    compute_capability: float | None = None
    libraries: tuple[str, ...] = ()
    strict_vram: bool = False
    ram_gb: float | None = None
    vcpus: int | None = None

    def __post_init__(self) -> None:
        force = msgspec.structs.force_setattr
        if self.vram_gb is not None:
            v = float(self.vram_gb)
            if v <= 0:
                raise ValueError(f"vram_gb must be positive, got {v}")
            force(self, "vram_gb", v)
        if self.compute_capability is not None:
            c = float(self.compute_capability)
            if c <= 0:
                raise ValueError(f"compute_capability must be positive, got {c}")
            force(self, "compute_capability", c)
        if self.libraries:
            force(self, "libraries", tuple(
                str(x).strip() for x in self.libraries if str(x).strip()
            ))
        if self.vram_gb is not None or self.compute_capability is not None:
            force(self, "gpu", True)
        if self.ram_gb is not None:
            r = float(self.ram_gb)
            if r <= 0:
                raise ValueError(f"ram_gb must be positive, got {r}")
            force(self, "ram_gb", r)
        if self.vcpus is not None:
            n = int(self.vcpus)
            if n <= 0:
                raise ValueError(f"vcpus must be positive, got {n}")
            force(self, "vcpus", n)


class NoWarmup(msgspec.Struct, frozen=True):
    """Opt a class out of the default boot warmup (gw#470) with a recorded
    reason: ``@endpoint(warmup=NoWarmup("engine captures CUDA graphs at
    boot"))``. Warmup is behavior — the opt-out lives in code, never in an
    env knob."""

    reason: str

    def __post_init__(self) -> None:
        reason = str(self.reason or "").strip()
        if not reason:
            raise ValueError("NoWarmup requires a non-empty reason")
        msgspec.structs.force_setattr(self, "reason", reason)


WarmupDecl = Union[NoWarmup, Mapping[str, Any]]


def _validate_warmup_decl(owner: str, warmup: Optional[WarmupDecl]) -> Optional[WarmupDecl]:
    if warmup is None or isinstance(warmup, NoWarmup):
        return warmup
    if isinstance(warmup, Mapping):
        for key in warmup:
            k = str(key or "").strip()
            if not k or not k.isidentifier():
                raise ValueError(
                    f"@endpoint {owner}: warmup= key {key!r} must be a handler "
                    "method name"
                )
        return dict(warmup)
    raise TypeError(
        f"@endpoint {owner}: warmup= must be a "
        "{method_name: payload} mapping or NoWarmup(reason), got "
        f"{type(warmup).__name__}"
    )


# th#1017 inference regimes: what checkpoint(s) a handler's CFG/scheduling
# code path was written for. Class handlers declare a {method_name:
# (regime, ...)} mapping (mirrors warmup=); the function form (one handler)
# declares a bare tuple. A method/function absent from the declaration gets
# DEFAULT_REGIMES ("standard",).
RegimesDecl = Union[Tuple[str, ...], Mapping[str, Tuple[str, ...]]]


def _validate_regime_tuple(owner: str, value: Any) -> Tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise TypeError(
            f"@endpoint {owner}: regimes= entries must be a tuple of regime "
            f"strings, got {type(value).__name__}"
        )
    regimes = tuple(str(v).strip() for v in value)
    if not regimes:
        raise ValueError(f"@endpoint {owner}: regimes= entry must not be empty")
    bad = [r for r in regimes if r not in REGIMES]
    if bad:
        raise ValueError(
            f"@endpoint {owner}: unknown regime(s) {bad!r} (valid: {REGIMES})"
        )
    return regimes


def _validate_class_regimes(
    owner: str, regimes: Optional[RegimesDecl], handler_names: "set[str]"
) -> Dict[str, Tuple[str, ...]]:
    if regimes is None:
        return {}
    if not isinstance(regimes, Mapping):
        raise TypeError(
            f"@endpoint {owner}: class regimes= must be a "
            "{method_name: (regime, ...)} mapping, got "
            f"{type(regimes).__name__}"
        )
    unknown = set(regimes) - handler_names
    if unknown:
        raise ValueError(
            f"@endpoint {owner}: regimes= names unknown handler method(s) "
            f"{sorted(unknown)} (known: {sorted(handler_names)})"
        )
    return {k: _validate_regime_tuple(f"{owner}.{k}", v) for k, v in regimes.items()}


def _validate_function_regimes(
    owner: str, regimes: Optional[RegimesDecl]
) -> Tuple[str, ...]:
    if regimes is None:
        return DEFAULT_REGIMES
    if isinstance(regimes, Mapping):
        raise TypeError(
            f"@endpoint function {owner!r}: regimes= must be a tuple of "
            "regime strings (functions have one handler, no per-method mapping)"
        )
    return _validate_regime_tuple(owner, regimes)


def _validate_handles(owner: str, handles: Any) -> Tuple[str, ...]:
    """th#1050 opt-in lane declaration: `handles=["fp8-w8a8-dynamic"]` marks
    that this endpoint's code BRANCHES on the executing lane (ctx.lane) —
    behavioral divergence only, never inventory. Tokens are concrete lane
    BODIES (no `+eager|+compiled`: execution is platform-managed). Nothing
    declared = fully platform-managed lanes, exactly as before."""
    from ..models import lanes as lanespec

    if handles is None:
        return ()
    if isinstance(handles, str) or not isinstance(handles, (list, tuple)):
        raise TypeError(
            f"@endpoint {owner}: handles= must be a list/tuple of lane body "
            f"strings, got {type(handles).__name__}"
        )
    known = lanespec.known_lane_bodies()
    out: list[str] = []
    for token in handles:
        t = str(token or "").strip().lower()
        if "+" in t:
            raise ValueError(
                f"@endpoint {owner}: handles= token {token!r} carries an "
                "execution axis; declare the lane body (execution is "
                "platform-managed)"
            )
        if t in lanespec.FAMILIES:
            raise ValueError(
                f"@endpoint {owner}: handles= token {token!r} is a coarse "
                f"family; declare a concrete lane body (one of {known})"
            )
        if t not in known:
            raise ValueError(
                f"@endpoint {owner}: handles= token {token!r} is not a known "
                f"lane body (known: {known})"
            )
        if t in out:
            raise ValueError(f"@endpoint {owner}: handles= repeats {token!r}")
        out.append(t)
    return tuple(out)


_CONFIG_TYPES: Dict[type, str] = {str: "string", int: "int", float: "float", bool: "bool"}


class ConfigParam(msgspec.Struct, frozen=True):
    """th#1087 declared config parameter: code declares the knob (name, type,
    default, constraints); the deployer sets values through the hub config
    API (write-time 422 on unknown/invalid); handlers read the effective
    value via ``ctx.config[name]``::

        @endpoint(config=[
            ConfigParam("scheduler", str, default="ddim", choices=["ddim", "euler_a"]),
            ConfigParam("default_steps", int, default=30, ge=1, le=150),
        ])
    """

    name: str
    type: type
    default: Any
    choices: tuple = ()
    ge: Optional[float] = None
    le: Optional[float] = None
    regex: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        force = msgspec.structs.force_setattr
        name = str(self.name or "").strip()
        if not name or not name.isidentifier():
            raise ValueError(f"ConfigParam name {self.name!r} must be an identifier")
        force(self, "name", name)
        if self.type not in _CONFIG_TYPES:
            raise TypeError(
                f"ConfigParam {name!r}: type must be one of "
                f"{sorted(t.__name__ for t in _CONFIG_TYPES)}, got {self.type!r}"
            )
        self._check_value("default", self.default)
        if self.type is float and isinstance(self.default, int) and not isinstance(self.default, bool):
            force(self, "default", float(self.default))
        choices = tuple(self.choices)
        for c in choices:
            self._check_value("choices entry", c)
        if choices:
            if len(set(choices)) != len(choices):
                raise ValueError(f"ConfigParam {name!r}: choices must be unique")
            if self.default not in choices:
                raise ValueError(
                    f"ConfigParam {name!r}: default {self.default!r} not in "
                    f"choices {list(choices)}"
                )
        force(self, "choices", choices)
        if (self.ge is not None or self.le is not None) and self.type not in (int, float):
            raise ValueError(f"ConfigParam {name!r}: ge/le apply to int/float only")
        if self.ge is not None and self.le is not None and self.ge > self.le:
            raise ValueError(f"ConfigParam {name!r}: ge {self.ge} > le {self.le}")
        regex = str(self.regex or "")
        if regex and self.type is not str:
            raise ValueError(f"ConfigParam {name!r}: regex applies to str only")
        if regex:
            try:
                re.compile(regex)
            except re.error as exc:
                raise ValueError(
                    f"ConfigParam {name!r}: invalid regex {regex!r}: {exc}"
                ) from exc
            if re.search(regex, self.default) is None:
                raise ValueError(
                    f"ConfigParam {name!r}: default {self.default!r} does not "
                    f"match regex {regex!r}"
                )
        force(self, "regex", regex)
        force(self, "description", str(self.description or "").strip())

    def _check_value(self, label: str, value: Any) -> None:
        ok = isinstance(value, self.type) and not (
            self.type is not bool and isinstance(value, bool)
        )
        if self.type is float and isinstance(value, int) and not isinstance(value, bool):
            ok = True
        if not ok:
            raise TypeError(
                f"ConfigParam {self.name!r}: {label} {value!r} is not "
                f"{self.type.__name__}"
            )
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if self.ge is not None and value < self.ge:
                raise ValueError(
                    f"ConfigParam {self.name!r}: {label} {value!r} < ge {self.ge}"
                )
            if self.le is not None and value > self.le:
                raise ValueError(
                    f"ConfigParam {self.name!r}: {label} {value!r} > le {self.le}"
                )

    def to_manifest(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "name": self.name,
            "type": _CONFIG_TYPES[self.type],
            "default": self.default,
        }
        if self.choices:
            out["choices"] = list(self.choices)
        if self.ge is not None:
            out["ge"] = self.ge
        if self.le is not None:
            out["le"] = self.le
        if self.regex:
            out["regex"] = self.regex
        if self.description:
            out["description"] = self.description
        return out


def _validate_config_decl(owner: str, config: Any) -> Tuple[ConfigParam, ...]:
    if config is None:
        return ()
    if isinstance(config, ConfigParam) or not isinstance(config, (list, tuple)):
        raise TypeError(
            f"@endpoint {owner}: config= must be a list/tuple of ConfigParam, "
            f"got {type(config).__name__}"
        )
    out: list[ConfigParam] = []
    seen: set[str] = set()
    for p in config:
        if not isinstance(p, ConfigParam):
            raise TypeError(
                f"@endpoint {owner}: config= entries must be ConfigParam, "
                f"got {type(p).__name__}"
            )
        if p.name in seen:
            raise ValueError(f"@endpoint {owner}: config= repeats {p.name!r}")
        seen.add(p.name)
        out.append(p)
    return tuple(out)


def _validate_env_decl(owner: str, env: Any) -> Tuple[str, ...]:
    """th#1087 D2: code declares the env names it reads; the hub validates
    config-layer env writes against this declaration (undeclared-write 422)."""
    if env is None:
        return ()
    if isinstance(env, str) or not isinstance(env, (list, tuple)):
        raise TypeError(
            f"@endpoint {owner}: env= must be a list/tuple of env-var name "
            f"strings, got {type(env).__name__}"
        )
    out: list[str] = []
    for name in env:
        n = str(name or "").strip()
        if (
            not n
            or len(n) > 64
            or not ("A" <= n[0] <= "Z")
            or any(c not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_" for c in n)
        ):
            raise ValueError(
                f"@endpoint {owner}: env= name {name!r} is not a valid "
                "environment variable name"
            )
        if n in out:
            raise ValueError(f"@endpoint {owner}: env= repeats {name!r}")
        out.append(n)
    return tuple(out)


class Compile(msgspec.Struct, frozen=True):
    """Opt-in torch.compile over pre-built per-SKU cache artifacts (#384).

    ``Compile(family="flux2-klein-4b", shapes=((768, 768), (1024, 1024)))``
    names the model FAMILY (caches key on the traced graph, so every
    fine-tune of a family shares one artifact) and the shape set the compile
    job warms. A shape row is ``(width, height)`` for image models or
    ``(width, height, frames)`` for video models (ie#381): video DiT graphs
    key on the token count, which includes the frame axis, so every
    (resolution, duration) preset pair is its own graph. Two-stage presets
    contribute BOTH their base and refined resolutions as rows.

    SHAPES DERIVE FROM THE PAYLOAD PRESET ENUM (ie#345 fleet policy): when
    the endpoint's payload uses size buckets, ``shapes`` must be exactly
    that bucket table — one source of truth, 100% cache coverage of legal
    requests. Endpoints still accepting free width/height use the family's
    dialect-default shapes until they adopt buckets. CFG is a graph shape
    too: CFG variants trace batch-2 graphs, distilled variants batch-1.
    ``guidance_scales`` declares the image regimes Forge must warm into the
    same family cell; for example ``(5.0, 0.0)`` captures CFG and no-CFG calls
    when they share one module graph. A LoRA-mutated graph needs its own lane.
    Empty preserves the pipeline's default. Declaring this does NOT force compilation: the
    worker arms torch.compile only when a verified cache artifact for
    (family, SKU, torch, triton) is seeded — otherwise it stays eager.
    See ``gen_worker.compile_cache``.
    """

    shapes: tuple[tuple[int, ...], ...]
    targets: tuple[str, ...] = ("transformer", "vae.decode")
    family: str = ""
    guidance_scales: tuple[float, ...] = ()
    # Regional compilation (diffusers compile_repeated_blocks): compile the
    # target's repeated transformer blocks instead of the whole forward.
    # REQUIRED for big fp8 layerwise-cast models (ie#381, measured on LTX
    # 22B/H100): whole-graph inductor planning co-materializes per-layer
    # bf16 upcast buffers and OOMs at the largest shapes; per-block graphs
    # bound that to one block. Also much faster cold compile (one block
    # graph per shape, reused across blocks). Cells record the mode — a
    # mode drift consumer stays eager (cache would miss anyway).
    regional: bool = False
    # gw#561: dynamic-LoRA endpoints declare the traced rank bucket. The
    # worker then serves the branch-bearing graph family: canonical zeroed
    # rank-<bucket> branches enabled at load (gw#547 compiled-lane
    # contract), only `<lane>-lora<bucket>` cells adopt, and adapter swaps
    # stay buffer copies — never a recompile. 0 = branchless (today).
    lora_bucket: int = 0

    def __post_init__(self) -> None:
        force = msgspec.structs.force_setattr
        shapes = tuple(tuple(int(v) for v in s) for s in self.shapes)
        if (
            not shapes
            or any(len(s) not in (2, 3) for s in shapes)
            or any(v <= 0 for s in shapes for v in s)
        ):
            raise ValueError(
                "Compile.shapes must be positive (w, h) or (w, h, frames) "
                f"rows, got {self.shapes!r}"
            )
        force(self, "shapes", shapes)
        targets = tuple(str(t).strip() for t in self.targets if str(t).strip())
        if not targets:
            raise ValueError("Compile.targets must not be empty")
        force(self, "targets", targets)
        force(self, "family", str(self.family or "").strip())
        guidance_scales = tuple(float(v) for v in self.guidance_scales)
        if any(not math.isfinite(v) or v < 0.0 for v in guidance_scales):
            raise ValueError(
                "Compile.guidance_scales must contain finite non-negative values, "
                f"got {self.guidance_scales!r}"
            )
        if len(set(guidance_scales)) != len(guidance_scales):
            raise ValueError("Compile.guidance_scales must not contain duplicates")
        force(self, "guidance_scales", guidance_scales)
        bucket = int(self.lora_bucket or 0)
        if bucket:
            from ..models.w8a8_lora import RANK_BUCKETS

            if bucket not in RANK_BUCKETS:
                raise ValueError(
                    f"Compile.lora_bucket must be 0 or one of {RANK_BUCKETS}, "
                    f"got {self.lora_bucket!r}"
                )
        force(self, "lora_bucket", bucket)


class EndpointDecl(msgspec.Struct, frozen=True, kw_only=True):
    """Metadata attached by ``@endpoint``; read by the registry walker."""

    kind: str = "inference"
    resources: Resources = msgspec.field(default_factory=Resources)
    models: Mapping[str, Binding] = msgspec.field(default_factory=dict)
    # Slot-declared entries in `models=`/`model=` (pgw#520): a subset of
    # `models`' keys, carrying the Slot's selected_by/default_checkpoint/
    # default_config metadata that `models` (a plain Binding map, for
    # back-compat with every existing model-injection call site) can't hold.
    slots: Mapping[str, Slot] = msgspec.field(default_factory=dict)
    runtime: Optional[str] = None
    name: Optional[str] = None  # function-shaped endpoints only
    is_function: bool = False
    compile: Optional[Compile] = None
    # th#826 call-out primitive: the function makes endpoint-to-endpoint
    # child calls (ctx.call_endpoint / ctx.workflow_checkpoint). The hub
    # mints the invoke_child credential ONLY for declaring functions.
    child_calls: bool = False
    # pgw#647 concurrency contract: handlers on ONE live instance run
    # SINGLE-FLIGHT by default (one binding set = one materialized graph
    # with mutable buffers — e.g. a resident LoRA branch; two concurrent
    # requests would corrupt each other). ``reentrant=True`` is the explicit
    # opt-in for classes whose handlers genuinely mutate no instance state.
    reentrant: bool = False
    # gw#470 boot warmup: None = auto-synthesize from each handler's payload
    # schema; {method: payload-or-None} = declared warmup payloads (None
    # skips that method); NoWarmup(reason) = class-level opt-out.
    warmup: Optional[WarmupDecl] = None
    # th#1017: per-handler declared regimes, attr_name -> (regime, ...).
    # Function-shaped endpoints key their single handler under "".
    regimes: Mapping[str, Tuple[str, ...]] = msgspec.field(default_factory=dict)
    # th#1050: opt-in declared lane bodies this endpoint's code branches on
    # (ctx.lane). Empty = platform-managed behavior only.
    handles: Tuple[str, ...] = ()
    # th#1051: declared compute-time formulas, attr_name -> RuntimeFormula
    # (function-shaped endpoints key their single handler under "").
    # Declared via the runtime= kwarg overload; payload-field validation
    # happens at registry walk time when payload types are resolved.
    runtime_formula: Mapping[str, RuntimeFormula] = msgspec.field(default_factory=dict)
    # th#1087: declared config parameters (ctx.config) + declared env names
    # (D2) — the mutable-config surface the hub validates writes against.
    config: Tuple[ConfigParam, ...] = ()
    env: Tuple[str, ...] = ()


ATTR = "__gen_worker_endpoint__"
VARIANT_ATTR = "__gen_worker_variant__"


class VariantDecl(msgspec.Struct, frozen=True):
    """``@variant_of`` marker (th#1004): this handler is the ``kind`` variant
    of sibling function ``of`` on the same endpoint."""

    of: str
    kind: str

    def __post_init__(self) -> None:
        force = msgspec.structs.force_setattr
        of = str(self.of or "").strip()
        kind = str(self.kind or "").strip().lower()
        if not of:
            raise ValueError("@variant_of requires a target function name")
        if not kind:
            raise ValueError("@variant_of requires a non-empty kind")
        force(self, "of", of)
        force(self, "kind", kind)


def variant_of(of: str, *, kind: str = "turbo") -> Callable[[T], T]:
    """Declare a handler as a variant of a sibling function (th#1004).

    The pairing is emitted into the discovery manifest (``variant_of`` /
    ``variant``) and surfaced on tensorhub's public endpoint info, so a
    platform can render e.g. a regular/turbo toggle instead of modeling the
    variant as a separate product::

        @variant_of("generate")  # kind="turbo"
        def generate_turbo(self, ctx, payload: TurboInput) -> ImageOutput: ...

    The target must be another function on the same endpoint and must not
    itself be a variant (validated at discovery time).
    """
    decl = VariantDecl(of=of, kind=kind)

    def apply(fn: T) -> T:
        if not inspect.isfunction(fn):
            raise TypeError(
                f"@variant_of decorates handler functions/methods, got "
                f"{type(fn).__name__}"
            )
        setattr(fn, VARIANT_ATTR, decl)
        return fn

    return apply


def _handler_params(fn: Callable[..., Any], *, is_method: bool) -> list[inspect.Parameter]:
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())
    if is_method:
        params = [p for p in params if p.name != "self"]
    return params


def _validate_handler_shape(owner: str, fn: Callable[..., Any], *, is_method: bool) -> None:
    params = _handler_params(fn, is_method=is_method)
    if len(params) < 2:
        raise TypeError(
            f"@endpoint: {owner} must accept (ctx, payload) "
            f"(got params {[p.name for p in params]}). Handlers take the "
            "request context first and a msgspec.Struct payload second; "
            "prefix non-handler methods with an underscore."
        )


def _reject_producer_generator(owner: str, fn: Callable[..., Any], kind: str) -> None:
    """Producer kinds (conversion/training/dataset) publish explicitly and
    return a result; a generator handler would stream chunks and publish
    NOTHING. Fail at decoration time instead of silently at runtime."""
    if kind == "inference":
        return
    if inspect.isgeneratorfunction(fn) or inspect.isasyncgenfunction(fn):
        raise TypeError(
            f"@endpoint(kind={kind!r}): {owner} must not be a generator. "
            "Producer endpoints write files locally, publish explicitly "
            "(gen_worker.convert.publish_flavors(ctx, flavors)), and return a "
            "result struct; streaming generators are inference-only."
        )


def _split_slot_like(key: str, v: "SlotLike") -> Tuple[Optional[Binding], Optional[Slot]]:
    """One ``models={}``/``model=`` value -> ``(binding_or_None, slot_or_None)``.

    A ``Slot`` contributes its ``default_checkpoint`` (if any) to the plain
    binding map every existing model-injection call site (executor, CLI,
    prefetch) already understands, PLUS itself to the slots map the pgw#520
    surfaces (discovery emission, the resolution chain) read. A bare binding
    contributes only to the binding map — it carries no Slot metadata."""
    if isinstance(v, Slot):
        return v.default_checkpoint, v
    if isinstance(v, BINDING_TYPES):
        return v, None
    raise TypeError(
        f"@endpoint models[{key!r}] must be a HF/Hub/Civitai/ModelScope "
        f"binding or Slot(...), got {type(v).__name__}"
    )


def _normalize_models(
    model: Optional["SlotLike"], models: Optional[Mapping[str, "SlotLike"]]
) -> Tuple[Dict[str, Binding], Dict[str, Slot]]:
    if model is not None and models is not None:
        raise ValueError("@endpoint: pass model= OR models=, not both")
    if model is not None:
        # Single-binding shorthand: the slot name is resolved from the
        # setup()/handler parameter name at class validation time.
        binding, slot = _split_slot_like("", model)
        out_models: Dict[str, Binding] = {"": binding} if binding is not None else {}
        out_slots: Dict[str, Slot] = {"": slot} if slot is not None else {}
        return out_models, out_slots
    out_models = {}
    out_slots = {}
    for key, v in (models or {}).items():
        k = str(key or "").strip()
        if not k or not k.isidentifier():
            raise ValueError(f"@endpoint models key {key!r} must be an identifier")
        binding, slot = _split_slot_like(k, v)
        if binding is not None:
            out_models[k] = binding
        if slot is not None:
            out_slots[k] = slot
    return out_models, out_slots


def _find_handler_methods(cls: type) -> list[tuple[str, Callable[..., Any]]]:
    out: list[tuple[str, Callable[..., Any]]] = []
    for attr, member in inspect.getmembers(cls, predicate=inspect.isfunction):
        if attr.startswith("_") or attr in RESERVED_METHODS:
            continue
        out.append((attr, member))
    if not out:
        raise TypeError(
            f"@endpoint class {cls.__name__!r}: no handler methods found. "
            "Define at least one public method taking (self, ctx, payload)."
        )
    for handler_name, handler_fn in out:
        _validate_handler_shape(f"{cls.__name__}.{handler_name}", handler_fn, is_method=True)
    return out


def _setup_params(cls: type) -> Optional[dict[str, inspect.Parameter]]:
    """Named params of ``setup`` (excluding self), or None when absent."""
    setup = inspect.getattr_static(cls, "setup", None)
    if setup is None:
        return None
    fn = setup.__func__ if isinstance(setup, (classmethod, staticmethod)) else setup
    sig = inspect.signature(fn)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    return {p.name: p for p in params}


def _is_server_handle_param(p: inspect.Parameter) -> bool:
    """True when the param is annotated ``ServerHandle`` (a runtime= injection
    target, not a model slot)."""
    ann = p.annotation
    if ann is inspect.Parameter.empty:
        return False
    from ..runtimes.server import ServerHandle

    if ann is ServerHandle:
        return True
    return isinstance(ann, str) and ann.split(".")[-1] == "ServerHandle"


def _resolve_single_slot(
    cls: type,
    models: Dict[str, Binding],
    slots: Dict[str, Slot],
    handlers: list[tuple[str, Callable[..., Any]]],
) -> Tuple[Dict[str, Binding], Dict[str, Slot]]:
    """Resolve the ``model=`` shorthand's slot name from setup()/handler
    params. Exactly one of ``models``/``slots`` holds the ``""`` key (a bare
    binding or a Slot, never both — ``_normalize_models`` is the only
    producer)."""
    if "" not in models and "" not in slots:
        return models, slots

    def _name_it() -> str:
        setup_kwargs = _setup_params(cls)
        if setup_kwargs:
            named = [
                n for n, p in setup_kwargs.items()
                if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
                and not _is_server_handle_param(p)
            ]
            if len(named) == 1:
                return named[0]
            raise ValueError(
                f"@endpoint class {cls.__name__!r}: model= needs exactly one "
                f"setup() parameter to name the slot (setup declares {named})."
            )
        # No setup: slot name comes from the (single) handler's injected param.
        injected: set[str] = set()
        for attr, method in handlers:
            for p in _handler_params(method, is_method=True)[2:]:
                injected.add(p.name)
        if len(injected) == 1:
            return injected.pop()
        raise ValueError(
            f"@endpoint class {cls.__name__!r}: model= with no setup() needs "
            f"exactly one injected handler parameter after (ctx, payload) to "
            f"name the slot (found {sorted(injected) or 'none'})."
        )

    name = _name_it()
    if "" in models:
        models[name] = models.pop("")
    if "" in slots:
        slots[name] = slots.pop("")
    return models, slots


def _validate_class_models(
    cls: type, models: Dict[str, Binding], slots: Dict[str, Slot]
) -> None:
    """Model slots must be consumable: by setup() params (stateful) or by
    handler params (per-call injection when there is no setup)."""
    all_keys = set(models) | set(slots)
    if not all_keys:
        return
    setup_kwargs = _setup_params(cls)
    if setup_kwargs is not None:
        has_var_kw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in setup_kwargs.values()
        )
        missing = [k for k in all_keys if k not in setup_kwargs and not has_var_kw]
        if missing:
            raise ValueError(
                f"@endpoint class {cls.__name__!r}: models slot(s) {missing} "
                f"match no setup() parameter (setup declares "
                f"{sorted(setup_kwargs)})."
            )


def _split_runtime_kwarg(
    owner: str, runtime: Any,
) -> Tuple[Optional[str], Dict[str, RuntimeFormula]]:
    """th#1051 runtime= overload: a str selects an engine runtime (vllm /
    llama-server, classes only); a RuntimeFormula declares the compute-time
    formula for every handler ("*"); a mapping declares per-handler formulas
    (classes only). Returns (engine_runtime, {attr_or_"*": formula})."""
    if runtime is None:
        return None, {}
    if isinstance(runtime, str):
        return runtime, {}
    if isinstance(runtime, RuntimeFormula):
        return None, {"*": runtime}
    if isinstance(runtime, Mapping):
        out: Dict[str, RuntimeFormula] = {}
        for attr, rf in runtime.items():
            if not isinstance(attr, str) or not isinstance(rf, RuntimeFormula):
                raise TypeError(
                    f"@endpoint {owner}: runtime= mapping must be "
                    f"{{handler_name: RuntimeFormula}}"
                )
            out[attr] = rf
        return None, out
    raise TypeError(
        f"@endpoint {owner}: runtime= must be an engine runtime string, a "
        f"RuntimeFormula, or a mapping of handler name -> RuntimeFormula, "
        f"got {type(runtime).__name__}"
    )


def _expand_formula_map(
    owner: str, formulas: Dict[str, RuntimeFormula], handler_attrs: "list[str]",
) -> Dict[str, RuntimeFormula]:
    if not formulas:
        return {}
    if "*" in formulas:
        if len(formulas) > 1:
            raise ValueError(
                f"@endpoint {owner}: runtime= cannot mix a bare RuntimeFormula "
                f"with per-handler entries"
            )
        return {attr: formulas["*"] for attr in handler_attrs}
    unknown = [a for a in formulas if a not in handler_attrs]
    if unknown:
        raise ValueError(
            f"@endpoint {owner}: runtime= names unknown handler(s) {unknown} "
            f"(handlers: {sorted(handler_attrs)})"
        )
    return dict(formulas)


def _decorate_class(
    cls: type,
    *,
    kind: str,
    resources: Resources,
    models: Dict[str, Binding],
    slots: Dict[str, Slot],
    runtime: Optional[str],
    runtime_formula: Optional[Dict[str, RuntimeFormula]] = None,
    compile: Optional[Compile] = None,
    child_calls: bool = False,
    reentrant: bool = False,
    warmup: Optional[WarmupDecl] = None,
    regimes: Optional[RegimesDecl] = None,
    handles: Optional[Any] = None,
    config: Optional[Any] = None,
    env: Optional[Any] = None,
) -> type:
    handlers = _find_handler_methods(cls)
    for attr, member in handlers:
        _reject_producer_generator(f"{cls.__name__}.{attr}", member, kind)
    models, slots = _resolve_single_slot(cls, models, slots, handlers)
    _validate_class_models(cls, models, slots)

    if runtime is not None and runtime not in ("vllm", "llama-server"):
        raise ValueError(
            f"@endpoint class {cls.__name__!r}: runtime must be 'vllm' or "
            f"'llama-server', got {runtime!r}"
        )

    decl = EndpointDecl(
        kind=kind, resources=resources, models=models, slots=slots,
        runtime=runtime, compile=compile, child_calls=child_calls,
        reentrant=reentrant,
        warmup=_validate_warmup_decl(cls.__name__, warmup),
        regimes=_validate_class_regimes(
            cls.__name__, regimes, {attr for attr, _ in handlers}
        ),
        handles=_validate_handles(cls.__name__, handles),
        runtime_formula=_expand_formula_map(
            cls.__name__, runtime_formula or {}, [attr for attr, _ in handlers]
        ),
        config=_validate_config_decl(cls.__name__, config),
        env=_validate_env_decl(cls.__name__, env),
    )
    setattr(cls, ATTR, decl)
    setattr(cls, "__gen_worker_handlers__", handlers)
    # gw#470: default-on boot warmup — fail unwarmable GPU inference classes
    # at import when type hints resolve here (walk time re-checks). Lazy
    # import: warmup.py imports this module.
    from ..warmup import validate_at_decoration

    validate_at_decoration(cls, decl)
    return cls


def _decorate_function(
    fn: Callable[..., Any],
    *,
    kind: str,
    resources: Resources,
    models: Dict[str, Binding],
    slots: Dict[str, Slot],
    runtime: Optional[str],
    runtime_formula: Optional[Dict[str, RuntimeFormula]] = None,
    name: Optional[str],
    compile: Optional[Compile] = None,
    child_calls: bool = False,
    reentrant: bool = False,
    warmup: Optional[WarmupDecl] = None,
    regimes: Optional[RegimesDecl] = None,
    handles: Optional[Any] = None,
    config: Optional[Any] = None,
    env: Optional[Any] = None,
) -> Callable[..., Any]:
    if reentrant:
        raise ValueError(
            f"@endpoint function {fn.__name__!r}: reentrant= applies to "
            "class endpoints only (stateless functions hold no instance "
            "state to single-flight)."
        )
    _validate_handler_shape(fn.__name__, fn, is_method=False)
    _reject_producer_generator(fn.__name__, fn, kind)
    if "" in models or "" in slots:
        injected = [p.name for p in _handler_params(fn, is_method=False)[2:]]
        if len(injected) != 1:
            raise ValueError(
                f"@endpoint function {fn.__name__!r}: model= needs exactly one "
                f"injected parameter after (ctx, payload) to name the slot "
                f"(found {injected or 'none'})."
            )
        slot_name = injected[0]
        if "" in models:
            models[slot_name] = models.pop("")
        if "" in slots:
            slots[slot_name] = slots.pop("")
    if runtime is not None:
        raise ValueError(
            f"@endpoint function {fn.__name__!r}: an engine runtime= requires "
            "a class with setup() (the engine server outlives single calls)."
        )
    if warmup is not None:
        raise ValueError(
            f"@endpoint function {fn.__name__!r}: warmup= requires a class "
            "with setup() (stateless functions hold nothing to warm)."
        )
    formulas = runtime_formula or {}
    if set(formulas) - {"*"}:
        raise ValueError(
            f"@endpoint function {fn.__name__!r}: runtime= takes a bare "
            "RuntimeFormula (per-handler mappings are for classes)"
        )
    decl = EndpointDecl(
        kind=kind, resources=resources, models=models, slots=slots,
        runtime=None, name=(name or fn.__name__), is_function=True,
        compile=compile, child_calls=child_calls,
        regimes={"": _validate_function_regimes(fn.__name__, regimes)},
        handles=_validate_handles(fn.__name__, handles),
        runtime_formula={"": formulas["*"]} if "*" in formulas else {},
        config=_validate_config_decl(fn.__name__, config),
        env=_validate_env_decl(fn.__name__, env),
    )
    setattr(fn, ATTR, decl)
    return fn


@overload
def endpoint(target: T) -> T: ...  # bare @endpoint on a class/function


@overload
def endpoint(
    *,
    kind: str = ...,
    model: Optional[SlotLike] = ...,
    models: Optional[Mapping[str, SlotLike]] = ...,
    resources: Optional[Resources] = ...,
    runtime: Union[str, RuntimeFormula, Mapping[str, RuntimeFormula], None] = ...,
    name: Optional[str] = ...,
    compile: Optional[Compile] = ...,
    child_calls: bool = ...,
    reentrant: bool = ...,
    warmup: Optional[WarmupDecl] = ...,
    regimes: Optional[RegimesDecl] = ...,
    handles: Optional[Any] = ...,
    config: Optional[Sequence[ConfigParam]] = ...,
    env: Optional[Sequence[str]] = ...,
) -> Callable[[T], T]: ...  # configured @endpoint(...) form


def endpoint(
    target: Optional[T] = None,
    *,
    kind: str = "inference",
    model: Optional[SlotLike] = None,
    models: Optional[Mapping[str, SlotLike]] = None,
    resources: Optional[Resources] = None,
    runtime: Union[str, RuntimeFormula, Mapping[str, RuntimeFormula], None] = None,
    name: Optional[str] = None,
    compile: Optional[Compile] = None,
    child_calls: bool = False,
    reentrant: bool = False,
    warmup: Optional[WarmupDecl] = None,
    regimes: Optional[RegimesDecl] = None,
    handles: Optional[Any] = None,
    config: Optional[Sequence[ConfigParam]] = None,
    env: Optional[Sequence[str]] = None,
) -> Any:
    """The one endpoint decorator. See the module docstring for shapes.

    ``model=``/``models=`` values are a ``Binding`` (a fixed pick — HF/Hub/
    Civitai/ModelScope) or a :class:`~gen_worker.api.slot.Slot` (a
    hub-resolved slot: ``selected_by=``, ``default_checkpoint=``,
    ``default_config=``; pgw#520). A bare binding is sugar for
    ``Slot(<inferred pipeline class>, default_checkpoint=ref)`` with no hub
    involvement — both forms can mix in one ``models={}`` dict.
    """
    if kind not in KINDS:
        raise ValueError(f"@endpoint kind must be one of {KINDS}, got {kind!r}")
    resources_value = resources if resources is not None else Resources()
    if resources is not None and not isinstance(resources, Resources):
        raise TypeError(
            f"@endpoint resources= must be a Resources, got {type(resources).__name__}"
        )
    if compile is not None and not isinstance(compile, Compile):
        raise TypeError(
            f"@endpoint compile= must be a Compile, got {type(compile).__name__}"
        )
    model_map, slot_map = _normalize_models(model, models)
    owner = getattr(target, "__name__", "<endpoint>") if target is not None else "<endpoint>"
    engine_runtime, runtime_formulas = _split_runtime_kwarg(owner, runtime)

    def apply(obj: Any) -> Any:
        if inspect.isclass(obj):
            if name is not None:
                raise ValueError("@endpoint name= applies to functions only; "
                                 "class handlers route by method name")
            return _decorate_class(
                obj, kind=kind, resources=resources_value, models=dict(model_map),
                slots=dict(slot_map), runtime=engine_runtime,
                runtime_formula=runtime_formulas, compile=compile,
                child_calls=child_calls, reentrant=reentrant, warmup=warmup,
                regimes=regimes, handles=handles, config=config, env=env,
            )
        if inspect.isfunction(obj):
            return _decorate_function(
                obj, kind=kind, resources=resources_value, models=dict(model_map),
                slots=dict(slot_map), runtime=engine_runtime,
                runtime_formula=runtime_formulas, name=name, compile=compile,
                child_calls=child_calls, reentrant=reentrant, warmup=warmup,
                regimes=regimes, handles=handles, config=config, env=env,
            )
        raise TypeError(
            f"@endpoint requires a function or class, got {type(obj).__name__}"
        )

    if target is not None:
        return apply(target)
    return apply


__all__ = [
    "Compile", "ConfigParam", "EndpointDecl", "NoWarmup", "RegimesDecl",
    "Resources", "SlotLike", "VariantDecl", "WarmupDecl", "endpoint",
    "variant_of",
]
