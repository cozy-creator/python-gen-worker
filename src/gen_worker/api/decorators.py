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
from typing import Any, Callable, Dict, Mapping, Optional, Tuple, TypeVar, Union, overload

import msgspec

from .binding import BINDING_TYPES, Binding
from .slot import Slot

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
    too: CFG variants trace batch-2 graphs, distilled variants batch-1 —
    a variant must never cross the boundary (clamp guidance_scale). Declaring this does NOT force compilation: the
    worker arms torch.compile only when a verified cache artifact for
    (family, SKU, torch, triton) is seeded — otherwise it stays eager.
    See ``gen_worker.compile_cache``.
    """

    shapes: tuple[tuple[int, ...], ...]
    targets: tuple[str, ...] = ("transformer", "vae.decode")
    family: str = ""
    # Regional compilation (diffusers compile_repeated_blocks): compile the
    # target's repeated transformer blocks instead of the whole forward.
    # REQUIRED for big fp8 layerwise-cast models (ie#381, measured on LTX
    # 22B/H100): whole-graph inductor planning co-materializes per-layer
    # bf16 upcast buffers and OOMs at the largest shapes; per-block graphs
    # bound that to one block. Also much faster cold compile (one block
    # graph per shape, reused across blocks). Cells record the mode — a
    # mode drift consumer stays eager (cache would miss anyway).
    regional: bool = False

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


ATTR = "__gen_worker_endpoint__"


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


def _decorate_class(
    cls: type,
    *,
    kind: str,
    resources: Resources,
    models: Dict[str, Binding],
    slots: Dict[str, Slot],
    runtime: Optional[str],
    compile: Optional[Compile] = None,
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
        runtime=runtime, compile=compile,
    )
    setattr(cls, ATTR, decl)
    setattr(cls, "__gen_worker_handlers__", handlers)
    return cls


def _decorate_function(
    fn: Callable[..., Any],
    *,
    kind: str,
    resources: Resources,
    models: Dict[str, Binding],
    slots: Dict[str, Slot],
    runtime: Optional[str],
    name: Optional[str],
    compile: Optional[Compile] = None,
) -> Callable[..., Any]:
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
            f"@endpoint function {fn.__name__!r}: runtime= requires a class "
            "with setup() (the engine server outlives single calls)."
        )
    decl = EndpointDecl(
        kind=kind, resources=resources, models=models, slots=slots,
        runtime=None, name=(name or fn.__name__), is_function=True,
        compile=compile,
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
    runtime: Optional[str] = ...,
    name: Optional[str] = ...,
    compile: Optional[Compile] = ...,
) -> Callable[[T], T]: ...  # configured @endpoint(...) form


def endpoint(
    target: Optional[T] = None,
    *,
    kind: str = "inference",
    model: Optional[SlotLike] = None,
    models: Optional[Mapping[str, SlotLike]] = None,
    resources: Optional[Resources] = None,
    runtime: Optional[str] = None,
    name: Optional[str] = None,
    compile: Optional[Compile] = None,
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

    def apply(obj: Any) -> Any:
        if inspect.isclass(obj):
            if name is not None:
                raise ValueError("@endpoint name= applies to functions only; "
                                 "class handlers route by method name")
            return _decorate_class(
                obj, kind=kind, resources=resources_value, models=dict(model_map),
                slots=dict(slot_map), runtime=runtime, compile=compile,
            )
        if inspect.isfunction(obj):
            return _decorate_function(
                obj, kind=kind, resources=resources_value, models=dict(model_map),
                slots=dict(slot_map), runtime=runtime, name=name, compile=compile,
            )
        raise TypeError(
            f"@endpoint requires a function or class, got {type(obj).__name__}"
        )

    if target is not None:
        return apply(target)
    return apply


__all__ = ["Compile", "EndpointDecl", "Resources", "SlotLike", "endpoint"]
