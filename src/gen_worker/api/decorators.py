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
* ``variants={name: (binding, Resources)}`` stamps one separately-routable,
  separately-placeable function per variant from a single handler body. With
  ``models={}`` the variant binding swaps the FIRST declared slot; remaining
  aux slots (e.g. a shared VAE) are inherited by every variant.
"""

from __future__ import annotations

import inspect
import typing
from typing import Any, Callable, Literal, Mapping, Optional, Sequence, TypeVar

import msgspec

from .binding import BINDING_TYPES, Binding

T = TypeVar("T")

KINDS = ("inference", "training", "dataset", "conversion")
RESERVED_METHODS = frozenset({"setup", "warmup", "shutdown"})


class Resources(msgspec.Struct, frozen=True, omit_defaults=True):
    """Hardware envelope for one function: ``Resources(gpu, vram_gb,
    compute_capability, libraries)``.

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
    """

    gpu: bool = False
    vram_gb: float | None = None
    compute_capability: float | None = None
    libraries: tuple[str, ...] = ()
    strict_vram: bool = False

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


class Variant(msgspec.Struct, frozen=True):
    """One resolved ``variants={}`` row: binding + optional Resources override."""

    name: str
    binding: Any
    resources: Resources | None = None


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
    variants: tuple[Variant, ...] = ()
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


def _normalize_models(
    model: Optional[Binding], models: Optional[Mapping[str, Binding]]
) -> dict[str, Binding]:
    if model is not None and models is not None:
        raise ValueError("@endpoint: pass model= OR models=, not both")
    if model is not None:
        if not isinstance(model, BINDING_TYPES):
            raise TypeError(
                f"@endpoint(model=) must be a HF/Hub/Civitai/ModelScope binding, "
                f"got {type(model).__name__}"
            )
        # Single-binding shorthand: the slot name is resolved from the
        # setup()/handler parameter name at class validation time.
        return {"": model}
    out: dict[str, Binding] = {}
    for key, b in (models or {}).items():
        k = str(key or "").strip()
        if not k or not k.isidentifier():
            raise ValueError(f"@endpoint models key {key!r} must be an identifier")
        if not isinstance(b, BINDING_TYPES):
            raise TypeError(
                f"@endpoint models[{k!r}] must be a HF/Hub/Civitai/ModelScope "
                f"binding, got {type(b).__name__}"
            )
        out[k] = b
    return out


def _normalize_variants(
    variants: Optional[Mapping[str, Any]], default_resources: Resources
) -> tuple[Variant, ...]:
    if not variants:
        return ()
    rows: list[Variant] = []
    seen: set[str] = set()
    for raw_name, value in variants.items():
        name = str(raw_name or "").strip()
        if not name:
            raise ValueError("@endpoint variants: empty variant name")
        if name in seen:
            raise ValueError(f"@endpoint variants: duplicate name {name!r}")
        seen.add(name)
        res: Resources | None = None
        if isinstance(value, tuple):
            if len(value) != 2 or not isinstance(value[1], Resources):
                raise TypeError(
                    f"@endpoint variants[{name!r}] must be a binding or "
                    "(binding, Resources) pair"
                )
            binding, res = value
        else:
            binding = value
        if not isinstance(binding, BINDING_TYPES):
            raise TypeError(
                f"@endpoint variants[{name!r}]: expected a HF/Hub/Civitai/"
                f"ModelScope binding, got {type(binding).__name__}"
            )
        rows.append(Variant(name=name, binding=binding, resources=res or default_resources))
    return tuple(rows)


def _literal_members(t: Any) -> Optional[tuple[Any, ...]]:
    if typing.get_origin(t) is Literal:
        return tuple(typing.get_args(t))
    return None


def _validate_variant_literal(cls_name: str, method: Callable[..., Any], names: Sequence[str]) -> None:
    """If the handler payload declares a ``variant`` field, it must be a
    ``Literal[...]`` whose members are a subset of the declared variant names
    (build-time validation of the discriminator, carried over from the old
    dispatch() Literal check)."""
    try:
        hints = typing.get_type_hints(method)
        params = _handler_params(method, is_method=True)
        payload_type = hints.get(params[1].name)
        field_hints = typing.get_type_hints(payload_type)
    except Exception:
        return
    field_type = field_hints.get("variant")
    if field_type is None:
        return
    members = _literal_members(field_type)
    if members is None:
        raise ValueError(
            f"@endpoint class {cls_name!r}: payload field 'variant' must be "
            f"Literal[...]-typed so variant names are validated (got {field_type!r})."
        )
    unknown = sorted({str(m) for m in members} - set(names))
    if unknown:
        raise ValueError(
            f"@endpoint class {cls_name!r}: payload 'variant' Literal member(s) "
            f"{unknown} match no declared variant (variants: {sorted(names)})."
        )


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
    for attr, member in out:
        _validate_handler_shape(f"{cls.__name__}.{attr}", member, is_method=True)
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
    models: dict[str, Binding],
    handlers: list[tuple[str, Callable[..., Any]]],
) -> dict[str, Binding]:
    """Resolve the ``model=`` shorthand's slot name from setup()/handler params."""
    if "" not in models:
        return models
    binding = models.pop("")
    setup_kwargs = _setup_params(cls)
    if setup_kwargs:
        named = [
            n for n, p in setup_kwargs.items()
            if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            and not _is_server_handle_param(p)
        ]
        if len(named) == 1:
            models[named[0]] = binding
            return models
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
        models[injected.pop()] = binding
        return models
    raise ValueError(
        f"@endpoint class {cls.__name__!r}: model= with no setup() needs "
        f"exactly one injected handler parameter after (ctx, payload) to name "
        f"the slot (found {sorted(injected) or 'none'})."
    )


def _validate_class_models(cls: type, models: dict[str, Binding]) -> None:
    """Model slots must be consumable: by setup() params (stateful) or by
    handler params (per-call injection when there is no setup)."""
    if not models:
        return
    setup_kwargs = _setup_params(cls)
    if setup_kwargs is not None:
        has_var_kw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in setup_kwargs.values()
        )
        missing = [k for k in models if k not in setup_kwargs and not has_var_kw]
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
    models: dict[str, Binding],
    variants: Optional[Mapping[str, Any]],
    runtime: Optional[str],
    compile: Optional[Compile] = None,
) -> type:
    handlers = _find_handler_methods(cls)
    for attr, member in handlers:
        _reject_producer_generator(f"{cls.__name__}.{attr}", member, kind)
    models = _resolve_single_slot(cls, models, handlers)
    _validate_class_models(cls, models)

    variant_rows = _normalize_variants(variants, resources)
    if variant_rows:
        if len(handlers) != 1:
            raise ValueError(
                f"@endpoint class {cls.__name__!r}: variants= requires exactly "
                f"ONE handler method to fan out over (found {len(handlers)})."
            )
        # Multi-slot classes: variants swap the FIRST declared slot; the
        # remaining (aux) slots are shared across all variants.
        _validate_variant_literal(
            cls.__name__, handlers[0][1], [v.name for v in variant_rows]
        )

    if runtime is not None and runtime not in ("vllm", "llama-server"):
        raise ValueError(
            f"@endpoint class {cls.__name__!r}: runtime must be 'vllm' or "
            f"'llama-server', got {runtime!r}"
        )

    decl = EndpointDecl(
        kind=kind, resources=resources, models=models,
        variants=variant_rows, runtime=runtime, compile=compile,
    )
    setattr(cls, ATTR, decl)
    setattr(cls, "__gen_worker_handlers__", handlers)
    return cls


def _decorate_function(
    fn: Callable[..., Any],
    *,
    kind: str,
    resources: Resources,
    models: dict[str, Binding],
    runtime: Optional[str],
    name: Optional[str],
    compile: Optional[Compile] = None,
) -> Callable[..., Any]:
    _validate_handler_shape(fn.__name__, fn, is_method=False)
    _reject_producer_generator(fn.__name__, fn, kind)
    if "" in models:
        binding = models.pop("")
        injected = [p.name for p in _handler_params(fn, is_method=False)[2:]]
        if len(injected) != 1:
            raise ValueError(
                f"@endpoint function {fn.__name__!r}: model= needs exactly one "
                f"injected parameter after (ctx, payload) to name the slot "
                f"(found {injected or 'none'})."
            )
        models[injected[0]] = binding
    if runtime is not None:
        raise ValueError(
            f"@endpoint function {fn.__name__!r}: runtime= requires a class "
            "with setup() (the engine server outlives single calls)."
        )
    decl = EndpointDecl(
        kind=kind, resources=resources, models=models,
        runtime=None, name=(name or fn.__name__), is_function=True,
        compile=compile,
    )
    setattr(fn, ATTR, decl)
    return fn


def endpoint(
    target: Optional[T] = None,
    *,
    kind: str = "inference",
    model: Optional[Binding] = None,
    models: Optional[Mapping[str, Binding]] = None,
    resources: Optional[Resources] = None,
    variants: Optional[Mapping[str, Any]] = None,
    runtime: Optional[str] = None,
    name: Optional[str] = None,
    compile: Optional[Compile] = None,
) -> Any:
    """The one endpoint decorator. See the module docstring for shapes."""
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
    model_map = _normalize_models(model, models)

    def apply(obj: Any) -> Any:
        if inspect.isclass(obj):
            if name is not None:
                raise ValueError("@endpoint name= applies to functions only; "
                                 "class handlers route by method name")
            return _decorate_class(
                obj, kind=kind, resources=resources_value, models=model_map,
                variants=variants, runtime=runtime, compile=compile,
            )
        if inspect.isfunction(obj):
            if variants:
                raise ValueError(
                    f"@endpoint function {obj.__name__!r}: variants= requires a "
                    "class (each variant needs its own setup state)."
                )
            return _decorate_function(
                obj, kind=kind, resources=resources_value, models=dict(model_map),
                runtime=runtime, name=name, compile=compile,
            )
        raise TypeError(
            f"@endpoint requires a function or class, got {type(obj).__name__}"
        )

    if target is not None:
        return apply(target)
    return apply


__all__ = ["Compile", "EndpointDecl", "Resources", "Variant", "endpoint"]
