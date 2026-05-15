"""Decorator API for class-shaped endpoints (#322 SDK foundation).

Hard cut from the previous function-shaped `@inference` /
`@conversion` decorators. Every endpoint is now a class declaring
lifecycle hooks (setup/warmup/shutdown) plus one-or-more
`@inference.function`-decorated invocable methods.

Two archetypes detected via discovery:
  - **SerialWorker** — sync class. One request fully owns the GPU until done.
    Covers: image/video diffusion, flow-matching audio, 3D.
  - **BatchedWorker** — async class. Many requests share each forward pass
    via continuous batching (vLLM/SGLang). Covers: LLMs, AR TTS.

The axis is autoregressive-vs-not, expressed in parallelization terms.

Lifecycle hooks (SDK calls these):
  - `setup(self, **models) -> None` — required, cheap. Load weights, apply
    torch.compile/cache/quant wrappers.
  - `warmup(self) -> None` — optional, expensive. Run forward(s) at every
    declared `allowed_shapes` to flush compile graphs. Worker stays in
    `warming` startup phase until this returns.
  - `shutdown(self) -> None` — required. Release CUDA graphs, engines, etc.

Inner decorators:
  - `@inference.function(...)` — marks a method as externally invocable
    (route at `org/endpoint/<method-name>`). Carries per-function metadata
    (timeout, allowed_shapes, etc).
  - `@inference.stage(name=..., gpu_class=...)` — marks a method as a
    pipeline stage. Today: in-process method call. Future: SDK can route
    to remote workers for disaggregated inference.

See progress.json #322 for the full design.
"""

from __future__ import annotations

import inspect
import typing
from typing import Any, Callable, Literal, Mapping, Optional, TypeVar, Union, overload

import msgspec

from .binding import Binding, Dispatch, Repo, _qualname

F = TypeVar("F", bound=Callable[..., Any])
C = TypeVar("C", bound=type)


def _force_setattr(obj: Any, name: str, value: Any) -> None:
    msgspec.structs.force_setattr(obj, name, value)


# ============================================================================
# Resources — declared per-class envelope + dynamic cost shape.
# Unchanged from the previous function-shape API.
# ============================================================================


class Resources(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    """Hardware envelope + dynamic cost shape for an endpoint class.

    Static placement envelope (hard gates):
      - ``accelerator``: ``"cuda"`` / ``"none"``.
      - ``requires_gpu``: bool.
      - ``min_vram_gb``: float, GiB.
      - ``min_compute_capability``: float, e.g. ``8.0`` for SM 8.0+.
      - ``required_libraries``: importable Python package names.

    Dynamic cost shape (admission + scheduling):
      - ``vram_must_fit``: ``"full_model"`` or ``"largest_component"``.
      - ``vram_base``: constant VRAM overhead in bytes.
      - ``vram_size_multiplier``: multiplier on ``source.size_facts[vram_must_fit]``.
      - ``vram_scales_with``: payload fields that grow VRAM.
      - ``runtime_scales_with``: payload fields that grow runtime.

    Per-request envelope (#322):
      - ``peak_vram_per_request_gb``: peak VRAM during one request. Scheduler
        uses this to know whether two concurrent requests fit on one GPU.
        SerialWorker endpoints typically set this == ``min_vram_gb`` since
        they don't share the GPU. BatchedWorker endpoints set it much lower
        per-request (the engine manages the shared pool).
    """

    # ----- Static placement envelope (hard gates) -------------------------
    accelerator: Literal["cuda", "none"] | None = None
    requires_gpu: bool | None = None
    min_vram_gb: float | None = None
    min_compute_capability: float | None = None
    required_libraries: tuple[str, ...] = ()

    # ----- Dynamic cost shape (admission + scheduling) --------------------
    vram_must_fit: Literal["full_model", "largest_component"] | None = None
    vram_base: int = 0
    vram_size_multiplier: float = 0.0
    vram_scales_with: tuple[str, ...] = ()
    runtime_scales_with: tuple[str, ...] = ()

    # ----- Per-request envelope (#322) ------------------------------------
    peak_vram_per_request_gb: float | None = None

    # ----- Derived wire-shape field (set in __post_init__) ----------------
    compute_capability: dict[str, str] | None = None

    def __post_init__(self) -> None:
        # Accelerator vocabulary: 'cuda' (GPU endpoints), 'none' (CPU-only
        # endpoints — CPU is the *absence* of an accelerator, not one), or
        # None (unset; treated as 'none' at scheduling time). Empty string
        # normalizes to None for convenience. The 'cpu' / 'gpu' shorthands
        # were dropped (#326) — they're oxymoronic ('cpu' as an accelerator
        # is a contradiction) and they masked typos.
        if self.accelerator is not None:
            accel = str(self.accelerator).strip().lower()
            if accel == "":
                _force_setattr(self, "accelerator", None)
            elif accel in ("none", "cuda"):
                _force_setattr(self, "accelerator", accel)
                if accel == "cuda" and self.requires_gpu is None:
                    _force_setattr(self, "requires_gpu", True)
            elif accel == "cpu":
                raise ValueError(
                    f"accelerator={self.accelerator!r} is invalid — "
                    "CPU endpoints use accelerator='none'. "
                    "Valid values: 'cuda', 'none', or None."
                )
            elif accel == "gpu":
                raise ValueError(
                    f"accelerator={self.accelerator!r} is invalid — "
                    "GPU endpoints use accelerator='cuda'. "
                    "Valid values: 'cuda', 'none', or None."
                )
            else:
                raise ValueError(
                    f"accelerator must be 'none' or 'cuda', got {self.accelerator!r}"
                )

        if self.min_compute_capability is not None:
            val = float(self.min_compute_capability)
            if val <= 0:
                raise ValueError(f"min_compute_capability must be positive, got {val}")
            _force_setattr(self, "min_compute_capability", val)
            _force_setattr(self, "compute_capability", {"min": f"{val:.1f}"})

        if self.min_vram_gb is not None:
            vram = float(self.min_vram_gb)
            if vram <= 0:
                raise ValueError(f"min_vram_gb must be positive, got {vram}")
            _force_setattr(self, "min_vram_gb", vram)

        if self.peak_vram_per_request_gb is not None:
            v = float(self.peak_vram_per_request_gb)
            if v <= 0:
                raise ValueError(
                    f"peak_vram_per_request_gb must be positive, got {v}"
                )
            _force_setattr(self, "peak_vram_per_request_gb", v)

        if self.required_libraries:
            libs = tuple(str(x).strip() for x in self.required_libraries if str(x).strip())
            _force_setattr(self, "required_libraries", libs)

        if self.vram_must_fit not in (None, "full_model", "largest_component"):
            raise ValueError(
                f"vram_must_fit must be 'full_model', 'largest_component', or None; "
                f"got {self.vram_must_fit!r}"
            )
        if self.vram_base < 0:
            raise ValueError(f"vram_base must be >= 0, got {self.vram_base}")
        if self.vram_size_multiplier < 0:
            raise ValueError(
                f"vram_size_multiplier must be >= 0, got {self.vram_size_multiplier}"
            )


# ============================================================================
# Marker structs that decorators attach to class/method objects.
# Discovery reads these to build the manifest.
# ============================================================================


class _FunctionSpec(msgspec.Struct, frozen=True, kw_only=True):
    """Metadata attached to a method by @inference.function."""

    name: str
    timeout_ms: int | None = None
    allowed_shapes: tuple[tuple[int, ...], ...] = ()
    rate_limit_per_invoker: int | None = None
    label: str | None = None
    description: str | None = None


class _StageSpec(msgspec.Struct, frozen=True, kw_only=True):
    """Metadata attached to a method by @inference.stage.

    Forward-compatibility hook for future disaggregated inference. Today
    the SDK runs the method in-process; tomorrow it can route the call to
    a remote worker on a different gpu_class.
    """

    name: str
    gpu_class: Literal["small", "large"] = "large"


class _EndpointClassSpec(msgspec.Struct, frozen=True, kw_only=True):
    """Metadata attached to the class itself by @inference / @training / etc.

    Discovery reads ``_gen_worker_endpoint_spec__`` to identify endpoints.
    """

    kind: Literal["inference", "training", "dataset", "conversion"]
    # Sub-kind for training endpoints (fine-tuning / quantization / etc.).
    sub_kind: str | None = None
    label: str | None = None
    description: str | None = None
    resources: Resources = msgspec.field(default_factory=Resources)
    # Model bindings — Repo / Dispatch — keyed by setup() kwarg name.
    models: Mapping[str, Binding] = msgspec.field(default_factory=dict)
    # Class-level shape constraint; per-method @inference.function may
    # override / add more.
    allowed_shapes: tuple[tuple[int, ...], ...] = ()
    # Engine runtime selector for BatchedWorker — None means SerialWorker.
    runtime: Literal["sglang", "vllm", None] = None
    # ----- SerialWorker cross-request micro-batching (#324) ---------------
    # When BOTH batch_window_ms and max_batch are declared on a SerialWorker
    # endpoint (and per-call latency is short — we trust the tenant), the
    # runtime collects concurrent requests in a time window and fires ONE
    # batched forward call. The tenant's @inference.function method gets
    # `payload` as a LIST when batched. Auto-disabled when any attached cache
    # wrapper has `breaks_cross_request_batching=True` (e.g. TeaCache —
    # nunchaku #597). Validated by TetriServe (https://arxiv.org/html/2510.01565).
    batch_window_ms: int | None = None
    max_batch: int | None = None
    # ----- Distilled-checkpoint awareness (#324) -------------------------
    # When True, the SDK will auto-substitute a Lightning/Turbo/Schnell/Sprint
    # variant of the declared model Repo(...) when one is published in the
    # same repository. The actual ref-resolution happens at tensorhub-resolve
    # time (orchestrator-side); this flag only surfaces tenant intent on the
    # discovered endpoint metadata so the orchestrator can act on it.
    # SLA-aware automatic selection is a follow-up depending on #320.
    prefer_distilled: bool = False


# ============================================================================
# Inner decorators — @inference.function and @inference.stage.
# Implemented as classmethods on a marker class below; the public API exposes
# them via attributes on the outer decorator object.
# ============================================================================


def _function_inner(
    fn: Optional[F] = None,
    *,
    name: Optional[str] = None,
    timeout_ms: Optional[int] = None,
    allowed_shapes: Optional[tuple[tuple[int, ...], ...]] = None,
    rate_limit_per_invoker: Optional[int] = None,
    label: Optional[str] = None,
    description: Optional[str] = None,
) -> Any:
    """Inner decorator marking a method as externally invocable.

    Usable as ``@inference.function`` (bare) or ``@inference.function(...)``.
    """

    def apply(method: F) -> F:
        spec = _FunctionSpec(
            name=name or method.__name__,
            timeout_ms=timeout_ms,
            allowed_shapes=tuple(tuple(s) for s in (allowed_shapes or ())),
            rate_limit_per_invoker=rate_limit_per_invoker,
            label=(label or "").strip() or None,
            description=(description or "").strip() or None,
        )
        setattr(method, "__gen_worker_function_spec__", spec)
        return method

    if fn is not None:
        return apply(fn)
    return apply


_VALID_GPU_CLASSES = ("small", "large")


def _stage_inner(
    fn: Optional[F] = None,
    *,
    name: Optional[str] = None,
    gpu_class: Literal["small", "large"] = "large",
) -> Any:
    """Inner decorator marking a method as a pipeline stage.

    Today: SDK calls the method in-process. Future: SDK may route to a
    remote worker for disaggregated inference. The contract is identical
    to local method calls, so tenant code is portable.

    Usable as ``@inference.stage(name='encode', gpu_class='small')``.

    Validation (raised at decoration time so tenants see the error during
    ``import``, not at bake / dispatch):

      - ``gpu_class`` must be ``'small'`` or ``'large'`` — the orchestrator's
        future remote-dispatch path uses this as a placement hint.
      - ``name`` (or the method name if not supplied) must produce a
        non-empty slug under ``slugify_name`` — same slug rules as
        ``@inference.function`` wire routes.
    """
    # gpu_class is typed ``Literal``, but msgspec.Struct does NOT enforce
    # Literal members at construction time. Validate explicitly so a typo
    # ('medium', 'big') fails fast at decoration time.
    if gpu_class not in _VALID_GPU_CLASSES:
        raise ValueError(
            f"@inference.stage: gpu_class must be one of "
            f"{list(_VALID_GPU_CLASSES)}, got {gpu_class!r}"
        )

    # Lazy import to avoid a circular dep between api.decorators and discovery.
    from gen_worker.discovery.names import slugify_name

    def apply(method: F) -> F:
        raw_name = name if name is not None else method.__name__
        if not isinstance(raw_name, str):
            raise TypeError(
                f"@inference.stage on {method.__name__!r}: name must be a str, "
                f"got {type(raw_name).__name__}"
            )
        slug = slugify_name(raw_name)
        if not slug:
            raise ValueError(
                f"@inference.stage on {method.__name__!r}: name {raw_name!r} "
                "produces an empty slug. Use a name with at least one "
                "alphanumeric character (slug rules: lowercased, _ -> -, "
                "non-[a-z0-9.] -> -)."
            )
        spec = _StageSpec(
            name=slug,
            gpu_class=gpu_class,
        )
        setattr(method, "__gen_worker_stage_spec__", spec)
        return method

    if fn is not None:
        return apply(fn)
    return apply


# ============================================================================
# Class-level validation pipeline.
# ============================================================================


def _payload_field_names(payload_type: type) -> set[str]:
    try:
        return {f.name for f in msgspec.structs.fields(payload_type)}
    except Exception:
        return set()


def _payload_field_type(payload_type: type, field_name: str) -> Any:
    hints: dict[str, Any] = {}
    try:
        hints = typing.get_type_hints(payload_type, include_extras=False)
    except Exception:
        hints = getattr(payload_type, "__annotations__", {}) or {}
    return hints.get(field_name)


def _literal_members(t: Any) -> Optional[tuple[Any, ...]]:
    origin = typing.get_origin(t)
    if origin is Literal:
        return tuple(typing.get_args(t))
    if origin is Union:
        members: list[Any] = []
        for arg in typing.get_args(t):
            if arg is type(None):
                continue
            sub = _literal_members(arg)
            if sub is None:
                return None
            members.extend(sub)
        if members:
            return tuple(members)
    return None


def _validate_setup_signature(cls: type) -> dict[str, Any]:
    """Return the model kwargs of ``setup``. setup must take (self, **models).

    The models keys must match the endpoint-class spec's ``models={}`` dict.
    """
    setup = getattr(cls, "setup", None)
    if setup is None:
        raise ValueError(
            f"@inference class {cls.__name__!r}: missing required `setup` method. "
            "Add `def setup(self, **models): ...`."
        )
    sig = inspect.signature(setup)
    # First param is self. Remaining params are model kwargs.
    params = list(sig.parameters.values())[1:]
    return {p.name: p for p in params}


def _validate_invocable_methods(cls: type) -> list[tuple[str, Callable[..., Any], _FunctionSpec]]:
    """Find @inference.function-decorated methods on the class."""
    out: list[tuple[str, Callable[..., Any], _FunctionSpec]] = []
    for attr_name in dir(cls):
        if attr_name.startswith("_"):
            continue
        member = getattr(cls, attr_name, None)
        if member is None or not callable(member):
            continue
        spec = getattr(member, "__gen_worker_function_spec__", None)
        if spec is not None:
            out.append((attr_name, member, spec))
    if not out:
        raise ValueError(
            f"@inference class {cls.__name__!r}: no @inference.function-decorated "
            "methods found. At least one invocable function is required."
        )
    return out


def _validate_stage_methods(cls: type) -> list[tuple[str, Callable[..., Any], _StageSpec]]:
    """Find @inference.stage-decorated methods. Optional — empty is fine.

    Rejects duplicate stage names within a class: two stages sharing a name
    would clash in the manifest's ``stages`` list and silently shadow each
    other in any future remote-dispatch routing table.
    """
    out: list[tuple[str, Callable[..., Any], _StageSpec]] = []
    seen_names: dict[str, str] = {}  # stage_name -> method attr_name
    for attr_name in dir(cls):
        if attr_name.startswith("_"):
            continue
        member = getattr(cls, attr_name, None)
        if member is None or not callable(member):
            continue
        spec = getattr(member, "__gen_worker_stage_spec__", None)
        if spec is None:
            continue
        prior = seen_names.get(spec.name)
        if prior is not None and prior != attr_name:
            raise ValueError(
                f"@inference class {cls.__name__!r}: duplicate stage name "
                f"{spec.name!r} on methods {prior!r} and {attr_name!r}. "
                "Stage names must be unique within a class (each name maps "
                "to one routable pipeline stage)."
            )
        seen_names[spec.name] = attr_name
        out.append((attr_name, member, spec))
    return out


def _validate_models_against_setup(
    cls_name: str,
    models: Mapping[str, Binding],
    setup_kwargs: Mapping[str, inspect.Parameter],
) -> dict[str, Binding]:
    """Each models={} key must correspond to a setup() kwarg."""
    if not models:
        return {}
    out: dict[str, Binding] = {}
    for key, binding in models.items():
        if not isinstance(binding, (Repo, Dispatch)):
            raise TypeError(
                f"@inference class {cls_name!r}: models[{key!r}] must be Repo or Dispatch; "
                f"got {type(binding).__name__}"
            )
        if key not in setup_kwargs and "kwargs" not in {p.kind.name.lower() for p in setup_kwargs.values()}:
            raise ValueError(
                f"@inference class {cls_name!r}: models[{key!r}] doesn't match any setup() "
                f"parameter. setup signature: {list(setup_kwargs.keys())}"
            )
        out[key] = binding
    return out


# ============================================================================
# Class detection: sync vs async (determines SerialWorker vs BatchedWorker).
# ============================================================================


def _is_async_class(cls: type, function_methods: list[tuple[str, Callable[..., Any], _FunctionSpec]]) -> bool:
    """A class is BatchedWorker (async) if its setup or any @inference.function method is async.

    SerialWorker is the default (sync class).
    """
    setup = getattr(cls, "setup", None)
    if setup is not None and inspect.iscoroutinefunction(setup):
        return True
    for _, method, _ in function_methods:
        if inspect.iscoroutinefunction(method) or inspect.isasyncgenfunction(method):
            return True
    return False


# ============================================================================
# Endpoint-class decorator factories.
# These are used by the @inference / @training / @dataset / @conversion
# public decorators below. Each binds a different `kind=` value.
# ============================================================================


def _make_endpoint_decorator(kind: Literal["inference", "training", "dataset", "conversion"]):
    """Build the @inference / @training / @dataset / @conversion decorator."""

    @overload
    def decorator(cls: C) -> C: ...
    @overload
    def decorator(
        cls: None = None,
        *,
        label: Optional[str] = None,
        description: Optional[str] = None,
        resources: Optional[Resources] = None,
        models: Optional[Mapping[str, Binding]] = None,
        allowed_shapes: Optional[tuple[tuple[int, ...], ...]] = None,
        runtime: Optional[Literal["sglang", "vllm"]] = None,
        sub_kind: Optional[str] = None,
        batch_window_ms: Optional[int] = None,
        max_batch: Optional[int] = None,
        prefer_distilled: bool = False,
        calibration: Optional[Mapping[str, str]] = None,
    ) -> Callable[[C], C]: ...
    def decorator(
        cls: Optional[C] = None,
        *,
        label: Optional[str] = None,
        description: Optional[str] = None,
        resources: Optional[Resources] = None,
        models: Optional[Mapping[str, Binding]] = None,
        allowed_shapes: Optional[tuple[tuple[int, ...], ...]] = None,
        runtime: Optional[Literal["sglang", "vllm"]] = None,
        sub_kind: Optional[str] = None,
        batch_window_ms: Optional[int] = None,
        max_batch: Optional[int] = None,
        prefer_distilled: bool = False,
        calibration: Optional[Mapping[str, str]] = None,
    ) -> Any:
        resources_value: Resources = resources if resources is not None else Resources()
        shapes_value = tuple(tuple(s) for s in (allowed_shapes or ()))

        def apply(target: C) -> C:
            if not inspect.isclass(target):
                raise TypeError(
                    f"@{kind} requires a class. Got {type(target).__name__}. "
                    "Migration: rewrite your function as a class with setup/generate/shutdown methods. "
                    "See progress.json #322 for migration guide."
                )

            cls_name = target.__name__
            setup_kwargs = _validate_setup_signature(target)
            function_methods = _validate_invocable_methods(target)
            stage_methods = _validate_stage_methods(target)
            validated_models = _validate_models_against_setup(
                cls_name, models or {}, setup_kwargs
            )

            is_async = _is_async_class(target, function_methods)
            archetype = "BatchedWorker" if is_async else "SerialWorker"

            if runtime is not None and not is_async:
                raise ValueError(
                    f"@{kind} class {cls_name!r}: runtime={runtime!r} requires async methods "
                    f"(BatchedWorker shape). Make setup/generate async, or remove runtime=."
                )
            if runtime is None and is_async and kind == "inference":
                # Async class without runtime= is allowed but unusual — warn via attribute,
                # not exception, so tenants can run async without a continuous-batching engine.
                pass

            # ----- Cross-request micro-batching (#324) ------------------
            # Validate batch_window_ms + max_batch — both must be declared
            # together; both apply only to SerialWorker (sync) endpoints.
            if batch_window_ms is not None or max_batch is not None:
                if (batch_window_ms is None) != (max_batch is None):
                    raise ValueError(
                        f"@{kind} class {cls_name!r}: batch_window_ms and max_batch must be "
                        f"declared together (got batch_window_ms={batch_window_ms!r}, "
                        f"max_batch={max_batch!r})."
                    )
                if is_async:
                    raise ValueError(
                        f"@{kind} class {cls_name!r}: batch_window_ms / max_batch are "
                        "SerialWorker-only — BatchedWorker (continuous-batching engine) "
                        "manages its own scheduler. Remove runtime= or remove batch kwargs."
                    )
                if batch_window_ms is not None and batch_window_ms <= 0:
                    raise ValueError(
                        f"@{kind} class {cls_name!r}: batch_window_ms must be > 0 "
                        f"(got {batch_window_ms})."
                    )
                if max_batch is not None and max_batch < 2:
                    raise ValueError(
                        f"@{kind} class {cls_name!r}: max_batch must be >= 2 to make "
                        f"cross-request batching meaningful (got {max_batch})."
                    )

            spec = _EndpointClassSpec(
                kind=kind,
                sub_kind=(sub_kind or "").strip() or None,
                label=(label or "").strip() or None,
                description=(description or "").strip() or None,
                resources=resources_value,
                models=validated_models,
                allowed_shapes=shapes_value,
                runtime=runtime,
                batch_window_ms=batch_window_ms,
                max_batch=max_batch,
                prefer_distilled=bool(prefer_distilled),
            )

            # Attach metadata for discovery.
            setattr(target, "__gen_worker_endpoint_spec__", spec)
            setattr(target, "__gen_worker_archetype__", archetype)
            setattr(target, "__gen_worker_function_methods__", function_methods)
            setattr(target, "__gen_worker_stage_methods__", stage_methods)
            # #332: per-scheme calibration policy for @conversion / @training
            # quantization endpoints. Free-form dict (scheme → 'required' |
            # 'beneficial' | 'unsupported'); the worker enforces this at
            # dispatch (rejects datasets paired with `unsupported` schemes).
            if calibration:
                setattr(target, "__gen_worker_calibration__", dict(calibration))
            return target

        if cls is not None:
            return apply(cls)
        return apply

    return decorator


# ============================================================================
# Public decorators.
# `inference` is a callable that ALSO has `.function` and `.stage` attributes.
# ============================================================================


class _InferenceDecorator:
    """Callable + namespace for `@inference` and its inner decorators.

    Usage:
        @inference(models={...}, resources=...)
        class MyEndpoint:
            @inference.function
            def generate(self, ctx, payload): ...

            @inference.stage(name='encode', gpu_class='small')
            def encode(self, prompt): ...
    """

    def __init__(self) -> None:
        self._outer = _make_endpoint_decorator("inference")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._outer(*args, **kwargs)

    @staticmethod
    def function(
        fn: Optional[F] = None,
        *,
        name: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        allowed_shapes: Optional[tuple[tuple[int, ...], ...]] = None,
        rate_limit_per_invoker: Optional[int] = None,
        label: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Any:
        return _function_inner(
            fn,
            name=name,
            timeout_ms=timeout_ms,
            allowed_shapes=allowed_shapes,
            rate_limit_per_invoker=rate_limit_per_invoker,
            label=label,
            description=description,
        )

    @staticmethod
    def stage(
        fn: Optional[F] = None,
        *,
        name: Optional[str] = None,
        gpu_class: Literal["small", "large"] = "large",
    ) -> Any:
        return _stage_inner(fn, name=name, gpu_class=gpu_class)


class _TrainingDecorator:
    """Callable + namespace for `@training`, mirrors `@inference`."""

    def __init__(self) -> None:
        self._outer = _make_endpoint_decorator("training")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._outer(*args, **kwargs)

    @staticmethod
    def function(*args: Any, **kwargs: Any) -> Any:
        return _function_inner(*args, **kwargs)

    @staticmethod
    def stage(*args: Any, **kwargs: Any) -> Any:
        return _stage_inner(*args, **kwargs)


class _DatasetDecorator:
    """Callable + namespace for `@dataset` (dataset-generation endpoints)."""

    def __init__(self) -> None:
        self._outer = _make_endpoint_decorator("dataset")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._outer(*args, **kwargs)

    @staticmethod
    def function(*args: Any, **kwargs: Any) -> Any:
        return _function_inner(*args, **kwargs)

    @staticmethod
    def stage(*args: Any, **kwargs: Any) -> Any:
        return _stage_inner(*args, **kwargs)


class _ConversionDecorator:
    """Callable + namespace for `@conversion(kind='format-conversion'|...)`."""

    def __init__(self) -> None:
        self._outer = _make_endpoint_decorator("conversion")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._outer(*args, **kwargs)

    @staticmethod
    def function(*args: Any, **kwargs: Any) -> Any:
        return _function_inner(*args, **kwargs)

    @staticmethod
    def stage(*args: Any, **kwargs: Any) -> Any:
        return _stage_inner(*args, **kwargs)


# Singletons exported as the public API.
inference = _InferenceDecorator()
training = _TrainingDecorator()
dataset = _DatasetDecorator()
conversion = _ConversionDecorator()


# ============================================================================
# Hard-cut migration shim — old decorator names import-error with a clear
# message pointing at the new shape.
# ============================================================================


def _migration_error(old_name: str, new_name: str) -> Callable[..., Any]:
    """Return a fake decorator that raises a migration error if called."""

    def stub(*args: Any, **kwargs: Any) -> Any:
        raise ImportError(
            f"`{old_name}` was removed in gen-worker SDK foundation (#322). "
            f"Use `{new_name}` with a class instead of a function. Migration:\n"
            f"  Before:\n"
            f"    @{old_name}(models={{'pipe': repo}})\n"
            f"    def my_fn(ctx, pipe, payload): ...\n"
            f"  After:\n"
            f"    @{new_name}(models={{'pipe': repo}})\n"
            f"    class MyEndpoint:\n"
            f"        def setup(self, pipe):\n"
            f"            self.pipe = pipe\n"
            f"        @{new_name}.function\n"
            f"        def my_fn(self, ctx, payload): ...\n"
            f"        def shutdown(self): pass\n"
            f"See progress.json #322 + #328 for the full migration guide."
        )

    return stub


inference_function = _migration_error("inference_function", "inference")
training_function = _migration_error("training_function", "training")
realtime_function = _migration_error("realtime_function", "inference")


__all__ = [
    "Resources",
    "inference",
    "training",
    "dataset",
    "conversion",
    # Hard-cut migration stubs (will raise ImportError on use):
    "inference_function",
    "training_function",
    "realtime_function",
    # Internal specs (for discovery):
    "_EndpointClassSpec",
    "_FunctionSpec",
    "_StageSpec",
]
