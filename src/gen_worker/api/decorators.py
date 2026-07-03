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
  - `shutdown(self) -> None` — OPTIONAL. A missing hook is a no-op: the
    runtime calls it only at process end (SIGTERM / Ctrl-C / drain), and
    `getattr(inst, "shutdown", None)` + callable already treats absence as
    nothing-to-do. For the common DI-injected endpoint it is pure ceremony —
    the model is framework-owned and freed by the model cache, and the OS
    reclaims VRAM on exit. Define it ONLY when you hold non-DI resources that
    need explicit release: engine subprocesses, CUDA graphs, threads, network
    connections, torch.distributed groups, scratch dirs. `setup` stays
    required.

Inner decorators:
  - `@invocable(...)` / `@inference.function(...)` — marks a method as
    externally invocable (route at `org/endpoint/<method-name>`). Carries
    per-function metadata (timeout, allowed_shapes, etc). `@invocable` is the
    kind-agnostic marker; the per-kind `<kind>.function` aliases remain for
    backward compatibility.
"""

from __future__ import annotations

import inspect
import typing
from typing import Any, Callable, Literal, Mapping, Optional, TypeVar, Union, overload

import msgspec

from .binding import Binding, Dispatch, Repo

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

        # ----- accelerator='none' + GPU resource axes is self-contradictory.
        # CPU-only endpoints (accelerator='none') legitimately exist — small
        # flow-matching audio, CPU classifiers — but they MUST NOT also
        # declare GPU resource axes. The combination is a copy/paste typo
        # (most often: a CPU port of a GPU endpoint where the resources
        # block wasn't pruned) and silently misroutes endpoints. Fail fast
        # at decoration time.
        if self.accelerator == "none":
            offenders: list[str] = []
            if self.requires_gpu is True:
                offenders.append(f"requires_gpu={self.requires_gpu!r}")
            if self.min_vram_gb is not None:
                offenders.append(f"min_vram_gb={self.min_vram_gb!r}")
            if self.min_compute_capability is not None:
                offenders.append(
                    f"min_compute_capability={self.min_compute_capability!r}"
                )
            if offenders:
                raise ValueError(
                    f"Resources(accelerator='none') cannot also declare "
                    f"{'/'.join(offenders)}. Use accelerator='cuda' for "
                    "GPU endpoints, or drop the GPU resource axes for "
                    "accelerator='none'."
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


class Case(msgspec.Struct, frozen=True, kw_only=True):
    """One row of an ``@inference(parametrize=[...])`` fan-out table (#339 §2).

    A multi-variant endpoint (e.g. the FLUX base+turbo x bf16/fp8/nvfp4 grid)
    is a set of near-identical classes that differ only by
    ``(function_name, Resources, model_ref, input_struct)`` — the body is the
    same shared ``_generate``. ``parametrize=`` stamps one separately-routable
    function per ``Case`` from a SINGLE class + single ``@invocable`` body, so
    authors stop hand-writing N near-identical classes.

    Each row becomes its own discovered function with its own placement
    (``resources``), model binding (``model``), and input type (``input``),
    all backed by the one decorated method body.

    Fields:
      - ``name``: the routable function name (slugified for the wire route).
      - ``resources``: per-row hardware envelope. Falls back to the class-level
        ``resources=`` when omitted.
      - ``model``: per-row model binding (``Repo`` / ``HFRepo`` / ...). When the
        class declares a single-slot ``models={...}`` the row's ``model``
        overrides that slot's binding; placement still resolves per row.
      - ``input``: per-row payload ``msgspec.Struct`` type. Falls back to the
        decorated method's declared payload annotation when omitted.
    """

    name: str
    resources: Resources | None = None
    model: Any = None
    input: Any = None


class _FunctionSpec(msgspec.Struct, frozen=True, kw_only=True):
    """Metadata attached to a method by @inference.function."""

    name: str
    timeout_ms: int | None = None
    allowed_shapes: tuple[tuple[int, ...], ...] = ()
    label: str | None = None
    description: str | None = None


class _BatchedInferenceFunctionSpec(msgspec.Struct, frozen=True, kw_only=True):
    """Metadata attached to a method by @batched_inference.function (#273).

    Parallels ``_FunctionSpec`` but for the LLM-serving BatchedWorker shape:
    the method MUST be an async generator yielding
    ``IncrementalTokenDelta | Done | Error`` signals. No ``allowed_shapes``
    field — batch-shape sweeping is meaningless for autoregressive engines
    (vLLM / SGLang manage their own continuous batching).
    """

    name: str
    timeout_ms: int | None = None
    label: str | None = None
    description: str | None = None


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
    # ----- Function fan-out (#339 §2) -------------------------------------
    # When non-empty, the class hosts a SINGLE @invocable body that is stamped
    # into one separately-routable function per Case row (distinct
    # function_name / Resources / model binding / input type). Empty = plain
    # @inference class (one function per @invocable method, the escape hatch).
    parametrize: tuple[Case, ...] = ()


# ============================================================================
# Inner decorators — @inference.function.
# Implemented as classmethods on a marker class below; the public API exposes
# them via attributes on the outer decorator object.
# ============================================================================


def _function_inner(
    fn: Optional[F] = None,
    *,
    name: Optional[str] = None,
    timeout_ms: Optional[int] = None,
    allowed_shapes: Optional[tuple[tuple[int, ...], ...]] = None,
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
            label=(label or "").strip() or None,
            description=(description or "").strip() or None,
        )
        setattr(method, "__gen_worker_function_spec__", spec)
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


def _validate_dispatch_literals(
    cls_name: str,
    validated_models: Mapping[str, Binding],
    function_methods: list[tuple[str, Callable[..., Any], "_FunctionSpec"]],
) -> None:
    """Enforce the documented ``dispatch()`` contract (api/binding.py): the
    discriminator ``field`` must be a ``Literal[...]``-typed member of the
    consuming handler's payload struct, and every table key must be one of
    the Literal's members. Skips handlers whose payload type is not an
    introspectable msgspec Struct."""
    dispatch_slots = {k: b for k, b in validated_models.items() if isinstance(b, Dispatch)}
    if not dispatch_slots:
        return
    for attr_name, method, _spec in function_methods:
        try:
            sig = inspect.signature(method)
        except (TypeError, ValueError):
            continue
        try:
            hints = typing.get_type_hints(method)
        except Exception:
            hints = {}
        ps = [p for p in sig.parameters.values() if p.name != "self"]
        if len(ps) < 2:
            continue
        consumed = {p.name for p in ps[2:]} & set(dispatch_slots)
        if not consumed:
            continue
        payload_type = hints.get(ps[1].name, ps[1].annotation)
        fields = _payload_field_names(payload_type)
        if not fields:
            continue
        for slot in sorted(consumed):
            d = dispatch_slots[slot]
            if d.field not in fields:
                raise ValueError(
                    f"@inference class {cls_name!r}: dispatch slot {slot!r} "
                    f"discriminates on payload field {d.field!r}, which does not "
                    f"exist on {payload_type.__name__!r} (fields: {sorted(fields)})."
                )
            field_type = _payload_field_type(payload_type, d.field)
            members = _literal_members(field_type)
            if members is None:
                raise ValueError(
                    f"@inference class {cls_name!r}: dispatch field {d.field!r} on "
                    f"{payload_type.__name__!r} must be Literal[...]-typed so table "
                    f"keys are validated (got {field_type!r})."
                )
            unknown = sorted(set(d.table) - {str(m) for m in members})
            if unknown:
                raise ValueError(
                    f"@inference class {cls_name!r}: dispatch table key(s) {unknown} "
                    f"for slot {slot!r} are not members of Literal{list(members)} "
                    f"on payload field {d.field!r}."
                )


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


def _validate_models_against_setup(
    cls_name: str,
    models: Mapping[str, Binding],
    setup_kwargs: Mapping[str, inspect.Parameter],
) -> dict[str, Binding]:
    """Classify each models={} slot as FIXED or DISPATCH and validate placement.

    Issue #337 contract — ``setup()`` builds what is shared and load-once;
    ``dispatch``/variant slots are resolved PER-REQUEST and injected into the
    HANDLER, not setup(). Therefore:

    - **FIXED slot** (a plain ``Repo``/``HFRepo``): loaded once at startup, so
      it MUST correspond to a ``setup()`` kwarg (or ``**kwargs``). Same as the
      historical contract.
    - **DISPATCH slot** (a ``Dispatch``, incl. a dispatch over
      ``SharedBase.variant(...)``): resolved per request. It MUST NOT appear as
      a ``setup()`` parameter — that is the motivating crash (``setup(self,
      pipeline)`` on a per-request slot called with no kwargs ->
      ``TypeError: setup() missing 1 required positional argument``). Surface
      it as a clear BUILD-TIME error instead of a runtime crash.

    An explicit named ``setup()`` param for a dispatch slot is the error; a
    bare ``def setup(self): ...`` (or ``**kwargs``) is correct and accepted.
    """
    if not models:
        return {}
    has_var_kwargs = "kwargs" in {p.kind.name.lower() for p in setup_kwargs.values()}
    out: dict[str, Binding] = {}
    for key, binding in models.items():
        if not isinstance(binding, (Repo, Dispatch)):
            raise TypeError(
                f"@inference class {cls_name!r}: models[{key!r}] must be Repo or Dispatch; "
                f"got {type(binding).__name__}"
            )
        if isinstance(binding, Dispatch):
            # Per-request slot: must NOT be a named setup() parameter. A
            # **kwargs catch-all on setup() does NOT count as "naming" the
            # slot, so it is allowed.
            if key in setup_kwargs:
                raise ValueError(
                    f"@inference class {cls_name!r}: models[{key!r}] is a per-request "
                    f"dispatch slot, but setup() declares a {key!r} parameter. A "
                    "dispatch/variant slot is resolved per REQUEST and injected into "
                    "the handler — it is NOT available at setup() time. Remove "
                    f"{key!r} from setup() and accept it on the handler instead:\n"
                    f"    def setup(self): ...                       # build shared base only\n"
                    f"    def generate(self, ctx, payload, {key}): ...  # SDK injects per request\n"
                    "(issue #337)"
                )
            out[key] = binding
            continue
        # Fixed slot: loaded once → must be a setup() kwarg (or **kwargs).
        if key not in setup_kwargs and not has_var_kwargs:
            raise ValueError(
                f"@inference class {cls_name!r}: models[{key!r}] (a fixed model) doesn't "
                f"match any setup() parameter. setup signature: {list(setup_kwargs.keys())}"
            )
        out[key] = binding
    return out

def _validate_handler_injection(
    cls_name: str,
    validated_models: Mapping[str, Binding],
    function_methods: list[tuple[str, Callable[..., Any], "_FunctionSpec"]],
    parametrize_value: tuple["Case", ...],
) -> None:
    """Validate per-request model slots against handler signatures (#337).

    Hard errors, by design (no bypass):
      * Every extra handler parameter (after ctx, payload) MUST correspond to
        an injectable model slot — an unmatched parameter can never be filled
        and would crash at call time (this is the #337 dispatch bug, caught at
        import instead of mid-request).
      * Every per-request (dispatch) slot MUST be consumed by at least one
        handler — a declared slot no handler accepts is dead.

    Fixed (Repo) slots are injected into setup() and validated by
    ``_validate_models_against_setup``; a non-parametrized handler that names a
    fixed slot therefore counts as an unmatched parameter here.

    Parametrize fan-out hosts ONE body with a single model slot named
    ``next(iter(models), "model")`` (mirrors discover.py). When the class
    declares that slot the name is authoritative and enforced; when it does not
    (slot supplied only per-Case), the name is not knowable here, so we enforce
    only the unsatisfiable case of more than one injected parameter.
    """
    RULE = (
        "static bindings are injected into setup(); dispatch bindings are "
        "injected into the handler per request"
    )

    def extra_params(method: Callable[..., Any]) -> Optional[list[str]]:
        try:
            sig = inspect.signature(method)
        except (TypeError, ValueError):
            return None  # uninspectable -> cannot prove a mismatch; skip
        ps = [p for p in sig.parameters.values() if p.name != "self"]
        if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in ps):
            return None  # **kwargs absorbs anything -> cannot prove a mismatch
        return [
            p.name for p in ps[2:]  # after ctx, payload
            if p.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]

    if parametrize_value:
        attr_name, method, _ = function_methods[0]
        names = extra_params(method)
        if names is None:
            return
        if validated_models:
            slot = next(iter(validated_models))
            for n in names:
                if n != slot:
                    raise ValueError(
                        f"@inference class {cls_name!r}: parametrized handler "
                        f"{attr_name!r} declares parameter {n!r}, but the single "
                        f"model slot is {slot!r}. ({RULE}.)"
                    )
        elif len(names) > 1:
            raise ValueError(
                f"@inference class {cls_name!r}: parametrized handler "
                f"{attr_name!r} declares {len(names)} injected parameters "
                f"{names}, but a parametrize fan-out has a single model slot. "
                f"({RULE}.)"
            )
        return

    dispatch_slots = {
        k for k, b in validated_models.items() if isinstance(b, Dispatch)
    }
    used: set[str] = set()
    for attr_name, method, _ in function_methods:
        names = extra_params(method)
        if names is None:
            used |= dispatch_slots  # cannot prove non-use; treat as satisfied
            continue
        for n in names:
            if n not in dispatch_slots:
                raise ValueError(
                    f"@inference class {cls_name!r}: handler {attr_name!r} "
                    f"declares parameter {n!r}, which matches no per-request "
                    f"(dispatch) model slot. Dispatch slot(s): "
                    f"{sorted(dispatch_slots) or '(none)'}. If {n!r} is a fixed "
                    f"model, receive it in setup() instead. ({RULE}.)"
                )
            used.add(n)
    missing = sorted(dispatch_slots - used)
    if missing:
        raise ValueError(
            f"@inference class {cls_name!r}: dispatch model slot(s) {missing} "
            f"are declared but no handler accepts them. Add a matching "
            f"parameter after (ctx, payload) on a handler. ({RULE}.)"
        )

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
        calibration: Optional[Mapping[str, str]] = None,
        parametrize: Optional[typing.Sequence[Case]] = None,
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
        calibration: Optional[Mapping[str, str]] = None,
        parametrize: Optional[typing.Sequence[Case]] = None,
    ) -> Any:
        resources_value: Resources = resources if resources is not None else Resources()
        shapes_value = tuple(tuple(s) for s in (allowed_shapes or ()))

        def apply(target: C) -> C:
            if not inspect.isclass(target):
                raise TypeError(
                    f"@{kind} requires a class. Got {type(target).__name__}. "
                    "Rewrite your function as a class with setup + @invocable "
                    "methods (see the README or examples/marco-polo)."
                )

            cls_name = target.__name__
            setup_kwargs = _validate_setup_signature(target)
            function_methods = _validate_invocable_methods(target)
            validated_models = _validate_models_against_setup(
                cls_name, models or {}, setup_kwargs
            )

            # ----- Function fan-out via parametrize= (#339 §2) ------------
            # One class + one @invocable body, stamped into N routable
            # functions. Validate the table here so authors see errors at
            # import time, not at discovery/dispatch.
            parametrize_value: tuple[Case, ...] = ()
            if parametrize:
                if len(function_methods) != 1:
                    raise ValueError(
                        f"@{kind} class {cls_name!r}: parametrize= requires "
                        f"exactly ONE @invocable method body to fan out over "
                        f"(found {len(function_methods)}). Each Case row stamps "
                        "that single body into its own routable function."
                    )
                seen_case_names: set[str] = set()
                rows: list[Case] = []
                for i, case in enumerate(parametrize):
                    if not isinstance(case, Case):
                        raise TypeError(
                            f"@{kind} class {cls_name!r}: parametrize[{i}] must "
                            f"be a gen_worker.Case, got {type(case).__name__}."
                        )
                    cname = (case.name or "").strip()
                    if not cname:
                        raise ValueError(
                            f"@{kind} class {cls_name!r}: parametrize[{i}] has "
                            "an empty name; each Case needs a unique function name."
                        )
                    if cname in seen_case_names:
                        raise ValueError(
                            f"@{kind} class {cls_name!r}: duplicate parametrize "
                            f"Case name {cname!r}; each row must route to a "
                            "distinct function."
                        )
                    seen_case_names.add(cname)
                    if case.model is not None and not isinstance(
                        case.model, (Repo, Dispatch)
                    ):
                        raise TypeError(
                            f"@{kind} class {cls_name!r}: parametrize[{i}].model "
                            f"must be a Repo/Dispatch binding, got "
                            f"{type(case.model).__name__}."
                        )
                    rows.append(case)
                parametrize_value = tuple(rows)

            # #337 contract: per-request model slots must line up with handler
            # signatures. Validate at import so a mismatch fails here with a
            # teaching message instead of crashing mid-request.
            _validate_handler_injection(
                cls_name, validated_models, function_methods, parametrize_value
            )
            # binding.py dispatch() contract: discriminator field is Literal-
            # typed on the payload and table keys are Literal members.
            _validate_dispatch_literals(cls_name, validated_models, function_methods)

            is_async = _is_async_class(target, function_methods)
            # #345 Improvement B: an async class is BatchedWorker ONLY when it
            # declares a continuous-batching engine via runtime= (sglang/vllm).
            # An async @inference class WITHOUT runtime= is now a SerialWorker
            # whose handlers run on the shared asyncio loop (_batched_loop) via
            # run_coroutine_threadsafe — high-concurrency I/O without a third
            # archetype. Sync classes are SerialWorker (ThreadPoolExecutor).
            # training/dataset/conversion stay sync-only (handled below).
            if runtime is not None:
                archetype = "BatchedWorker"
            else:
                archetype = "SerialWorker"

            if runtime is not None and not is_async:
                raise ValueError(
                    f"@{kind} class {cls_name!r}: runtime={runtime!r} requires async methods "
                    f"(BatchedWorker shape). Make setup/generate async, or remove runtime=."
                )
            # #345 Improvement B: async @inference without runtime= is a
            # first-class SerialWorker async endpoint (no warning needed). For
            # non-inference kinds (training/dataset/conversion) async remains
            # unsupported and is rejected at registration in the worker.

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
                parametrize=parametrize_value,
            )

            # Attach metadata for discovery.
            setattr(target, "__gen_worker_endpoint_spec__", spec)
            setattr(target, "__gen_worker_archetype__", archetype)
            setattr(target, "__gen_worker_function_methods__", function_methods)
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
# `inference` is a callable that ALSO has a `.function` attribute.
# ============================================================================


class _InferenceDecorator:
    """Callable + namespace for `@inference` and its inner decorators.

    Usage:
        @inference(models={...}, resources=...)
        class MyEndpoint:
            @inference.function
            def generate(self, ctx, payload): ...
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
        label: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Any:
        return _function_inner(
            fn,
            name=name,
            timeout_ms=timeout_ms,
            allowed_shapes=allowed_shapes,
            label=label,
            description=description,
        )



# ============================================================================
# @batched_inference — LLM-serving BatchedWorker shape (#273).
# Parallel to @inference but:
#   * async-generator methods required (inspect.isasyncgenfunction).
#   * No stages — BatchedWorker hosts a single long-lived engine.
#   * Yields typed signals: IncrementalTokenDelta | Done | Error.
#   * Tenant constructs the engine in setup(); SDK does not pick the engine.
# ============================================================================


def _batched_function_inner(
    fn: Optional[F] = None,
    *,
    name: Optional[str] = None,
    timeout_ms: Optional[int] = None,
    label: Optional[str] = None,
    description: Optional[str] = None,
) -> Any:
    """Inner decorator for @batched_inference.function (#273).

    Marks an async-generator method as the externally-invocable entry
    point of a BatchedWorker LLM-serving class. The method MUST satisfy
    ``inspect.isasyncgenfunction(method)`` — i.e. ``async def`` with at
    least one ``yield``. Plain coroutines and sync methods are rejected
    at decoration time with a clear error.

    Usage::

        @batched_inference.function
        async def caption(self, ctx, payload):
            async for token in self.engine.generate(...):
                yield IncrementalTokenDelta(text=token)
            yield Done()
    """

    def apply(method: F) -> F:
        if not inspect.isasyncgenfunction(method):
            qname = getattr(
                method, "__qualname__", getattr(method, "__name__", "<method>")
            )
            if inspect.iscoroutinefunction(method):
                hint = (
                    "method is a plain coroutine (``async def`` without "
                    "``yield``). BatchedWorker methods must be async "
                    "GENERATORS — add a ``yield`` statement (yield Done() at "
                    "minimum)."
                )
            elif inspect.isfunction(method) or inspect.ismethod(method):
                hint = (
                    "method is sync. Use ``@inference`` for sync (SerialWorker) "
                    "endpoints, or rewrite this method as ``async def`` with at "
                    "least one ``yield``."
                )
            else:
                hint = (
                    f"got {type(method).__name__!r}; expected an async-generator "
                    "method (``async def`` with ``yield``)."
                )
            raise TypeError(
                f"@batched_inference.function rejected {qname!r}: {hint}"
            )
        spec = _BatchedInferenceFunctionSpec(
            name=name or method.__name__,
            timeout_ms=timeout_ms,
            label=(label or "").strip() or None,
            description=(description or "").strip() or None,
        )
        setattr(method, "__gen_worker_batched_inference_function_spec__", spec)
        return method

    if fn is not None:
        return apply(fn)
    return apply


def _validate_batched_setup_signature(cls: type) -> dict[str, inspect.Parameter]:
    """Return the model kwargs of ``setup``. setup must take (self, **models).

    Same contract as ``_validate_setup_signature`` but reports under
    ``@batched_inference`` framing for clearer errors.
    """
    setup = getattr(cls, "setup", None)
    if setup is None:
        raise ValueError(
            f"@batched_inference class {cls.__name__!r}: missing required "
            "`setup` method. Add `def setup(self, **models): ...` and "
            "construct your engine (vLLM / SGLang / etc.) inside it."
        )
    sig = inspect.signature(setup)
    params = list(sig.parameters.values())[1:]
    return {p.name: p for p in params}


def _validate_batched_invocable_methods(
    cls: type,
) -> list[tuple[str, Callable[..., Any], _BatchedInferenceFunctionSpec]]:
    """Find @batched_inference.function-decorated methods on the class."""
    out: list[tuple[str, Callable[..., Any], _BatchedInferenceFunctionSpec]] = []
    for attr_name in dir(cls):
        if attr_name.startswith("_"):
            continue
        member = getattr(cls, attr_name, None)
        if member is None or not callable(member):
            continue
        spec = getattr(
            member, "__gen_worker_batched_inference_function_spec__", None
        )
        if spec is not None:
            # Re-validate the async-generator constraint at class-decoration
            # time too — guards against a tenant decorating the method
            # correctly but then re-assigning a sync replacement before
            # class construction lands.
            if not inspect.isasyncgenfunction(member):
                raise TypeError(
                    f"@batched_inference class {cls.__name__!r}: method "
                    f"{attr_name!r} carries @batched_inference.function but "
                    "is not an async generator. Add a ``yield`` to make it one."
                )
            out.append((attr_name, member, spec))
    if not out:
        raise ValueError(
            f"@batched_inference class {cls.__name__!r}: no "
            "@batched_inference.function-decorated methods found. At least "
            "one async-generator method is required."
        )
    return out


def _make_batched_inference_decorator():
    """Build the @batched_inference class-level decorator (#273)."""

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
    ) -> Callable[[C], C]: ...
    def decorator(
        cls: Optional[C] = None,
        *,
        label: Optional[str] = None,
        description: Optional[str] = None,
        resources: Optional[Resources] = None,
        models: Optional[Mapping[str, Binding]] = None,
    ) -> Any:
        resources_value: Resources = (
            resources if resources is not None else Resources()
        )

        def apply(target: C) -> C:
            if not inspect.isclass(target):
                raise TypeError(
                    f"@batched_inference requires a class. Got "
                    f"{type(target).__name__}. BatchedWorker is a long-lived "
                    "engine host — express it as a class with setup / "
                    "warmup / shutdown lifecycle methods."
                )

            cls_name = target.__name__
            setup_kwargs = _validate_batched_setup_signature(target)
            function_methods = _validate_batched_invocable_methods(target)

            # Reuse the same models-vs-setup validator — the contract is
            # identical: each models={} key must correspond to a setup()
            # kwarg.
            validated_models = _validate_models_against_setup(
                cls_name, models or {}, setup_kwargs
            )

            spec = _EndpointClassSpec(
                kind="inference",
                sub_kind=None,
                label=(label or "").strip() or None,
                description=(description or "").strip() or None,
                resources=resources_value,
                models=validated_models,
                allowed_shapes=(),
                # `runtime` stays None — engine choice is the tenant's job
                # (they construct vLLM / SGLang / transformers inside
                # setup()). The SDK only routes the request to the class
                # instance; the engine is opaque to it.
                runtime=None,
                batch_window_ms=None,
                max_batch=None,
            )

            setattr(target, "__gen_worker_endpoint_spec__", spec)
            setattr(target, "__gen_worker_archetype__", "BatchedWorker")
            # Parallel-to-@inference marker so the worker dispatcher can
            # route requests through the @batched_inference codepath
            # without overloading the @inference function-methods slot.
            setattr(
                target,
                "__gen_worker_batched_inference_function_methods__",
                function_methods,
            )
            return target

        if cls is not None:
            return apply(cls)
        return apply

    return decorator


class _BatchedInferenceDecorator:
    """Callable + namespace for `@batched_inference` and `.function` (#273).

    Parallel to ``@inference``, but enforces the LLM-serving BatchedWorker
    shape:
      * setup() constructs the tenant's engine (vLLM / SGLang / transformers).
        SDK does NOT pick or own the engine.
      * @batched_inference.function methods MUST be async generators yielding
        ``IncrementalTokenDelta | Done | Error`` signals.

    Usage::

        from gen_worker import (
            batched_inference, Resources,
            IncrementalTokenDelta, Done, Error,
        )

        @batched_inference(
            models={'llm': Repo("org/llama-3b")},
            resources=Resources(accelerator='cuda', min_vram_gb=24),
        )
        class JoyCaptionGenerate:
            def setup(self, llm):
                self.engine = build_engine(llm)

            @batched_inference.function
            async def caption(self, ctx, payload):
                async for tok in self.engine.generate(payload):
                    if ctx.cancelled():
                        break
                    yield IncrementalTokenDelta(text=tok)
                yield Done()

            def shutdown(self):
                self.engine.shutdown()
    """

    def __init__(self) -> None:
        self._outer = _make_batched_inference_decorator()

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._outer(*args, **kwargs)

    @staticmethod
    def function(
        fn: Optional[F] = None,
        *,
        name: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        label: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Any:
        return _batched_function_inner(
            fn,
            name=name,
            timeout_ms=timeout_ms,
            label=label,
            description=description,
        )


class _TrainingDecorator:
    """Callable + namespace for `@training`, mirrors `@inference`."""

    def __init__(self) -> None:
        self._outer = _make_endpoint_decorator("training")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._outer(*args, **kwargs)

    @staticmethod
    def function(*args: Any, **kwargs: Any) -> Any:
        return _function_inner(*args, **kwargs)



class _DatasetDecorator:
    """Callable + namespace for `@dataset` (dataset-generation endpoints)."""

    def __init__(self) -> None:
        self._outer = _make_endpoint_decorator("dataset")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._outer(*args, **kwargs)

    @staticmethod
    def function(*args: Any, **kwargs: Any) -> Any:
        return _function_inner(*args, **kwargs)



class _ConversionDecorator:
    """Callable + namespace for `@conversion(kind='format-conversion'|...)`."""

    def __init__(self) -> None:
        self._outer = _make_endpoint_decorator("conversion")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._outer(*args, **kwargs)

    @staticmethod
    def function(*args: Any, **kwargs: Any) -> Any:
        return _function_inner(*args, **kwargs)



# ============================================================================
# @invocable — single kind-agnostic method marker (#339 §5).
#
# `@inference.function` / `@training.function` / `@conversion.function` /
# `@dataset.function` all resolve to the same kind-agnostic `_function_inner`
# — four names for one marker, with the kind living on the CLASS decorator.
# `@invocable` collapses those to a single neutral name. It is ADDITIVE: the
# per-kind `.function` aliases keep working unchanged (production endpoints
# depend on them).
# ============================================================================


class _InvocableDecorator:
    """Kind-agnostic externally-callable method marker.

    Functionally identical to ``@inference.function`` / ``@<kind>.function``
    — it attaches the same ``__gen_worker_function_spec__`` that discovery and
    dispatch already read, so an ``@invocable`` method is discovered and routed
    exactly like a ``<kind>.function`` one. The endpoint *kind* is declared on
    the class decorator (``@inference`` / ``@training`` / ...); the method
    marker does not need to repeat it.

    Usage::

        @inference(models={...}, resources=...)
        class MyEndpoint:
            def setup(self, pipe):
                self.pipe = pipe

            @invocable(name="generate")
            def generate(self, ctx, payload): ...
    """

    def __call__(
        self,
        fn: Optional[F] = None,
        *,
        name: Optional[str] = None,
        timeout_ms: Optional[int] = None,
        allowed_shapes: Optional[tuple[tuple[int, ...], ...]] = None,
        label: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Any:
        return _function_inner(
            fn,
            name=name,
            timeout_ms=timeout_ms,
            allowed_shapes=allowed_shapes,
            label=label,
            description=description,
        )



# Singletons exported as the public API.
invocable = _InvocableDecorator()
inference = _InferenceDecorator()
batched_inference = _BatchedInferenceDecorator()
training = _TrainingDecorator()
dataset = _DatasetDecorator()
conversion = _ConversionDecorator()


__all__ = [
    "Resources",
    "Case",
    "invocable",
    "inference",
    "batched_inference",
    "training",
    "dataset",
    "conversion",
    # Internal specs (for discovery):
    "_EndpointClassSpec",
    "_FunctionSpec",
    "_BatchedInferenceFunctionSpec",
]
