from typing import Any, Callable, Dict, List, Optional, Sequence, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


_VRAM_MUST_FIT_VALUES = frozenset({"full_model", "largest_component"})


class ScalingHints:
    """Per-function scaling hints used by the orchestrator's VRAM-aware
    placement + cost/latency-aware scheduling (gen-orchestrator #320).

    The tenant tells the platform WHICH dimensions matter — not by how much.
    The orchestrator learns the coefficients from observed runs.

    VRAM at submit is gated as::

        required_vram = vram_base
                      + vram_size_multiplier × source.size_facts[vram_must_fit]
                      + Σ vram_coef[f] × payload[f]   for f in vram_scales_with

    Runtime per gpu_class is modeled as::

        runtime_ms[class] = base[class] + Σ runtime_coef[class][f] × payload[f]
                            for f in runtime_scales_with

    All `vram_coef[f]` / `runtime_coef[class][f]` start at 0 and are learned
    from observed `peak_vram_bytes` and `wall_time_ms` after each successful
    run. The tenant only declares the *dimensions* that influence cost; the
    orchestrator fits the magnitudes.

    Args:
        vram_must_fit: Anchor for the size-scaled VRAM term. Allowed:
            ``"full_model"`` (whole bf16 must fit) or
            ``"largest_component"`` (only the largest single component must
            fit — for streaming-load paths). Tensorhub records both numbers
            on each checkpoint at ingest.
        vram_base: Constant VRAM overhead (bytes) — CUDA kernels, working
            memory, etc. Default 0.
        vram_size_multiplier: Multiplier on ``source.size_facts[vram_must_fit]``.
            Use 1.0 for "model must fit exactly", >1 for headroom. Default 0.
        vram_scales_with: Payload field names that grow VRAM (e.g.
            ``["batch_size", "width", "height"]``). Coefficients learned.
        runtime_scales_with: Payload field names that grow runtime (e.g.
            ``["num_inference_steps", "batch_size"]``). Coefficients learned
            per gpu_class.
    """

    def __init__(
        self,
        *,
        vram_must_fit: Optional[str] = None,
        vram_base: int = 0,
        vram_size_multiplier: float = 0.0,
        vram_scales_with: Optional[Sequence[str]] = None,
        runtime_scales_with: Optional[Sequence[str]] = None,
    ) -> None:
        if vram_must_fit is not None:
            v = str(vram_must_fit).strip()
            if v and v not in _VRAM_MUST_FIT_VALUES:
                raise ValueError(
                    f"vram_must_fit must be one of {sorted(_VRAM_MUST_FIT_VALUES)} "
                    f"(or None); got {vram_must_fit!r}"
                )
            self.vram_must_fit: Optional[str] = v or None
        else:
            self.vram_must_fit = None
        if int(vram_base) < 0:
            raise ValueError(f"vram_base must be >= 0, got {vram_base}")
        self.vram_base: int = int(vram_base)
        if float(vram_size_multiplier) < 0:
            raise ValueError(f"vram_size_multiplier must be >= 0, got {vram_size_multiplier}")
        self.vram_size_multiplier: float = float(vram_size_multiplier)
        self.vram_scales_with: List[str] = [
            str(x).strip() for x in (vram_scales_with or []) if str(x).strip()
        ]
        self.runtime_scales_with: List[str] = [
            str(x).strip() for x in (runtime_scales_with or []) if str(x).strip()
        ]

    def is_empty(self) -> bool:
        return (
            self.vram_must_fit is None
            and self.vram_base == 0
            and self.vram_size_multiplier == 0.0
            and not self.vram_scales_with
            and not self.runtime_scales_with
        )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if self.vram_must_fit is not None:
            out["vram_must_fit"] = self.vram_must_fit
        if self.vram_base:
            out["vram_base"] = self.vram_base
        if self.vram_size_multiplier:
            out["vram_size_multiplier"] = self.vram_size_multiplier
        if self.vram_scales_with:
            out["vram_scales_with"] = list(self.vram_scales_with)
        if self.runtime_scales_with:
            out["runtime_scales_with"] = list(self.runtime_scales_with)
        return out

    def __repr__(self) -> str:
        return f"ScalingHints({self.to_dict()})"


def _build_scaling_hints(
    *,
    scaling_hints: Optional["ScalingHints"],
    vram_must_fit: Optional[str],
    vram_base: Optional[int],
    vram_size_multiplier: Optional[float],
    vram_scales_with: Optional[Sequence[str]],
    runtime_scales_with: Optional[Sequence[str]],
) -> "ScalingHints":
    """Resolve `scaling_hints=` vs flat kwargs into one ScalingHints object.

    Tenants may pass a `scaling_hints=ScalingHints(...)` object OR flat
    kwargs (`vram_must_fit=...`, etc.). Passing both is an error.
    """
    flat_given = (
        vram_must_fit is not None
        or vram_base is not None
        or vram_size_multiplier is not None
        or vram_scales_with is not None
        or runtime_scales_with is not None
    )
    if scaling_hints is not None and flat_given:
        raise ValueError(
            "pass either scaling_hints=ScalingHints(...) OR flat kwargs "
            "(vram_must_fit, vram_base, vram_size_multiplier, vram_scales_with, "
            "runtime_scales_with), not both"
        )
    if scaling_hints is not None:
        return scaling_hints
    return ScalingHints(
        vram_must_fit=vram_must_fit,
        vram_base=int(vram_base) if vram_base is not None else 0,
        vram_size_multiplier=float(vram_size_multiplier) if vram_size_multiplier is not None else 0.0,
        vram_scales_with=vram_scales_with,
        runtime_scales_with=runtime_scales_with,
    )


class ResourceRequirements:
    """
    Specifies the resource requirements for a worker function.

    Worker resources may include per-function hints used by schedulers
    (for example, requires_gpu or low-precision profile support).
    """
    def __init__(
        self,
        kind: Optional[str] = None,
        accelerator: Optional[str] = None,
        cuda_compute_min: Optional[float] = None,
        requires_gpu: Optional[bool] = None,
        min_vram_gb: Optional[float] = None,
        required_libraries: Optional[Sequence[str]] = None,
    ) -> None:
        self.kind = str(kind or "").strip()
        self._requirements: Dict[str, Any] = {}
        if self.kind:
            self._requirements["kind"] = self.kind
        if accelerator is not None:
            accel = str(accelerator or "").strip().lower()
            if accel == "gpu":
                accel = "cuda"
            elif accel == "cpu":
                accel = "none"
            if accel not in ("", "none", "cuda"):
                raise ValueError(f"accelerator must be 'none' or 'cuda', got {accelerator!r}")
            if accel:
                self._requirements["accelerator"] = accel
                if accel == "cuda" and requires_gpu is None:
                    requires_gpu = True
        if cuda_compute_min is not None:
            val = float(cuda_compute_min)
            if val <= 0:
                raise ValueError(f"cuda_compute_min must be positive, got {val}")
            self._requirements["cuda_compute_min"] = f"{val:.1f}"
            self._requirements["compute_capability"] = {"min": f"{val:.1f}"}
        if requires_gpu is not None:
            self._requirements["requires_gpu"] = bool(requires_gpu)
        if min_vram_gb is not None:
            vram = float(min_vram_gb)
            if vram <= 0:
                raise ValueError(f"min_vram_gb must be positive, got {vram}")
            self._requirements["min_vram_gb"] = vram
        if required_libraries is not None:
            libs = [str(x).strip() for x in required_libraries if str(x).strip()]
            if libs:
                self._requirements["required_libraries"] = libs

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the defined requirements."""
        return self._requirements

    def __repr__(self) -> str:
        return f"ResourceRequirements({self._requirements})"


def inference_function(
    fn: Optional[F] = None,
    *,
    label: Optional[str] = None,
    description: Optional[str] = None,
    resources: Optional[ResourceRequirements] = None,
    scaling_hints: Optional[ScalingHints] = None,
    # Flat-kwarg variants of ScalingHints, for ergonomic decorator use.
    # Passing these AND scaling_hints= is an error.
    vram_must_fit: Optional[str] = None,
    vram_base: Optional[int] = None,
    vram_size_multiplier: Optional[float] = None,
    vram_scales_with: Optional[Sequence[str]] = None,
    runtime_scales_with: Optional[Sequence[str]] = None,
) -> Any:
    """Mark a function as an inference endpoint (tensorhub #232).

    Usable as ``@inference_function`` (bare) or ``@inference_function(...)``
    with kwargs.

    Args:
        label: Optional short label surfaced in the endpoint UI / search
            (e.g. ``"text-to-image"``). Non-functional.
        description: Optional free-text description. Non-functional.
        resources: Per-function ResourceRequirements. Hardware declarations
            such as ``accelerator="cuda"``, compute capability, VRAM, and
            optional library requirements are used by the worker to advertise
            only functions runnable on the current host.
        scaling_hints: Per-function ScalingHints — declares which dimensions
            drive VRAM and runtime so the orchestrator can predict and learn.
            See :class:`ScalingHints`. Alternatively, pass the flat kwargs
            ``vram_must_fit``, ``vram_base``, ``vram_size_multiplier``,
            ``vram_scales_with``, ``runtime_scales_with`` directly on the
            decorator.
    """
    if resources is None:
        resources = ResourceRequirements()
    hints = _build_scaling_hints(
        scaling_hints=scaling_hints,
        vram_must_fit=vram_must_fit,
        vram_base=vram_base,
        vram_size_multiplier=vram_size_multiplier,
        vram_scales_with=vram_scales_with,
        runtime_scales_with=runtime_scales_with,
    )

    def apply(func: F) -> F:
        setattr(func, "_is_inference_function", True)
        setattr(func, "_function_label", (label or "").strip() or None)
        setattr(func, "_function_description", (description or "").strip() or None)
        setattr(func, "_worker_resources", resources)
        setattr(func, "_scaling_hints", hints)
        return func

    if fn is not None:
        return apply(fn)
    return apply


# NOTE: @training_function lives in ``gen_worker.conversion``. It's a richer
# decorator than @inference_function: it handles the reserved-name contract
# (ctx / source / datasets) and signature-introspected dispatch. Keeping it out
# of the top-level package avoids presenting conversion internals to inference
# endpoint authors.


# Realtime/WebSocket endpoints retain their own decorator. Tenants handling
# realtime sessions declare ``@realtime_function`` (renamed from
# ``@worker_websocket``) for the same reason inference + training got their
# own decorator names — the kind is part of the tenant's function metadata.
def realtime_function(
    fn: Optional[F] = None,
    *,
    label: Optional[str] = None,
    description: Optional[str] = None,
    resources: Optional[ResourceRequirements] = None,
    scaling_hints: Optional[ScalingHints] = None,
    vram_must_fit: Optional[str] = None,
    vram_base: Optional[int] = None,
    vram_size_multiplier: Optional[float] = None,
    vram_scales_with: Optional[Sequence[str]] = None,
    runtime_scales_with: Optional[Sequence[str]] = None,
) -> Any:
    """Mark an async function as a WebSocket realtime handler.

    Scaling hints work the same way as on @inference_function — see
    :class:`ScalingHints`.
    """
    if resources is None:
        resources = ResourceRequirements()
    hints = _build_scaling_hints(
        scaling_hints=scaling_hints,
        vram_must_fit=vram_must_fit,
        vram_base=vram_base,
        vram_size_multiplier=vram_size_multiplier,
        vram_scales_with=vram_scales_with,
        runtime_scales_with=runtime_scales_with,
    )

    def apply(func: F) -> F:
        setattr(func, "_is_worker_websocket", True)
        setattr(func, "_function_label", (label or "").strip() or None)
        setattr(func, "_function_description", (description or "").strip() or None)
        setattr(func, "_worker_resources", resources)
        setattr(func, "_scaling_hints", hints)
        return func

    if fn is not None:
        return apply(fn)
    return apply
