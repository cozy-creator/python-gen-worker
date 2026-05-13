from typing import Any, Callable, Literal, Optional, Sequence, TypeVar, overload

import msgspec

F = TypeVar("F", bound=Callable[..., Any])


def _force_setattr(obj: Any, name: str, value: Any) -> None:
    msgspec.structs.force_setattr(obj, name, value)


class ScalingHints(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    """Per-endpoint scaling hints used by the orchestrator's VRAM-aware
    placement + cost/latency-aware scheduling (gen-orchestrator #320).

    The tenant declares WHICH payload dimensions drive VRAM and runtime; the
    orchestrator learns the coefficients from observed runs.

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

    Attributes:
        vram_must_fit: Anchor for the size-scaled VRAM term. ``"full_model"``
            (whole bf16 must fit) or ``"largest_component"`` (only the largest
            single component must fit — for streaming-load paths). Tensorhub
            records both numbers on each checkpoint at ingest. Selects which
            entry in ``source.size_facts`` the orchestrator multiplies by
            ``vram_size_multiplier``.
        vram_base: Constant VRAM overhead in bytes (CUDA kernels, working
            memory, etc.). Default 0.
        vram_size_multiplier: Multiplier on ``source.size_facts[vram_must_fit]``.
            Use 1.0 for "model must fit exactly", >1 for headroom. Default 0.0.
        vram_scales_with: Tuple of payload field paths that grow VRAM
            (e.g. ``("batch_size", "width", "height")``). Coefficients learned.
        runtime_scales_with: Tuple of payload field paths that grow runtime
            (e.g. ``("num_inference_steps", "batch_size")``). Coefficients
            learned per gpu_class.
    """

    vram_must_fit: Literal["full_model", "largest_component"] | None = None
    vram_base: int = 0
    vram_size_multiplier: float = 0.0
    vram_scales_with: tuple[str, ...] = ()
    runtime_scales_with: tuple[str, ...] = ()

    def __post_init__(self) -> None:
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

    def is_empty(self) -> bool:
        return (
            self.vram_must_fit is None
            and self.vram_base == 0
            and self.vram_size_multiplier == 0.0
            and not self.vram_scales_with
            and not self.runtime_scales_with
        )


class ResourceRequirements(msgspec.Struct, frozen=True, kw_only=True, omit_defaults=True):
    """Resource requirements for an endpoint pod.

    Passed to :func:`inference_function` / :func:`training_function` to declare
    the hardware shape the function needs. Frozen + kw_only — matches Go struct
    semantics (immutable, fields by name).

    Wire shape (consumed by tensorhub builder + gen-orchestrator scheduler):

    - ``kind``: optional free-form label.
    - ``accelerator``: ``"cuda"`` | ``"none"``. ``"gpu"`` and ``"cpu"`` are
      normalized to ``"cuda"`` / ``"none"`` on construction.
    - ``cuda_compute_min``: minimum CUDA compute capability (e.g. ``8.0``).
      Emitted on the wire as a formatted ``"8.0"`` string plus
      ``compute_capability: {"min": "8.0"}`` for the Go scheduler.
    - ``requires_gpu``: bool. Auto-set to ``True`` when ``accelerator="cuda"``
      and not otherwise specified.
    - ``min_vram_gb``: minimum VRAM in GiB.
    - ``required_libraries``: tuple of library names the function imports
      (``("bitsandbytes", "torchao")`` etc.) — used by the worker to advertise
      only functions runnable on the current host.
    """

    kind: str | None = None
    accelerator: Literal["cuda", "none"] | None = None
    cuda_compute_min: float | None = None
    # Derived wire-shape field — set in __post_init__ when cuda_compute_min is
    # provided. Kept on the struct so msgspec.to_builtins emits it. Consumers
    # (tensorhub function_requirements.go, gen-orchestrator resolver) read
    # both this and cuda_compute_min.
    compute_capability: dict[str, str] | None = None
    requires_gpu: bool | None = None
    min_vram_gb: float | None = None
    required_libraries: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # Normalize kind: empty string → None (so omit_defaults drops it).
        if self.kind is not None:
            k = str(self.kind).strip()
            _force_setattr(self, "kind", k or None)

        # Normalize accelerator: "gpu" → "cuda", "cpu" → "none", "" → None.
        if self.accelerator is not None:
            accel = str(self.accelerator).strip().lower()
            if accel == "gpu":
                accel = "cuda"
            elif accel == "cpu":
                accel = "none"
            if accel == "":
                _force_setattr(self, "accelerator", None)
            elif accel in ("none", "cuda"):
                _force_setattr(self, "accelerator", accel)
                # accelerator="cuda" implies requires_gpu=True unless caller said otherwise.
                if accel == "cuda" and self.requires_gpu is None:
                    _force_setattr(self, "requires_gpu", True)
            else:
                raise ValueError(
                    f"accelerator must be 'none' or 'cuda', got {self.accelerator!r}"
                )

        # Validate + derive compute_capability from cuda_compute_min.
        if self.cuda_compute_min is not None:
            val = float(self.cuda_compute_min)
            if val <= 0:
                raise ValueError(f"cuda_compute_min must be positive, got {val}")
            _force_setattr(self, "cuda_compute_min", val)
            _force_setattr(self, "compute_capability", {"min": f"{val:.1f}"})

        if self.min_vram_gb is not None:
            vram = float(self.min_vram_gb)
            if vram <= 0:
                raise ValueError(f"min_vram_gb must be positive, got {vram}")
            _force_setattr(self, "min_vram_gb", vram)

        # Strip empty entries from required_libraries.
        if self.required_libraries:
            libs = tuple(str(x).strip() for x in self.required_libraries if str(x).strip())
            _force_setattr(self, "required_libraries", libs)


@overload
def inference_function(fn: F) -> F: ...
@overload
def inference_function(
    fn: None = None,
    *,
    label: Optional[str] = None,
    description: Optional[str] = None,
    resources: Optional[ResourceRequirements] = None,
    scaling_hints: Optional[ScalingHints] = None,
) -> Callable[[F], F]: ...
def inference_function(
    fn: Optional[F] = None,
    *,
    label: Optional[str] = None,
    description: Optional[str] = None,
    resources: Optional[ResourceRequirements] = None,
    scaling_hints: Optional[ScalingHints] = None,
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
        scaling_hints: Per-function :class:`ScalingHints` — declares which
            dimensions drive VRAM and runtime so the orchestrator can
            predict and learn.
    """
    if resources is None:
        resources = ResourceRequirements()

    def apply(func: F) -> F:
        setattr(func, "_is_inference_function", True)
        setattr(func, "_function_label", (label or "").strip() or None)
        setattr(func, "_function_description", (description or "").strip() or None)
        setattr(func, "_worker_resources", resources)
        setattr(func, "_scaling_hints", scaling_hints)
        return func

    if fn is not None:
        return apply(fn)
    return apply


# NOTE: @training_function lives in ``gen_worker.conversion``. It's a richer
# decorator than @inference_function: it handles the reserved-name contract
# (ctx / source / datasets) and signature-introspected dispatch. Keeping it out
# of the top-level package avoids presenting conversion internals to inference
# endpoint authors.
