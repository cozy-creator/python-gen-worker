import inspect
import typing
from typing import Any, Callable, Dict, Optional, Sequence, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


# Concurrency modes for @inference_function / @training_function (tensorhub
# #232). These are the scheduler-facing capability declarations:
#   - "sequential": one request at a time per worker (safe default).
#   - "batched":    worker accepts batched requests grouped by the scheduler.
#   - "concurrent": worker handles N reentrant requests in parallel.
# Worker-reported FunctionCapacity at handshake tells the scheduler how many
# to dispatch within each mode.
CONCURRENCY_MODES = ("sequential", "batched", "concurrent")


def _infer_concurrency_mode(func: Callable[..., Any], *, is_training: bool) -> str:
    """Signature-based default when the decorator omits ``concurrency=``.

    Training functions always default to ``sequential`` — training workloads
    are typically long-running, non-reentrant, one-at-a-time per worker.

    Inference functions infer from the signature:
      - ``list[Input] → list[Output]`` (second positional is list, return is list)
        → ``batched``
      - Everything else → ``sequential``. Tenants who have verified their
        function is reentrant bump explicitly to ``concurrent``.
    """
    if is_training:
        return "sequential"
    try:
        hints = typing.get_type_hints(func, include_extras=True)
    except Exception:
        hints = {}
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return "sequential"
    params = [p for p in sig.parameters.values() if p.name not in ("self", "cls")]
    # Skip the ctx param; consider the first tenant-declared input.
    if params and params[0].name in ("ctx", "context"):
        params = params[1:]
    if not params:
        return "sequential"
    first_ann = hints.get(params[0].name)
    return_ann = hints.get("return")
    if _is_list_type(first_ann) and _is_list_type(return_ann):
        return "batched"
    return "sequential"


def _is_list_type(ann: Any) -> bool:
    if ann is None:
        return False
    # Bare ``list`` annotation (no element type) — common enough that we
    # accept it as batched too.
    if ann is list:
        return True
    origin = typing.get_origin(ann)
    return origin is list

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
    concurrency: Optional[str] = None,
    label: Optional[str] = None,
    description: Optional[str] = None,
    resources: Optional[ResourceRequirements] = None,
) -> Any:
    """Mark a function as an inference endpoint (tensorhub #232).

    Usable as ``@inference_function`` (bare) or ``@inference_function(...)``
    with kwargs.

    Args:
        concurrency: ``"sequential"`` | ``"batched"`` | ``"concurrent"``.
            Omitted → inferred from signature (``list→list`` → batched;
            everything else → sequential).
        label: Optional short label surfaced in the endpoint UI / search
            (e.g. ``"text-to-image"``). Non-functional.
        description: Optional free-text description. Non-functional.
        resources: Per-function ResourceRequirements. Hardware declarations
            such as ``accelerator="cuda"``, compute capability, VRAM, and
            optional library requirements are used by the worker to advertise
            only functions runnable on the current host.
    """
    if concurrency is not None and concurrency not in CONCURRENCY_MODES:
        raise ValueError(
            f"@inference_function: concurrency={concurrency!r} not in {CONCURRENCY_MODES}"
        )
    if resources is None:
        resources = ResourceRequirements()

    def apply(func: F) -> F:
        mode = concurrency or _infer_concurrency_mode(func, is_training=False)
        setattr(func, "_is_inference_function", True)
        setattr(func, "_concurrency_mode", mode)
        setattr(func, "_function_label", (label or "").strip() or None)
        setattr(func, "_function_description", (description or "").strip() or None)
        setattr(func, "_worker_resources", resources)
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
) -> Any:
    """Mark an async function as a WebSocket realtime handler."""
    if resources is None:
        resources = ResourceRequirements()

    def apply(func: F) -> F:
        setattr(func, "_is_worker_websocket", True)
        setattr(func, "_function_label", (label or "").strip() or None)
        setattr(func, "_function_description", (description or "").strip() or None)
        setattr(func, "_worker_resources", resources)
        return func

    if fn is not None:
        return apply(fn)
    return apply
