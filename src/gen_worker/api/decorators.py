from typing import Any, Callable, Dict, Optional, Sequence, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


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
    """
    if resources is None:
        resources = ResourceRequirements()

    def apply(func: F) -> F:
        setattr(func, "_is_inference_function", True)
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
