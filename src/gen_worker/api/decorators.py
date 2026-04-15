from typing import Any, Callable, Dict, Mapping, Optional, Sequence, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

class ResourceRequirements:
    """
    Specifies the resource requirements for a worker function.

    Worker resources may include per-function hints used by schedulers
    (for example, requires_gpu or low-precision profile support).
    """
    def __init__(
        self,
        batch_size_min: Optional[int] = None,
        batch_size_target: Optional[int] = None,
        batch_size_max: Optional[int] = None,
        prefetch_depth: Optional[int] = None,
        max_wait_ms: Optional[int] = None,
        memory_hint_mb: Optional[int] = None,
        kind: Optional[str] = None,
        compute_capability_min: Optional[float] = None,
        requires_gpu: Optional[bool] = None,
        min_vram_gb: Optional[float] = None,
        supported_conversion_profiles: Optional[Sequence[str]] = None,
        supported_precisions: Optional[Sequence[str]] = None,
        runtime_hints: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.batch_size_min = batch_size_min
        self.batch_size_target = batch_size_target
        self.batch_size_max = batch_size_max
        self.prefetch_depth = prefetch_depth
        self.max_wait_ms = max_wait_ms
        self.memory_hint_mb = memory_hint_mb
        self.kind = str(kind or "").strip()
        self.compute_capability_min = compute_capability_min
        self.requires_gpu = requires_gpu
        self.min_vram_gb = min_vram_gb
        self._requirements: Dict[str, Any] = {}
        if batch_size_min is not None:
            self._requirements["batch_size_min"] = int(batch_size_min)
        if batch_size_target is not None:
            self._requirements["batch_size_target"] = int(batch_size_target)
        if batch_size_max is not None:
            self._requirements["batch_size_max"] = int(batch_size_max)
        if prefetch_depth is not None:
            self._requirements["prefetch_depth"] = int(prefetch_depth)
        if max_wait_ms is not None:
            self._requirements["max_wait_ms"] = int(max_wait_ms)
        if memory_hint_mb is not None:
            self._requirements["memory_hint_mb"] = int(memory_hint_mb)
        if self.kind:
            self._requirements["kind"] = self.kind
        if compute_capability_min is not None:
            val = float(compute_capability_min)
            if val <= 0:
                raise ValueError(f"compute_capability_min must be positive, got {val}")
            self._requirements["compute_capability"] = {"min": f"{val:.1f}"}
        if requires_gpu is not None:
            self._requirements["requires_gpu"] = bool(requires_gpu)
        if min_vram_gb is not None:
            vram = float(min_vram_gb)
            if vram <= 0:
                raise ValueError(f"min_vram_gb must be positive, got {vram}")
            self._requirements["min_vram_gb"] = vram
        if supported_conversion_profiles is not None:
            profiles = [str(x).strip() for x in supported_conversion_profiles if str(x).strip()]
            if profiles:
                self._requirements["supported_conversion_profiles"] = profiles
        if supported_precisions is not None:
            precisions = [str(x).strip() for x in supported_precisions if str(x).strip()]
            if precisions:
                self._requirements["supported_precisions"] = precisions
        if runtime_hints is not None:
            for key, value in dict(runtime_hints).items():
                if key in self._requirements:
                    continue
                self._requirements[str(key)] = value

    def to_dict(self) -> Dict[str, Any]:
        """Returns a dictionary representation of the defined requirements."""
        return self._requirements

    def __repr__(self) -> str:
        return f"ResourceRequirements({self._requirements})"


def worker_function(
    resources: Optional[ResourceRequirements] = None,
) -> Callable[[F], F]:
    """
    Decorator to mark a function as a worker task and associate resource requirements.

    Args:
        resources: An optional ResourceRequirements object describing the function's needs.
    """
    if resources is None:
        resources = ResourceRequirements() # Default empty requirements

    def decorator(func: F) -> F:
        # Attach metadata directly to the function object.
        # The SDK's runner component will look for these attributes.
        setattr(func, '_is_worker_function', True)
        setattr(func, '_worker_resources', resources)

        # Return the original function, now marked with attributes.
        #
        # Important: do not wrap the function; we want `inspect.signature()`
        # to reflect the tenant-authored callable, and self-wrapping can create
        # `inspect.unwrap()` loops.
        return func

    return decorator


def worker_websocket(
    resources: Optional[ResourceRequirements] = None,
) -> Callable[[F], F]:
    """
    Decorator to mark an async function as a WebSocket realtime handler.

    WebSocket handlers are invoked via a worker-owned socket interface (no FastAPI
    dependency) when the scheduler/orchestrator starts a realtime session.
    """
    if resources is None:
        resources = ResourceRequirements()

    def decorator(func: F) -> F:
        setattr(func, "_is_worker_websocket", True)
        setattr(func, "_worker_resources", resources)
        return func

    return decorator
