from typing import Any, Callable, Dict, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

class ResourceRequirements:
    """
    Specifies the resource requirements for a worker function.

    Note: GPU/CPU is a release/image-level decision (Dockerfile + Cozy Hub
    placement constraints), not a per-function setting.
    """
    def __init__(
        self,
        max_concurrency: Optional[int] = None,
        batch_size_min: Optional[int] = None,
        batch_size_target: Optional[int] = None,
        batch_size_max: Optional[int] = None,
        prefetch_depth: Optional[int] = None,
        max_wait_ms: Optional[int] = None,
        memory_hint_mb: Optional[int] = None,
        stage_profile: Optional[str] = None,
        stage_traits: Optional[list[str]] = None,
    ) -> None:
        self.max_concurrency = max_concurrency
        self.batch_size_min = batch_size_min
        self.batch_size_target = batch_size_target
        self.batch_size_max = batch_size_max
        self.prefetch_depth = prefetch_depth
        self.max_wait_ms = max_wait_ms
        self.memory_hint_mb = memory_hint_mb
        self.stage_profile = stage_profile
        self.stage_traits = list(stage_traits or [])
        self._requirements: Dict[str, Any] = {}
        if max_concurrency is not None:
            self._requirements["max_concurrency"] = max_concurrency
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
        if stage_profile:
            self._requirements["stage_profile"] = str(stage_profile).strip()
        if self.stage_traits:
            self._requirements["stage_traits"] = [str(x).strip() for x in self.stage_traits if str(x).strip()]

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
