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
    ) -> None:
        self.max_concurrency = max_concurrency
        self._requirements = {}
        if max_concurrency is not None:
            self._requirements["max_concurrency"] = max_concurrency

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
