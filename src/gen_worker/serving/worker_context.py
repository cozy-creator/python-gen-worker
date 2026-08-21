"""The ONE place the worker builds a production :class:`LoadContext`."""

from __future__ import annotations

from typing import Any, Callable, Optional

from .context import DeployBinding, LoadContext, LoaderEngine
from .placement import serving_device

__all__ = ["worker_load_context"]


def worker_load_context(
    *,
    binding: DeployBinding,
    model_type: Optional[type] = None,
    lane: Any = None,
    resolved: Any = None,
    compile_sink: Optional[Callable[[Any], Any]] = None,
    engine: Optional[LoaderEngine] = None,
    device: str = "",
    io: str = "buffered",
    weight_budget_bytes: int = 0,
) -> LoadContext[Any]:
    """A ``LoadContext`` carrying every decision the WORKER owns."""

    return LoadContext(
        binding=binding,
        model_type=model_type,
        lane=lane,
        resolved=resolved,
        engine=engine,
        compile_sink=compile_sink,
        device=str(device or "") or serving_device(),
        io=str(io or "buffered"),
        weight_budget_bytes=weight_budget_bytes,
    )
