from __future__ import annotations

import inspect
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Optional, get_args, get_origin, Annotated


class ModelRefSource(str, Enum):
    # FIXED means the model key is fixed by the function signature and does not
    # depend on the request payload.
    FIXED = "fixed"
    # Deprecated alias for FIXED (kept for compatibility with older workers/examples).
    RELEASE = "release"
    PAYLOAD = "payload"


@dataclass(frozen=True)
class ModelRef:
    """
    Metadata marker for signature-driven model selection/injection.

    This is intended to be used inside `typing.Annotated[..., ModelRef(...)]`.
    """

    source: ModelRefSource
    key: str


@dataclass(frozen=True)
class InjectionSpec:
    param_name: str
    param_type: Any
    model_ref: ModelRef


def parse_injection(annotation: Any) -> Optional[tuple[Any, ModelRef]]:
    """
    Returns (base_type, model_ref) if annotation is Annotated[base_type, ModelRef(...)],
    otherwise None.
    """

    origin = get_origin(annotation)
    if origin is not Annotated:
        return None
    args = get_args(annotation)
    if not args:
        return None
    base = args[0]
    meta = args[1:]
    for m in meta:
        if isinstance(m, ModelRef):
            return base, m
    return None


def type_qualname(t: Any) -> str:
    if hasattr(t, "__module__") and hasattr(t, "__qualname__"):
        return f"{t.__module__}.{t.__qualname__}"
    if hasattr(t, "__module__") and hasattr(t, "__name__"):
        return f"{t.__module__}.{t.__name__}"
    return repr(t)


def is_async_callable(fn: Callable[..., Any]) -> bool:
    return inspect.iscoroutinefunction(fn) or inspect.isasyncgenfunction(fn)
