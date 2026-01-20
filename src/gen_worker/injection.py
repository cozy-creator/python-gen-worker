from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, get_args, get_origin, Annotated


class ModelRefSource(str, Enum):
    DEPLOYMENT = "deployment"
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
class ModelArtifacts:
    """
    A worker-injected handle to on-disk artifacts for a resolved model ref.

    This is for "non-standard" runtimes where the worker cannot (or should not)
    construct an in-memory pipeline/model object itself.

    `files` keys are deployment-defined (source of truth is deployment config),
    e.g. {"checkpoint_path": Path(...), "config_path": Path(...)}.
    """

    model_id: str
    root_dir: Path
    files: Mapping[str, Path]
    metadata: Mapping[str, Any] | None = None

    def path(self, key: str) -> Path:
        p = self.files.get(key)
        if p is None:
            raise KeyError(f"missing artifact key: {key!r}")
        return p

    def get_path(self, key: str, default: Optional[Path] = None) -> Optional[Path]:
        return self.files.get(key, default)


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


def import_object(path: str) -> Any:
    """
    Import an object from "module:qualname" or "module.attr".
    """

    raw = path.strip()
    if ":" in raw:
        mod, qual = raw.split(":", 1)
    else:
        mod, qual = raw.rsplit(".", 1)
    module = importlib.import_module(mod)
    obj: Any = module
    for part in qual.split("."):
        obj = getattr(obj, part)
    return obj


def resolve_loader(path: str) -> Callable[..., Any]:
    obj = import_object(path)
    if not callable(obj):
        raise TypeError(f"loader is not callable: {path}")
    return obj


def type_qualname(t: Any) -> str:
    if hasattr(t, "__module__") and hasattr(t, "__qualname__"):
        return f"{t.__module__}.{t.__qualname__}"
    if hasattr(t, "__module__") and hasattr(t, "__name__"):
        return f"{t.__module__}.{t.__name__}"
    return repr(t)


def is_async_callable(fn: Callable[..., Any]) -> bool:
    return inspect.iscoroutinefunction(fn) or inspect.isasyncgenfunction(fn)


_RUNTIME_LOADERS: dict[str, Callable[..., Any]] = {}


def register_runtime_loader(runtime_type: Any, loader: Callable[..., Any]) -> None:
    """
    Register a tenant-provided loader hook for a custom injected runtime type.

    The worker/model-manager can call these loaders to build a VRAM-resident
    runtime handle from ModelArtifacts and then cache/inject that handle.
    """
    qn = type_qualname(runtime_type)
    if not callable(loader):
        raise TypeError("loader must be callable")
    _RUNTIME_LOADERS[qn] = loader


def get_registered_runtime_loader(runtime_type: Any) -> Optional[Callable[..., Any]]:
    return _RUNTIME_LOADERS.get(type_qualname(runtime_type))
