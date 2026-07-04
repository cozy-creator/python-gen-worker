"""The ONE decorator walker: decorated classes -> EndpointSpec table.

Shared by the worker runtime (dispatch), build-time discovery (endpoint.lock),
and the local CLI. Signature inspection lives here and nowhere else.
"""

from __future__ import annotations

import collections.abc as cabc
import inspect
import types as py_types
import typing
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import msgspec

from .api.binding import Binding, Dispatch, Repo
from .api.decorators import Resources
from .discovery.walk import find_endpoint_classes

_ITER_ORIGINS = (
    typing.Iterator, typing.Iterable, typing.AsyncIterator, typing.AsyncIterable,
    cabc.Iterator, cabc.Iterable, cabc.AsyncIterator, cabc.AsyncIterable,
)


def _is_struct(t: Any) -> bool:
    try:
        return isinstance(t, type) and issubclass(t, msgspec.Struct)
    except Exception:
        return False


@dataclass(frozen=True)
class EndpointSpec:
    """One externally-routable function: a bound (class, method) plus wire shape."""

    name: str                     # routable function name (pre-slug)
    cls: type
    attr_name: str                # python attribute name on the class
    method: Callable[..., Any]    # unbound function object
    kind: str                     # inference | training | dataset | conversion
    payload_type: type            # msgspec.Struct
    output_mode: str              # "single" | "stream"
    output_type: Optional[type] = None   # struct returned (single mode)
    delta_type: Optional[type] = None    # struct yielded (stream mode; None if union)
    is_async: bool = False        # coroutine or async generator
    is_async_gen: bool = False
    ctx_param: str = "ctx"
    payload_param: str = "payload"
    resources: Resources = field(default_factory=Resources)
    models: Dict[str, Binding] = field(default_factory=dict)  # slot -> binding
    timeout_ms: Optional[int] = None
    sub_kind: Optional[str] = None
    runtime: Optional[str] = None
    label: Optional[str] = None
    description: Optional[str] = None
    allowed_shapes: tuple = ()
    module: str = ""              # declaring module
    walked_module: str = ""       # top-level package the class was found under

    @property
    def needs_gpu(self) -> bool:
        r = self.resources
        return bool(
            r.accelerator == "cuda"
            or r.requires_gpu is True
            or r.min_vram_gb is not None
            or r.min_compute_capability is not None
        )

    @property
    def fixed_models(self) -> Dict[str, Repo]:
        return {k: b for k, b in self.models.items() if isinstance(b, Repo)}

    @property
    def dispatch_models(self) -> Dict[str, Dispatch]:
        return {k: b for k, b in self.models.items() if isinstance(b, Dispatch)}


def _inspect_return(cls: type, attr: str, ret: Any) -> tuple[str, Optional[type], Optional[type]]:
    """-> (output_mode, output_type, delta_type)."""
    if _is_struct(ret):
        return "single", ret, None
    origin = typing.get_origin(ret)
    if origin in _ITER_ORIGINS:
        args = typing.get_args(ret) or ()
        item = args[0] if args else None
        if _is_struct(item):
            return "stream", item, item
        # Union of signal structs (batched_inference shape): still a stream.
        if typing.get_origin(item) in (typing.Union, py_types.UnionType):
            return "stream", None, None
        raise ValueError(
            f"{cls.__name__}.{attr}: streaming return must be "
            f"(Async)Iterator[msgspec.Struct], got {ret!r}"
        )
    raise ValueError(
        f"{cls.__name__}.{attr}: return type must be msgspec.Struct or "
        f"(Async)Iterator[msgspec.Struct], got {ret!r}"
    )


def _spec_for_method(
    cls: type,
    ep_spec: Any,
    attr_name: str,
    method: Callable[..., Any],
    fn_name: str,
    *,
    timeout_ms: Optional[int],
    label: Optional[str],
    description: Optional[str],
    allowed_shapes: tuple,
    payload_override: Optional[type],
    models: Dict[str, Binding],
    resources: Resources,
    walked_module: str,
) -> EndpointSpec:
    hints = typing.get_type_hints(method, include_extras=False)
    sig = inspect.signature(method)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    if len(params) < 2:
        raise ValueError(
            f"{cls.__name__}.{attr_name}: must accept (self, ctx, payload); "
            f"got params {[p.name for p in params]}"
        )
    ctx_param, payload_param = params[0].name, params[1].name

    if payload_override is not None:
        if not _is_struct(payload_override):
            raise ValueError(
                f"{cls.__name__}.{attr_name}: Case input= must be a msgspec.Struct"
            )
        payload_type = payload_override
    else:
        payload_type = hints.get(payload_param)
        if not _is_struct(payload_type):
            raise ValueError(
                f"{cls.__name__}.{attr_name}: payload param {payload_param!r} "
                f"must be annotated with a msgspec.Struct (got {payload_type!r})"
            )

    ret = hints.get("return")
    if ret is None:
        raise ValueError(f"{cls.__name__}.{attr_name}: missing return type annotation")
    output_mode, output_type, delta_type = _inspect_return(cls, attr_name, ret)

    return EndpointSpec(
        name=fn_name,
        cls=cls,
        attr_name=attr_name,
        method=method,
        kind=str(getattr(ep_spec, "kind", "inference") or "inference"),
        payload_type=payload_type,
        output_mode=output_mode,
        output_type=output_type,
        delta_type=delta_type,
        is_async=bool(
            inspect.iscoroutinefunction(method) or inspect.isasyncgenfunction(method)
        ),
        is_async_gen=inspect.isasyncgenfunction(method),
        ctx_param=ctx_param,
        payload_param=payload_param,
        resources=resources,
        models=models,
        timeout_ms=timeout_ms,
        sub_kind=getattr(ep_spec, "sub_kind", None),
        runtime=getattr(ep_spec, "runtime", None),
        label=label,
        description=description,
        allowed_shapes=allowed_shapes,
        module=getattr(cls, "__module__", "") or "",
        walked_module=walked_module,
    )


def extract_specs(cls: type, *, walked_module: str = "") -> List[EndpointSpec]:
    """All EndpointSpecs declared by one decorated class (parametrize expanded)."""
    ep_spec = getattr(cls, "__gen_worker_endpoint_spec__", None)
    if ep_spec is None:
        return []
    class_models: Dict[str, Binding] = dict(getattr(ep_spec, "models", {}) or {})
    class_res: Resources = getattr(ep_spec, "resources", None) or Resources()
    class_shapes = tuple(getattr(ep_spec, "allowed_shapes", ()) or ())
    walked = walked_module or (getattr(cls, "__module__", "") or "")

    methods = list(getattr(cls, "__gen_worker_function_methods__", []) or [])
    methods += list(
        getattr(cls, "__gen_worker_batched_inference_function_methods__", []) or []
    )
    out: List[EndpointSpec] = []

    parametrize = tuple(getattr(ep_spec, "parametrize", ()) or ())
    if parametrize and methods:
        attr_name, method, fn_spec = methods[0]
        for case in parametrize:
            models = class_models
            if getattr(case, "model", None) is not None:
                slot = next(iter(class_models), "model")
                models = dict(class_models)
                models[slot] = case.model
            out.append(_spec_for_method(
                cls, ep_spec, attr_name, method, str(case.name),
                timeout_ms=fn_spec.timeout_ms,
                label=fn_spec.label, description=fn_spec.description,
                allowed_shapes=tuple(fn_spec.allowed_shapes or ()) or class_shapes,
                payload_override=getattr(case, "input", None),
                models=models,
                resources=case.resources if case.resources is not None else class_res,
                walked_module=walked,
            ))
        return out

    for attr_name, method, fn_spec in methods:
        out.append(_spec_for_method(
            cls, ep_spec, attr_name, method, str(fn_spec.name or attr_name),
            timeout_ms=fn_spec.timeout_ms,
            label=getattr(fn_spec, "label", None),
            description=getattr(fn_spec, "description", None),
            allowed_shapes=tuple(getattr(fn_spec, "allowed_shapes", ()) or ()) or class_shapes,
            payload_override=None,
            models=class_models,
            resources=class_res,
            walked_module=walked,
        ))
    return out


def collect_endpoints(module_names: List[str]) -> List[EndpointSpec]:
    """Walk top-level modules (+ submodules) and return every EndpointSpec."""
    out: List[EndpointSpec] = []
    for found in find_endpoint_classes(list(module_names)):
        out.extend(extract_specs(found.cls, walked_module=found.walked_module))
    return out


def collect_from_namespace(module: Any) -> List[EndpointSpec]:
    """Extract EndpointSpecs from a single already-imported module's namespace."""
    out: List[EndpointSpec] = []
    seen: set[int] = set()
    for obj in list(vars(module).values()):
        if not inspect.isclass(obj) or id(obj) in seen:
            continue
        seen.add(id(obj))
        out.extend(extract_specs(obj))
    return out
