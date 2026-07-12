"""The ONE decorator walker: ``@endpoint`` objects -> EndpointSpec table.

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

from .api.binding import Binding
from .api.decorators import ATTR, Compile, EndpointDecl, Resources
from .discovery.names import slugify_name
from .discovery.walk import find_endpoints

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
    """One externally-routable function: a handler plus wire shape.

    ``cls`` is None for function-shaped (stateless) endpoints — the worker
    calls ``method`` directly with no instance/setup.
    """

    name: str                     # routable function name (canonical slug)
    method: Callable[..., Any]    # unbound function object
    kind: str                     # inference | training | dataset | conversion
    payload_type: type            # msgspec.Struct
    output_mode: str              # "single" | "stream"
    cls: Optional[type] = None
    attr_name: str = ""           # python attribute name on the class
    output_type: Optional[type] = None   # struct returned (single mode)
    delta_type: Optional[type] = None    # struct yielded (stream mode; None if union)
    is_async: bool = False        # coroutine or async generator
    is_async_gen: bool = False
    ctx_param: str = "ctx"
    payload_param: str = "payload"
    resources: Resources = field(default_factory=Resources)
    models: Dict[str, Binding] = field(default_factory=dict)  # slot -> binding
    timeout_ms: Optional[int] = None
    runtime: Optional[str] = None
    compile: Optional[Compile] = None  # opt-in torch.compile spec (#384)
    module: str = ""              # declaring module
    walked_module: str = ""       # top-level package the object was found under

    @property
    def needs_gpu(self) -> bool:
        return bool(self.resources.gpu)

    @property
    def instance_key(self) -> Any:
        """Specs sharing this key share one class instance (same class + same
        resolved binding set)."""
        return (self.cls, tuple(sorted(self.models.items())))


def _inspect_return(owner: str, ret: Any) -> tuple[str, Optional[type], Optional[type]]:
    """-> (output_mode, output_type, delta_type)."""
    if _is_struct(ret):
        return "single", ret, None
    origin = typing.get_origin(ret)
    if origin in _ITER_ORIGINS:
        args = typing.get_args(ret) or ()
        item = args[0] if args else None
        if _is_struct(item):
            return "stream", item, item
        # Union of delta structs: still a stream.
        if typing.get_origin(item) in (typing.Union, py_types.UnionType):
            return "stream", None, None
        raise ValueError(
            f"{owner}: streaming return must be "
            f"(Async)Iterator[msgspec.Struct], got {ret!r}"
        )
    raise ValueError(
        f"{owner}: return type must be msgspec.Struct or "
        f"(Async)Iterator[msgspec.Struct], got {ret!r}"
    )


def _spec_for_handler(
    *,
    fn_name: str,
    method: Callable[..., Any],
    decl: EndpointDecl,
    cls: Optional[type],
    attr_name: str,
    models: Dict[str, Binding],
    resources: Resources,
    walked_module: str,
) -> EndpointSpec:
    owner = f"{cls.__name__}.{attr_name}" if cls is not None else method.__name__
    hints = typing.get_type_hints(method, include_extras=False)
    sig = inspect.signature(method)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    if len(params) < 2:
        raise ValueError(
            f"{owner}: must accept (ctx, payload); got params "
            f"{[p.name for p in params]}"
        )
    ctx_param, payload_param = params[0].name, params[1].name

    payload_type = hints.get(payload_param)
    if not isinstance(payload_type, type) or not _is_struct(payload_type):
        raise ValueError(
            f"{owner}: payload param {payload_param!r} must be annotated "
            f"with a msgspec.Struct (got {payload_type!r})"
        )
    ret = hints.get("return")
    if ret is None:
        raise ValueError(f"{owner}: missing return type annotation")
    output_mode, output_type, delta_type = _inspect_return(owner, ret)

    # The wire/dispatch name is the SLUG (matches the discovery manifest and
    # tensorhub's canonical function names — `_` -> `-`): the orchestrator's
    # RunJob.function_name and the worker's advertised available_functions
    # must agree, and the platform normalizes to slugs everywhere.
    slug = slugify_name(fn_name)
    if not slug:
        raise ValueError(f"{owner}: function name {fn_name!r} cannot be normalized")

    return EndpointSpec(
        name=slug,
        method=method,
        kind=decl.kind,
        payload_type=payload_type,
        output_mode=output_mode,
        cls=cls,
        attr_name=attr_name,
        output_type=output_type,
        delta_type=delta_type,
        is_async=bool(
            inspect.iscoroutinefunction(method) or inspect.isasyncgenfunction(method)
        ),
        is_async_gen=inspect.isasyncgenfunction(method),
        ctx_param=ctx_param,
        payload_param=payload_param,
        resources=resources,
        models=dict(models),
        runtime=decl.runtime,
        compile=decl.compile,
        module=getattr(cls or method, "__module__", "") or "",
        walked_module=walked_module,
    )


def extract_specs(obj: Any, *, walked_module: str = "") -> List[EndpointSpec]:
    """All EndpointSpecs declared by one decorated object (one per handler)."""
    decl: Optional[EndpointDecl] = getattr(obj, ATTR, None)
    if decl is None:
        return []
    walked = walked_module or (getattr(obj, "__module__", "") or "")

    if decl.is_function:
        return [_spec_for_handler(
            fn_name=decl.name or obj.__name__,
            method=obj,
            decl=decl,
            cls=None,
            attr_name="",
            models=dict(decl.models),
            resources=decl.resources,
            walked_module=walked,
        )]

    cls = obj
    handlers: list[tuple[str, Callable[..., Any]]] = list(
        getattr(cls, "__gen_worker_handlers__", []) or []
    )
    out: List[EndpointSpec] = []

    for attr_name, method in handlers:
        out.append(_spec_for_handler(
            fn_name=attr_name, method=method, decl=decl, cls=cls,
            attr_name=attr_name, models=dict(decl.models),
            resources=decl.resources, walked_module=walked,
        ))
    return out


def collect_endpoints(module_names: List[str]) -> List[EndpointSpec]:
    """Walk top-level modules (+ submodules) and return every EndpointSpec."""
    out: List[EndpointSpec] = []
    for found in find_endpoints(list(module_names)):
        out.extend(extract_specs(found.obj, walked_module=found.walked_module))
    _assert_unique_names(out)
    return out


def collect_from_namespace(module: Any) -> List[EndpointSpec]:
    """Extract EndpointSpecs from a single already-imported module's namespace."""
    out: List[EndpointSpec] = []
    seen: set[int] = set()
    for obj in list(vars(module).values()):
        if not (inspect.isclass(obj) or inspect.isfunction(obj)) or id(obj) in seen:
            continue
        seen.add(id(obj))
        out.extend(extract_specs(obj))
    _assert_unique_names(out)
    return out


def _assert_unique_names(specs: List[EndpointSpec]) -> None:
    seen: Dict[str, EndpointSpec] = {}
    for s in specs:
        prior = seen.get(s.name)
        if prior is not None and prior.method is not s.method:
            raise ValueError(
                f"duplicate routable function name {s.name!r} "
                f"({prior.module} and {s.module})"
            )
        seen[s.name] = s
