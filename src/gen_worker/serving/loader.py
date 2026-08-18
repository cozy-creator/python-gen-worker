"""Catalog-free endpoint loading (pgw#1372): the module IS the surface.

``endpoint.toml`` names the author's module (``main = "sdxl.main"``); the
module carries ONE endpoint class; the class's public methods taking a
msgspec payload are the routable handlers. No ModelSpec, no family registry,
no codegen imports — the loader reads author code and the one marker the
``@endpoint`` decorator stamps.

The decorator itself belongs to the author-surface lane; the SEAM is
:data:`ENDPOINT_ATTR`: whatever decorates the class stamps an
:class:`EndpointDeclaration` there. A module whose single setup-bearing
class carries no marker still loads — as an eager-permanent endpoint with no
lanes — because the EAGER path must work standalone first.
"""

from __future__ import annotations

import importlib
import inspect
import sys
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Optional, Tuple, Type, get_type_hints

import msgspec

#: The one attribute the @endpoint decorator stamps on the author's class.
ENDPOINT_ATTR = "__cozy_endpoint__"


class EndpointLoadError(RuntimeError):
    """The endpoint module cannot state a loadable endpoint class."""


@dataclass(frozen=True, slots=True)
class EndpointDeclaration:
    """What ``@endpoint(lanes=..., samples=...)`` stamps: the author surface.

    ``lanes`` are execution lanes (torchcg ``Lane``-shaped: name, compile
    target paths, tensor-layout contract, dtype); an empty tuple is the
    explicit eager-permanent shape. ``samples`` is the trace-coverage
    callable — payloads the publish-time discovery drives through the
    handlers. Held structurally (no isinstance on Lane) so the author's
    ``torchcg`` and the worker's vendored copy never fight over class
    identity.
    """

    lanes: Tuple[Any, ...] = ()
    samples: Optional[Callable[[], Tuple[Any, ...]]] = None


@dataclass(frozen=True, slots=True)
class Handler:
    """One routable handler: ``def name(self, ctx, payload: Struct) -> Out``."""

    name: str
    fn: Callable[..., Any]
    payload_type: Type[msgspec.Struct]


@dataclass(frozen=True, slots=True)
class LoadedEndpoint:
    """The author's endpoint, loaded: class + declaration + handler table."""

    module_name: str
    cls: type
    declaration: EndpointDeclaration
    handlers: Dict[str, Handler] = field(default_factory=dict)

    def lane(self, contract: str = "") -> Any:
        """The lane for one contract handle, or the single declared lane.
        A lane IS a tensorfs contract reference (Paul's main_v2 review
        ruling); ambiguity refuses — a multi-lane endpoint's active lane is
        the deploy's pick, never a default."""
        lanes = self.declaration.lanes
        if not lanes:
            return None
        declared = sorted(str(getattr(lane, "contract", lane)) for lane in lanes)
        if not contract:
            if len(lanes) == 1:
                return lanes[0]
            raise EndpointLoadError(
                f"{self.module_name}: {len(lanes)} lanes declared "
                f"({declared}); the active lane must be named by contract"
            )
        for lane in lanes:
            if getattr(lane, "contract", None) == contract:
                return lane
        raise EndpointLoadError(
            f"{self.module_name}: no lane {contract!r} (declared: {declared})"
        )


def _handlers_of(cls: type) -> Dict[str, Handler]:
    handlers: Dict[str, Handler] = {}
    for name, fn in vars(cls).items():
        if name.startswith("_") or name in ("setup", "teardown") or not inspect.isfunction(fn):
            continue
        parameters = list(inspect.signature(fn).parameters.values())
        if len(parameters) != 3:  # self, ctx, payload — the one handler shape
            continue
        try:
            hints = get_type_hints(fn)
        except Exception as exc:  # noqa: BLE001 — a bad hint is the author's bug
            raise EndpointLoadError(
                f"{cls.__name__}.{name}: unresolvable type hints: {exc}"
            ) from exc
        payload_type = hints.get(parameters[2].name)
        if not (isinstance(payload_type, type) and issubclass(payload_type, msgspec.Struct)):
            continue
        handlers[name] = Handler(name=name, fn=fn, payload_type=payload_type)
    if not handlers:
        raise EndpointLoadError(
            f"{cls.__name__} declares no handlers: a handler is a public method "
            f"(self, ctx, payload) whose payload annotation is a msgspec.Struct"
        )
    return handlers


def _verify_hooks(cls: type) -> None:
    """The base-class behavior contract, held: an overridden hook keeps the
    typed ``(self, ctx)`` shape. Framework capabilities arrive via ctx ONLY
    (Paul ruling 2026-08-18) — a hook that asks for more is asking the base
    to be a toolbox, and the refusal names it."""
    for hook in ("setup", "teardown"):
        fn = vars(cls).get(hook)
        if fn is None:
            continue  # inherited no-op
        if not inspect.isfunction(fn):
            raise EndpointLoadError(
                f"{cls.__name__}.{hook} must be a plain method, got "
                f"{type(fn).__name__}"
            )
        positional = [
            parameter
            for parameter in inspect.signature(fn).parameters.values()
            if parameter.kind
            in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional) != 2:
            raise EndpointLoadError(
                f"{cls.__name__}.{hook} must accept exactly (self, ctx); "
                f"framework capabilities arrive via ctx only"
            )


def _endpoint_class(module: ModuleType) -> tuple[type, EndpointDeclaration]:
    from .endpoint import Endpoint

    own_classes = [
        value
        for value in vars(module).values()
        if isinstance(value, type) and value.__module__ == module.__name__
    ]
    marked = [cls for cls in own_classes if ENDPOINT_ATTR in vars(cls)]
    if len(marked) > 1:
        raise EndpointLoadError(
            f"{module.__name__} marks {len(marked)} endpoint classes; one module, "
            f"one endpoint"
        )
    if marked:
        cls = marked[0]
        declaration = vars(cls)[ENDPOINT_ATTR]
        if not isinstance(declaration, EndpointDeclaration):
            raise EndpointLoadError(
                f"{module.__name__}.{cls.__name__}.{ENDPOINT_ATTR} is "
                f"{type(declaration).__name__}, not an EndpointDeclaration"
            )
    else:
        # Unmarked bridge shape: the single Endpoint subclass is the
        # endpoint, eager-permanent — plain author code runs before the
        # decorator lane lands (the always-runnable eager bridge).
        subclasses = [
            value for value in own_classes
            if issubclass(value, Endpoint) and value is not Endpoint
        ]
        if len(subclasses) != 1:
            raise EndpointLoadError(
                f"{module.__name__} defines {len(subclasses)} Endpoint "
                f"subclasses; expected exactly one endpoint class "
                f"(class MyEndpoint(gen_worker.Endpoint))"
            )
        cls, declaration = subclasses[0], EndpointDeclaration()
    if not issubclass(cls, Endpoint):
        raise EndpointLoadError(
            f"{module.__name__}.{cls.__name__} does not inherit "
            f"gen_worker.Endpoint — the base class is REQUIRED (Paul ruling "
            f"2026-08-18): class {cls.__name__}(Endpoint)"
        )
    _verify_hooks(cls)
    return cls, declaration


def load_endpoint_module(module_name: str) -> LoadedEndpoint:
    """Import the author's module and state its endpoint class + handlers."""

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise EndpointLoadError(
            f"endpoint module {module_name!r} does not import: {exc}"
        ) from exc
    cls, declaration = _endpoint_class(module)
    return LoadedEndpoint(
        module_name=module_name,
        cls=cls,
        declaration=declaration,
        handlers=_handlers_of(cls),
    )


def load_endpoint(endpoint_dir: str | Path) -> LoadedEndpoint:
    """Load an endpoint from its directory: ``endpoint.toml``'s ``main =``.

    A ``src/`` layout that is not already importable joins ``sys.path`` —
    the local/test convenience; production installs the package and the
    module imports without help.
    """

    root = Path(endpoint_dir)
    manifest = root / "endpoint.toml"
    if not manifest.is_file():
        raise EndpointLoadError(f"{root} has no endpoint.toml")
    try:
        parsed = tomllib.loads(manifest.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise EndpointLoadError(f"{manifest} is not valid TOML: {exc}") from exc
    main = parsed.get("main")
    if not isinstance(main, str) or not main.strip():
        raise EndpointLoadError(f"{manifest} declares no `main = \"pkg.module\"`")
    src = root / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return load_endpoint_module(main.strip())


__all__ = [
    "ENDPOINT_ATTR",
    "EndpointDeclaration",
    "EndpointLoadError",
    "Handler",
    "LoadedEndpoint",
    "load_endpoint",
    "load_endpoint_module",
]
