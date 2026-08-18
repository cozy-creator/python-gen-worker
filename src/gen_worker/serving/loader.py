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

    def lane(self, name: str = "") -> Any:
        """The named lane, or the single declared lane. Ambiguity refuses:
        a multi-lane endpoint's active lane is the deploy's pick, never a
        default."""
        lanes = self.declaration.lanes
        if not lanes:
            return None
        if not name:
            if len(lanes) == 1:
                return lanes[0]
            raise EndpointLoadError(
                f"{self.module_name}: {len(lanes)} lanes declared "
                f"({sorted(str(getattr(lane, 'name', lane)) for lane in lanes)}); "
                f"the active lane must be named"
            )
        for lane in lanes:
            if getattr(lane, "name", None) == name:
                return lane
        raise EndpointLoadError(
            f"{self.module_name}: no lane named {name!r} "
            f"(declared: {sorted(str(getattr(lane, 'name', lane)) for lane in lanes)})"
        )


def _handlers_of(cls: type) -> Dict[str, Handler]:
    handlers: Dict[str, Handler] = {}
    for name, fn in vars(cls).items():
        if name.startswith("_") or name == "setup" or not inspect.isfunction(fn):
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


def _endpoint_class(module: ModuleType) -> tuple[type, EndpointDeclaration]:
    own_classes = [
        value
        for value in vars(module).values()
        if isinstance(value, type) and value.__module__ == module.__name__
    ]
    marked = [cls for cls in own_classes if getattr(cls, ENDPOINT_ATTR, None) is not None]
    if len(marked) > 1:
        raise EndpointLoadError(
            f"{module.__name__} marks {len(marked)} endpoint classes; one module, "
            f"one endpoint"
        )
    if marked:
        declaration = getattr(marked[0], ENDPOINT_ATTR)
        if not isinstance(declaration, EndpointDeclaration):
            raise EndpointLoadError(
                f"{module.__name__}.{marked[0].__name__}.{ENDPOINT_ATTR} is "
                f"{type(declaration).__name__}, not an EndpointDeclaration"
            )
        return marked[0], declaration
    # Unmarked bridge shape: the single class with a setup() is the endpoint,
    # eager-permanent. Stated so plain author code runs before the decorator
    # lane lands — the always-runnable eager bridge.
    with_setup = [cls for cls in own_classes if callable(getattr(cls, "setup", None))]
    if len(with_setup) == 1:
        return with_setup[0], EndpointDeclaration()
    raise EndpointLoadError(
        f"{module.__name__} has {'no' if not with_setup else len(with_setup)} "
        f"endpoint class: expected exactly one class stamped by @endpoint "
        f"(or exactly one class with a setup method)"
    )


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
