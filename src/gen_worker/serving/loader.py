from __future__ import annotations

import importlib
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Tuple

from .entrypoints import ENTRYPOINT_ATTR, EntrypointSpec
from .model import (
    ModelDeclarationError,
    lane_handle,
    model_declared_lanes,
    model_lanes,
    model_type,
)


class EndpointLoadError(RuntimeError):
    """The endpoint module does not state a loadable serve surface."""


@dataclass(frozen=True, slots=True)
class LoadedEndpoint:
    """The author's module, loaded and statically extracted."""

    module_name: str
    entrypoints: Dict[str, EntrypointSpec]
    models: Tuple[type, ...]

    def model_type_of(self, model_cls: type) -> type:
        return model_type(model_cls)

    def lanes_of(self, model_cls: type) -> Tuple[Any, ...]:
        return model_lanes(model_cls)

    def lane(self, model_cls: type, contract: str = "") -> Any:
        """The active lane for one model class: the deploy's pick by contract handle, or the single declared lane."""
        lanes = self.lanes_of(model_cls)
        if not lanes:
            return None
        declared = sorted(lane_handle(lane) for lane in lanes)
        if not contract:
            if len(lanes) == 1:
                return lanes[0]
            raise EndpointLoadError(
                f"{self.module_name}.{model_cls.__name__}: {len(lanes)} lanes "
                f"declared ({declared}); ask `resolve()` for the boot ladder's "
                "pick — `lane()` answers only a lane named by contract or a "
                "single declared one"
            )
        for lane in lanes:
            if lane_handle(lane) == contract:
                return lane
        raise EndpointLoadError(
            f"{self.module_name}.{model_cls.__name__}: no lane {contract!r} "
            f"(declared: {declared})"
        )

    def resolve(
        self, model_cls: type, *, card: Any, verdicts: Any, gates: Any,
        contract: str = "",
    ) -> Any:
        """THE boot pick: rank this model's declared lanes and choose one."""
        from . import lane_ladder

        lanes = self.lanes_of(model_cls)
        if not lanes:
            return None
        if contract:
            lanes = tuple(
                lane for lane in lanes if lane_handle(lane) == contract)
            if not lanes:
                raise EndpointLoadError(
                    f"{self.module_name}.{model_cls.__name__}: no lane "
                    f"{contract!r} to pin (declared: "
                    f"{sorted(lane_handle(row) for row in self.lanes_of(model_cls))})"
                )
        handles = {lane_handle(lane) for lane in lanes}
        declared = tuple(
            row for row in model_declared_lanes(model_cls)
            if row.contract_id in handles
        )
        return lane_ladder.resolve_lane(
            declared=declared, card=card, verdicts=verdicts, gates=gates,
        )


def _surface_of(module: ModuleType) -> tuple[Dict[str, EntrypointSpec], Tuple[type, ...]]:
    entrypoints: Dict[str, EntrypointSpec] = {}
    models: Dict[type, None] = {}
    for value in vars(module).values():
        spec = getattr(value, ENTRYPOINT_ATTR, None)
        if not isinstance(spec, EntrypointSpec):
            continue
        if spec.fn.__module__ != module.__name__:
            continue
        entrypoints[spec.name] = spec
        for cls in spec.model_classes:
            models.setdefault(cls)
    if not entrypoints:
        raise EndpointLoadError(
            f"{module.__name__} declares no entrypoints: an entrypoint is a "
            "module-level @entrypoint function "
            "(ctx: RequestContext, payload: msgspec.Struct) plus zero or "
            "more slots"
        )
    for cls in models:
        try:
            model_type(cls)
            model_lanes(cls)
        except ModelDeclarationError as exc:
            raise EndpointLoadError(str(exc)) from exc
    return entrypoints, tuple(models)


def load_endpoint_module(module_name: str) -> LoadedEndpoint:
    """Import the author's module and state its serve surface."""

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise EndpointLoadError(
            f"endpoint module {module_name!r} does not import: {exc}"
        ) from exc
    entrypoints, models = _surface_of(module)
    return LoadedEndpoint(
        module_name=module_name, entrypoints=entrypoints, models=models,
    )


def endpoint_module_name(endpoint_dir: str | Path) -> str:
    """The module name ``endpoint.toml`` declares (``main =``), WITHOUT importing it."""

    manifest = Path(endpoint_dir) / "endpoint.toml"
    if not manifest.is_file():
        raise EndpointLoadError(f"{Path(endpoint_dir)} has no endpoint.toml")
    try:
        parsed = tomllib.loads(manifest.read_text())
    except tomllib.TOMLDecodeError as exc:
        raise EndpointLoadError(f"{manifest} is not valid TOML: {exc}") from exc
    main = parsed.get("main")
    if not isinstance(main, str) or not main.strip():
        raise EndpointLoadError(f"{manifest} declares no `main = \"pkg.module\"`")
    return main.strip()


def load_endpoint(endpoint_dir: str | Path) -> LoadedEndpoint:
    """Load an endpoint from its directory: ``endpoint.toml``'s ``main =``."""

    root = Path(endpoint_dir)
    main = endpoint_module_name(root)
    src = root / "src"
    if src.is_dir() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return load_endpoint_module(main)


__all__ = [
    "EndpointLoadError",
    "LoadedEndpoint",
    "endpoint_module_name",
    "load_endpoint",
    "load_endpoint_module",
]
