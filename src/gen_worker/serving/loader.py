"""Catalog-free endpoint loading (pgw#1372/pgw#1382): the module IS the surface.

``endpoint.toml`` names the author's module (``main = "sdxl.main"``); an
"endpoint" is the deployable-unit NOUN, not a class: the module's
``@entrypoint`` functions plus the Model classes their annotations
reference. Loading imports the module and READS what the decorator and the
``Model`` class header stamped — payload schema per entrypoint, model class
per slot, (model type, lanes) per model class — no author code executed
beyond import, no ModelSpec, no family registry, no codegen.
"""

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
    """The author's module, loaded and statically extracted.

    ``entrypoints`` routes by function name; ``models`` are the referenced
    model classes in first-reference order (each carrying its model type and
    lanes in the class header).
    """

    module_name: str
    entrypoints: Dict[str, EntrypointSpec]
    models: Tuple[type, ...]

    def model_type_of(self, model_cls: type) -> type:
        return model_type(model_cls)

    def lanes_of(self, model_cls: type) -> Tuple[Any, ...]:
        return model_lanes(model_cls)

    def lane(self, model_cls: type, contract: str = "") -> Any:
        """The active lane for one model class: the deploy's pick by
        contract handle, or the single declared lane.

        pgw#1599: `lanes=` is required, so the empty case is gone. Ambiguity
        still refuses — a multi-lane model's active lane is the RESOLVER's
        pick (pgw#1606's boot-time ladder), never a default chosen here."""
        lanes = self.lanes_of(model_cls)
        if not lanes:
            return None
        declared = sorted(lane_handle(lane) for lane in lanes)
        if not contract:
            if len(lanes) == 1:
                return lanes[0]
            # pgw#1606: this used to raise, and that refusal is why no Model
            # class in the fleet has ever declared more than one lane — a
            # multi-lane endpoint simply could not boot on a pod, because the
            # only writers of `contract` are the local CLI and the daemon
            # (`entrypoint.py` builds `Worker(...)` with no `lane=` at all).
            # Picking among declared lanes is PLATFORM work and it now has a
            # home: `serving.lane_ladder.resolve_lane`. Use `resolve()` below,
            # which ranks them and says why. Reaching here means a caller
            # asked for "the" lane of a multi-lane model without running the
            # ladder, which is a wiring defect, not a deployment state.
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
        """THE boot pick: rank this model's declared lanes and choose one.

        pgw#1606. Returns a `lane_ladder.ResolvedLane` — the chosen lane, the
        reason, and the rejected rungs in order — or ``None`` for a model that
        declares no lanes (`eager_only`), which has nothing to resolve.

        A non-empty ``contract`` is an OPERATOR OVERRIDE (the local CLI's
        ``--lane``): it pins the candidate set to one lane, and the ladder
        still evaluates that lane's floor, gate and bytes so a pinned lane
        that cannot run says why instead of failing later and elsewhere.
        """
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
                    f"{sorted(lane_handle(l) for l in self.lanes_of(model_cls))})"
                )
        # pgw#1599's DeclaredLane rows, filtered to the (possibly pinned)
        # candidate set. Read, never rebuilt: the stamp is not re-parsed and
        # `min_sm` is not re-derived, so this consumer cannot disagree with
        # the declaration about a floor.
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
            continue  # re-exported from elsewhere: not this module's surface
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
    """The module name ``endpoint.toml`` declares (``main =``), WITHOUT importing it.

    Still the ONE reader of ``main =`` (pgw#1537's drift concern):
    :func:`load_endpoint` resolves the name through this very function, so a
    caller that needs only the NAME — ``gen-worker compile`` publishes and
    looks up the graph-set document by it — shares the reader without paying
    the author-module import (torch, diffusers: measured ~4.5 s on a warm
    all-present compile, pgw#1546) that ``load_endpoint``'s callers actually
    need.
    """

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
    """Load an endpoint from its directory: ``endpoint.toml``'s ``main =``.

    A ``src/`` layout that is not already importable joins ``sys.path`` —
    the local/test convenience; production installs the package and the
    module imports without help.
    """

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
