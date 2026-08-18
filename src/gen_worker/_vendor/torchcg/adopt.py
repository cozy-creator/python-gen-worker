"""Adopt-first swap-in: compiled graphs replace module forwards (pgw#1372).

The serving worker never derives and never compiles here (trace-once-at-
publish ruling, 2026-08-18): the caller hands the release's
``GraphSetDocument``, a ``GraphStore`` and the author's own live objects.
Adoption asserts the exact-env audit (``assert_exact_env`` -- a mismatch is a
build-system bug refusing loudly, never a compat decision), fetches every
artifact that exists at (graph, env), and arms each behind a per-module
dispatcher installed on the module the author's path names -- the module
actually CALLED, so the swap intercepts ``Module.__call__``'s own
``self.forward`` read.

Partial-hit is the contract: a miss or an unreadable row is a HOLE, never a
boot failure. ``LaneAdoption.holes`` is the ordered mint work-list the
background mint (pgw#1371's mechanism) consumes, arming each artifact as it
lands via :meth:`LaneAdoption.arm`. A call no armed graph matches runs the
module's own eager forward -- the author's code stays the serve host.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .document import GraphRecord, GraphSetDocument, LaneGraphs
from .graph_identity import EnvIdentity
from .lane import LaneError, resolve_target
from .requirements import assert_exact_env
from .store import GraphStore, StoreError

#: Turns one fetched artifact into the callable that replaces the target
#: module's forward for its graph class. The real AOTInductor loader arrives
#: with the runtime mint wave; the seam keeps adoption testable without a GPU.
ArtifactLoader = Callable[[Path, GraphRecord], Callable[..., Any]]

HOLE_MISS = "miss"


class AdoptError(RuntimeError):
    """Adoption cannot even start: bad lane name, bad target, bad document."""


@dataclass(frozen=True, slots=True)
class Hole:
    """One graph this env still needs minted, with the reason it is a hole."""

    record: GraphRecord
    reason: str


def _is_concrete(record: GraphRecord) -> bool:
    return all(
        all(isinstance(dimension, int) for dimension in row.shape)
        for row in record.ingress.inputs
    )


def _structure_key(record: GraphRecord) -> tuple[Any, ...]:
    """The dispatchable identity of a record's tensor structure.

    Two graph classes whose tensor feeds are identical (they differ only in
    baked-in literals) cannot be told apart at call time; arming both would
    be a coin flip, so the dispatcher refuses the pair instead.
    """

    return tuple(
        (row.param, row.path, row.dtype, row.shape) for row in record.ingress.inputs
    )


def _matches(record: GraphRecord, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    import torch

    for row in record.ingress.inputs:
        resolved, value = row.resolve(args, kwargs)
        if not resolved or not isinstance(value, torch.Tensor):
            return False
        if str(value.dtype).removeprefix("torch.") != row.dtype:
            return False
        if len(value.shape) != len(row.shape):
            return False
        for got, want in zip(value.shape, row.shape, strict=True):
            if isinstance(want, int) and int(got) != want:
                return False
    return True


class AmbiguousStructure(Exception):
    """Two graph classes share one tensor structure; neither can be dispatched."""

    def __init__(self, armed: GraphRecord) -> None:
        self.armed = armed
        super().__init__(f"structure collides with armed graph {armed.graph}")


class _ForwardDispatcher:
    """One module's serve-path router: armed graphs by call structure, else eager.

    Installed as the module's instance ``forward`` so ``Module.__call__``
    (which reads ``self.forward``) routes through it -- hooks and everything
    else on the module keep working, and removing the instance attribute
    restores the eager class forward untouched.
    """

    __slots__ = ("eager_forward", "_entries")

    def __init__(self, module: Any) -> None:
        self.eager_forward = module.forward
        self._entries: list[tuple[GraphRecord, Callable[..., Any]]] = []

    def arm(self, record: GraphRecord, compiled: Callable[..., Any]) -> None:
        key = _structure_key(record)
        for armed, _ in self._entries:
            if armed.graph == record.graph:
                raise AdoptError(f"graph {record.graph} is already armed on this module")
            if _structure_key(armed) == key:
                raise AmbiguousStructure(armed)
        # Concrete-shape records dispatch before symbolic ones so an exact
        # bucket wins over a dynamic-range graph that also spans it.
        if _is_concrete(record):
            insert_at = sum(1 for armed, _ in self._entries if _is_concrete(armed))
            self._entries.insert(insert_at, (record, compiled))
        else:
            self._entries.append((record, compiled))

    def disarm(self, graph: str) -> None:
        self._entries = [(r, c) for r, c in self._entries if r.graph != graph]

    def armed_graphs(self) -> tuple[str, ...]:
        return tuple(record.graph for record, _ in self._entries)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        for record, compiled in self._entries:
            if _matches(record, args, kwargs):
                return compiled(*args, **kwargs)
        return self.eager_forward(*args, **kwargs)


@dataclass(slots=True)
class LaneAdoption:
    """One lane's boot adoption: what armed, what is eager, what to mint.

    ``holes`` is THE handoff to the background mint: canonical document order
    (the ``LaneGraphs`` ordering -- sorted by (target, graph)), one row per
    graph this (lane, env) still needs, each carrying its ``GraphRecord``
    (graph hash + target + ingress) and a stated reason. The mint publishes
    per-graph and calls :meth:`arm` as each lands -- never all-or-nothing.
    """

    lane: LaneGraphs
    env: EnvIdentity
    adopted: tuple[GraphRecord, ...]
    holes: tuple[Hole, ...]
    ambiguous: tuple[GraphRecord, ...]
    _loader: ArtifactLoader
    _artifacts_dir: Path
    _dispatchers: dict[str, _ForwardDispatcher] = field(default_factory=dict)

    def armed_graphs(self, target: str) -> tuple[str, ...]:
        dispatcher = self._dispatchers.get(target)
        return dispatcher.armed_graphs() if dispatcher else ()

    def arm(self, record: GraphRecord, artifact: Path) -> None:
        """Arm one late-landing mint (or live-encounter graph) onto its target.

        The mint lane calls this per graph as each publish completes; the
        record's target must be one of this lane's dispatched modules.
        """

        dispatcher = self._dispatchers.get(record.target)
        if dispatcher is None:
            raise AdoptError(
                f"record targets {record.target!r}, which this lane never dispatched "
                f"(targets: {sorted(self._dispatchers)!r})"
            )
        compiled = self._loader(Path(artifact), record)
        try:
            dispatcher.arm(record, compiled)
        except AmbiguousStructure as exc:
            # Neither twin can be dispatched honestly: de-arm the survivor too
            # and serve both structures eager. A re-mint cannot fix this, so
            # the pair is reported, not queued.
            dispatcher.disarm(exc.armed.graph)
            self.adopted = tuple(r for r in self.adopted if r.graph != exc.armed.graph)
            self.ambiguous = (*self.ambiguous, exc.armed, record)
            return
        self.adopted = (*self.adopted, record)
        self.holes = tuple(hole for hole in self.holes if hole.record.graph != record.graph)


def adopt_lane(
    store: GraphStore,
    document: GraphSetDocument,
    lane_name: str,
    roots: Mapping[str, object],
    sm: str,
    *,
    loader: ArtifactLoader,
    artifacts_dir: str | Path,
    installed: Mapping[str, str],
) -> LaneAdoption:
    """Adopt one lane's compiled graphs onto the author's live objects.

    Runs the exact-env audit first (``EnvironmentMismatch`` propagates -- a
    loud typed refusal), resolves every declared target to a real module
    (a typo must not become an eager-forever lane), installs one dispatcher
    per target, then fetches and arms per graph. Misses and unreadable rows
    become ordered ``holes``; nothing here is all-or-nothing.
    """

    import torch

    lane = next((row for row in document.lanes if row.name == lane_name), None)
    if lane is None:
        available = sorted(row.name for row in document.lanes)
        raise AdoptError(
            f"document has no lane {lane_name!r} "
            f"({'lanes: ' + repr(available) if available else 'eager-permanent document'})"
        )
    env = EnvIdentity(closure=document.closure, sm=sm)
    assert_exact_env(env, installed=installed, sm=sm)

    dispatchers: dict[str, _ForwardDispatcher] = {}
    for path in lane.targets:
        try:
            module = resolve_target(roots, path)
        except LaneError as exc:
            raise AdoptError(str(exc)) from exc
        if not isinstance(module, torch.nn.Module):
            raise AdoptError(
                f"lane {lane.name!r} target {path!r} resolves to "
                f"{type(module).__name__}, which is not a torch.nn.Module"
            )
        dispatcher = _ForwardDispatcher(module)
        module.forward = dispatcher  # instance attr: Module.__call__ reads it
        dispatchers[path] = dispatcher

    adoption = LaneAdoption(
        lane=lane,
        env=env,
        adopted=(),
        holes=(),
        ambiguous=(),
        _loader=loader,
        _artifacts_dir=Path(artifacts_dir),
        _dispatchers=dispatchers,
    )
    holes: list[Hole] = []
    for record in lane.graphs:
        destination = adoption._artifacts_dir / env.value / f"{record.graph}.so"
        try:
            artifact = store.fetch_artifact(record.graph, env, destination)
        except StoreError as exc:
            holes.append(Hole(record=record, reason=f"store_error: {exc}"))
            continue
        if artifact is None:
            holes.append(Hole(record=record, reason=HOLE_MISS))
            continue
        adoption.arm(record, artifact)
    # arm() above already pruned adopted holes; state the remainder in
    # canonical document order.
    adoption.holes = tuple(holes)
    return adoption


__all__ = [
    "AdoptError",
    "ArtifactLoader",
    "HOLE_MISS",
    "Hole",
    "LaneAdoption",
    "adopt_lane",
]
