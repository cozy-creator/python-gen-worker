"""Adopt-first swap-in behind ``ctx.compile`` (pgw#1372, ruling 2026-08-18).

Module marking is IMPERATIVE — the author writes the torch.compile idiom::

    self.pipe.unet = ctx.compile(self.pipe.unet)

At publish time that call records + hooks the module (the derive's half); at
serve time it lands here: :class:`AdoptSession` holds one boot's adoption
state for the active lane, and ``session.adopt(module)`` consults the store
per graph — hit -> the module comes back with the compiled graph armed
behind a per-module dispatcher; miss -> the module comes back UNCHANGED
(eager) and the graph is registered as a :class:`Hole`, the ordered
work-list the background mint (pgw#1371's mechanism) consumes, arming each
artifact as it lands via :meth:`AdoptSession.arm`. ``adopt`` on a pipeline
container walks its ``components`` mapping.

The exact-env audit runs at session CONSTRUCTION — before any author code
touches an artifact — and a mismatch is a loud ``EnvironmentMismatch``
(a build-system bug surfacing, never a compat decision). Everything after is
partial-hit by construction: an unreadable row is a hole, never a boot
failure, and a call no armed graph matches runs the module's own eager
forward — the author's code stays the serve host.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .document import GraphRecord, GraphSetDocument, LaneGraphs
from .graph_identity import EnvIdentity
from .requirements import assert_exact_env
from .store import GraphStore, StoreError

if TYPE_CHECKING:  # torch-free import closure: transform.py is torch-shaped
    from .transform import TransformSet

#: Turns one fetched artifact into the callable that replaces the target
#: module's forward for its graph class. The real AOTInductor loader arrives
#: with the runtime mint wave; the seam keeps adoption testable without a GPU.
ArtifactLoader = Callable[[Path, GraphRecord], Callable[..., Any]]

HOLE_MISS = "miss"


class AdoptError(RuntimeError):
    """Adoption cannot even start: bad lane name, bad target, bad document."""


class Hole:
    """One graph this env still needs minted, with the reason it is a hole."""

    __slots__ = ("record", "reason")

    def __init__(self, record: GraphRecord, reason: str) -> None:
        self.record = record
        self.reason = reason

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return f"Hole({self.record.graph}, {self.reason!r})"


class AmbiguousStructure(Exception):
    """Two graph classes share one tensor structure; neither can be dispatched."""

    def __init__(self, armed: GraphRecord) -> None:
        self.armed = armed
        super().__init__(f"structure collides with armed graph {armed.graph}")


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


def _forward_parameters(module: Any) -> frozenset[str]:
    return frozenset(
        parameter.name
        for parameter in inspect.signature(module.forward).parameters.values()
        if parameter.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    )


class AdoptSession:
    """One boot's adoption for the active (lane, sm) — ``ctx.compile``'s engine.

    A record is CLAIMED by the first ``adopt``-ed module whose forward
    signature admits every parameter the record's ingress names; runtime
    dispatch still verifies the full call structure per call, so a record on
    a signature-twin module can never serve a wrong answer — it simply never
    matches and the call runs eager.

    ``store`` may be ``None`` (metadata known, no artifact source yet):
    every claimed record is then a hole — the mint work-list still forms.
    """

    def __init__(
        self,
        store: GraphStore | None,
        document: GraphSetDocument,
        contract: str,
        sm: str,
        *,
        loader: ArtifactLoader,
        artifacts_dir: str | Path,
        installed: Mapping[str, str],
        transforms: TransformSet | None = None,
    ) -> None:
        lane = next((row for row in document.lanes if row.contract == contract), None)
        if lane is None:
            available = sorted(row.contract for row in document.lanes)
            raise AdoptError(
                f"document has no lane {contract!r} "
                f"({'lanes: ' + repr(available) if available else 'eager-permanent document'})"
            )
        self.lane: LaneGraphs = lane
        self.env = EnvIdentity(closure=document.closure, sm=sm)
        # The exact-env audit, BEFORE any artifact or module is touched.
        assert_exact_env(self.env, installed=installed, sm=sm)
        # And the PASS audit, in the same breath and for the same reason
        # (tcg#52): a pass name is a cg-graph-v1 derivation input, so a boot
        # whose ran-pass set differs from the document's lane row is about to
        # arm graphs derived from a module it does not have. Refusing here is
        # what makes "a pass after export" unrepresentable on the serve side
        # -- the set is stated before any author code is adopted.
        self.passes = self._reconcile_passes(lane, transforms)
        self._store = store
        self._loader = loader
        self._artifacts_dir = Path(artifacts_dir)
        self._unclaimed: dict[str, GraphRecord] = {
            record.graph: record for record in lane.graphs
        }
        self._dispatchers: list[tuple[Any, _ForwardDispatcher]] = []
        self._home: dict[str, _ForwardDispatcher] = {}
        self._adopted: set[str] = set()
        self._holes: dict[str, Hole] = {}
        self._ambiguous: set[str] = set()

    @staticmethod
    def _reconcile_passes(
        lane: LaneGraphs, transforms: TransformSet | None
    ) -> tuple[str, ...]:
        from .transform import TransformOrderError, require_transform_set

        try:
            return require_transform_set(lane.contract, transforms, lane.passes)
        except TransformOrderError as exc:
            raise AdoptError(str(exc)) from exc

    # -- ctx.compile --------------------------------------------------------

    def adopt(self, target: Any) -> Any:
        """The serve half of ``ctx.compile``: returns ``target``, armed.

        An ``nn.Module`` is adopted directly; a pipeline-shaped container
        (anything with a ``components`` mapping) has each nn.Module
        component adopted and comes back itself. Anything else is a typed
        refusal — a silently ignored mark would be an eager-forever module
        nobody stated.
        """

        import torch

        if isinstance(target, torch.nn.Module):
            self._adopt_module(target)
            return target
        components = getattr(target, "components", None)
        if isinstance(components, Mapping):
            for component in components.values():
                if isinstance(component, torch.nn.Module):
                    self._adopt_module(component)
            return target
        raise AdoptError(
            f"ctx.compile expects an nn.Module or a pipeline with a "
            f"`components` mapping, got {type(target).__name__}"
        )

    def _dispatcher_for(self, module: Any) -> _ForwardDispatcher:
        for known, dispatcher in self._dispatchers:
            if known is module:
                return dispatcher
        dispatcher = _ForwardDispatcher(module)
        module.forward = dispatcher  # instance attr: Module.__call__ reads it
        self._dispatchers.append((module, dispatcher))
        return dispatcher

    def _adopt_module(self, module: Any) -> None:
        from .transform import seal_for_export

        # Export-bound from here: a pass handed this module later is refused
        # from the pass side too (TransformSession.run), so the ordering holds
        # whichever end someone starts from.
        seal_for_export(module)
        signature = _forward_parameters(module)
        claimed = [
            record
            for record in self._unclaimed.values()
            if set(record.ingress.parameters) <= signature
        ]
        if not claimed:
            return  # marked module, no graphs for it in this document: eager
        dispatcher = self._dispatcher_for(module)
        for record in claimed:
            del self._unclaimed[record.graph]
            self._home[record.graph] = dispatcher
            if self._store is None:
                self._holes[record.graph] = Hole(record, HOLE_MISS)
                continue
            destination = self._artifacts_dir / self.env.value / f"{record.graph}.so"
            try:
                artifact = self._store.fetch_artifact(record.graph, self.env, destination)
            except StoreError as exc:
                self._holes[record.graph] = Hole(record, f"store_error: {exc}")
                continue
            if artifact is None:
                self._holes[record.graph] = Hole(record, HOLE_MISS)
                continue
            self._arm(dispatcher, record, artifact)

    def _arm(self, dispatcher: _ForwardDispatcher, record: GraphRecord, artifact: Path) -> None:
        compiled = self._loader(Path(artifact), record)
        try:
            dispatcher.arm(record, compiled)
        except AmbiguousStructure as exc:
            # Neither twin can be dispatched honestly: de-arm the survivor too
            # and serve both structures eager. A re-mint cannot fix this, so
            # the pair is reported, not queued.
            dispatcher.disarm(exc.armed.graph)
            self._adopted.discard(exc.armed.graph)
            self._ambiguous.update((exc.armed.graph, record.graph))
            return
        self._adopted.add(record.graph)
        self._holes.pop(record.graph, None)

    # -- the mint handoff ---------------------------------------------------

    def arm(self, record: GraphRecord, artifact: Path) -> None:
        """Arm one late-landing mint onto the module that claimed its graph.

        The background mint calls this per graph as each publish completes —
        partial-hit, no reboot."""

        dispatcher = self._home.get(record.graph)
        if dispatcher is None:
            raise AdoptError(
                f"graph {record.graph} was never claimed by a ctx.compile-ed "
                f"module; there is nothing to arm it onto"
            )
        self._arm(dispatcher, record, artifact)

    def _canonical(self, graphs: set[str] | Mapping[str, Any]) -> tuple[GraphRecord, ...]:
        return tuple(record for record in self.lane.graphs if record.graph in graphs)

    @property
    def adopted(self) -> tuple[GraphRecord, ...]:
        return self._canonical(self._adopted)

    @property
    def holes(self) -> tuple[Hole, ...]:
        """THE ordered mint work-list: canonical document order, full
        GraphRecord per row (graph hash + ingress), reason stated."""
        return tuple(
            self._holes[record.graph]
            for record in self.lane.graphs
            if record.graph in self._holes
        )

    @property
    def ambiguous(self) -> tuple[GraphRecord, ...]:
        return self._canonical(self._ambiguous)

    @property
    def unclaimed(self) -> tuple[GraphRecord, ...]:
        """Records no ``ctx.compile`` call claimed — stated, never dropped:
        an unclaimed record has no module to arm onto, so it is NOT mint
        work; it is evidence the author stopped marking a module the derive
        once observed."""
        return self._canonical(self._unclaimed)


__all__ = [
    "AdoptError",
    "AdoptSession",
    "ArtifactLoader",
    "HOLE_MISS",
    "Hole",
]
