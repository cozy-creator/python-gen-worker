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
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .artifact import ArtifactError
from .document import GraphRecord, GraphSetDocument, LaneGraphs
from .requirements import assert_exact_env
from .runner import ConstantBindingError
from .store import GraphStore, StoreError

if TYPE_CHECKING:  # torch-free import closure: transform.py is torch-shaped
    from .transform import TransformSet

logger = logging.getLogger(__name__)

#: Turns one fetched artifact into the callable that replaces the target
#: module's forward for its graph specialization.
#:
#: Three arguments, and every one of them is load-bearing (tcg#58): the
#: ARTIFACT is weightless by sealed policy, the RECORD's ingress is what maps
#: the author's nested call onto the package's flat positional feeds, and the
#: live MODULE is the only place the constant table's values exist on the
#: device. A loader handed fewer than three cannot produce a callable that
#: serves -- which is exactly the shape every consumer wrote until tcg#58.
#: :func:`torchcg.serve.aoti_loader` is the production implementation and the
#: default; the seam stays open only so adoption is testable without a GPU.
ArtifactLoader = Callable[[Path, GraphRecord, Any], Callable[..., Any]]

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


class UnclaimedMark:
    """A module the author MARKED with ``ctx.compile`` that fit no record.

    tcg#70. This used to be a bare ``return`` with a comment on it — a marked
    module that claims nothing is eager FOREVER, and nothing anywhere said so.
    Together with a lane whose records also go unclaimed it produces
    ``adopted=0, holes=0``: two zeros indistinguishable from "there was nothing
    to do", over a store holding every artifact the document asked for.

    A hole would be the wrong shape for it — a hole is a graph the mint should
    build, and these graphs are already built. What is broken is the MATCH, so
    what gets recorded is both sides of it: the module's own forward signature,
    and the parameters the nearest record needed that this signature does not
    have. That difference IS the diagnosis, and it costs one set subtraction.
    """

    __slots__ = ("module", "parameters", "nearest", "missing")

    def __init__(
        self,
        module: str,
        parameters: tuple[str, ...],
        nearest: str,
        missing: tuple[str, ...],
    ) -> None:
        self.module = module
        self.parameters = parameters
        self.nearest = nearest
        self.missing = missing

    def __repr__(self) -> str:  # pragma: no cover - diagnostics
        return (
            f"UnclaimedMark({self.module}, missing={list(self.missing)}, "
            f"nearest={self.nearest[-12:] if self.nearest else None})"
        )

    def describe(self) -> str:
        """One operator-facing line naming what did not match and why."""

        if not self.parameters:
            # THE MOST LIKELY CAUSE GETS THE MOST DIRECT SENTENCE. A denoiser
            # with no NAMED forward parameters does not exist; a `*args,
            # **kwargs` wrapper installed over one does, and every parameter
            # name the match is made on disappears behind it. Measured exactly
            # this way (pgw#1534): an OOM-retry wrapper installed at load time
            # turned a 13-parameter forward into an empty set, and every record
            # in the lane went unclaimed in silence.
            return (
                f"{self.module}: marked with ctx.compile, and its forward "
                f"accepts NO named parameters — so no record can name anything "
                f"it takes. That is the signature of a `*args, **kwargs` "
                f"wrapper installed over the real forward: a wrapper must "
                f"carry the wrapped signature (functools.wraps, or an explicit "
                f"__signature__), or it silently disables adoption for this "
                f"module"
            )
        if not self.nearest:
            return (
                f"{self.module}: marked with ctx.compile and this lane has no "
                f"records left to match it against"
            )
        return (
            f"{self.module}: marked with ctx.compile, matched NO graph in this "
            f"lane. Nearest record {self.nearest[-12:]} needs "
            f"{sorted(self.missing)!r}, which this module's forward does not "
            f"accept; it accepts {sorted(self.parameters)!r}"
        )


class AmbiguousStructure(Exception):
    """Two graph specializations share one tensor structure; neither can be dispatched."""

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

    Two graph specializations whose tensor feeds are identical (they differ only in
    baked-in literals) cannot be told apart at call time; arming both would
    be a coin flip, so the dispatcher refuses the pair instead.
    """

    return tuple(
        (row.param, row.path, row.dtype, row.shape) for row in record.ingress.inputs
    )


def _row_mismatch(row: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str | None:
    """Why this ONE ingress row refuses this call, or ``None`` when it passes.

    THE single source for both the match decision and the diagnosis (tcg#76):
    ``_matches`` is ``every row answers None``, and the first-mismatch trace
    prints exactly what this function refused on — one predicate, so the guard
    and its explanation cannot drift into telling two stories.
    """
    import torch

    resolved, value = row.resolve(args, kwargs)
    if not resolved:
        return (
            f"input {row.param!r} (position {row.position}, path {row.path!r}) "
            f"never resolved from the call — absent from args/kwargs"
        )
    if not isinstance(value, torch.Tensor):
        received = repr(value)
        if len(received) > 60:
            received = received[:60] + "…"
        return (
            f"input {row.param!r}: received py-type {type(value).__name__} "
            f"({received}), expected a torch.Tensor {row.dtype} {tuple(row.shape)}"
        )
    dtype = str(value.dtype).removeprefix("torch.")
    if dtype != row.dtype:
        return (
            f"input {row.param!r}: dtype {dtype} != expected {row.dtype} "
            f"(received shape {tuple(value.shape)})"
        )
    if len(value.shape) != len(row.shape):
        return (
            f"input {row.param!r}: rank {len(value.shape)} (shape "
            f"{tuple(value.shape)}) != expected rank {len(row.shape)} (shape "
            f"{tuple(row.shape)})"
        )
    for got, want in zip(value.shape, row.shape, strict=True):
        if isinstance(want, int) and int(got) != want:
            return (
                f"input {row.param!r}: shape {tuple(value.shape)} != expected "
                f"{tuple(row.shape)}"
            )
    return None


def _matches(record: GraphRecord, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    return all(
        _row_mismatch(row, args, kwargs) is None for row in record.ingress.inputs
    )


def _first_mismatch(
    record: GraphRecord, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> tuple[int, str]:
    """(rows passed before the first failure, that failure's sentence).

    A record with no failing row never reaches here — ``_matches`` would have
    dispatched it.
    """
    passed = 0
    for row in record.ingress.inputs:
        sentence = _row_mismatch(row, args, kwargs)
        if sentence is not None:
            return passed, sentence
        passed += 1
    return passed, "every enumerated row matched (a non-row guard refused?)"


class _ForwardDispatcher:
    """One module's serve-path router: armed graphs by call structure, else eager.

    Installed as the module's instance ``forward`` so ``Module.__call__``
    (which reads ``self.forward``) routes through it -- hooks and everything
    else on the module keep working, and removing the instance attribute
    restores the eager class forward untouched.
    """

    __slots__ = ("eager_forward", "module", "_entries", "_mismatch_reported")

    def __init__(self, module: Any) -> None:
        #: The module this dispatcher fronts, kept because arming needs it:
        #: the loader binds the compiled graph's constants to THESE tensors
        #: (tcg#58), and the late-mint :meth:`AdoptSession.arm` reaches the
        #: module only through here.
        self.module = module
        self.eager_forward = module.forward
        self._entries: list[tuple[GraphRecord, Callable[..., Any]]] = []
        #: tcg#76: the first all-miss call is DIAGNOSED, once. Not per call —
        #: the fall-through is per-request hot path — and not never, which is
        #: what "armed 14, entered 0" cost a full night of counter-reading:
        #: three confident root causes died against a store whose dispatcher
        #: knew the exact divergent input on every single call and said
        #: nothing.
        self._mismatch_reported = False

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
        if self._entries and not self._mismatch_reported:
            self._mismatch_reported = True
            self._say_first_mismatch(args, kwargs)
        return self.eager_forward(*args, **kwargs)

    def _say_first_mismatch(self, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
        """Name the FIRST divergence of the BEST-matching armed record (tcg#76).

        Fires on the first call no armed graph matches, once per module per
        boot, at WARNING — the level a bare ``gen-worker up`` actually
        surfaces. "Armed and never entered" is a per-call silence by design
        (eager is correct), but an UNEXPLAINED one is a diagnosis this object
        already holds and used to discard: every guard refusal knows exactly
        which input diverged and what it received.

        Best-matching = the record that passes the MOST rows before its first
        failure (ties broken by graph hash, so two boots print the same
        record). Never raises: a diagnostic that can take down a serving
        forward is worse than the silence it replaces.
        """
        try:
            best: tuple[int, str, GraphRecord, str] | None = None
            for record, _compiled in self._entries:
                passed, sentence = _first_mismatch(record, args, kwargs)
                candidate = (-passed, record.graph, record, sentence)
                if best is None or candidate < (best[0], best[1], best[2], best[3]):
                    best = candidate
            if best is None:  # pragma: no cover — guarded by `self._entries`
                return
            _negative_passed, _graph, record, sentence = best
            rows = len(record.ingress.inputs)
            logger.warning(
                "torchcg dispatch: NO armed graph matched a call on %s — %d "
                "graph(s) armed, serving eager. Closest record %s (%d/%d "
                "input row(s) matched); first divergence: %s. Reported once "
                "per module; further unmatched calls fall through silently.",
                type(self.module).__name__, len(self._entries),
                record.graph[-16:], -_negative_passed, rows, sentence,
            )
        except Exception:  # noqa: BLE001 — the diagnostic must never cost a call
            logger.debug("torchcg dispatch: first-mismatch trace failed", exc_info=True)


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


def _describe_unclaimed(
    module: Any, signature: frozenset[str], records: Iterable[GraphRecord]
) -> UnclaimedMark:
    """Why this marked module fits nothing, in the terms the match is made in.

    "Nearest" is the record needing the FEWEST parameters this signature lacks
    — the one whose gap is most likely to be the real story. Ties break on the
    graph hash so the message is deterministic across boots, because an operator
    comparing two runs must not be reading a set-iteration order.
    """

    best: tuple[int, str, tuple[str, ...]] | None = None
    for record in records:
        missing = tuple(sorted(set(record.ingress.parameters) - signature))
        candidate = (len(missing), record.graph, missing)
        if best is None or candidate < best:
            best = candidate
    return UnclaimedMark(
        module=type(module).__name__,
        parameters=tuple(sorted(signature)),
        nearest="" if best is None else best[1],
        missing=() if best is None else best[2],
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
        loader: ArtifactLoader | None = None,
        artifacts_dir: str | Path,
        stack: Mapping[str, str] | Sequence[tuple[str, str]],
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
        self.env = document.env(sm)
        # The exact-env audit, BEFORE any artifact or module is touched. Both
        # sides are the endpoint's own lockfile stack now (pgw#1489), so this
        # fires on a real compile-stack difference and on nothing else.
        assert_exact_env(self.env, stack=stack, sm=sm)
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
        self._unclaimed_marks: list[UnclaimedMark] = []

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
            # RECORDED, not returned from silently (tcg#70). Eager is a safe
            # answer and stays one; an unexplained eager is not.
            self._unclaimed_marks.append(
                _describe_unclaimed(module, signature, self._unclaimed.values())
            )
            return
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

    def _load(self, dispatcher: _ForwardDispatcher, artifact: Path, record: GraphRecord) -> Any:
        """The loader call, with the module it must bind against (tcg#58)."""

        loader = self._loader
        if loader is None:
            from .serve import aoti_loader

            loader = aoti_loader
        return loader(Path(artifact), record, dispatcher.module)

    def _arm(self, dispatcher: _ForwardDispatcher, record: GraphRecord, artifact: Path) -> None:
        try:
            compiled = self._load(dispatcher, Path(artifact), record)
        except (ArtifactError, ConstantBindingError) as exc:
            # Bytes that will not verify, or a constant table that will not
            # bind to THIS module, is a hole with a stated reason -- never a
            # dead boot and never a silent arm. Both refusals are typed at
            # their source precisely so this arm can be narrow: anything else
            # escaping the loader is a bug and keeps escaping.
            self._holes[record.graph] = Hole(record, f"{type(exc).__name__}: {exc}")
            return
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

    @property
    def unclaimed_marks(self) -> tuple[UnclaimedMark, ...]:
        """The MODULE side of the same fact: every ``ctx.compile``-ed module
        that matched no record, with the signature gap that explains it.

        :attr:`unclaimed` says which graphs found no home; this says which
        marks found no graph. Reading only one of them is how ``adopted=0,
        holes=0`` became a silent verdict (tcg#70) — a boot needs both to tell
        "the author stopped marking a module" from "the module the author
        marks no longer has the forward the derive traced"."""
        return tuple(self._unclaimed_marks)

    def silently_eager(self) -> bool:
        """Nothing armed, nothing to mint, and marks that matched nothing.

        The exact state that used to print as two zeros. It is not an error —
        eager costs speed, never correctness — but it is never a quiet success
        either, and a caller that reports adoption MUST say so."""
        return bool(self._unclaimed_marks) and not self._adopted and not self._holes


__all__ = [
    "AdoptError",
    "AdoptSession",
    "ArtifactLoader",
    "HOLE_MISS",
    "Hole",
    "UnclaimedMark",
]
