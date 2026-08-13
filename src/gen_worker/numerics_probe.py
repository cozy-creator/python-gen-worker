"""The thing that MEASURES (pgw#868 cross-cutting; pgw#848 CP12).

Everything this program built refuses a compiled graph for being UNUSABLE — wrong `sm`,
wrong torch, unbound constants, ingress outside the declared envelope. Nothing refused one
for being WRONG. :mod:`gen_worker.numerics_ladder` has owned the verdict since
pgw#817 and `Compile.numerics_floor` has declared the bar since pgw#812, but
``compare_outputs`` had zero consumers: no code anywhere ran the armed compiled graph and
the eager forward it replaces on the same inputs. So the gate could not be
"wired" — ``gate()`` opens ``if comparison is None: return None``, and calling
it with nothing to compare passes every compiled graph, always, while looking correct in
the diff AND in the call graph.

This module is that missing measurement, and it is built to ONE constraint
first: **bisectability**. Three confident diagnoses in this program were wrong
because two fuses were burning at once, so a whole-compiled graph "fail" nobody can split
is the artifact that produces the fourth. Every number here is taken against a
single named :class:`ProbeAxis` — one packaged COMPILED_GRAPH, one shape ROW, one LANE,
one recorded SEED — and every verdict carries the axis, the thresholds and
their source with it, the way :class:`~gen_worker.aot_compile_pool.PoolWidth`
carries its readings.

**Placement: MINT time, and nowhere else** (DESIGN-RULINGS §4.32, landed as
pgw#1141). This module was written for adopt time and said so — *"what must
never exist is a mint-only gate"* — and Paul overruled it: every failure it has
ever caught (a baked ``conv_out.bias``, timestep dtype scars) was an AUTHOR
defect in endpoint code or config, so re-measuring on every adopter taxes the
whole fleet forever for one author's one-time mistake. The single caller is
``provision.arm_aot(verify_numerics=True)``, set by ``adopt_delegated_mint`` on
the pod that compiled the bytes, before they publish. Adoption materializes,
arms and serves. That the measurement moved without a new implementation is the
property :func:`measure_axis`'s signature was built for: it takes the two
callables and a contract, never a pipeline.

**Parity by construction.** The probe feed comes from
:func:`gen_worker.aot_inputs.builder_for` — the mint's OWN input builder,
driven by an :class:`~gen_worker.aot_contract.ExportSpec` reconstructed from
the compiled graph's own recorded compiled graph coordinate. A probe that built its inputs some
other way would be measuring a call the artifact was never traced for, which is
the gw#391 / ie#381 bug in a third hat. It comes from ``aot_contract`` and not
``aot_mint`` for a measured reason — see :func:`build_feed`.

**Fail closed.** A probe that cannot run is NOT a pass. Every failure path
raises :class:`ProbeUnavailable` with a named reason, and the arm treats it
exactly like a refusal: stay eager, say why on the wire. Staying eager is the
ordinary miss policy for every other adopt gate, so the cost of a probe bug is
an un-armed compiled graph — never a silently-degraded one.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple

from . import numerics_ladder
from .numerics_ladder import Comparison, Thresholds
from . import aot_serve
from . import aot_inputs
from .aot_contract import ExportSpec

logger = logging.getLogger(__name__)

#: The lifted-adapter call inputs. An compiled graph whose contract does not declare
#: them is a BRANCHLESS graph class (pgw#790) and must be probed with
#: ``lora_bucket=0``, or the feed carries a pair the class refuses.
_ADAPTER_INPUTS = ("lora_a", "lora_b")


class ProbeUnavailable(RuntimeError):
    """The measurement could not be TAKEN — never that it passed.

    ``reason`` is the phase a refusal reports, so an unmeasurable compiled graph is as
    countable fleet-wide as a failing one. The distinction matters: "this compiled graph
    reproduces eager to 0.9997" and "nobody could ask" are different facts, and
    collapsing them is how an absent gate becomes an invisible one.
    """

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(detail or reason)
        self.reason = str(reason)


class CompiledGraphNumericsRefused(RuntimeError):
    """This compiled graph does not reproduce the eager forward it replaces."""

    reason = "compiled_graph_numerics_below_floor"

    def __init__(self, detail: str, comparison: Optional[Comparison] = None) -> None:
        super().__init__(detail)
        self.comparison = comparison


# ---------------------------------------------------------------------------
# The axis — the unit a verdict is allowed to be about
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeAxis:
    """ONE named axis: one compiled graph x one shape row x one lane.

    Reconstructed from the compiled graph's own ``metadata.compiled graphs`` block, so the axis
    names a coordinate the ARTIFACT declares rather than one the probe chose.
    :attr:`name` is stable across pods and processes, which is what makes a
    verdict re-runnable by whoever reads it: ``probe_compiled_graph(..., only=name)``.
    """

    compiled_graph: str
    target: str
    fork: Tuple[Tuple[str, Any], ...] = ()
    class_dims: Tuple[Tuple[str, int], ...] = ()
    execution_lane: str = ""
    lora_bucket: int = 0

    @property
    def name(self) -> str:
        return self.compiled_graph

    @property
    def seed(self) -> int:
        """Deterministic per-axis RNG seed, derived from the axis itself.

        Not a constant: two axes of one compiled graph must not be fed the same random
        latent, or a shape-independent bug looks like agreement. Not random
        either — a verdict nobody can reproduce is a verdict somebody
        re-litigates from a screenshot.
        """
        digest = hashlib.sha256(
            f"{self.compiled_graph}|{self.execution_lane}|{self.lora_bucket}".encode()).hexdigest()
        return int(digest[:8], 16)

    @property
    def shape_row(self) -> str:
        if not self.class_dims:
            return "(collapsed)"
        return ",".join(f"{n}={int(v)}" for n, v in self.class_dims)

    def __str__(self) -> str:
        return (f"compiled_graph={self.compiled_graph} target={self.target} "
                f"row={self.shape_row} lane={self.execution_lane or '(plain)'} "
                f"bucket={self.lora_bucket} seed={self.seed}")


def axes_from_meta(meta: Mapping[str, Any]) -> Tuple[ProbeAxis, ...]:
    """Every probeable axis of one compiled graph, read off its own metadata.

    ONE axis, because pgw#1176 made one artifact one graph class: the mint
    parity gate runs per compiled graph, at the moment that compiled graph exists, against the
    eager callable it was traced from — never "after all 36". That is what
    makes §4.33's ~8 GiB true by construction: exactly ONE compiled runner is
    resident beside the already-resident weights while the probe runs.

    The signature stays a TUPLE so every caller's shape is unchanged; what
    changed is that its length is now one by construction rather than by luck.
    """

    compiled_graph = aot_serve.compiled_graph_from_meta(dict(meta))
    compiled_graphs = {str(compiled_graph.get("name") or ""): compiled_graph}
    execution_lane = str(meta.get("precision") or "")
    compiled_graph_bucket = int(meta.get("lora_bucket") or 0)
    axes: List[ProbeAxis] = []
    for name in sorted(compiled_graphs):
        block = compiled_graphs[name]
        contract = aot_serve.contract_from_meta(block)
        declared = {spec.name for spec in contract.inputs}
        # pgw#790: the branchless class declares no adapter inputs and REFUSES
        # them. Feeding it the lifted pair would be refused at ingress, which
        # reads as an unmeasurable axis when it is really a probe that built
        # the wrong call. The contract discriminates; nothing here guesses.
        bucket = compiled_graph_bucket if all(
            n in declared for n in _ADAPTER_INPUTS) else 0
        axes.append(ProbeAxis(
            compiled_graph=str(name),
            target=str(block.get("target") or ""),
            fork=tuple((str(n), v) for n, v in (block.get("fork") or ())),
            class_dims=tuple(
                (str(n), int(v)) for n, v in (block.get("class_dims") or ())),
            execution_lane=execution_lane,
            lora_bucket=bucket,
        ))
    if not axes:
        raise ProbeUnavailable(
            "no_probeable_axis",
            "the compiled_graph packages no compiled_graph to compare against eager")
    return tuple(axes)


# ---------------------------------------------------------------------------
# The feed — the mint's own builder, at the artifact's own coordinate
# ---------------------------------------------------------------------------


class _EagerView:
    """The module as it was BEFORE the wrap, for signature purposes only.

    MEASURED the first time this probe ran end to end: the arm has already
    swapped ``module.<attr>`` for the compiled graph's dispatch by the time the gate
    runs, so ``aot_declaration._positionalize`` — which binds the declared feed
    to slots by reading ``inspect.signature`` of that attribute — sees the
    wrapper's ``(*args, **kwargs)`` and refuses every declared name as "not a
    parameter". The feed must be built against the signature the artifact was
    TRACED against, which is the eager callable the wrap retained.

    A facade rather than a temporary restore: un-swapping a live module to run
    a probe would hand the compiled graph's traffic back to eager for the duration, on a
    pod that is concurrently serving (pgw#784). Everything except ``attr``
    delegates to the real module, so config, parameters, buffers and dtype all
    come off the thing actually being measured.
    """

    def __init__(self, module: Any, attr: str, eager: Callable[..., Any]) -> None:
        object.__setattr__(self, "_module", module)
        object.__setattr__(self, "_attr", str(attr or "forward"))
        object.__setattr__(self, "_eager", eager)

    def __getattr__(self, name: str) -> Any:
        if name == object.__getattribute__(self, "_attr"):
            return object.__getattribute__(self, "_eager")
        return getattr(object.__getattribute__(self, "_module"), name)


def eager_view(module: Any, attr: str, eager: Callable[..., Any]) -> Any:
    """``module`` with its pre-wrap callable back under ``attr``."""
    return _EagerView(module, attr, eager)


def build_feed(module: Any, family: str, axis: ProbeAxis) -> Tuple[Any, ...]:
    """The all-positional example call for ``axis``, built the way the MINT
    builds it.

    The RNG is forked, not reseeded: a serving pod's tenant traffic owns the
    global generator (pgw#784 — the tenant keeps being served throughout an
    arm), and a probe that shifted it would change a paying request's output
    to check a compiled graph.
    """
    import torch

    try:
        builder = aot_inputs.builder_for(family, axis.target)
    except Exception as exc:  # noqa: BLE001 — any refusal is unmeasurable
        raise ProbeUnavailable(
            "no_input_contract",
            f"{axis}: no export-input contract to build a probe feed from "
            f"({type(exc).__name__}: {exc})") from exc
    # `aot_contract`, NEVER `aot_mint` — MEASURED (pgw#793/#811 guards, red on
    # the first run of this module): `provision` is a `static_code_closure`
    # entrypoint, the closure walk is an AST walk that follows function-level
    # imports, and the closure memo records it on every artifact.
    # Importing the mint DRIVER here dragged `aot_wrapper_split` /
    # `aot_run_impl_split` into compiled graph identity, which both issues forbid: a
    # compile-time transform must re-key nothing. The vocabulary now lives in a
    # leaf module and the driver stays out.
    spec = ExportSpec(
        family=str(family), target=axis.target,
        lora_bucket=int(axis.lora_bucket),
        fork=axis.fork, class_dims=axis.class_dims)
    _dtype, device = aot_inputs.module_dtype_device(module)
    devices = [device] if getattr(device, "type", "") == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.manual_seed(axis.seed)
            args, kwargs = builder(module, spec)
    except ProbeUnavailable:
        raise
    except Exception as exc:  # noqa: BLE001 — any refusal is unmeasurable
        raise ProbeUnavailable(
            "feed_unbuildable",
            f"{axis}: the declared example feed could not be built "
            f"({type(exc).__name__}: {exc})") from exc
    if kwargs:
        # `declared_inputs` documents an ALWAYS-empty kwargs dict: all-positional
        # feeds are a mint obligation, and the serve marshal is positional. A
        # non-empty one means the two paths have drifted.
        raise ProbeUnavailable(
            "feed_not_positional",
            f"{axis}: the input builder returned keyword feeds {sorted(kwargs)!r}; "
            f"the packaged call convention is positional")
    return tuple(args)


# ---------------------------------------------------------------------------
# The measurement
# ---------------------------------------------------------------------------


def measure_axis(
    module: Any,
    eager: Callable[..., Any],
    compiled: Callable[..., Any],
    axis: ProbeAxis,
    *,
    family: str,
    thresholds: Thresholds,
    feed: Optional[Sequence[Any]] = None,
) -> Comparison:
    """Run ONE axis through both forwards on the SAME inputs and compare.

    Deliberately takes two callables rather than a pipeline, so the identical
    measurement runs at mint time (child process, both forwards resident) and
    at adopt time (this module's caller) with no second implementation.

    ``compiled`` must be the compiled graph's OWN runner, never the dispatch: dispatch
    selects by ingress, so a probe that went through it could silently measure
    a different graph class than the axis names.
    """
    import torch

    args = tuple(feed) if feed is not None else build_feed(module, family, axis)
    try:
        with torch.no_grad():
            reference = eager(*args)
    except Exception as exc:  # noqa: BLE001
        raise ProbeUnavailable(
            "eager_forward_failed",
            f"{axis}: the EAGER reference forward raised "
            f"({type(exc).__name__}: {exc}) — with no reference there is "
            f"nothing to compare the compiled_graph against") from exc
    try:
        with torch.no_grad():
            subject = compiled(*args)
    except Exception as exc:  # noqa: BLE001
        raise ProbeUnavailable(
            "compiled_graph_forward_failed",
            f"{axis}: the ARMED COMPILED_GRAPH raised on its own declared feed "
            f"({type(exc).__name__}: {exc})") from exc
    try:
        return numerics_ladder.compare_outputs(
            reference, subject, thresholds=thresholds,
            reference_label="eager", subject_label=axis.name)
    except ValueError as exc:
        # A structural mismatch (dropped output, different shape) is a REFUSAL
        # in waiting, not an unmeasurable axis — but it is not a cosine either,
        # so it cannot ride the ladder. Named, fail-closed, on its own phase.
        raise ProbeUnavailable(
            "output_structure_differs", f"{axis}: {exc}") from exc


@dataclass(frozen=True)
class AxisVerdict:
    """One axis's own result — carried whole, including when it failed."""

    axis: ProbeAxis
    comparison: Optional[Comparison] = None
    reason: str = ""
    detail: str = ""
    elapsed_ms: int = 0

    @property
    def measured(self) -> bool:
        return self.comparison is not None

    def __str__(self) -> str:
        if self.comparison is None:
            return f"{self.axis.name}: UNMEASURED {self.reason} ({self.detail})"
        c = self.comparison
        return (f"{self.axis.name}[{self.axis.shape_row}]: cos={c.cosine:.5f} "
                f"ret={c.retention:.4f} rel_l2={c.rel_l2:.4f} "
                f"{c.verdict} {self.elapsed_ms}ms")


@dataclass(frozen=True)
class CompiledGraphNumerics:
    """The whole compiled graph's verdict, and every axis it is made of."""

    family: str
    compiled_graph_key: str
    thresholds: Thresholds
    threshold_source: str
    verdicts: Tuple[AxisVerdict, ...]
    axes_total: int
    elapsed_ms: int = 0

    @property
    def measured(self) -> bool:
        """True only when every axis of THIS report produced a comparison.

        pgw#1176 DELETED the all-axes-of-the-compiled graph rule this used to state
        ("partial coverage is not a pass; a compiled graph arming on a subset of its own
        graph classes is a silent subset of what the compiled graph key advertises").
        That rule was the verification half of the wrong atom: it made one
        unmeasurable class condemn 35 measured ones. A report is now one graph
        class, so the predicate below says exactly what it always meant —
        absent evidence is never a pass — without a collection to be partial
        about.

        Both remaining clauses are load-bearing and stay: a report short of
        its own axis count has lost a verdict somewhere, and an errored axis
        is not a measured one.
        """
        return (bool(self.verdicts)
                and len(self.verdicts) == self.axes_total
                and all(v.measured for v in self.verdicts))

    @property
    def unmeasured(self) -> Tuple[AxisVerdict, ...]:
        return tuple(v for v in self.verdicts if not v.measured)

    def worst(self) -> Optional[AxisVerdict]:
        """The axis the whole-compiled graph verdict is decided on — NAMED, always.

        The gate judges the worst axis rather than a pooled number: pooling 36
        compiled graphs would let one destroyed graph class hide behind 35 intact ones,
        which is the same failure mode the ladder's norm-weighted aggregate
        exists to prevent one level down.
        """
        rows = [v for v in self.verdicts if v.comparison is not None]
        if not rows:
            return None
        return min(rows, key=lambda v: v.comparison.cosine)  # type: ignore[union-attr]

    def comparison(self) -> Optional[Comparison]:
        worst = self.worst()
        return worst.comparison if worst is not None else None

    def context(self) -> str:
        """Everything a reader needs to re-run this verdict, in one line."""
        worst = self.worst()
        head = (
            f"family={self.family} key={self.compiled_graph_key or '?'} "
            f"axes={len(self.verdicts)}/{self.axes_total} "
            f"thresholds={self.thresholds} source={self.threshold_source} "
            f"took={self.elapsed_ms}ms")
        if worst is not None:
            head += f" | worst axis: {worst.axis}"
        rows = "; ".join(str(v) for v in self.verdicts[:12])
        return f"{head} | per-axis: {rows}"


def probe_compiled_graph(
    pipeline: Any,
    cfg: Any,
    meta: Mapping[str, Any],
    *,
    only: str = "",
) -> CompiledGraphNumerics:
    """Measure an ARMED pipeline against the eager forward it replaced.

    ``only`` names one axis (``ProbeAxis.name``, i.e. the packaged compiled graph name)
    and is the bisection interface: a whole-compiled graph refusal must be re-runnable
    one axis at a time, from a hub row, without editing anything.

    Raises :class:`ProbeUnavailable` when the compiled graph cannot be probed at all —
    never returns a report that could be mistaken for a pass.
    """

    family = str(getattr(cfg, "family", "") or meta.get("family") or "")
    thresholds = numerics_ladder.declared_thresholds(cfg)
    # READ, never re-derive. This used to ask `cfg.numerics_floor` again and
    # so could disagree with the band it was reporting on: a family declaring
    # only `numerics_warn` was judged at its DECLARED band while this said
    # `sdk-default`. `declared_thresholds` is the one authority and stamps its
    # own answer (pgw#1150).
    source = thresholds.source
    targets = aot_serve.armed_targets(pipeline)
    if not targets:
        raise ProbeUnavailable(
            "not_armed",
            f"family={family}: the pipeline carries no armed AOT target, so "
            f"there is no compiled_graph to compare against eager")
    axes = axes_from_meta(meta)
    if only:
        axes = tuple(a for a in axes if a.name == only)
        if not axes:
            raise ProbeUnavailable(
                "no_such_axis",
                f"family={family}: no packaged compiled_graph named {only!r}")
    started = time.monotonic()
    verdicts: List[AxisVerdict] = []
    for axis in axes:
        row = targets.get(axis.target)
        if not row:
            verdicts.append(AxisVerdict(
                axis=axis, reason="target_not_armed",
                detail=f"no armed target {axis.target!r} "
                       f"(armed: {sorted(targets)!r})"))
            continue
        state = row.get("state") or {}
        eager = state.get("original")
        dispatch = state.get("runner")
        runner = next(
            (r for n, r in getattr(dispatch, "runners", ()) if n == axis.compiled_graph),
            None)
        if not callable(eager) or runner is None:
            verdicts.append(AxisVerdict(
                axis=axis, reason="compiled_graph_not_loaded",
                detail=f"the armed target {axis.target!r} retains no eager "
                       f"callable or no runner for compiled_graph {axis.compiled_graph!r}"))
            continue
        t0 = time.monotonic()
        try:
            comparison = measure_axis(
                eager_view(
                    row.get("module"), str(row.get("attr") or "forward"), eager),
                eager, runner, axis, family=family, thresholds=thresholds)
        except ProbeUnavailable as exc:
            verdicts.append(AxisVerdict(
                axis=axis, reason=exc.reason, detail=str(exc),
                elapsed_ms=int((time.monotonic() - t0) * 1000)))
            continue
        verdicts.append(AxisVerdict(
            axis=axis, comparison=comparison,
            elapsed_ms=int((time.monotonic() - t0) * 1000)))
    report = CompiledGraphNumerics(
        family=family, compiled_graph_key=str(meta.get("compiled_graph_key") or ""),
        thresholds=thresholds, threshold_source=source,
        verdicts=tuple(verdicts), axes_total=len(axes),
        elapsed_ms=int((time.monotonic() - started) * 1000))
    with _LAST_LOCK:
        global _LAST
        _LAST = report
    return report


_LAST_LOCK = threading.Lock()
_LAST: Optional[CompiledGraphNumerics] = None


def last_report() -> Optional[CompiledGraphNumerics]:
    """The most recent :func:`probe_compiled_graph` report in this process, or None.

    The GATE (``provision.gate_compiled_graph_numerics``) answers yes/no and confesses
    the numbers to the activity wire; a caller in the same process that needs
    the NUMBER — pgw#1150's author-CI harness, which has to write the measured
    cosine into an ``author-ci.toml`` ``[proof]`` block — would otherwise have
    to re-run the probe and record a second measurement of a different moment,
    or parse the number back out of an event string. Neither is the gate's own
    verdict. Read-only: nothing here changes what the gate decides.
    """
    with _LAST_LOCK:
        return _LAST


__all__ = [
    "AxisVerdict",
    "CompiledGraphNumerics",
    "CompiledGraphNumericsRefused",
    "ProbeAxis",
    "ProbeUnavailable",
    "axes_from_meta",
    "build_feed",
    "eager_view",
    "last_report",
    "measure_axis",
    "probe_compiled_graph",
]
