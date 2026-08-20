"""pgw#1355: a cold boot states its OWN decomposition, as one readable fact.

e2e#1892 measured an 877 s cold serve and had to assemble the breakdown BY
HAND, from four different event kinds:

    41 s   queue          the request row's own timestamps
    ~18 s  boot/imports   `worker_boot_phases` (its own hub table, its own route)
    1.1 s  snapshot pull  `load_phase` on that same table
    2.5 s  model load     ditto
    804 s  keyset derive  `boot_adopt.duration_ms` on `worker_activity_events`
    6.2 s  compute        the request row again
    2.4 s  finalize       the request row again

Every one of those numbers already existed. What did not exist was a place
where the SHAPE of a cold boot is stated once, so this module is a JOIN made
durable at the source rather than a new clock. It reads the spans the worker
already records — :mod:`gen_worker.boot_phases` (which already carries
start/end offsets from OS process start), ``boot_adopt``'s derive wall, the
``snapshot_pull`` roll-up — folds them into a CLOSED stage vocabulary, and
emits one versioned ``boot_stages`` event series on the channel everything
else is already read from.

# Why a second reporter, when `boot_phases` is already good

``boot_phases`` is a good instrument on a channel nobody joins against, with
three holes:

1. **It rides ``worker_boot_phases``** — a different hub table, a different
   admin route (``admin_worker_boot_phases.go``) and a different reader from
   ``worker_activity_events``, which is where ``boot_adopt``, ``snapshot_pull``,
   ``compile_child`` and every other worker fact lands. Answering "where did
   this pod's cold start go" therefore costs a cross-table join that no caller
   had, so no caller did it.
2. **Its window closes too early.** ``first_request_servable`` is marked when a
   StateDelta advertises a function, which on an eager-first boot happens
   before the model is loaded at all. pgw#1353 measured the consequence: the
   ladder goes ``first_request_servable`` and then **871 s of silence** to
   ``eager_ready``, and the 805 s keyset derive that fills the silence has no
   span on any channel.
3. **Its vocabulary is open.** 21 phase names, added as producers appeared.
   That is right for a decomposition ladder and wrong for a report a renderer
   in another repository has to bind to.

So this module is deliberately NOT a replacement. ``boot_phases`` keeps every
row at full fidelity; ``boot_stages`` is the reduction of it that fits on one
line of an operator's screen, closed enough for a Go renderer to parse, and on
the channel that renderer already reads.

# Overlap is first-class, and the arithmetic is a UNION

Stages run CONCURRENTLY on a real pod — a snapshot pull overlaps a component
load overlaps a derive — and a report that serialized them would be inventing a
boot that did not happen. Every span carries ``t0_ms``/``t1_ms`` as offsets from
OS process start, so concurrency is visible rather than inferred.

The consequence is the rule this module refuses to break: **the total is the
UNION of the spans, never their sum.** ``boot_phases`` learned this the
expensive way (its own header records a summing reconciliation that "explained"
3,338 ms of a 909 ms fetch). A sum that exceeds wall is not a rounding problem,
it is a report that has stopped describing time. So:

    span_sum_ms  = the sum of every span's duration    (>= union whenever
                                                        anything overlapped)
    critical_path_ms = the UNION — wall in which SOMETHING was running
    overlap_ms   = span_sum_ms - critical_path_ms      (>= 0, by construction)
    unmeasured_ms = wall_ms - critical_path_ms         (the honest hole)

``unmeasured_ms`` is NAMED, never smeared across the stages: "unmeasured" and
"zero" are different answers, and the hole is the hint about where the next
instrument belongs.

# The vocabulary is CLOSED

:class:`Stage` is an enum and :func:`record` REFUSES a name that is not in it.
An open vocabulary is right for the ladder underneath and wrong here: a
renderer in another repo binds to these tokens, and a stage invented at a call
site is a column that appears in production and in no reader. A new stage is a
deliberate edit to this file, which is the review this deserves.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Dict, Final, List, Mapping, Optional, Sequence, Tuple

from . import boot_phases
from .pb import worker_scheduler_pb2 as pb

logger = logging.getLogger(__name__)

#: Wire envelope version. A reader that does not recognise it must say so
#: rather than parse the fields it happens to know: this event is read by a Go
#: renderer in another repository, on a fleet that upgrades independently of it.
SCHEMA_VERSION: Final[int] = 1

#: The kind, on `worker_activity_events` via the th#1839 route. Mirrored in
#: `activity.KIND_BOOT_STAGES`; defined there with the rest of the vocabulary.
KIND: Final[str] = "boot_stages"

#: The terminal roll-up's phase. Per-stage rows are `stage:<name>`; a reader
#: after the whole table wants exactly one row and this is it.
PHASE_READY: Final[str] = "ready"

#: Prefix for the per-stage rows.
PHASE_STAGE_PREFIX: Final[str] = "stage:"

#: How many packed span runs the roll-up carries before it truncates loudly.
#: `detail` is capped at 2000 chars hub-side; a run costs ~24. A boot with more
#: disjoint runs than this has a finding of its own, and the per-stage rows
#: carry the full fidelity either way.
MAX_PACKED_RUNS: Final[int] = 32

#: How many per-stage rows one boot may put on the wire.
#:
#: NOT a stylistic cap. `boot_phases` buffers up to 2048 rows, and a
#: pathological boot (a checkpoint with hundreds of components, a pod that
#: reconnected repeatedly) would turn `emit` into a two-thousand-message burst
#: on the worker->hub stream at exactly the moment the pod is trying to start
#: serving. The report must never be able to cost more than the thing it
#: reports on.
#:
#: The ROLL-UP is never dropped — it is emitted after the children and carries
#: the packed table, so a truncated series still answers the question the whole
#: event exists for. What is lost is per-span detail, and `rows_truncated=` says
#: so rather than leaving a reader to wonder why a stage has fewer rows than
#: spans.
MAX_STAGE_ROWS: Final[int] = 64


class Stage(StrEnum):
    """The closed cold-boot stage vocabulary.

    Ordered as a cold boot walks them. The order is a RENDERING convenience,
    not a claim about sequence — several of these genuinely overlap, which is
    the whole reason the spans carry offsets.
    """

    #: OS process creation -> the first gen_worker module that can observe a
    #: clock. Interpreter startup, `site`, and the import chain up to the
    #: recorder itself. No span can cover this: nothing can open a span before
    #: the module that opens spans is imported, so it is measured as the
    #: difference between two timestamps rather than bracketed.
    PROCESS_BOOT = "process_boot"
    #: The recorder exists -> the SDK is usable. `import torch`, endpoint-module
    #: discovery, executor construction, and the environment seal that derives
    #: the toolchain/sm key axes everything downstream needs. Process setup, as
    #: distinct from any model work.
    IMPORTS = "imports"
    #: SDK ready -> the first local work this worker was told to do. The gRPC
    #: dial, the Hello/HelloAck round trip and the wait for a residency order —
    #: real boot seconds in which the worker deliberately does nothing, so no
    #: span can cover them and leaving them unnamed reads as an instrument hole
    #: rather than as the hub round trip it is.
    HUB_HANDSHAKE = "hub_handshake"
    #: Bytes onto local disk. Per component where the pull decomposes, because
    #: four components inside one 200 s pull that each measure 180 s were
    #: OVERLAPPED, and that is a different finding from four sequential 50 s
    #: ones. pgw#1351's `snapshot_pull` roll-up says what those bytes cost the
    #: wire; this says when they were moving.
    SNAPSHOT_PULL = "snapshot_pull"
    #: Weights on disk -> weights in VRAM. Per component where the load
    #: decomposes.
    MODEL_LOAD = "model_load"
    #: The compiled-graph key set: DERIVED by tracing, or READ from shipped
    #: data or this machine's memo. `keys_from` is the axis that tells them
    #: apart and it is the single most load-bearing attribute in this table —
    #: pgw#1353 measured `keys_from=traced` costing 805 s on every sdxl pod,
    #: against milliseconds for a read.
    KEYSET = "keyset"
    #: pgw#1372 adopt-first boot: the release graph document read plus the
    #: per-graph `[release x sm]` artifact pull and swap-in. The NEW flow's
    #: replacement for `keyset` — there is no derive to time, so the span
    #: carries `graphs_from=release` and per-outcome counts
    #: (`artifact_from_store` / `artifact_from_eager`); `keyset` survives
    #: only until the old boot ladder is deleted (pgw#1373).
    ADOPT_PULL = "adopt_pull"
    #: Getting a compiled graph: the boot-adopt decision, the fetch it orders
    #: and the staging/verification before the first dlopen.
    ADOPT = "adopt"
    #: Making a fetched compiled graph the served path: entry admission, constant bind,
    #: ingress arming, the parity sweep.
    ARM = "arm"
    #: The warm forwards. An ARMED warm pays the call; an unarmed one pays a
    #: compile, and they are recorded as the same stage because they are the
    #: same position in the boot — the attributes say which happened.
    WARMUP = "warmup"
    #: The terminal milestone: the first instant this worker could serve a
    #: request at all. A zero-width span at `wall_ms`, so the vocabulary can
    #: name the boundary without pretending it had a duration.
    READY = "ready"


#: Every stage, in walk order — the render order and the enum's declaration
#: order are the same thing on purpose.
STAGES: Final[Tuple[Stage, ...]] = tuple(Stage)

#: Which stage each :mod:`gen_worker.boot_phases` phase folds into.
#:
#: A phase absent from this map contributes NOTHING to the table rather than
#: being guessed into a bucket, and :func:`unmapped_phases` names the gap so a
#: new boot phase with no home is a test failure here instead of a silent
#: omission in production. That is the same rule `boot_phases.phase_class`
#: applies to its own classification, for the same reason.
_STAGE_BY_PHASE: Final[Mapping[str, Stage]] = {
    boot_phases.PHASE_ENV_ESTABLISH: Stage.IMPORTS,
    boot_phases.PHASE_LIB_MEMO: Stage.IMPORTS,
    boot_phases.PHASE_WEIGHTS_FETCH: Stage.SNAPSHOT_PULL,
    boot_phases.PHASE_COMPONENT_FETCH: Stage.SNAPSHOT_PULL,
    # pgw#1555. SNAPSHOT_PULL and not a stage of its own: it answers the same
    # question the pull does — how long until this pod holds the tree — and on
    # a warm volume it REPLACES the pull entirely, so its seconds belong in the
    # bucket a reader compares warm boots against cold ones in.
    boot_phases.PHASE_RESIDENCY_CHECK: Stage.SNAPSHOT_PULL,
    boot_phases.PHASE_PIPELINE_LOAD: Stage.MODEL_LOAD,
    boot_phases.PHASE_DECLARATION_COMPOSE: Stage.KEYSET,
    boot_phases.PHASE_TRACE_FOR_KEY: Stage.KEYSET,
    boot_phases.PHASE_KEY_FOLD: Stage.KEYSET,
    boot_phases.PHASE_GRAPH_FETCH: Stage.ADOPT,
    boot_phases.PHASE_GRAPH_VERIFY: Stage.ADOPT,
    boot_phases.PHASE_GRAPH_HUB_RTT: Stage.ADOPT,
    boot_phases.PHASE_GRAPH_ARM: Stage.ARM,
    boot_phases.PHASE_ENTRY_ADMIT: Stage.ARM,
    boot_phases.PHASE_WARMUP: Stage.WARMUP,
}

#: Boot phases that are deliberately NOT folded into a stage, with the reason.
#: They are CUMULATIVE milestones — each measures from process start, so it
#: covers wall clock the spans already account for, and adding one as a span
#: would double-count the whole boot. Two of them are used as BOUNDARIES
#: instead (see :func:`collect`).
_MILESTONE_PHASES: Final[Mapping[str, str]] = {
    boot_phases.PHASE_SDK_READY:
        "boundary: splits process_boot from imports",
    boot_phases.PHASE_HELLO:
        "boundary: the hub handshake's close",
    boot_phases.PHASE_EAGER_READY:
        "the terminal — becomes Stage.READY and the roll-up's wall_ms",
    boot_phases.PHASE_FIRST_REQUEST_SERVABLE:
        "reported as the roll-up's servable_ms attribute, not as a span",
    boot_phases.PHASE_COMPILED_SWAP:
        "lands after the boot closes; the eager-serving window, not a stage",
}


def unmapped_phases() -> Tuple[str, ...]:
    """Boot phases with neither a stage nor a documented exemption.

    A boot phase that reaches production with no home here silently drops its
    seconds out of this table. Asserted by the suite, so adding a phase to
    ``boot_phases`` without deciding where it belongs is red rather than
    invisible.
    """
    return tuple(sorted(
        phase for phase in boot_phases.PHASES
        if phase not in _STAGE_BY_PHASE and phase not in _MILESTONE_PHASES
    ))


class UnknownStageError(ValueError):
    """A stage name that is not in the closed vocabulary.

    Raised, never logged-and-dropped: the caller is inventing a column, and a
    report that quietly accepts one is a report whose vocabulary is not closed
    after all. Telemetry never breaks the work it measures, so every EMISSION
    path catches broadly — but the refusal happens first, at the call site that
    can actually be fixed.
    """


def _token(value: object) -> str:
    """A ``k=v`` value with no whitespace in it, and never empty.

    The same rule pgw#1351's ``snapshot_pull`` detail follows, and for the same
    reason: an empty value ends the token at the ``=`` and silently merges the
    next pair into it, while a value containing a space splits one pair into
    two. Both produce a detail line that parses without error and means
    something else.
    """
    cleaned = "".join(ch for ch in str(value if value is not None else "")
                      if not ch.isspace())
    return cleaned or "-"


@dataclass(frozen=True, slots=True)
class StageSpan:
    """One interval of one stage, in ms from OS process start.

    ``t0_ms``/``t1_ms`` rather than a duration, because a duration cannot say
    whether two stages ran at the same time and that is the question this whole
    module exists to answer.
    """

    stage: Stage
    t0_ms: int
    t1_ms: int
    #: What produced this span — the originating boot phase, or the module that
    #: recorded it directly. Carried so nothing is lost in the reduction: a
    #: reader that wants the underlying ladder knows which row to go find.
    label: str = ""
    #: Stage-specific facts as ``k=v``-safe values (``keys_from``, ``classes``,
    #: ``component``, ``bytes``, ``outcome``). Never load-bearing for the
    #: arithmetic — the table reconciles on the intervals alone.
    attrs: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.stage not in STAGES:
            raise UnknownStageError(
                f"{self.stage!r} is not a boot stage; the vocabulary is closed "
                f"and is {[s.value for s in STAGES]}")
        if self.t1_ms < self.t0_ms:
            raise ValueError(
                f"stage {self.stage.value} ends before it starts "
                f"(t0_ms={self.t0_ms} t1_ms={self.t1_ms})")

    @property
    def duration_ms(self) -> int:
        return self.t1_ms - self.t0_ms


def _union_runs(intervals: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Merge intervals into disjoint runs, in order.

    The one arithmetic primitive this module has, and every total is computed
    from it: overlapping time is counted ONCE. A sum is never a total here.
    """
    runs: List[Tuple[int, int]] = []
    for start, end in sorted(intervals):
        if runs and start <= runs[-1][1]:
            if end > runs[-1][1]:
                runs[-1] = (runs[-1][0], end)
            continue
        runs.append((start, end))
    return runs


def _union_ms(intervals: Sequence[Tuple[int, int]]) -> int:
    return sum(end - start for start, end in _union_runs(intervals))


@dataclass(frozen=True, slots=True)
class BootStageTable:
    """A cold boot's stages, and the totals that must reconcile.

    ``wall_ms`` is the boot's own clock — process start to the terminal
    milestone — and is NOT derived from the spans. That is deliberate: a total
    computed from the parts can never fail to close, so it could never report
    an instrument hole, and the hole is the finding.
    """

    spans: Tuple[StageSpan, ...]
    wall_ms: int
    #: Process start -> `first_request_servable`, when the boot reached it.
    #: Reported beside the wall rather than as a stage: on an eager-first boot
    #: it lands long before the pod can actually serve, and pgw#1353's 871 s of
    #: silence is exactly the gap between the two.
    servable_ms: int = 0
    #: True when the packed roll-up dropped runs it could not fit.
    truncated: bool = False

    @property
    def span_sum_ms(self) -> int:
        """The sum of every span. Exceeds :attr:`critical_path_ms` whenever
        anything overlapped, and is reported only so the difference can be."""
        return sum(s.duration_ms for s in self.spans)

    @property
    def critical_path_ms(self) -> int:
        """Wall in which SOMETHING was running — the union, never the sum.

        This is the number a cold-start budget is spent against: shortening a
        stage that ran entirely inside another stage's window buys nothing, and
        only a union can say so.
        """
        return _union_ms([(s.t0_ms, s.t1_ms) for s in self.spans])

    @property
    def overlap_ms(self) -> int:
        """How much of this boot was parallel. Non-negative by construction."""
        return max(0, self.span_sum_ms - self.critical_path_ms)

    @property
    def unmeasured_ms(self) -> int:
        """Wall no stage explained. NAMED, never smeared across the stages."""
        return max(0, self.wall_ms - self.critical_path_ms)

    @property
    def accounted_pct(self) -> int:
        if self.wall_ms <= 0:
            return 0
        return int(round(100.0 * min(1.0, self.critical_path_ms / self.wall_ms)))

    def runs(self) -> List[Tuple[Stage, int, int]]:
        """Disjoint runs per stage, in CHRONOLOGICAL order.

        Merged per stage rather than globally: a stage that ran twice with a
        gap is a different fact from one that ran once across both, and
        collapsing them would hide it.

        Sorted by start time rather than by the enum, because this table is
        read as a timeline. Enum order looks right until a stage runs out of
        position — on a real cold sdxl boot `model_load` happens THIRTEEN
        MINUTES after `keyset` starts, and printing it above `keyset` because
        it sits earlier in the vocabulary makes the reader distrust the
        column they came for. The enum is the tiebreak for runs that begin
        together, so the order is still total and still stable.
        """
        out: List[Tuple[Stage, int, int]] = []
        for stage in STAGES:
            intervals = [
                (s.t0_ms, s.t1_ms) for s in self.spans if s.stage is stage]
            if not intervals:
                continue
            for start, end in _union_runs(intervals):
                out.append((stage, start, end))
        out.sort(key=lambda row: (row[1], row[2], STAGES.index(row[0])))
        return out

    def busy_ms(self, stage: Stage) -> int:
        """One stage's own union — its wall, overlaps within it counted once."""
        return _union_ms(
            [(s.t0_ms, s.t1_ms) for s in self.spans if s.stage is stage])

    def attr(self, key: str) -> str:
        """The first value any span carries for ``key``, or ``""``.

        How ``keys_from`` reaches the roll-up: it is a property of the boot,
        recorded on the stage that learned it.
        """
        for span in self.spans:
            value = span.attrs.get(key, "")
            if value:
                return value
        return ""


# --- the direct recorder ----------------------------------------------------
# For stages with no `boot_phases` span of their own. pgw#1353's keyset derive
# is the load-bearing case: it runs during a REQUEST, after the boot window
# closed, so `boot_phases.in_boot()` is already False and no ladder row covers
# its 805 seconds.

_lock = threading.Lock()
_recorded: List[StageSpan] = []


def record(
    stage: Stage,
    *,
    t0_ms: int,
    t1_ms: int,
    label: str = "",
    **attrs: object,
) -> None:
    """Record one stage span directly, in ms from OS process start.

    For a stage the ladder cannot see. Prefer instrumenting ``boot_phases`` when
    the work IS inside the boot window — this is not a second way to say the
    same thing, it is the way to say something the ladder structurally cannot.

    A span recorded here and a ladder row covering the same wall do not
    double-count: every total is a union.

    Raises :class:`UnknownStageError` on a name outside the vocabulary. Never
    raises for any other reason — a boot that succeeded is never failed by the
    thing measuring it.
    """
    span = StageSpan(
        stage=stage, t0_ms=max(0, int(t0_ms)), t1_ms=max(0, int(t1_ms)),
        label=label,
        attrs={k: _token(v) for k, v in attrs.items() if v not in (None, "")},
    )
    try:
        with _lock:
            _recorded.append(span)
    except Exception:  # pragma: no cover — telemetry never breaks a boot
        logger.debug("boot stage span dropped", exc_info=True)


def record_ending_now(
    stage: Stage, *, duration_ms: int, label: str = "", **attrs: object,
) -> None:
    """:func:`record` a span that just CLOSED, given only its duration.

    The shape almost every existing internal measurement has: a caller times a
    block with a monotonic clock and knows the wall it took, not the offsets it
    occupied. Reading the process clock at the close recovers the interval, and
    is the minimal instrument a stage needs to join this table.

    A duration LONGER than the process has existed is a contradiction — the two
    numbers came from different clocks, or the caller passed a wall it did not
    measure here. The span is clamped to the process start, because a negative
    offset is not representable, and the clamp CONFESSES as ``clamped_ms``: a
    silently shortened span is a wrong measurement that reads as a right one,
    which is the defect this whole vocabulary keeps closing.
    """
    end = boot_phases.process_uptime_ms()
    wall = max(0, int(duration_ms))
    lost = max(0, wall - end)
    if lost:
        attrs = {**attrs, "clamped_ms": lost}
    record(stage, t0_ms=end - wall + lost, t1_ms=end, label=label, **attrs)


def recorded() -> Tuple[StageSpan, ...]:
    """Spans recorded directly (tests, diagnostics)."""
    with _lock:
        return tuple(_recorded)


def reset_for_tests() -> None:
    """Drop recorded spans and the once-only emission latch. Test-only."""
    global _emitted
    with _lock:
        _recorded.clear()
        _emitted = False


# --- collection -------------------------------------------------------------


def collect(
    *,
    rows: Optional[List["pb.BootPhase"]] = None,
    extra: Sequence[StageSpan] = (),
) -> BootStageTable:
    """Fold this process's boot into the stage table.

    ``rows`` overrides the ladder with one captured off the wire, so a test can
    ask the SAME arithmetic about a modified boot — delete a stage's rows from a
    real ladder and the verdict must move. ``extra`` adds spans without
    recording them globally, which is how a fixture builds a boot that never
    ran.
    """
    table = boot_phases.phase_table(rows)
    milestones: Dict[str, int] = {
        row.phase: row.duration_ms for row in table if row.cumulative}

    spans: List[StageSpan] = []
    for row in table:
        if row.cumulative:
            continue
        stage = _STAGE_BY_PHASE.get(row.phase)
        if stage is None:
            continue
        attrs: Dict[str, str] = {}
        if row.function:
            attrs["component"] = _token(row.function)
        if row.bytes:
            attrs["bytes"] = str(row.bytes)
        if row.source:
            attrs["source"] = _token(row.source)
        if row.outcome and row.outcome != boot_phases.OUTCOME_OK:
            attrs["outcome"] = _token(row.outcome)
        if row.reason:
            attrs["reason"] = _token(row.reason)
        spans.append(StageSpan(
            stage=stage, t0_ms=row.start_ms, t1_ms=row.end_ms,
            label=row.phase, attrs=attrs))

    # The two windows no span can cover, recovered from the milestones that
    # bound them. `boot_phases` already names both in its own reconciliation;
    # this states them as INTERVALS so they take their place in the table
    # instead of living in a footer.
    sdk_ready = milestones.get(boot_phases.PHASE_SDK_READY)
    if sdk_ready is not None:
        import_ms = min(boot_phases.module_import_ms(), sdk_ready)
        spans.append(StageSpan(
            stage=Stage.PROCESS_BOOT, t0_ms=0, t1_ms=import_ms,
            label="interpreter+import"))
        spans.append(StageSpan(
            stage=Stage.IMPORTS, t0_ms=import_ms, t1_ms=sdk_ready,
            label=boot_phases.PHASE_SDK_READY))
        # SDK ready -> the first LOCAL work. A `hello` milestone bounds it
        # exactly; without one, the first top-level span does, and with neither
        # the stage is simply absent — which is the honest answer, not a zero.
        handshake_end = milestones.get(boot_phases.PHASE_HELLO)
        if handshake_end is None:
            starts = [
                row.start_ms for row in table
                if not row.cumulative and not row.parent_ordinal
                and row.start_ms >= sdk_ready]
            handshake_end = min(starts) if starts else None
        if handshake_end is not None and handshake_end >= sdk_ready:
            spans.append(StageSpan(
                stage=Stage.HUB_HANDSHAKE, t0_ms=sdk_ready,
                t1_ms=handshake_end,
                label=boot_phases.PHASE_HELLO if boot_phases.PHASE_HELLO
                in milestones else "first_local_work"))

    with _lock:
        spans.extend(_recorded)
    spans.extend(extra)

    # The wall is the terminal milestone's own cumulative measurement, not a
    # number derived from the spans: a total computed from the parts can never
    # fail to close, so it could never report a hole.
    wall_ms = milestones.get(boot_phases.PHASE_EAGER_READY, 0)
    servable_ms = milestones.get(boot_phases.PHASE_FIRST_REQUEST_SERVABLE, 0)
    if wall_ms <= 0:
        # No terminal milestone: the boot has not reached ready. The table is
        # still worth having (an operator looking at a stuck pod wants exactly
        # this), and the wall falls back to the furthest span end so the
        # reconciliation describes what HAS happened rather than dividing by
        # zero.
        wall_ms = max((s.t1_ms for s in spans), default=0)
    if wall_ms > 0:
        spans.append(StageSpan(
            stage=Stage.READY, t0_ms=wall_ms, t1_ms=wall_ms, label="eager_ready"))

    ordered = tuple(sorted(
        spans, key=lambda s: (STAGES.index(s.stage), s.t0_ms, s.t1_ms)))
    return BootStageTable(
        spans=ordered, wall_ms=wall_ms, servable_ms=servable_ms)


# --- the wire codec ---------------------------------------------------------
# `detail` is free text the th#1839 route serves VERBATIM, so the grammar is
# the contract: space-separated `k=v` whose values never contain spaces, which
# `(\w+)=(\S+)` parses entirely. e2e's `detailKV` already implements exactly
# that, so the renderer needs no new parser — only this packing.


def pack_runs(table: BootStageTable) -> Tuple[str, bool]:
    """The stage table as ONE ``k=v``-safe token: ``name:t0-t1,name:t0-t1``.

    Packed into the roll-up so reading a boot's shape is a ONE-ROW query. The
    per-stage rows carry the same intervals with their attributes; this is the
    summary that makes the join unnecessary, not a replacement for them.

    Returns the token and whether it was truncated. Truncation is reported, not
    silent: a table that dropped runs is a table whose union no longer closes,
    and a reader must be able to tell that from a boot that simply had fewer
    stages.
    """
    runs = table.runs()
    truncated = len(runs) > MAX_PACKED_RUNS
    kept = runs[:MAX_PACKED_RUNS]
    packed = ",".join(
        f"{stage.value}:{start}-{end}" for stage, start, end in kept)
    return (packed or "-"), truncated


def parse_runs(packed: str) -> Tuple[Tuple[Stage, int, int], ...]:
    """Inverse of :func:`pack_runs`. The round trip is what the suite asserts.

    A run naming a stage outside the vocabulary raises: a reader that silently
    skipped it would under-report a boot rather than say it does not understand
    the fleet it is reading, and the second is the answer an operator can act
    on.
    """
    if not packed or packed == "-":
        return ()
    out: List[Tuple[Stage, int, int]] = []
    for entry in packed.split(","):
        name, _, window = entry.partition(":")
        start_s, _, end_s = window.partition("-")
        try:
            stage = Stage(name)
        except ValueError as exc:
            raise UnknownStageError(
                f"{name!r} is not a boot stage; this reader does not understand "
                f"the fleet that emitted it") from exc
        out.append((stage, int(start_s), int(end_s)))
    return tuple(out)


def _packed_or_refused(table: BootStageTable) -> Tuple[str, bool, bool]:
    """Pack the table and PROVE the packed form parses back.

    The renderer that consumes ``runs=`` lives in another repository, on a
    release cadence this worker knows nothing about. If a packing bug ever
    ships, the symptom over there is a parse error on a pod that is otherwise
    healthy — days later, in somebody else's lane, with no way to tell a broken
    emitter from a broken reader.

    So the emitter checks its own output against its own parser before shipping
    it. A token that does not round-trip is DROPPED and confessed as
    ``runs_unpackable=1``, never shipped in the hope that the reader copes:
    pgw#1339's rule, that a degradation is loud and the surviving report still
    serves. Every scalar total is stated independently of ``runs=``, so a
    reader that loses the packed table still gets the wall, the critical path
    and the overlap.

    Returns ``(packed, truncated, unpackable)``.
    """
    packed, truncated = pack_runs(table)
    try:
        parsed = parse_runs(packed)
    except Exception:
        logger.warning(
            "[boot] the packed stage table does not parse back and was DROPPED "
            "— the totals still ship; this is an emitter bug (pgw#1355)",
            exc_info=True)
        return "-", truncated, True
    expected = table.runs()
    if not truncated and list(parsed) != [
            (stage, start, end) for stage, start, end in expected]:
        logger.warning(
            "[boot] the packed stage table round-tripped to a DIFFERENT table "
            "and was dropped: %d run(s) in, %d out (pgw#1355)",
            len(expected), len(parsed))
        return "-", truncated, True
    return packed, truncated, False


def rollup_detail(table: BootStageTable) -> str:
    """The terminal roll-up's ``detail`` line."""
    packed, truncated, unpackable = _packed_or_refused(table)
    parts = [
        f"v={SCHEMA_VERSION}",
        f"wall_ms={table.wall_ms}",
        f"critical_path_ms={table.critical_path_ms}",
        f"span_sum_ms={table.span_sum_ms}",
        f"overlap_ms={table.overlap_ms}",
        f"unmeasured_ms={table.unmeasured_ms}",
        f"accounted_pct={table.accounted_pct}",
        f"servable_ms={table.servable_ms}",
        f"stages={len({s.stage for s in table.spans})}",
        f"spans={len(table.spans)}",
    ]
    # pgw#1373: `keys_from` was the keyset ladder's axis and the ladder is
    # deleted, so nothing writes it. A roll-up that keeps ASKING for a key
    # nothing produces reads as "the boot had no key source" rather than "that
    # question no longer exists".
    for key in ("classes", "family"):
        value = table.attr(key)
        if value:
            parts.append(f"{key}={value}")
    if truncated:
        parts.append("runs_truncated=1")
    if unpackable:
        parts.append("runs_unpackable=1")
    parts.append(f"runs={packed}")
    return " ".join(parts)


def stage_detail(span: StageSpan) -> str:
    """One per-stage row's ``detail`` line."""
    parts = [
        f"v={SCHEMA_VERSION}",
        f"stage={span.stage.value}",
        f"t0_ms={span.t0_ms}",
        f"t1_ms={span.t1_ms}",
    ]
    if span.label:
        parts.append(f"label={_token(span.label)}")
    parts.extend(f"{k}={_token(v)}" for k, v in sorted(span.attrs.items()))
    return " ".join(parts)


# --- emission ---------------------------------------------------------------

_emitted = False


def emit(table: Optional[BootStageTable] = None, *, family: str = "") -> bool:
    """Report this boot's stages: one row per span, then ONE terminal roll-up.

    A series with a roll-up rather than a single event, for the reason
    `aot_mint_phases` and `jit_compile` already use that shape: the roll-up is
    what a reader groups on, and the children are what it drills into. The
    roll-up is emitted LAST so a reader that sees it knows the series is
    complete.

    Once per process. Returns whether it emitted — False on a second call, so a
    caller sitting on a `mark_once` boundary cannot double-report.

    Never raises. A boot that reached ready is not failed by the report of it.
    """
    global _emitted
    with _lock:
        if _emitted:
            return False
        _emitted = True
    try:
        from . import activity as activity_mod

        if table is None:
            table = collect()
        # Longest spans first, so a truncated series keeps the rows an operator
        # would actually have read. Dropping the tail of an arbitrary order
        # would be as likely to discard the 805 s derive as a 2 ms fold.
        ordered = sorted(
            table.spans, key=lambda s: s.duration_ms, reverse=True)
        dropped = max(0, len(ordered) - MAX_STAGE_ROWS)
        for span in ordered[:MAX_STAGE_ROWS]:
            activity_mod.emit_event(
                KIND, stage_detail(span),
                phase=PHASE_STAGE_PREFIX + span.stage.value,
                duration_ms=span.duration_ms, family=family)
        detail = rollup_detail(table)
        if dropped:
            detail = f"{detail} rows_truncated={dropped}"
        activity_mod.emit_event(
            KIND, detail, phase=PHASE_READY,
            duration_ms=table.wall_ms, family=family)
        logger.info("[boot] stages:\n%s", render(table))
        return True
    except Exception:  # pragma: no cover — telemetry never breaks a boot
        logger.debug("boot stages report failed", exc_info=True)
        return False


# --- rendering --------------------------------------------------------------


def render(table: BootStageTable) -> str:
    """The stage table as fixed-width text — what a runbook pastes.

    The same table the e2e renderer prints from the wire, so an operator with a
    pod shell and an operator reading the hub see the same shape.
    """
    lines = [
        f"{'stage':<15} {'t0_ms':>9} {'t1_ms':>9} {'busy_ms':>9} "
        f"{'bar':<32} what",
    ]
    scale = max(1, table.wall_ms)
    for stage, start, end in table.runs():
        attrs = " ".join(
            sorted({
                f"{k}={v}"
                for span in table.spans if span.stage is stage
                for k, v in span.attrs.items()
            }))
        head = int(round(32.0 * start / scale))
        width = max(1, int(round(32.0 * (end - start) / scale))) if end > start else 0
        bar = " " * head + "#" * min(width, max(0, 32 - head))
        lines.append(
            f"{stage.value:<15} {start:>9} {end:>9} {end - start:>9} "
            f"{bar:<32} {attrs[:60]}")
    lines.append("")
    lines.append(
        f"wall_ms={table.wall_ms} critical_path_ms={table.critical_path_ms} "
        f"span_sum_ms={table.span_sum_ms} overlap_ms={table.overlap_ms} "
        f"unmeasured_ms={table.unmeasured_ms} "
        f"accounted_pct={table.accounted_pct}%")
    return "\n".join(lines)


__all__ = [
    "KIND",
    "MAX_PACKED_RUNS",
    "PHASE_READY",
    "PHASE_STAGE_PREFIX",
    "SCHEMA_VERSION",
    "STAGES",
    "BootStageTable",
    "Stage",
    "StageSpan",
    "UnknownStageError",
    "collect",
    "emit",
    "pack_runs",
    "parse_runs",
    "record",
    "record_ending_now",
    "recorded",
    "render",
    "reset_for_tests",
    "rollup_detail",
    "stage_detail",
    "unmapped_phases",
]
