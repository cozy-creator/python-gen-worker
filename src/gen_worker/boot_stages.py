from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Dict, Final, List, Mapping, Optional, Sequence, Tuple

from . import boot_phases
from .pb import worker_scheduler_pb2 as pb

logger = logging.getLogger(__name__)

SCHEMA_VERSION: Final[int] = 1

KIND: Final[str] = "boot_stages"

PHASE_READY: Final[str] = "ready"

PHASE_STAGE_PREFIX: Final[str] = "stage:"

MAX_PACKED_RUNS: Final[int] = 32

MAX_STAGE_ROWS: Final[int] = 64


class Stage(StrEnum):
    """The closed cold-boot stage vocabulary."""

    PROCESS_BOOT = "process_boot"
    IMPORTS = "imports"
    HUB_HANDSHAKE = "hub_handshake"
    SNAPSHOT_PULL = "snapshot_pull"
    MODEL_LOAD = "model_load"
    KEYSET = "keyset"
    ADOPT_PULL = "adopt_pull"
    ADOPT = "adopt"
    ARM = "arm"
    WARMUP = "warmup"
    READY = "ready"


STAGES: Final[Tuple[Stage, ...]] = tuple(Stage)

_STAGE_BY_PHASE: Final[Mapping[str, Stage]] = {
    boot_phases.PHASE_ENV_ESTABLISH: Stage.IMPORTS,
    boot_phases.PHASE_LIB_MEMO: Stage.IMPORTS,
    boot_phases.PHASE_WEIGHTS_FETCH: Stage.SNAPSHOT_PULL,
    boot_phases.PHASE_COMPONENT_FETCH: Stage.SNAPSHOT_PULL,
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
    """Boot phases with neither a stage nor a documented exemption."""
    return tuple(sorted(
        phase for phase in boot_phases.PHASES
        if phase not in _STAGE_BY_PHASE and phase not in _MILESTONE_PHASES
    ))


class UnknownStageError(ValueError):
    """A stage name that is not in the closed vocabulary."""


def _token(value: object) -> str:
    cleaned = "".join(ch for ch in str(value if value is not None else "")
                      if not ch.isspace())
    return cleaned or "-"


@dataclass(frozen=True, slots=True)
class StageSpan:
    """One interval of one stage, in ms from OS process start."""

    stage: Stage
    t0_ms: int
    t1_ms: int
    label: str = ""
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
    """A cold boot's stages, and the totals that must reconcile."""

    spans: Tuple[StageSpan, ...]
    wall_ms: int
    servable_ms: int = 0
    truncated: bool = False

    @property
    def span_sum_ms(self) -> int:
        """The sum of every span."""
        return sum(s.duration_ms for s in self.spans)

    @property
    def critical_path_ms(self) -> int:
        """Wall in which SOMETHING was running — the union, never the sum."""
        return _union_ms([(s.t0_ms, s.t1_ms) for s in self.spans])

    @property
    def overlap_ms(self) -> int:
        """How much of this boot was parallel."""
        return max(0, self.span_sum_ms - self.critical_path_ms)

    @property
    def unmeasured_ms(self) -> int:
        """Wall no stage explained."""
        return max(0, self.wall_ms - self.critical_path_ms)

    @property
    def accounted_pct(self) -> int:
        if self.wall_ms <= 0:
            return 0
        return int(round(100.0 * min(1.0, self.critical_path_ms / self.wall_ms)))

    def runs(self) -> List[Tuple[Stage, int, int]]:
        """Disjoint runs per stage, in CHRONOLOGICAL order."""
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
        """The first value any span carries for ``key``, or ``""``."""
        for span in self.spans:
            value = span.attrs.get(key, "")
            if value:
                return value
        return ""


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
    """Record one stage span directly, in ms from OS process start."""
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
    """:func:`record` a span that just CLOSED, given only its duration."""
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
    """Drop recorded spans and the once-only emission latch."""
    global _emitted
    with _lock:
        _recorded.clear()
        _emitted = False


def collect(
    *,
    rows: Optional[List["pb.BootPhase"]] = None,
    extra: Sequence[StageSpan] = (),
) -> BootStageTable:
    """Fold this process's boot into the stage table."""
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

    sdk_ready = milestones.get(boot_phases.PHASE_SDK_READY)
    if sdk_ready is not None:
        import_ms = min(boot_phases.module_import_ms(), sdk_ready)
        spans.append(StageSpan(
            stage=Stage.PROCESS_BOOT, t0_ms=0, t1_ms=import_ms,
            label="interpreter+import"))
        spans.append(StageSpan(
            stage=Stage.IMPORTS, t0_ms=import_ms, t1_ms=sdk_ready,
            label=boot_phases.PHASE_SDK_READY))
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

    wall_ms = milestones.get(boot_phases.PHASE_EAGER_READY, 0)
    servable_ms = milestones.get(boot_phases.PHASE_FIRST_REQUEST_SERVABLE, 0)
    if wall_ms <= 0:
        wall_ms = max((s.t1_ms for s in spans), default=0)
    if wall_ms > 0:
        spans.append(StageSpan(
            stage=Stage.READY, t0_ms=wall_ms, t1_ms=wall_ms, label="eager_ready"))

    ordered = tuple(sorted(
        spans, key=lambda s: (STAGES.index(s.stage), s.t0_ms, s.t1_ms)))
    return BootStageTable(
        spans=ordered, wall_ms=wall_ms, servable_ms=servable_ms)


# detail is free text served VERBATIM by the hub route, so the grammar is the contract: space-separated k=v whose values contain no spaces, parsed entirely by (\w+)=(\S+) — e2e's detailKV already implements exactly that.


def pack_runs(table: BootStageTable) -> Tuple[str, bool]:
    """The stage table as ONE ``k=v``-safe token: ``name:t0-t1,name:t0-t1``."""
    runs = table.runs()
    truncated = len(runs) > MAX_PACKED_RUNS
    kept = runs[:MAX_PACKED_RUNS]
    packed = ",".join(
        f"{stage.value}:{start}-{end}" for stage, start, end in kept)
    return (packed or "-"), truncated


def parse_runs(packed: str) -> Tuple[Tuple[Stage, int, int], ...]:
    """Inverse of :func:`pack_runs`."""
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


_emitted = False


def emit(table: Optional[BootStageTable] = None, *, family: str = "") -> bool:
    """Report this boot's stages: one row per span, then ONE terminal roll-up."""
    global _emitted
    with _lock:
        if _emitted:
            return False
        _emitted = True
    try:
        from . import activity as activity_mod

        if table is None:
            table = collect()
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


def render(table: BootStageTable) -> str:
    """The stage table as fixed-width text — what a runbook pastes."""
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
