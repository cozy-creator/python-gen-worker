"""Per-phase BOOT telemetry."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import (
    Any, Awaitable, Callable, Dict, Iterable, Iterator, List, Optional, Tuple,
)

from .pb import worker_scheduler_pb2 as pb

logger = logging.getLogger(__name__)

# Phase vocabulary is wire-shared with tensorhub's bootphase.go. Every name here must have a production producer on the shipping path.
PHASE_HELLO = "hello"
PHASE_WEIGHTS_FETCH = "weights_fetch"
PHASE_PIPELINE_LOAD = "pipeline_load"
PHASE_WARMUP = "warmup"
PHASE_GRAPH_FETCH = "graph_fetch"
PHASE_GRAPH_ARM = "graph_arm"
PHASE_FIRST_REQUEST_SERVABLE = "first_request_servable"

PHASE_SDK_READY = "sdk_ready"
PHASE_COMPONENT_FETCH = "component_fetch"
PHASE_RESIDENCY_CHECK = "residency_check"
PHASE_ENV_ESTABLISH = "env_establish"
PHASE_LIB_MEMO = "lib_memo"
PHASE_DECLARATION_COMPOSE = "declaration_compose"
PHASE_TRACE_FOR_KEY = "trace_for_key"
PHASE_KEY_FOLD = "key_fold"
PHASE_GRAPH_HUB_RTT = "graph_hub_rtt"
PHASE_GRAPH_VERIFY = "graph_verify"
PHASE_ENTRY_ADMIT = "entry_admit"
PHASE_EAGER_READY = "eager_ready"
PHASE_COMPILED_SWAP = "compiled_swap"

SERVABLE_PHASES = frozenset({PHASE_FIRST_REQUEST_SERVABLE})

CUMULATIVE_PHASES = frozenset({
    PHASE_HELLO,
    PHASE_SDK_READY,
    PHASE_EAGER_READY,
    PHASE_COMPILED_SWAP,
    PHASE_FIRST_REQUEST_SERVABLE,
})

CLASS_FETCH = "fetch"
CLASS_COMPILE = "compile"
CLASS_LOAD = "load"
CLASS_SETUP = "setup"

_CLASS_BY_PHASE: Dict[str, str] = {
    PHASE_HELLO: CLASS_SETUP,
    PHASE_WEIGHTS_FETCH: CLASS_FETCH,
    PHASE_GRAPH_FETCH: CLASS_FETCH,
    PHASE_GRAPH_ARM: CLASS_LOAD,
    PHASE_PIPELINE_LOAD: CLASS_LOAD,
    PHASE_WARMUP: CLASS_COMPILE,
    PHASE_FIRST_REQUEST_SERVABLE: CLASS_SETUP,
    PHASE_SDK_READY: CLASS_SETUP,
    PHASE_COMPONENT_FETCH: CLASS_FETCH,
    PHASE_RESIDENCY_CHECK: CLASS_FETCH,
    PHASE_ENV_ESTABLISH: CLASS_SETUP,
    PHASE_LIB_MEMO: CLASS_SETUP,
    PHASE_DECLARATION_COMPOSE: CLASS_SETUP,
    PHASE_TRACE_FOR_KEY: CLASS_COMPILE,
    PHASE_KEY_FOLD: CLASS_SETUP,
    PHASE_GRAPH_HUB_RTT: CLASS_SETUP,
    PHASE_GRAPH_VERIFY: CLASS_LOAD,
    PHASE_ENTRY_ADMIT: CLASS_LOAD,
    PHASE_EAGER_READY: CLASS_SETUP,
    PHASE_COMPILED_SWAP: CLASS_SETUP,
}

PHASES: frozenset = frozenset(_CLASS_BY_PHASE)


def phase_class(phase: str, ordinal: int = 0) -> str:
    """Classification for ``phase``; unknown phases classify as "" and are reported unattributed rather than guessed into a bucket."""
    if ordinal:
        override = _class_override.get(ordinal)
        if override:
            return override
    return _CLASS_BY_PHASE.get(phase, "")


OUTCOME_OK = "ok"
OUTCOME_REFUSED = "refused"
OUTCOME_FAILED = "failed"
OUTCOME_SKIPPED = "skipped"

SOURCE_CAS = "cas"
SOURCE_VOLUME = "volume"
SOURCE_R2 = "r2"
SOURCE_HF_CACHE = "hf_cache"
SOURCE_INFLIGHT_SHARE = "inflight_share"
SOURCE_LOCAL = "local"

_MAX_BUFFERED_ROWS = 2048

BOOT_ID: str = uuid.uuid4().hex

_lock = threading.Lock()
_stack_var: ContextVar[tuple] = ContextVar("boot_phase_stack", default=())
_ordinal = 0
_rows: List[pb.BootPhase] = []
_sink: Optional[Callable[[pb.BootPhase], None]] = None
_truncated = False
_servable_ms: Optional[int] = None
_hello_seen = False
_pending_servable: Optional[Dict[str, Any]] = None
_class_override: Dict[int, str] = {}
_milestone_ms: Dict[str, int] = {}
_servable_probe: Optional[Callable[[], bool]] = None

_process_start_unix: float = 0.0


def _resolve_process_start() -> float:
    try:
        import psutil

        return float(psutil.Process().create_time())
    except Exception:
        return time.time()


_process_start_unix = _resolve_process_start()

_module_import_unix: float = time.time()


def process_start_unix() -> float:
    """Wall-clock OS process start for this worker."""
    return _process_start_unix


def module_import_ms() -> int:
    """ms from OS process start to this module's import."""
    return max(0, int(round((_module_import_unix - _process_start_unix) * 1000.0)))


def process_uptime_ms() -> int:
    """Milliseconds since OS process start."""
    return max(0, int(round((time.time() - _process_start_unix) * 1000.0)))


def _next_ordinal() -> int:
    global _ordinal
    with _lock:
        _ordinal += 1
        return _ordinal


def _stack() -> tuple:
    return _stack_var.get()


def _emit(row: pb.BootPhase) -> None:
    try:
        with _lock:
            if len(_rows) < _MAX_BUFFERED_ROWS:
                _rows.append(row)
            else:
                global _truncated
                _truncated = True
                return
            sink = _sink
        if sink is not None:
            sink(row)
        else:
            logger.info(
                "[boot] %s%s ordinal=%d %s%s",
                row.phase,
                " done" if row.terminal else "",
                row.ordinal,
                f"{row.duration_ms}ms " if row.terminal else "",
                row.reason or row.outcome,
            )
    except Exception:
        logger.debug("boot phase row dropped", exc_info=True)


def bind_sink(
    emit: Callable[["pb.WorkerMessage"], Awaitable[None]],
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Route boot rows onto the worker->hub stream and FLUSH everything recorded before the stream existed."""
    def sink(row: pb.BootPhase) -> None:
        async def _ship() -> None:
            await emit(pb.WorkerMessage(boot_phase=row))

        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is loop:
            loop.create_task(_ship())
        elif not loop.is_closed():
            asyncio.run_coroutine_threadsafe(_ship(), loop)

    global _sink
    with _lock:
        _sink = sink
        pending = list(_rows)
    for row in pending:
        try:
            sink(row)
        except Exception:
            logger.debug("boot phase flush dropped", exc_info=True)


def unbind_sink() -> None:
    """Drop the sink (test teardown, shutdown)."""
    global _sink
    with _lock:
        _sink = None


class BootSpan:
    """One open boot phase."""

    __slots__ = ("phase", "ordinal", "parent", "_started", "_row", "_closed",
                 "_bytes", "_source", "_outcome", "_reason", "_detail")

    def __init__(self, phase: str, ordinal: int, parent: int, row: pb.BootPhase) -> None:
        self.phase = phase
        self.ordinal = ordinal
        self.parent = parent
        self._started = time.monotonic()
        self._row = row
        self._closed = False
        self._bytes = 0
        self._source = ""
        self._outcome = OUTCOME_OK
        self._reason = ""
        self._detail = ""

    def bytes_moved(self, n: int, source: str = "") -> None:
        """Record bytes this phase moved and where they came from."""
        if n > 0:
            self._bytes += int(n)
        if source:
            self._source = source

    def note(self, detail: str) -> None:
        """Attach identifiers (ref=/key=/fn=/lane=)."""
        self._detail = detail[:2000]

    def classify(self, reason: str, detail: str = "") -> None:
        """Attach the countable reason token WITHOUT calling this a refusal."""
        self._reason = reason[:300]
        if detail:
            self._detail = detail[:2000]

    def refused(self, reason: str, detail: str = "") -> None:
        """A TYPED refusal: this phase declined and the worker serves something else."""
        self._outcome = OUTCOME_REFUSED
        self._reason = reason[:300]
        if detail:
            self._detail = detail[:2000]

    def skipped(self, reason: str = "") -> None:
        self._outcome = OUTCOME_SKIPPED
        if reason:
            self._reason = reason[:300]

    def close(self, exc: Optional[BaseException] = None) -> None:
        if self._closed:
            return
        self._closed = True
        duration_ms = int(round(max(0.0, time.monotonic() - self._started) * 1000.0))
        outcome, reason, detail = self._outcome, self._reason, self._detail
        if exc is not None:
            outcome = OUTCOME_FAILED
            reason = str(getattr(exc, "reason", "") or "") or type(exc).__name__
            detail = f"{detail} {type(exc).__name__}: {exc}".strip()[:2000]
        _emit(pb.BootPhase(
            boot_id=BOOT_ID,
            ordinal=self.ordinal,
            parent_ordinal=self.parent,
            phase=self.phase,
            terminal=True,
            started_at_unix_ms=self._row.started_at_unix_ms,
            duration_ms=duration_ms,
            process_uptime_ms=process_uptime_ms(),
            bytes=self._bytes,
            source=self._source,
            ref=self._row.ref,
            artifact_kind=self._row.artifact_kind,
            artifact_key=self._row.artifact_key,
            function=self._row.function,
            outcome=outcome,
            reason=reason[:300],
            detail=detail[:2000],
        ))


def open_span(
    phase: str,
    *,
    ref: str = "",
    function: str = "",
    artifact_kind: str = "",
    artifact_key: str = "",
    parent: Optional[int] = None,
    klass: str = "",
) -> BootSpan:
    """Open a phase span without a ``with`` block (for phases whose start and end are in different call frames, e.g."""
    ordinal = _next_ordinal()
    stack = _stack()
    if parent is None:
        parent = stack[-1] if stack else 0
    if klass:
        with _lock:
            _class_override[ordinal] = klass
    row = pb.BootPhase(
        boot_id=BOOT_ID,
        ordinal=ordinal,
        parent_ordinal=parent,
        phase=phase,
        terminal=False,
        started_at_unix_ms=int(time.time() * 1000),
        process_uptime_ms=process_uptime_ms(),
        ref=ref,
        function=function,
        artifact_kind=artifact_kind,
        artifact_key=artifact_key,
    )
    _emit(row)
    return BootSpan(phase, ordinal, parent, row)


@contextmanager
def span(
    phase: str,
    *,
    ref: str = "",
    function: str = "",
    artifact_kind: str = "",
    artifact_key: str = "",
    parent: Optional[int] = None,
    klass: str = "",
) -> Iterator[BootSpan]:
    """Bracket a boot phase."""
    handle = open_span(
        phase, ref=ref, function=function, artifact_kind=artifact_kind,
        artifact_key=artifact_key, parent=parent, klass=klass,
    )
    token = _stack_var.set(_stack_var.get() + (handle.ordinal,))
    try:
        yield handle
    except BaseException as exc:
        _stack_var.reset(token)
        handle.close(exc)
        raise
    else:
        _stack_var.reset(token)
        handle.close()


class ComponentSpans:
    """Per-component fetch spans that open on the first byte and close on the last."""

    __slots__ = ("_remaining", "_open", "_parent", "_ref", "_lock")

    def __init__(
        self,
        expected: Dict[str, int],
        *,
        parent: Optional[int] = None,
        ref: str = "",
    ) -> None:
        self._remaining = {k: int(v) for k, v in expected.items() if int(v) > 0}
        self._open: Dict[str, BootSpan] = {}
        self._parent = parent
        self._ref = ref
        self._lock = threading.Lock()

    def start(self, component: str) -> None:
        with self._lock:
            if component in self._open or component not in self._remaining:
                return
            self._open[component] = open_span(
                PHASE_COMPONENT_FETCH, ref=self._ref, function=component,
                parent=self._parent)

    def finish(
        self, component: str, *, bytes_moved: int = 0, source: str = "",
    ) -> None:
        with self._lock:
            left = self._remaining.get(component)
            if left is None:
                return
            span_handle = self._open.get(component)
            if span_handle is not None and (bytes_moved or source):
                span_handle.bytes_moved(int(bytes_moved), source)
            left -= 1
            if left > 0:
                self._remaining[component] = left
                return
            self._remaining.pop(component, None)
            handle = self._open.pop(component, None)
        if handle is not None:
            handle.close()

    def close_all(self, reason: str = "") -> None:
        """Close every still-open component span (an aborted fetch)."""
        with self._lock:
            handles = list(self._open.items())
            self._open.clear()
            self._remaining.clear()
        for name, handle in handles:
            if reason:
                handle.refused(reason, f"component={name}")
            handle.close()


@contextmanager
def parent_scope(ordinal: int) -> Iterator[None]:
    """Make ``ordinal`` the implicit parent for spans opened in this context."""
    token = _stack_var.set(_stack_var.get() + (int(ordinal),))
    try:
        yield
    finally:
        _stack_var.reset(token)


def mark(
    phase: str,
    *,
    duration_ms: int = 0,
    since_process_start: bool = False,
    ref: str = "",
    function: str = "",
    artifact_kind: str = "",
    artifact_key: str = "",
    bytes_moved: int = 0,
    source: str = "",
    outcome: str = OUTCOME_OK,
    reason: str = "",
    detail: str = "",
    klass: str = "",
    parent: Optional[int] = None,
) -> None:
    """Record an instantaneous boot MILESTONE as a single closed row."""
    if phase in CUMULATIVE_PHASES:
        since_process_start = True
    if phase in SERVABLE_PHASES:
        with _lock:
            if not _hello_seen:
                global _pending_servable
                if _pending_servable is None:
                    _pending_servable = dict(
                        duration_ms=duration_ms,
                        since_process_start=since_process_start,
                        ref=ref, function=function,
                        artifact_kind=artifact_kind, artifact_key=artifact_key,
                        bytes_moved=bytes_moved,
                        source=source, outcome=outcome, reason=reason,
                        detail=detail,
                    )
                    logger.info(
                        "[boot] %s HELD until hello — a worker the hub cannot "
                        "reach yet is not servable (pgw#797)", phase)
                return
    ordinal = _next_ordinal()
    if since_process_start:
        parent = 0
    elif parent is None:
        parent = _stack()[-1] if _stack() else 0
    if klass:
        with _lock:
            _class_override[ordinal] = klass
    if since_process_start:
        duration_ms = process_uptime_ms()
    if since_process_start:
        with _lock:
            _milestone_ms.setdefault(phase, duration_ms)
    if phase in SERVABLE_PHASES:
        global _servable_ms
        with _lock:
            if _servable_ms is None:
                _servable_ms = process_uptime_ms()
        if not detail:
            detail = " ".join(
                f"{k}={v}" for k, v in sorted(reconciliation().items()))
    _emit(pb.BootPhase(
        boot_id=BOOT_ID,
        ordinal=ordinal,
        parent_ordinal=parent,
        phase=phase,
        terminal=True,
        started_at_unix_ms=int(time.time() * 1000),
        duration_ms=max(0, int(duration_ms)),
        process_uptime_ms=process_uptime_ms(),
        bytes=max(0, int(bytes_moved)),
        source=source,
        ref=ref,
        function=function,
        artifact_kind=artifact_kind,
        artifact_key=artifact_key,
        outcome=outcome,
        reason=reason[:300],
        detail=detail[:2000],
        cumulative=bool(since_process_start),
    ))
    if phase == PHASE_HELLO:
        note_hello()


def note_hello() -> None:
    """The worker->hub stream is up."""
    global _hello_seen, _pending_servable
    with _lock:
        if _hello_seen:
            return
        _hello_seen = True
        pending, _pending_servable = _pending_servable, None
    if pending is not None:
        pending["detail"] = ""
        mark(PHASE_FIRST_REQUEST_SERVABLE, **pending)


def mark_once(phase: str, **kw: Any) -> bool:
    """:func:`mark` the phase only if it has never been recorded in this process."""
    with _lock:
        seen = any(r.phase == phase and r.terminal for r in _rows)
        if not seen and phase in SERVABLE_PHASES and _pending_servable is not None:
            seen = True
    if seen:
        return False
    mark(phase, **kw)
    return True


def servable_ms() -> Optional[int]:
    """Process start -> first-request-servable, in ms; None if not yet."""
    with _lock:
        return _servable_ms


def bind_servable_probe(probe: Optional[Callable[[], bool]]) -> None:
    """Tell the recorder how to ask whether this worker can serve right now."""
    global _servable_probe
    with _lock:
        _servable_probe = probe


def in_boot() -> bool:
    """True while this worker CANNOT SERVE — not merely before its first :data:`PHASE_FIRST_REQUEST_SERVABLE`."""
    with _lock:
        if _servable_ms is None:
            return True
        probe = _servable_probe
    if probe is None:
        return False
    try:
        return not probe()
    except Exception:  # noqa: BLE001 — a broken probe must not gate the work
        logger.debug("servable probe raised; treating boot as closed", exc_info=True)
        return False


def _union_ms(intervals: List[Tuple[int, int]]) -> int:
    total = 0
    end_so_far = -1
    for start, end in sorted(intervals):
        if end <= end_so_far:
            continue
        total += end - max(start, end_so_far if end_so_far >= 0 else start)
        end_so_far = end
    return max(0, total)


def _deduped(rows: List[pb.BootPhase]) -> List[pb.BootPhase]:
    seen: Dict[Tuple[int, bool], pb.BootPhase] = {}
    for row in rows:
        seen.setdefault((row.ordinal, row.terminal), row)
    return list(seen.values())


def reconciliation(
    rows: Optional[List[pb.BootPhase]] = None,
) -> Dict[str, int]:
    """Boot totals, per the rule that an instrument must close."""
    given = rows is not None
    with _lock:
        if rows is None:
            rows = list(_rows)
        truncated = _truncated
        total = _servable_ms
    if given:
        total = next(
            (r.duration_ms for r in rows
             if r.terminal and r.phase in SERVABLE_PHASES), None)
    rows = _deduped(rows)
    exclusive = 0
    per_class: Dict[str, int] = {}
    children: Dict[int, int] = {}
    intervals: List[Tuple[int, int]] = []
    for row in rows:
        if row.terminal and not row.cumulative and row.parent_ordinal:
            children[row.parent_ordinal] = children.get(row.parent_ordinal, 0) + row.duration_ms
    for row in rows:
        if not row.terminal or row.cumulative:
            continue
        own = max(0, row.duration_ms - children.get(row.ordinal, 0))
        exclusive += own
        intervals.append(
            (max(0, row.process_uptime_ms - row.duration_ms),
             row.process_uptime_ms))
        kind = phase_class(row.phase, row.ordinal)
        per_class["class." + (kind or "unattributed")] = (
            per_class.get("class." + (kind or "unattributed"), 0) + own
        )
    measured = _union_ms(intervals)
    out: Dict[str, int] = {"measured_ms": measured}
    out.update(per_class)
    if exclusive > measured:
        out["concurrency_ms"] = exclusive - measured
    if given:
        milestones = {
            r.phase: r.duration_ms for r in rows
            if r.terminal and r.cumulative}
    else:
        with _lock:
            milestones = dict(_milestone_ms)
    named = 0
    sdk = milestones.get(PHASE_SDK_READY)
    if sdk is not None:
        pre_sdk_spans = _union_ms([
            (max(0, r.process_uptime_ms - r.duration_ms), r.process_uptime_ms)
            for r in rows
            if r.terminal and not r.cumulative and r.process_uptime_ms <= sdk])
        named_sdk = max(0, sdk - pre_sdk_spans)
        out["named.sdk_import_ms"] = named_sdk
        named += named_sdk
        starts = [
            max(0, row.process_uptime_ms - row.duration_ms)
            for row in rows
            if row.terminal and not row.cumulative and not row.parent_ordinal
            and row.process_uptime_ms - row.duration_ms >= sdk
        ]
        if starts:
            handshake = max(0, min(starts) - sdk)
            out["named.hub_handshake_ms"] = handshake
            named += handshake
    for phase in (PHASE_EAGER_READY, PHASE_COMPILED_SWAP):
        if phase in milestones:
            out["milestone." + phase + "_ms"] = milestones[phase]
    if PHASE_EAGER_READY in milestones and PHASE_COMPILED_SWAP in milestones:
        out["eager_serving_ms"] = max(
            0, milestones[PHASE_COMPILED_SWAP] - milestones[PHASE_EAGER_READY])
    if total is not None:
        out["total_ms"] = total
        out["named_ms"] = named
        out["residual_ms"] = max(0, total - measured - named)
        out["accounted_pct"] = int(round(
            100.0 * min(1.0, (measured + named) / total))) if total > 0 else 100
    if truncated:
        out["flag.rows_truncated"] = 1
    return out


@dataclass(frozen=True)
class PhaseRow:
    """One line of the boot's phase table."""

    phase: str
    ordinal: int
    parent_ordinal: int
    klass: str
    duration_ms: int
    exclusive_ms: int
    start_ms: int
    end_ms: int
    cumulative: bool
    bytes: int
    source: str
    ref: str
    function: str
    outcome: str
    reason: str
    detail: str


def phase_table(
    rows: Optional[List[pb.BootPhase]] = None,
) -> List[PhaseRow]:
    """The boot decomposition, in emission order, with children subtracted."""
    if rows is None:
        with _lock:
            rows = list(_rows)
    rows = _deduped(rows)
    children: Dict[int, int] = {}
    for row in rows:
        if row.terminal and not row.cumulative and row.parent_ordinal:
            children[row.parent_ordinal] = (
                children.get(row.parent_ordinal, 0) + row.duration_ms)
    out: List[PhaseRow] = []
    for row in rows:
        if not row.terminal:
            continue
        out.append(PhaseRow(
            phase=row.phase,
            ordinal=row.ordinal,
            parent_ordinal=row.parent_ordinal,
            klass=phase_class(row.phase, row.ordinal),
            duration_ms=row.duration_ms,
            exclusive_ms=(
                row.duration_ms if row.cumulative
                else max(0, row.duration_ms - children.get(row.ordinal, 0))),
            start_ms=(
                0 if row.cumulative
                else max(0, row.process_uptime_ms - row.duration_ms)),
            end_ms=row.process_uptime_ms,
            cumulative=row.cumulative,
            bytes=row.bytes,
            source=row.source,
            ref=row.ref,
            function=row.function,
            outcome=row.outcome,
            reason=row.reason,
            detail=row.detail,
        ))
    return out


def render_phase_table(rows: Optional[List[pb.BootPhase]] = None) -> str:
    """The phase table as fixed-width text — what a runbook pastes."""
    table = phase_table(rows)
    lines = [
        f"{'phase':<22} {'class':<8} {'start_ms':>9} {'end_ms':>9} "
        f"{'dur_ms':>9} {'excl_ms':>9} {'bytes':>13}  what"
    ]
    for row in table:
        lines.append(
            f"{row.phase:<22} {row.klass:<8} {row.start_ms:>9} "
            f"{row.end_ms:>9} {row.duration_ms:>9} {row.exclusive_ms:>9} "
            f"{row.bytes:>13}  "
            f"{(row.function or row.ref or row.reason or row.detail)[:70]}")
    for key, value in sorted(reconciliation(rows).items()):
        lines.append(f"{key:<22} {value:>9}")
    return "\n".join(lines)


SHAPE_EAGER: frozenset = frozenset({
    PHASE_SDK_READY, PHASE_HELLO,
    PHASE_WEIGHTS_FETCH, PHASE_COMPONENT_FETCH, PHASE_PIPELINE_LOAD,
    PHASE_EAGER_READY, PHASE_FIRST_REQUEST_SERVABLE,
})
SHAPE_ENTRYPOINT: frozenset = SHAPE_EAGER | frozenset({
    PHASE_ENV_ESTABLISH, PHASE_LIB_MEMO,
})
SHAPE_ADOPT: frozenset = SHAPE_ENTRYPOINT | frozenset({
    PHASE_GRAPH_FETCH, PHASE_GRAPH_VERIFY, PHASE_ENTRY_ADMIT, PHASE_GRAPH_ARM,
    PHASE_COMPILED_SWAP,
})
SHAPE_SELF_MINT: frozenset = SHAPE_ADOPT | frozenset({
    PHASE_DECLARATION_COMPOSE, PHASE_TRACE_FOR_KEY, PHASE_KEY_FOLD,
    PHASE_GRAPH_HUB_RTT,
}) - frozenset({PHASE_GRAPH_FETCH})

DEFAULT_RESIDUAL_TOLERANCE_PCT = 5.0


@dataclass(frozen=True)
class BootCompleteness:
    """Does this boot's phase table actually account for the boot? Two independent failures, reported separately because they have different fixes: a phase that never emitted (``missing`` — an instrument ..."""

    shape: Tuple[str, ...]
    missing: Tuple[str, ...]
    total_ms: int
    measured_ms: int
    named_ms: int
    residual_ms: int
    residual_pct: float
    tolerance_pct: float

    @property
    def reconciles(self) -> bool:
        return self.residual_pct <= self.tolerance_pct

    @property
    def complete(self) -> bool:
        return not self.missing and self.reconciles and self.total_ms > 0

    def explain(self) -> str:
        """One paragraph naming every reason this table is not complete."""
        if self.total_ms <= 0:
            return ("the boot never closed: no `first_request_servable` "
                    "milestone, so there is no wall clock to reconcile against")
        parts: List[str] = []
        if self.missing:
            parts.append(
                "phases with NO row on this boot: " + ", ".join(self.missing))
        if not self.reconciles:
            parts.append(
                f"{self.residual_ms} ms of {self.total_ms} ms "
                f"({self.residual_pct:.1f}%) is explained by no phase and no "
                f"named segment — tolerance is {self.tolerance_pct:.1f}%")
        return "; ".join(parts) or "complete"


def completeness(
    shape: Iterable[str] = SHAPE_EAGER,
    *,
    rows: Optional[List[pb.BootPhase]] = None,
    tolerance_pct: float = DEFAULT_RESIDUAL_TOLERANCE_PCT,
) -> BootCompleteness:
    """Verdict on this boot's decomposition against the shape it drove."""
    expect = tuple(sorted(shape))
    seen = {row.phase for row in phase_table(rows)}
    recon = reconciliation(rows)
    total = int(recon.get("total_ms", 0))
    measured = int(recon.get("measured_ms", 0))
    named = int(recon.get("named_ms", 0))
    residual = int(recon.get("residual_ms", max(0, total - measured - named)))
    return BootCompleteness(
        shape=expect,
        missing=tuple(p for p in expect if p not in seen),
        total_ms=total,
        measured_ms=measured,
        named_ms=named,
        residual_ms=residual,
        residual_pct=(100.0 * residual / total) if total > 0 else 100.0,
        tolerance_pct=float(tolerance_pct),
    )


def recorded_rows() -> List[pb.BootPhase]:
    """Every row recorded so far, in order (tests, diagnostics)."""
    with _lock:
        return list(_rows)


def reset_for_tests() -> None:
    """Clear all recorder state."""
    global _ordinal, _truncated, _servable_ms, _sink
    global _hello_seen, _pending_servable, _servable_probe
    with _lock:
        _rows.clear()
        _ordinal = 0
        _truncated = False
        _servable_ms = None
        _sink = None
        _hello_seen = False
        _pending_servable = None
        _servable_probe = None
        _class_override.clear()
        _milestone_ms.clear()
    _stack_var.set(())


__all__ = [
    "BOOT_ID",
    "BootSpan",
    "bind_sink",
    "bind_servable_probe",
    "in_boot",
    "unbind_sink",
    "span",
    "open_span",
    "mark",
    "mark_once",
    "note_hello",
    "parent_scope",
    "phase_class",
    "process_start_unix",
    "process_uptime_ms",
    "reconciliation",
    "recorded_rows",
    "reset_for_tests",
    "servable_ms",
    "BootCompleteness",
    "ComponentSpans",
    "PhaseRow",
    "completeness",
    "phase_table",
    "render_phase_table",
    "CUMULATIVE_PHASES",
    "DEFAULT_RESIDUAL_TOLERANCE_PCT",
    "SHAPE_ADOPT",
    "SHAPE_EAGER",
    "SHAPE_ENTRYPOINT",
    "SHAPE_SELF_MINT",
    "PHASES",
    "PHASE_HELLO",
    "PHASE_WEIGHTS_FETCH",
    "PHASE_PIPELINE_LOAD",
    "PHASE_WARMUP",
    "PHASE_GRAPH_FETCH",
    "PHASE_GRAPH_ARM",
    "PHASE_FIRST_REQUEST_SERVABLE",
    "PHASE_SDK_READY",
    "PHASE_COMPONENT_FETCH",
    "PHASE_ENV_ESTABLISH",
    "PHASE_LIB_MEMO",
    "PHASE_DECLARATION_COMPOSE",
    "PHASE_TRACE_FOR_KEY",
    "PHASE_KEY_FOLD",
    "PHASE_GRAPH_HUB_RTT",
    "PHASE_GRAPH_VERIFY",
    "PHASE_ENTRY_ADMIT",
    "PHASE_EAGER_READY",
    "PHASE_COMPILED_SWAP",
    "OUTCOME_OK",
    "OUTCOME_REFUSED",
    "OUTCOME_FAILED",
    "OUTCOME_SKIPPED",
    "SOURCE_CAS",
    "SOURCE_VOLUME",
    "SOURCE_R2",
    "SOURCE_HF_CACHE",
    "SOURCE_INFLIGHT_SHARE",
    "SOURCE_LOCAL",
    "CLASS_FETCH",
    "CLASS_COMPILE",
    "CLASS_LOAD",
    "CLASS_SETUP",
]
