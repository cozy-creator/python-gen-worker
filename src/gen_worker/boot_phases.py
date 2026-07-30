"""Per-phase BOOT telemetry (pgw#764 / th#1293).

Cold-boot time is the number the hub prices every capacity buy against and the
number every "why was TTFR five minutes" investigation has had to hand-derive
from log archaeology. The hub can already see the boot from OUTSIDE — pod
create -> worker hello -> first assignment — but those bounds say nothing about
which of weights fetch, cell fetch, cell load or warmup owns the seconds
INSIDE. Only the worker knows that, and today it dies with the pod.

This module is the boot-time analogue of :mod:`gen_worker.stage_timing` (the
per-REQUEST measurement spine, th#1111), and it deliberately copies that
module's two trustworthiness properties:

* **It reconciles.** Spans nest, and a nested span's time is charged to the
  CHILD, never twice. So measured phases + ``residual`` == the whole boot
  window. A boot instrument that reports 120s of phases inside a 190s boot
  cannot answer the question it exists for.
* **It classifies.** Every phase is FETCH, COMPILE, LOAD or SETUP, so
  "this release's boots are network-bound" is a query, not a hunch.

## Why it buffers

``activity.bind_sink`` is called from ``Executor.ensure_setup`` — i.e. AFTER
weights are on disk. The boot window is precisely the window in which no sink
exists yet, so an event emitted the ordinary way lands on a logger that
hub-spawned workers do not expose. Worse, ``transport.SendQueue`` clears queued
events on every reconnect. So this module holds its rows and flushes them once
a sink is bound; rows recorded before that survive, in order, and a phase
recorded during a disconnect is still delivered on the next connect.

Rows are OPENED at phase start and CLOSED at phase end (th#1269's build-phase
shape), so a pod that dies mid-boot still shows exactly where its time went —
an open row with no terminal is itself the finding.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Awaitable, Callable, Dict, Iterator, List, Optional

from .pb import worker_scheduler_pb2 as pb

logger = logging.getLogger(__name__)

# --- phase vocabulary (wire-shared with tensorhub's bootphase.go) -----------
# Reuses the entrypoint's existing de-facto boot phase names (_log_startup_phase)
# rather than inventing a second vocabulary for the same milestones.
PHASE_PROCESS_START = "process_start"
PHASE_ENV_SEAL = "env_seal"
PHASE_MANIFEST_LOAD = "manifest_load"
PHASE_CACHE_PREFLIGHT = "cache_preflight"
PHASE_CUDA_PROBE = "cuda_probe"
PHASE_HELLO = "hello"
PHASE_WEIGHTS_FETCH = "weights_fetch"
PHASE_PIPELINE_LOAD = "pipeline_load"
#: pgw#797: the warmup forwards, split OUT of `pipeline_load` and nested under
#: it, so `pipeline_load` becomes weights->VRAM by subtraction and "what does a
#: cell save on warmup" is a column instead of an estimate.
PHASE_WARMUP = "warmup"
#: One warm forward. The first iteration dominates (allocator growth, CUDA
#: module load, cuDNN/cuBLAS algorithm selection), so a warmup total without the
#: shape of the sequence cannot answer what a cell removes.
PHASE_WARMUP_ITERATION = "warmup_iteration"
PHASE_CELL_DISCOVER = "cell_discover"
PHASE_CELL_FETCH = "cell_fetch"
PHASE_CELL_LOAD = "cell_load"
PHASE_CELL_ARM = "cell_arm"
PHASE_FIRST_REQUEST_SERVABLE = "first_request_servable"
PHASE_FIRST_REQUEST = "first_request"
PHASE_WARM_COMPLETE = "warm_complete"

#: The boot's closing milestone: the phase whose completion means this worker
#: can serve. Everything after it is optimization, not boot.
SERVABLE_PHASES = frozenset({PHASE_FIRST_REQUEST_SERVABLE})

# Phase classification — which resource a phase spends. Mirrors
# stage_timing's GPU_BUSY/SMALL_GPU/GPU_IDLE intent at boot scale.
CLASS_FETCH = "fetch"      # network / disk bytes
CLASS_COMPILE = "compile"  # inductor / AOT compile
CLASS_LOAD = "load"        # weights -> VRAM, cell load+arm
CLASS_SETUP = "setup"      # probes, seals, manifests, handshake

_CLASS_BY_PHASE: Dict[str, str] = {
    PHASE_PROCESS_START: CLASS_SETUP,
    PHASE_ENV_SEAL: CLASS_SETUP,
    PHASE_MANIFEST_LOAD: CLASS_SETUP,
    PHASE_CACHE_PREFLIGHT: CLASS_SETUP,
    PHASE_CUDA_PROBE: CLASS_SETUP,
    PHASE_HELLO: CLASS_SETUP,
    PHASE_WEIGHTS_FETCH: CLASS_FETCH,
    PHASE_CELL_DISCOVER: CLASS_SETUP,
    PHASE_CELL_FETCH: CLASS_FETCH,
    PHASE_CELL_LOAD: CLASS_LOAD,
    PHASE_CELL_ARM: CLASS_LOAD,
    PHASE_PIPELINE_LOAD: CLASS_LOAD,
    # pgw#797: an UNARMED warm pays the compile; an ARMED one pays only the
    # call. The default is the expensive reading; `span(..., klass=)` overrides
    # per row, which is why classification is a lookup and not a constant.
    PHASE_WARMUP: CLASS_COMPILE,
    PHASE_WARMUP_ITERATION: CLASS_COMPILE,
    PHASE_WARM_COMPLETE: CLASS_COMPILE,
    PHASE_FIRST_REQUEST_SERVABLE: CLASS_SETUP,
    PHASE_FIRST_REQUEST: CLASS_COMPILE,
}


def phase_class(phase: str, ordinal: int = 0) -> str:
    """Classification for ``phase``; unknown phases classify as "" and are
    reported unattributed rather than guessed into a bucket.

    A row may override its phase's default class (pgw#797) — passing its
    ``ordinal`` consults that override.
    """
    if ordinal:
        override = _class_override.get(ordinal)
        if override:
            return override
    return _CLASS_BY_PHASE.get(phase, "")


# --- outcomes ---------------------------------------------------------------
OUTCOME_OK = "ok"
OUTCOME_REFUSED = "refused"   # a TYPED refusal; the worker serves something else
OUTCOME_FAILED = "failed"     # an exception escaped the span
OUTCOME_SKIPPED = "skipped"

# --- weights/artifact sources (wire-shared) --------------------------------
SOURCE_CAS = "cas"
SOURCE_VOLUME = "volume"
SOURCE_R2 = "r2"
SOURCE_HF_CACHE = "hf_cache"
SOURCE_INFLIGHT_SHARE = "inflight_share"
SOURCE_LOCAL = "local"

#: Memory bound on buffered rows. NOT a timeout — a cap on how much a boot that
#: never connects may retain. A pathological boot (thousands of checkpoints)
#: truncates loudly via `flag.rows_truncated` instead of growing without bound.
_MAX_BUFFERED_ROWS = 2048

#: One id per worker PROCESS boot. Generated at import, which is as close to
#: process start as this module can observe.
BOOT_ID: str = uuid.uuid4().hex

_lock = threading.Lock()
_stack_var: ContextVar[tuple] = ContextVar("boot_phase_stack", default=())
_ordinal = 0
_rows: List[pb.BootPhase] = []
_sink: Optional[Callable[[pb.BootPhase], None]] = None
_truncated = False
_servable_ms: Optional[int] = None
#: pgw#797 ordering contract. `hello` and `first_request_servable` are both
#: CUMULATIVE milestones off the same origin, so `servable - hello` is a phase
#: of the boot and must be >= 0. On 0.78.0 it was NEGATIVE on every real boot
#: (servable 8.9s, hello 13.2s) because `Lifecycle.startup()` runs concurrently
#: with the transport and closed the boot before the stream existed. A worker
#: the hub cannot reach is not servable BY DEFINITION, so the recorder holds a
#: servable close that arrives before `hello` and emits it when `hello` lands.
#: The inversion is then not merely unlikely, it is unrepresentable.
_hello_seen = False
_pending_servable: Optional[Dict[str, Any]] = None
#: Per-ordinal classification override (pgw#797: an armed warm is LOAD, an
#: unarmed one is COMPILE — same phase name, different resource).
_class_override: Dict[int, str] = {}

_process_start_unix: float = 0.0


def _resolve_process_start() -> float:
    """OS process creation time, so the cost BEFORE the first recorded phase
    (interpreter startup, torch import) is visible instead of hiding inside an
    unexplained residual. Falls back to this module's import time."""
    try:
        import psutil

        return float(psutil.Process().create_time())
    except Exception:
        # No psutil (or a sandboxed /proc): fall back to import time, which
        # under-reports the interpreter+import cost rather than inventing it.
        return time.time()


_process_start_unix = _resolve_process_start()


def process_start_unix() -> float:
    """Wall-clock OS process start for this worker."""
    return _process_start_unix


def process_uptime_ms() -> int:
    """Milliseconds since OS process start."""
    return max(0, int(round((time.time() - _process_start_unix) * 1000.0)))


def _next_ordinal() -> int:
    global _ordinal
    with _lock:
        _ordinal += 1
        return _ordinal


def _stack() -> tuple:
    """The enclosing span ordinals, innermost last.

    A ContextVar, not a thread-local: boot work is asyncio, and several setup /
    mint / adopt tasks run interleaved on the ONE worker thread. A thread-local
    stack makes them share one stack, so a span opened by task B while task A's
    span is open is recorded as A's CHILD — which does not merely mislabel the
    row, it makes the ladder stop reconciling (B's time is subtracted from A's
    exclusive total). A ContextVar is copied per task, so each task nests
    against its own creator (pgw#797).
    """
    return _stack_var.get()


def _emit(row: pb.BootPhase) -> None:
    """Record a row and ship it if a sink is bound. Never raises: telemetry
    must not be able to break the boot it measures."""
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
    """Route boot rows onto the worker->hub stream and FLUSH everything
    recorded before the stream existed.

    ``emit`` is the async WorkerMessage sender and ``loop`` its event loop —
    the same contract as :func:`gen_worker.activity.bind_sink`. Flushing is the
    whole point: the boot window is the window with no sink, so a boot
    recorder that only forwarded live rows would report only its own tail.
    """
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
    # Replay in recorded order. Rows are cheap and few; a duplicate delivery
    # after a reconnect is harmless because the hub upserts on
    # (boot_id, ordinal, terminal).
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
    """One open boot phase. Prefer the :func:`span` context manager."""

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
        """Record bytes this phase moved and where they came from. The single
        most load-bearing boot fact after duration: the same release boots in
        wildly different times off a warm volume vs a cold R2 pull."""
        if n > 0:
            self._bytes += int(n)
        if source:
            self._source = source

    def note(self, detail: str) -> None:
        """Attach identifiers (ref=/key=/fn=/lane=) per the pgw#760 doctrine."""
        self._detail = detail[:2000]

    def refused(self, reason: str, detail: str = "") -> None:
        """A TYPED refusal: this phase declined and the worker serves something
        else. Not an error — the reason token is the countable fact."""
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
            # A classified refusal type (AdoptError/ConstantsUnboundError) carries
            # .reason; fall back to the exception class so a phase is never
            # closed with an unnamed failure.
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
            graph_class=self._row.graph_class,
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
    graph_class: str = "",
    parent: Optional[int] = None,
    klass: str = "",
) -> BootSpan:
    """Open a phase span without a ``with`` block (for phases whose start and
    end are in different call frames, e.g. eager-ready vs warm-complete).

    ``parent`` names the enclosing span's ordinal EXPLICITLY. The implicit
    thread-local stack is right for straight-line code but cannot express
    nesting across ``await`` boundaries where sibling tasks share the thread —
    and a boot ladder whose parent links are wrong stops reconciling silently
    (a child charged to the wrong parent inflates one phase and deflates
    another). Pass it wherever the parent is actually known (pgw#797).
    """
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
        graph_class=graph_class,
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
    graph_class: str = "",
    parent: Optional[int] = None,
    klass: str = "",
) -> Iterator[BootSpan]:
    """Bracket a boot phase. Nested spans charge their time to the CHILD, so
    the recorded phases reconcile against the whole boot window.

    An exception closes the span as ``failed`` (carrying the classified
    ``.reason`` when the exception has one) and then propagates — this is a
    measurement, never a behavior change.
    """
    handle = open_span(
        phase, ref=ref, function=function, artifact_kind=artifact_kind,
        artifact_key=artifact_key, graph_class=graph_class,
        parent=parent, klass=klass,
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


def mark(
    phase: str,
    *,
    duration_ms: int = 0,
    since_process_start: bool = False,
    ref: str = "",
    function: str = "",
    artifact_kind: str = "",
    artifact_key: str = "",
    graph_class: str = "",
    bytes_moved: int = 0,
    source: str = "",
    outcome: str = OUTCOME_OK,
    reason: str = "",
    detail: str = "",
    klass: str = "",
) -> None:
    """Record an instantaneous boot MILESTONE as a single closed row.

    ``since_process_start=True`` measures the milestone from OS process start —
    which is what "time to first-request-servable" means and what the
    autoscaler's cold-boot horizon needs.

    A boot-CLOSING milestone (:data:`SERVABLE_PHASES`) recorded before ``hello``
    is HELD, not emitted: see the ``_hello_seen`` note. It is released, with its
    time re-read at release, by :func:`note_hello`.
    """
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
                        graph_class=graph_class, bytes_moved=bytes_moved,
                        source=source, outcome=outcome, reason=reason,
                        detail=detail,
                    )
                    logger.info(
                        "[boot] %s HELD until hello — a worker the hub cannot "
                        "reach yet is not servable (pgw#797)", phase)
                return
    ordinal = _next_ordinal()
    # A CUMULATIVE milestone measures from process start, so it is never part
    # of any span and must never be recorded as one's child: `reconciliation`
    # and every hub-side reader subtract a child's duration from its parent's
    # exclusive time, and a whole-boot number charged against an 870ms
    # `pipeline_load` drives that to zero. Seen live while authoring pgw#797 —
    # the servable mark rides a StateDelta task created inside the setup span,
    # so it INHERITED that span's context. Cumulative rows are top-level, by
    # construction rather than by the caller remembering.
    parent = 0 if since_process_start else (_stack()[-1] if _stack() else 0)
    if klass:
        with _lock:
            _class_override[ordinal] = klass
    if since_process_start:
        duration_ms = process_uptime_ms()
    if phase in SERVABLE_PHASES:
        global _servable_ms
        with _lock:
            if _servable_ms is None:
                _servable_ms = process_uptime_ms()
        if not detail:
            # The recorder owns its own reconciliation string. A caller that
            # formatted it BEFORE the milestone was actually released (pgw#797
            # holds one) would ship a reconciliation for a different instant.
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
        graph_class=graph_class,
        outcome=outcome,
        reason=reason[:300],
        detail=detail[:2000],
        cumulative=bool(since_process_start),
    ))
    if phase == PHASE_HELLO:
        # The gate opens on the hello ROW existing, not on which helper wrote
        # it. Hooking this to `mark_once` instead left `mark(PHASE_HELLO)` — a
        # real call shape, used in the committed pgw#764 suite — recording hello
        # without opening the gate, so a later close stayed held forever.
        note_hello()


def note_hello() -> None:
    """The worker->hub stream is up. Releases any boot close held by
    :func:`mark` (pgw#797).

    Called from the ``hello`` milestone itself, so "hello was recorded" and
    "the ordering gate is open" cannot drift apart.
    """
    global _hello_seen, _pending_servable
    with _lock:
        if _hello_seen:
            return
        _hello_seen = True
        pending, _pending_servable = _pending_servable, None
    if pending is not None:
        # Time re-read at RELEASE: the held value measured a moment at which
        # the hub could not dispatch to this worker, so it was never the
        # cold-boot number. `detail` is dropped so the recorder recomputes the
        # reconciliation for the instant it actually emits.
        pending["detail"] = ""
        mark(PHASE_FIRST_REQUEST_SERVABLE, **pending)


def mark_once(phase: str, **kw: Any) -> bool:
    """:func:`mark` the phase only if it has never been recorded in this
    process. Returns True if it was recorded now.

    Boot milestones are once-per-process by definition, but some of the call
    sites that know about them fire repeatedly — ``on_hello_ack`` runs again on
    every RECONNECT, and "process start -> hello" measured on the third
    reconnect of a six-hour-old worker is not a boot number at all. Recording
    it again would put a second, much larger `hello` row in the series and
    quietly corrupt every boot aggregate that reads it.
    """
    with _lock:
        seen = any(r.phase == phase and r.terminal for r in _rows)
        if not seen and phase in SERVABLE_PHASES and _pending_servable is not None:
            seen = True  # already requested, still held behind `hello`
    if seen:
        return False
    mark(phase, **kw)  # `mark` opens the ordering gate on a hello row itself
    return True


def servable_ms() -> Optional[int]:
    """Process start -> first-request-servable, in ms; None if not yet.

    This is THE cold-boot number: the wall clock from the OS creating this
    process to the hub being allowed to dispatch to it.
    """
    with _lock:
        return _servable_ms


def in_boot() -> bool:
    """True until :data:`PHASE_FIRST_REQUEST_SERVABLE` is marked.

    The gate for instrumenting a call site that runs BOTH during boot and in
    steady state — the weights materializer is the load-bearing case (pgw#789):
    it owns the ~230s of a cold boot, and it also runs every time the hub
    delivers a new ref hours later. Recording the steady-state calls would put
    non-boot spans in a boot ladder (so `residual_ms` stops reconciling) and
    grow the table without bound. Boot ends where the boot number ends.
    """
    with _lock:
        return _servable_ms is None


def reconciliation() -> Dict[str, int]:
    """Boot totals, per the th#1111 rule that an instrument must close.

    ``residual`` is the boot window no phase explained. It is REPORTED, never
    smeared across the measured phases: "unmeasured" and "zero" are different
    answers, and the residual is the honest hint about where the next
    instrument belongs.
    """
    with _lock:
        rows = list(_rows)
        truncated = _truncated
        total = _servable_ms
    exclusive = 0
    per_class: Dict[str, int] = {}
    children: Dict[int, int] = {}
    for row in rows:
        if row.terminal and not row.cumulative and row.parent_ordinal:
            children[row.parent_ordinal] = children.get(row.parent_ordinal, 0) + row.duration_ms
    for row in rows:
        # A cumulative milestone measures the SAME wall clock the spans already
        # account for, so adding it to the span sum would double-count the
        # whole boot. It is the total, not a part of it.
        if not row.terminal or row.cumulative:
            continue
        own = max(0, row.duration_ms - children.get(row.ordinal, 0))
        exclusive += own
        kind = phase_class(row.phase, row.ordinal)
        per_class["class." + (kind or "unattributed")] = (
            per_class.get("class." + (kind or "unattributed"), 0) + own
        )
    out: Dict[str, int] = {"measured_ms": exclusive}
    out.update(per_class)
    if total is not None:
        out["total_ms"] = total
        out["residual_ms"] = max(0, total - exclusive)
    if truncated:
        out["flag.rows_truncated"] = 1
    return out


def recorded_rows() -> List[pb.BootPhase]:
    """Every row recorded so far, in order (tests, diagnostics)."""
    with _lock:
        return list(_rows)


def reset_for_tests() -> None:
    """Clear all recorder state. Test-only."""
    global _ordinal, _truncated, _servable_ms, _sink
    global _hello_seen, _pending_servable
    with _lock:
        _rows.clear()
        _ordinal = 0
        _truncated = False
        _servable_ms = None
        _sink = None
        _hello_seen = False
        _pending_servable = None
        _class_override.clear()
    _stack_var.set(())


__all__ = [
    "BOOT_ID",
    "BootSpan",
    "bind_sink",
    "in_boot",
    "unbind_sink",
    "span",
    "open_span",
    "mark",
    "mark_once",
    "note_hello",
    "phase_class",
    "process_start_unix",
    "process_uptime_ms",
    "reconciliation",
    "recorded_rows",
    "reset_for_tests",
    "servable_ms",
    "PHASE_PROCESS_START",
    "PHASE_ENV_SEAL",
    "PHASE_MANIFEST_LOAD",
    "PHASE_CACHE_PREFLIGHT",
    "PHASE_CUDA_PROBE",
    "PHASE_HELLO",
    "PHASE_WEIGHTS_FETCH",
    "PHASE_PIPELINE_LOAD",
    "PHASE_WARMUP",
    "PHASE_WARMUP_ITERATION",
    "PHASE_CELL_DISCOVER",
    "PHASE_CELL_FETCH",
    "PHASE_CELL_LOAD",
    "PHASE_CELL_ARM",
    "PHASE_FIRST_REQUEST_SERVABLE",
    "PHASE_FIRST_REQUEST",
    "PHASE_WARM_COMPLETE",
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
