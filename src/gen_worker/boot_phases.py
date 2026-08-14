"""Per-phase BOOT telemetry.

The hub can see a boot from OUTSIDE — pod create -> worker hello -> first
assignment — but those bounds say nothing about which of weights fetch, cell
fetch, cell load or warmup owns the seconds INSIDE. Only the worker knows.

This module is the boot-time analogue of :mod:`gen_worker.stage_timing` (the
per-REQUEST measurement spine), and it deliberately copies that module's two
trustworthiness properties:

* **It reconciles.** Spans nest, and a nested span's time is charged to the
  CHILD, never twice. So measured phases + named segments + ``residual`` ==
  the whole boot window.

  The reconciliation is a UNION, not a sum: once phases decompose per
  component they run concurrently, and a summing reconciliation "explains"
  3,338 ms of a 909 ms fetch — closing the ladder by over-counting, which is
  worse than visibly not closing it. ``measured_ms`` is the wall time covered
  by at least one span; the gap between that and the exclusive sum is reported
  as ``concurrency_ms``.
* **It classifies.** Every phase is FETCH, COMPILE, LOAD or SETUP, so
  "this release's boots are network-bound" is a query, not a hunch.
* **It NAMES its residual**. Two windows no span can cover — the
  interpreter+import wall before the recorder exists, and the hub handshake in
  which the worker deliberately does no local work — are named segments, not
  an unexplained lump. What survives both is the honest hole.

## Why it buffers

``activity.bind_sink`` is called from ``Executor.ensure_setup`` — i.e. AFTER
weights are on disk. The boot window is precisely the window in which no sink
exists yet, so an event emitted the ordinary way lands on a logger that
hub-spawned workers do not expose. Worse, ``transport.SendQueue`` clears queued
events on every reconnect. So this module holds its rows and flushes them once
a sink is bound; rows recorded before that survive, in order, and a phase
recorded during a disconnect is still delivered on the next connect.

Rows are OPENED at phase start and CLOSED at phase end, so a pod that dies
mid-boot still shows exactly where its time went — an open row with no
terminal is itself the finding.
"""

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

# --- phase vocabulary (wire-shared with tensorhub's bootphase.go) -----------
# Every name here has a production producer on the shipping path. A declared
# phase with no producer is not "coverage we have not gotten to": every reader
# of the ladder sees a name that can only ever report nothing, which is a
# default read as a fact.
PHASE_HELLO = "hello"
PHASE_WEIGHTS_FETCH = "weights_fetch"
PHASE_PIPELINE_LOAD = "pipeline_load"
#: The warmup forwards, split OUT of `pipeline_load` and nested under it, so
#: `pipeline_load` becomes weights->VRAM by subtraction and "what does a cell
#: save on warmup" is a column instead of an estimate.
#:
#: Emitted ONLY when warm work actually runs: a skipped warmup emits no row,
#: because "nobody warmed" and "warming was free" are different answers.
PHASE_WARMUP = "warmup"
PHASE_CELL_FETCH = "cell_fetch"
#: The arm of ONE delivered or discovered cell. Its duration is the same
#: quantity the hub stores as the adoption's `duration_ms`, measured once, in
#: the one place that does the arming.
PHASE_CELL_ARM = "cell_arm"
PHASE_FIRST_REQUEST_SERVABLE = "first_request_servable"

# --- the per-COMPONENT decomposition ---------------------------------------
# The phases above are LEG-grade: they answer "fetch or compile" and nothing
# finer. Each name below answers one question leg-grade phases cannot, and each
# has exactly one production producer — the rule above is not relaxed for being
# new.

#: Process start -> the SDK is usable (interpreter + torch import + endpoint
#: discovery + executor construction). CUMULATIVE. Names a window no span can
#: cover: nothing can start a span before the code that opens spans is
#: imported.
PHASE_SDK_READY = "sdk_ready"
#: One component of one ref's weights. Child of `weights_fetch`, opened at the
#: component's first byte and closed at its last, so CONCURRENCY IS VISIBLE:
#: four components inside a 200 s `weights_fetch` that each measure 180 s were
#: overlapped, and that is the fact an overlap optimization needs.
PHASE_COMPONENT_FETCH = "component_fetch"
#: `env_seal.establish` — the settings declaration digest, the boot-frozen
#: loaded-library digest and the sm/host-ISA derivation that together make the
#: `toolchain` and `sm` key axes. Guessed at "ms"; this proves it.
PHASE_ENV_ESTABLISH = "env_establish"
#: The library-digest MEMO path, hit or miss, inside `env_establish`. `reason`
#: is `hit` or `miss` and `detail` carries the covered/total library counts, so
#: the saving a memo buys is a subtraction between two real rows rather than an
#: estimate.
PHASE_LIB_MEMO = "lib_memo"
#: Composing the export declaration a mint will trace against.
PHASE_DECLARATION_COMPOSE = "declaration_compose"
#: ONE graph CLASS traced for the key. `function` is the entry name and
#: `detail` carries `nodes=` — a class's trace cost is meaningless without the
#: graph size it paid for. Never a roll-up: 36 classes is 36 rows.
PHASE_TRACE_FOR_KEY = "trace_for_key"
#: Per-class hashing + the fold into `combined_graph_hash`.
PHASE_KEY_FOLD = "key_fold"
#: One worker->hub cell control-plane round trip (publish-intent /
#: publish-complete). `function` names the leg. NOTE: there is no worker-side
#: key LOOKUP to time (the hub resolves the arm), so this is the whole of the
#: hub RTT the cell path actually pays on a boot.
PHASE_CELL_HUB_RTT = "cell_hub_rtt"
#: Staging + contract verification of a downloaded cell, before the first
#: dlopen. The first half of admission.
PHASE_CELL_VERIFY = "cell_verify"
#: ONE entry's admission: contract parse, constant bind, ingress-assertion
#: arming and the admission-drift parity check against the artifact's own
#: generated guards. The second half of admission, and the per-entry parity
#: sweep, measured where it happens.
PHASE_ENTRY_ADMIT = "entry_admit"
#: CUMULATIVE. The first instant this worker could have served a request at
#: all, compiled or not — the first of the two user-visible timestamps.
PHASE_EAGER_READY = "eager_ready"
#: CUMULATIVE. The instant a compiled cell became the served path. The second
#: user-visible timestamp, and the only honest measure of how long a pod serves
#: eager before its cell arrives. May land AFTER the boot closes, which is
#: exactly the fact worth having.
PHASE_COMPILED_SWAP = "compiled_swap"

#: The boot's closing milestone: the phase whose completion means this worker
#: can serve. Everything after it is optimization, not boot.
SERVABLE_PHASES = frozenset({PHASE_FIRST_REQUEST_SERVABLE})

#: Milestones measured from process start rather than spans of their own. They
#: are excluded from the phase SUM (they cover wall clock the spans already
#: account for) and reported beside it.
CUMULATIVE_PHASES = frozenset({
    PHASE_HELLO,
    PHASE_SDK_READY,
    PHASE_EAGER_READY,
    PHASE_COMPILED_SWAP,
    PHASE_FIRST_REQUEST_SERVABLE,
})

# Phase classification — which resource a phase spends. Mirrors
# stage_timing's GPU_BUSY/SMALL_GPU/GPU_IDLE intent at boot scale.
CLASS_FETCH = "fetch"      # network / disk bytes
CLASS_COMPILE = "compile"  # inductor / AOT compile
CLASS_LOAD = "load"        # weights -> VRAM, cell load+arm
CLASS_SETUP = "setup"      # probes, seals, manifests, handshake

_CLASS_BY_PHASE: Dict[str, str] = {
    PHASE_HELLO: CLASS_SETUP,
    PHASE_WEIGHTS_FETCH: CLASS_FETCH,
    PHASE_CELL_FETCH: CLASS_FETCH,
    PHASE_CELL_ARM: CLASS_LOAD,
    PHASE_PIPELINE_LOAD: CLASS_LOAD,
    # An UNARMED warm pays the compile; an ARMED one pays only the call. The
    # default is the expensive reading; `span(..., klass=)` overrides per row,
    # which is why classification is a lookup and not a constant.
    PHASE_WARMUP: CLASS_COMPILE,
    PHASE_FIRST_REQUEST_SERVABLE: CLASS_SETUP,
    PHASE_SDK_READY: CLASS_SETUP,
    PHASE_COMPONENT_FETCH: CLASS_FETCH,
    PHASE_ENV_ESTABLISH: CLASS_SETUP,
    PHASE_LIB_MEMO: CLASS_SETUP,
    PHASE_DECLARATION_COMPOSE: CLASS_SETUP,
    PHASE_TRACE_FOR_KEY: CLASS_COMPILE,
    PHASE_KEY_FOLD: CLASS_SETUP,
    PHASE_CELL_HUB_RTT: CLASS_SETUP,
    PHASE_CELL_VERIFY: CLASS_LOAD,
    PHASE_ENTRY_ADMIT: CLASS_LOAD,
    PHASE_EAGER_READY: CLASS_SETUP,
    PHASE_COMPILED_SWAP: CLASS_SETUP,
}

#: The complete vocabulary. Exported so a test can assert the declaration and
#: the production producers are the SAME set.
PHASES: frozenset = frozenset(_CLASS_BY_PHASE)


def phase_class(phase: str, ordinal: int = 0) -> str:
    """Classification for ``phase``; unknown phases classify as "" and are
    reported unattributed rather than guessed into a bucket.

    A row may override its phase's default class — passing its
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
#: Ordering contract. `hello` and `first_request_servable` are both CUMULATIVE
#: milestones off the same origin, so `servable - hello` is a phase of the boot
#: and must be >= 0. `Lifecycle.startup()` runs concurrently with the transport
#: and can close the boot before the stream exists; a worker the hub cannot
#: reach is not servable BY DEFINITION, so the recorder HOLDS a servable close
#: that arrives before `hello` and emits it when `hello` lands. The inversion is
#: then not merely unlikely, it is unrepresentable.
_hello_seen = False
_pending_servable: Optional[Dict[str, Any]] = None
#: Per-ordinal classification override (an armed warm is LOAD, an unarmed one
#: is COMPILE — same phase name, different resource).
_class_override: Dict[int, str] = {}
#: Every CUMULATIVE milestone's ms-from-process-start, first write wins. Read
#: by :func:`reconciliation` and :func:`completeness`.
_milestone_ms: Dict[str, int] = {}

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
    against its own creator.
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
        """Attach identifiers (ref=/key=/fn=/lane=)."""
        self._detail = detail[:2000]

    def classify(self, reason: str, detail: str = "") -> None:
        """Attach the countable reason token WITHOUT calling this a refusal.

        `memo hit` / `memo miss` and `cell cached` / `cell fetched`
        are the two branches of a SUCCESSFUL phase, and both are the fact worth
        counting. Before this the only way to put a token on a row was
        :meth:`refused`, which sets ``outcome=refused`` — so a hub-side count of
        refusals would have been polluted by every memo hit.
        """
        self._reason = reason[:300]
        if detail:
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
    """Open a phase span without a ``with`` block (for phases whose start and
    end are in different call frames, e.g. eager-ready vs warm-complete).

    ``parent`` names the enclosing span's ordinal EXPLICITLY. The implicit
    thread-local stack is right for straight-line code but cannot express
    nesting across ``await`` boundaries where sibling tasks share the thread —
    and a boot ladder whose parent links are wrong stops reconciling silently
    (a child charged to the wrong parent inflates one phase and deflates
    another). Pass it wherever the parent is actually known.
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
    """Bracket a boot phase. Nested spans charge their time to the CHILD, so
    the recorded phases reconcile against the whole boot window.

    An exception closes the span as ``failed`` (carrying the classified
    ``.reason`` when the exception has one) and then propagates — this is a
    measurement, never a behavior change.
    """
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
    """Per-component fetch spans that open on the first byte and close on the
    last.

    Weights download is the biggest slice of most cold boots and the platform
    could only ever say how long the WHOLE ref took. A per-component span is
    not a finer roll-up of the same number: the components download
    CONCURRENTLY, so four 180 s components inside a 200 s `weights_fetch` is a
    completely different finding from four sequential 50 s ones, and only
    start/end per component can tell them apart.

    ``expected`` is {component: file count}. A component closes when its last
    file is accounted, so a span never re-opens after closing — a refcount
    that touched zero mid-fetch would otherwise split one component into two
    rows and make the ladder stop reconciling.
    """

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
    """Make ``ordinal`` the implicit parent for spans opened in this context.

    :func:`open_span` (as opposed to :func:`span`) does NOT push onto the
    nesting stack — it cannot, because its close happens in another frame. So
    a decomposition opened deep inside such a span, across `await` boundaries
    and module boundaries, has no way to find its parent and lands at the top
    level, where its time is added to the ladder a second time and
    `reconciliation` stops closing. Threading an ordinal through six call
    frames is the alternative; a ContextVar scope is the same fact stated once
    (and ContextVars copy per task, so concurrent fetches nest correctly).
    """
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
    """Record an instantaneous boot MILESTONE as a single closed row.

    ``since_process_start=True`` measures the milestone from OS process start —
    which is what "time to first-request-servable" means and what the
    autoscaler's cold-boot horizon needs.

    ``parent`` names the enclosing span's ordinal EXPLICITLY, for the same
    reason :func:`open_span` takes one: a phase measured across an ``await``
    boundary cannot read its parent off the implicit stack. `warmup` is exactly
    that shape — it is decided and measured around the warm call but only
    recorded once its cost is known to be real.

    A boot-CLOSING milestone (:data:`SERVABLE_PHASES`) recorded before ``hello``
    is HELD, not emitted: see the ``_hello_seen`` note. It is released, with its
    time re-read at release, by :func:`note_hello`.
    """
    # A milestone is cumulative BY NAME, not by the caller passing the flag.
    # `eager_ready` is marked from a setup task that runs inside the
    # `pipeline_load` span, so a caller who forgot the flag would charge a
    # whole-boot duration against a sub-second parent and drive its exclusive
    # time to zero. Making it structural means the flag can only ever be
    # redundant, never wrong.
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
    # A CUMULATIVE milestone measures from process start, so it is never part
    # of any span and must never be recorded as one's child: `reconciliation`
    # and every hub-side reader subtract a child's duration from its parent's
    # exclusive time, and a whole-boot number charged against an 870ms
    # `pipeline_load` drives that to zero. Cumulative rows are top-level by
    # construction rather than by the caller remembering.
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
        # Every cumulative milestone is remembered, not just the servable one:
        # `eager_ready` and `compiled_swap` are the two user-visible timestamps
        # and the interval between them is how long a pod served eager while
        # its cell arrived.
        with _lock:
            _milestone_ms.setdefault(phase, duration_ms)
    if phase in SERVABLE_PHASES:
        global _servable_ms
        with _lock:
            if _servable_ms is None:
                _servable_ms = process_uptime_ms()
        if not detail:
            # The recorder owns its own reconciliation string. A caller that
            # formatted it BEFORE the milestone was actually released (a
            # servable close can be held) would ship one for a different instant.
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
        # The gate opens on the hello ROW existing, not on which helper wrote
        # it: a bare `mark(PHASE_HELLO)` must open it too, or a later close
        # stays held forever.
        note_hello()


def note_hello() -> None:
    """The worker->hub stream is up. Releases any boot close held by
    :func:`mark`.

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
    steady state — the weights materializer is the load-bearing case:
    it owns the ~230s of a cold boot, and it also runs every time the hub
    delivers a new ref hours later. Recording the steady-state calls would put
    non-boot spans in a boot ladder (so `residual_ms` stops reconciling) and
    grow the table without bound. Boot ends where the boot number ends.
    """
    with _lock:
        return _servable_ms is None


def _union_ms(intervals: List[Tuple[int, int]]) -> int:
    """Total wall time covered by ``intervals``, overlaps counted ONCE."""
    total = 0
    end_so_far = -1
    for start, end in sorted(intervals):
        if end <= end_so_far:
            continue
        total += end - max(start, end_so_far if end_so_far >= 0 else start)
        end_so_far = end
    return max(0, total)


def _deduped(rows: List[pb.BootPhase]) -> List[pb.BootPhase]:
    """One row per (ordinal, terminal) — the hub's own upsert key.

    A ladder read off the wire carries duplicates by design: `bind_sink`
    re-flushes every buffered row on each RECONNECT so a boot that lost its
    stream still delivers, and the hub upserts. A reader that does not dedupe
    counts a reconnecting boot's phases twice.
    """
    seen: Dict[Tuple[int, bool], pb.BootPhase] = {}
    for row in rows:
        seen.setdefault((row.ordinal, row.terminal), row)
    return list(seen.values())


def reconciliation(
    rows: Optional[List[pb.BootPhase]] = None,
) -> Dict[str, int]:
    """Boot totals, per the rule that an instrument must close.

    ``residual`` is the boot window no phase explained. It is REPORTED, never
    smeared across the measured phases: "unmeasured" and "zero" are different
    answers, and the residual is the honest hint about where the next
    instrument belongs.

    ``rows`` overrides this process's recorded ladder, so a test can ask the
    SAME arithmetic about a modified ladder (delete one phase's rows from a
    real boot and the verdict must go red).
    """
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
        # A cumulative milestone measures the SAME wall clock the spans already
        # account for, so adding it to the span sum would double-count the
        # whole boot. It is the total, not a part of it.
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
    # `measured_ms` is the UNION of the span intervals, not the sum of their
    # exclusive times: the two differ the moment anything runs concurrently,
    # and a ladder that closes by over-counting is worse than one that visibly
    # does not close. The difference answers "how much of this boot was
    # parallel", so it is reported rather than discarded.
    measured = _union_ms(intervals)
    out: Dict[str, int] = {"measured_ms": measured}
    out.update(per_class)
    if exclusive > measured:
        out["concurrency_ms"] = exclusive - measured
    # NAME the residual instead of reporting one lump. Two named segments cover
    # what no span can: the interpreter+import window (no span can open before
    # the module that opens spans is imported) and the post-servable tail a
    # compiled swap lands in. What survives both is the only honest
    # `residual_ms`.
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
        # `sdk_ready` is cumulative, so it covers any span that closed inside
        # it; only the part no span explained is named here.
        pre_sdk_spans = _union_ms([
            (max(0, r.process_uptime_ms - r.duration_ms), r.process_uptime_ms)
            for r in rows
            if r.terminal and not r.cumulative and r.process_uptime_ms <= sdk])
        named_sdk = max(0, sdk - pre_sdk_spans)
        out["named.sdk_import_ms"] = named_sdk
        named += named_sdk
        # The SECOND named window: SDK ready -> the first thing this worker was
        # told to do. That is the gRPC dial, the Hello/HelloAck round trip and
        # the wait for a DesiredResidency — real boot seconds in which the
        # worker deliberately does no local work, so no span can cover them and
        # leaving them in `residual_ms` reads as an instrument hole rather than
        # as the hub round trip it is. It is big enough to matter: 1,238 ms of a
        # 23,208 ms boot, i.e. on its own enough to fail the ~5% acceptance.
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
        # The interval a pod served EAGER while its cell was being made or
        # fetched. The single number the compiled-serving campaign is about.
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
    #: Wall time of the span, children included.
    duration_ms: int
    #: ``duration_ms`` minus the children's — what THIS phase itself spent.
    exclusive_ms: int
    #: ms from process start to the phase's OPEN and CLOSE. Both, because the
    #: whole point of a per-component decomposition is telling four overlapping
    #: 180 s fetches from four sequential 50 s ones, and only an interval can.
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
    """The boot decomposition, in emission order, with children subtracted.

    Open (non-terminal) rows are EXCLUDED — a phase with no close has no
    duration, and a table that silently rendered it as 0 would be the "default
    read as a fact" defect this vocabulary keeps closing. The open row itself
    still ships on the wire, where it is the finding.
    """
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
            # A cumulative milestone starts at process start by definition; a
            # span's open is its close minus its own duration, both read off
            # the one process clock.
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
    """The phase table as fixed-width text — what a runbook pastes.

    ``rows`` is a captured ladder (off the wire, or another process's), not a
    pre-rendered table: the reconciliation footer must be computed from the
    SAME rows as the body, or one boot's phases print under another boot's
    totals.
    """
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


#: The phases a boot of each SHAPE must produce. A boot's shape is what it
#: DID, so there is no single "complete" set: a memo-hit adopt boot legitimately
#: has no `trace_for_key` rows, and asserting otherwise would make the
#: completeness check unusable exactly where it matters. Callers name the shape
#: they drove; :func:`completeness` reports what that shape is missing.
SHAPE_EAGER: frozenset = frozenset({
    PHASE_SDK_READY, PHASE_HELLO,
    PHASE_WEIGHTS_FETCH, PHASE_COMPONENT_FETCH, PHASE_PIPELINE_LOAD,
    PHASE_EAGER_READY, PHASE_FIRST_REQUEST_SERVABLE,
})
#: A boot that came up through `python -m gen_worker.entrypoint` — i.e. every
#: pod. `env_establish` and its nested `lib_memo` are produced by
#: `env_seal.establish`, which the entrypoint, the mint child and the
#: entry-compile child all call and an EMBEDDED worker (the in-process test
#: harness, a library caller) does not. Kept as a separate shape rather than
#: folded into SHAPE_EAGER so an embedded boot is not asked for a phase it
#: legitimately cannot have — and so a POD boot still is.
SHAPE_ENTRYPOINT: frozenset = SHAPE_EAGER | frozenset({
    PHASE_ENV_ESTABLISH, PHASE_LIB_MEMO,
})
#: A boot that ADOPTED a cell the hub named: no trace, no fold — it pays a
#: download and an admission instead.
SHAPE_ADOPT: frozenset = SHAPE_ENTRYPOINT | frozenset({
    PHASE_CELL_FETCH, PHASE_CELL_VERIFY, PHASE_ENTRY_ADMIT, PHASE_CELL_ARM,
    PHASE_COMPILED_SWAP,
})
#: A boot that MINTED its own cell: declaration, per-class trace, fold, the
#: publish round trips, then the same admission as an adopt.
SHAPE_SELF_MINT: frozenset = SHAPE_ADOPT | frozenset({
    PHASE_DECLARATION_COMPOSE, PHASE_TRACE_FOR_KEY, PHASE_KEY_FOLD,
    PHASE_CELL_HUB_RTT,
}) - frozenset({PHASE_CELL_FETCH})

#: A boot whose phases explain less of the wall than this is not decomposed.
#: The acceptance bar: the phases sum to within ~5% of wall.
DEFAULT_RESIDUAL_TOLERANCE_PCT = 5.0


@dataclass(frozen=True)
class BootCompleteness:
    """Does this boot's phase table actually account for the boot?

    Two independent failures, reported separately because they have different
    fixes: a phase that never emitted (``missing`` — an instrument hole) and a
    boot whose measured phases do not add up to its wall clock (``residual_pct``
    — an unmeasured window). A table can be missing nothing and still explain
    half the boot.
    """

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
    """Verdict on this boot's decomposition against the shape it drove.

    ``rows`` reads a ladder captured off the wire instead of this process's
    own — how a test asserts the PRODUCTION boot's table rather than the
    recorder's memory of it.
    """
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
        _milestone_ms.clear()
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
    "PHASE_CELL_FETCH",
    "PHASE_CELL_ARM",
    "PHASE_FIRST_REQUEST_SERVABLE",
    "PHASE_SDK_READY",
    "PHASE_COMPONENT_FETCH",
    "PHASE_ENV_ESTABLISH",
    "PHASE_LIB_MEMO",
    "PHASE_DECLARATION_COMPOSE",
    "PHASE_TRACE_FOR_KEY",
    "PHASE_KEY_FOLD",
    "PHASE_CELL_HUB_RTT",
    "PHASE_CELL_VERIFY",
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
