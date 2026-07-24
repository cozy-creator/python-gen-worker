"""Per-stage timing for one served request (th#1111).

``runtime_ms`` used to be one opaque number covering input fetch, text
encode, denoise, VAE decode, image encode, credential stamp and upload, and
the GPU-permit wait appeared in no metric at all. This module is the
measurement spine: framework hooks (``io.write_image``, the output stream's
finalize, the credential stamp, the executor's permit acquire) and endpoint
brackets (``with ctx.stage("text_encode")``) all land here, and
:meth:`StageTimer.snapshot` renders them as ``JobMetrics.stage_ms``.

Two properties the design is built around:

* **It reconciles.** Stage totals are EXCLUSIVE (a nested stage's time is
  charged to the child, never twice), so measured stages + ``resid.*`` sum to
  ``total.handler``, which equals ``runtime_ms``.
* **It classifies.** Every stage is GPU-BUSY, SMALL-GPU or GPU-IDLE, which is
  what makes ``class.gpu_busy / total.handler`` the per-request half of
  pgw#652's hot-fraction metric — the number that says how much of a
  request's wall clock the device was actually computing.

Derived from the same intervals: ``total.prep`` (handler start -> first
denoise step) and ``total.tail`` (last denoise step -> handler end), the two
numbers pipelining (th#1108 / pgw#652) is sized against.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Dict, Iterator, List, Mapping, Optional, Tuple

# Stage classification. GPU_BUSY stages are the device actually computing;
# SMALL_GPU are short device ops that leave most SMs free (overlap candidates);
# GPU_IDLE are CPU/network stages where the device is idle unless another
# request is pipelined into it.
GPU_BUSY = "gpu_busy"
SMALL_GPU = "small_gpu"
GPU_IDLE = "gpu_idle"

_CLASS_BY_STAGE: Dict[str, str] = {
    # device compute
    "denoise": GPU_BUSY,
    "refine": GPU_BUSY,
    "upsample": GPU_BUSY,
    "vae_decode": GPU_BUSY,
    "vae_encode": GPU_BUSY,
    "compute": GPU_BUSY,
    # short device ops
    "text_encode": SMALL_GPU,
    "text_encode_2": SMALL_GPU,
    "scheduler_setup": SMALL_GPU,
    "latent_prepare": SMALL_GPU,
    "adapter_activate": SMALL_GPU,
    # CPU / network
    "gpu_permit_wait": GPU_IDLE,
    "input_fetch": GPU_IDLE,
    "setup_wait": GPU_IDLE,
    "image_encode": GPU_IDLE,
    "video_encode": GPU_IDLE,
    "audio_encode": GPU_IDLE,
    "credential_stamp": GPU_IDLE,
    "upload": GPU_IDLE,
    "output_serialize": GPU_IDLE,
}

#: Stages whose window defines "denoise" for prep/tail derivation.
_DENOISE_STAGES = frozenset({"denoise", "refine", "upsample"})

#: Hard cap on recorded intervals so a pathological handler (a stage inside a
#: per-step loop) cannot grow unbounded.
_MAX_INTERVALS = 512


def stage_class(name: str) -> str:
    """Classification for ``name``; unknown stages are GPU_IDLE-neutral and
    reported under ``class.unattributed`` instead of being guessed."""
    return _CLASS_BY_STAGE.get(name, "")


class StageTimer:
    """Thread-safe stage recorder for ONE request.

    The handler runs on a worker thread while uploads may fan out to more
    threads, so open-stage stacks are per-thread; totals and the interval log
    are shared under one lock.
    """

    __slots__ = (
        "_lock", "_local", "_totals", "_intervals", "_pre",
        "_steps", "_handler_start", "_handler_end", "_truncated",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._local = threading.local()
        self._totals: Dict[str, float] = {}
        self._intervals: List[Tuple[str, float, float]] = []
        self._pre: Dict[str, float] = {}
        # stage name -> [(step_index, monotonic_at_step_end), ...]
        self._steps: Dict[str, List[Tuple[int, float]]] = {}
        self._handler_start: Optional[float] = None
        self._handler_end: Optional[float] = None
        self._truncated = False

    # -- recording ---------------------------------------------------------

    def handler_open(self) -> None:
        with self._lock:
            if self._handler_start is None:
                self._handler_start = time.monotonic()

    def handler_close(self) -> None:
        with self._lock:
            self._handler_end = time.monotonic()

    def record_pre(self, name: str, seconds: float) -> None:
        """Record a stage that ran BEFORE the handler window (the GPU-permit
        wait, input fetch): reported, but never part of the ``runtime_ms``
        reconciliation."""
        if seconds <= 0:
            return
        with self._lock:
            self._pre[name] = self._pre.get(name, 0.0) + float(seconds)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Bracket a stage. Nested stages are charged exclusively: the parent
        keeps only the time not spent inside its children."""
        name = _stage_name(name)
        if not name:
            yield
            return
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        start = time.monotonic()
        frame = [start, 0.0]  # [start, children_total]
        stack.append(frame)
        try:
            yield
        finally:
            stack.pop()
            end = time.monotonic()
            elapsed = max(0.0, end - start)
            exclusive = max(0.0, elapsed - frame[1])
            if stack:
                stack[-1][1] += elapsed
            with self._lock:
                self._totals[name] = self._totals.get(name, 0.0) + exclusive
                if len(self._intervals) < _MAX_INTERVALS:
                    self._intervals.append((name, start, end))
                else:
                    self._truncated = True

    def mark_step(self, stage: str, index: int) -> None:
        """Record the END of denoise step ``index`` (1-based) for ``stage``.
        Wired from ``diffusers_step_callback``, so every endpoint using the
        shared callback gets denoise timing with no code change."""
        stage = str(stage or "denoise").strip() or "denoise"
        now = time.monotonic()
        with self._lock:
            marks = self._steps.setdefault(stage, [])
            if len(marks) < _MAX_INTERVALS:
                marks.append((int(index), now))

    # -- rendering ---------------------------------------------------------

    def snapshot(
        self,
        handler_start: Optional[float] = None,
        handler_end: Optional[float] = None,
    ) -> Dict[str, int]:
        """Render ``stage_ms``. ``handler_start``/``handler_end`` override the
        recorded window so the executor can anchor the map to the exact
        interval ``runtime_ms`` measures."""
        with self._lock:
            totals = dict(self._totals)
            intervals = list(self._intervals)
            pre = dict(self._pre)
            steps = {k: list(v) for k, v in self._steps.items()}
            start = handler_start if handler_start is not None else self._handler_start
            end = handler_end if handler_end is not None else self._handler_end
            truncated = self._truncated

        out: Dict[str, int] = {}
        for name, seconds in pre.items():
            out[name] = _ms(seconds)
        if start is None:
            return out
        if end is None:
            end = time.monotonic()
        handler_total = max(0.0, end - start)

        # Denoise window, explicit brackets first, else derived from the
        # per-step marks (the shared diffusers callback).
        denoise_start: Optional[float] = None
        denoise_end: Optional[float] = None
        estimated = False
        for name, t0, t1 in intervals:
            if name in _DENOISE_STAGES:
                denoise_start = t0 if denoise_start is None else min(denoise_start, t0)
                denoise_end = t1 if denoise_end is None else max(denoise_end, t1)
        explicit_window = denoise_start is not None
        step_mean = 0.0
        for name, marks in steps.items():
            if not marks:
                continue
            marks.sort(key=lambda m: m[1])
            first, last = marks[0][1], marks[-1][1]
            n = len(marks)
            mean = (last - first) / (n - 1) if n > 1 else 0.0
            step_mean = max(step_mean, mean)
            if explicit_window:
                continue
            # The callback fires AFTER each step, so step 1's start is one
            # mean step before its mark. With a single step the start is
            # unknowable from marks alone — clamp to the mark itself.
            estimated = True
            ds = max(start, first - mean)
            denoise_start = ds if denoise_start is None else min(denoise_start, ds)
            denoise_end = last if denoise_end is None else max(denoise_end, last)
            totals[name] = max(totals.get(name, 0.0), max(0.0, last - ds))
            if len(intervals) < _MAX_INTERVALS:
                intervals.append((name, ds, last))
        if step_mean > 0:
            out["denoise.step_mean"] = _ms(step_mean)
        if estimated:
            out["flag.denoise_estimated"] = 1
        if truncated:
            out["flag.intervals_truncated"] = 1

        measured = 0.0
        for name, seconds in totals.items():
            out[name] = _ms(seconds)
            measured += seconds

        out["total.handler"] = _ms(handler_total)
        if denoise_start is not None and denoise_end is not None:
            prep = max(0.0, denoise_start - start)
            tail = max(0.0, end - denoise_end)
            out["total.prep"] = _ms(prep)
            out["total.tail"] = _ms(tail)
            out["total.denoise"] = _ms(
                sum(v for k, v in totals.items() if k in _DENOISE_STAGES)
            )
            # What the prep/tail windows did NOT explain — for images the tail
            # residual is essentially the VAE decode, the single biggest
            # un-bracketed stage in the fleet.
            out["resid.prep"] = _ms(
                prep - _clipped(intervals, start, denoise_start))
            out["resid.tail"] = _ms(
                tail - _clipped(intervals, denoise_end, end))

        residual = handler_total - measured
        out["resid.unattributed"] = _ms(max(0.0, residual))
        if residual < 0:
            # Concurrent stages (parallel uploads) can sum past wall clock;
            # say so rather than silently clamping.
            out["resid.overlap"] = _ms(-residual)

        classes: Dict[str, float] = {GPU_BUSY: 0.0, SMALL_GPU: 0.0, GPU_IDLE: 0.0}
        unclassified = max(0.0, residual)
        for name, seconds in totals.items():
            kind = stage_class(name)
            if kind:
                classes[kind] += seconds
            else:
                unclassified += seconds
        for kind, seconds in classes.items():
            out["class." + kind] = _ms(seconds)
        out["class.unattributed"] = _ms(unclassified)
        return out


def _clipped(
    intervals: List[Tuple[str, float, float]], lo: float, hi: float
) -> float:
    """Seconds of ``intervals`` falling inside ``[lo, hi]``. Overlapping
    intervals are unioned so a parallel fan-out cannot over-explain a window."""
    if hi <= lo:
        return 0.0
    spans = sorted(
        (max(lo, t0), min(hi, t1)) for _, t0, t1 in intervals if t1 > lo and t0 < hi
    )
    total = 0.0
    cur_lo: Optional[float] = None
    cur_hi = 0.0
    for s0, s1 in spans:
        if cur_lo is None:
            cur_lo, cur_hi = s0, s1
            continue
        if s0 > cur_hi:
            total += cur_hi - cur_lo
            cur_lo, cur_hi = s0, s1
        else:
            cur_hi = max(cur_hi, s1)
    if cur_lo is not None:
        total += cur_hi - cur_lo
    return total


#: Stages recorded OUTSIDE the handler window: reported, never part of the
#: ``runtime_ms`` reconciliation.
PRE_HANDLER_STAGES = frozenset({"gpu_permit_wait", "input_fetch", "setup_wait"})


def reconciliation(stage_ms: Mapping[str, int]) -> Tuple[int, int]:
    """``(attributed_ms, runtime_ms)`` for a ``stage_ms`` map.

    Attributed = every in-handler stage plus ``resid.unattributed``. The two
    are equal by construction; a divergence means a hook double-counts, which
    is the one failure mode that would make the instrument lie.
    """
    total = int(stage_ms.get("total.runtime", stage_ms.get("total.handler", 0)))
    attributed = int(stage_ms.get("resid.unattributed", 0))
    for key, value in stage_ms.items():
        if "." in key or key in PRE_HANDLER_STAGES:
            continue
        attributed += int(value)
    return attributed, total


def stage_ms_for_metrics(timer: Optional[StageTimer], runtime_ms: int) -> Dict[str, int]:
    """Render ``timer`` for ``JobMetrics.stage_ms``, closed against
    ``runtime_ms``.

    The handler window opens a hair after ``runtime_ms`` starts — the executor
    validates the compile fence, pins refs and activates per-request adapters
    in between. That prologue is sub-millisecond for adapter-free requests and
    is real (adapter activation) when LoRAs ride the request, so it is
    reported as ``slot_prologue`` rather than smeared, which keeps the exact
    invariant: every emitted stage + ``resid.unattributed`` == ``runtime_ms``.
    """
    if timer is None:
        return {}
    out = timer.snapshot()
    if not out:
        return out
    handler = out.get("total.handler", 0)
    gap = int(runtime_ms) - handler
    if handler > 0 and gap > 0:
        out["slot_prologue"] = gap
        out["class." + SMALL_GPU] = out.get("class." + SMALL_GPU, 0) + gap
    out["total.runtime"] = max(0, int(runtime_ms))
    return out


@contextmanager
def stage_of(ctx: object, name: str) -> Iterator[None]:
    """Bracket a stage on ``ctx``'s timer; a no-op for contexts that carry
    none (CLI dispatch, endpoint unit tests with a stub context)."""
    timer = getattr(ctx, "_stages", None)
    if not isinstance(timer, StageTimer):
        yield
        return
    with timer.stage(name):
        yield


def _stage_name(name: str) -> str:
    """Endpoint-supplied stage names share one flat map with the derived
    rollups, so ``.`` (the namespace separator for ``total.``/``class.``/
    ``resid.``/``flag.``) is folded away."""
    return str(name or "").strip().replace(".", "_")


def _ms(seconds: float) -> int:
    return int(round(max(0.0, float(seconds)) * 1000.0))


__all__ = [
    "StageTimer",
    "stage_of",
    "stage_ms_for_metrics",
    "reconciliation",
    "PRE_HANDLER_STAGES",
    "stage_class",
    "GPU_BUSY",
    "GPU_IDLE",
    "SMALL_GPU",
]
