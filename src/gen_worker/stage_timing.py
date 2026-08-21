"""Per-stage timing for one served request."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Dict, Iterator, List, Mapping, Optional, Tuple

GPU_BUSY = "gpu_busy"
SMALL_GPU = "small_gpu"
GPU_IDLE = "gpu_idle"

_CLASS_BY_STAGE: Dict[str, str] = {
    "denoise": GPU_BUSY,
    "refine": GPU_BUSY,
    "upsample": GPU_BUSY,
    "vae_decode": GPU_BUSY,
    "vae_encode": GPU_BUSY,
    "compute": GPU_BUSY,
    "text_encode": SMALL_GPU,
    "text_encode_2": SMALL_GPU,
    "scheduler_setup": SMALL_GPU,
    "latent_prepare": SMALL_GPU,
    "adapter_activate": SMALL_GPU,
    "gpu_permit_wait": GPU_IDLE,
    "child_call_wait": GPU_IDLE,
    "input_fetch": GPU_IDLE,
    "setup_wait": GPU_IDLE,
    "image_encode": GPU_IDLE,
    "video_encode": GPU_IDLE,
    "output_integrity": GPU_IDLE,
    "audio_encode": GPU_IDLE,
    "credential_stamp": GPU_IDLE,
    "upload": GPU_IDLE,
    "output_serialize": GPU_IDLE,
}

_DENOISE_STAGES = frozenset({"denoise", "refine", "upsample"})

_MAX_INTERVALS = 512


def stage_class(name: str) -> str:
    """Classification for ``name``; unknown stages are GPU_IDLE-neutral and reported under ``class.unattributed`` instead of being guessed."""
    return _CLASS_BY_STAGE.get(name, "")


class StageTimer:
    """Thread-safe stage recorder for ONE request."""

    __slots__ = (
        "_lock", "_local", "_totals", "_intervals", "_pre", "_phases",
        "_steps", "_step_seen", "_handler_start", "_handler_end", "_truncated",
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._local = threading.local()
        self._totals: Dict[str, float] = {}
        self._intervals: List[Tuple[str, float, float]] = []
        self._pre: Dict[str, float] = {}
        self._phases: Dict[Tuple[str, str], float] = {}
        self._steps: Dict[str, List[Tuple[int, float]]] = {}
        self._step_seen: Dict[str, set] = {}
        self._handler_start: Optional[float] = None
        self._handler_end: Optional[float] = None
        self._truncated = False

    def handler_open(self) -> None:
        with self._lock:
            if self._handler_start is None:
                self._handler_start = time.monotonic()

    def handler_close(self) -> None:
        with self._lock:
            self._handler_end = time.monotonic()

    def record_pre(self, name: str, seconds: float) -> None:
        """Record a stage that ran BEFORE the handler window (the GPU-permit wait, input fetch): reported, but never part of the ``runtime_ms`` reconciliation."""
        if seconds <= 0:
            return
        with self._lock:
            self._pre[name] = self._pre.get(name, 0.0) + float(seconds)

    def record_phase(self, stage: str, phase: str, seconds: float) -> None:
        """Record a SUB-PHASE of ``stage``, reported as ``stage.phase``."""
        stage = _stage_name(stage)
        phase = _stage_name(phase)
        if not stage or not phase or seconds <= 0:
            return
        key = (stage, phase)
        with self._lock:
            self._phases[key] = self._phases.get(key, 0.0) + float(seconds)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Bracket a stage."""
        name = _stage_name(name)
        if not name:
            yield
            return
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        start = time.monotonic()
        frame = [start, 0.0]
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
        """Record the END of denoise step ``index`` (1-based) for ``stage``."""
        stage = str(stage or "denoise").strip() or "denoise"
        index = int(index)
        now = time.monotonic()
        with self._lock:
            seen = self._step_seen.setdefault(stage, set())
            if index in seen:
                return
            marks = self._steps.setdefault(stage, [])
            if len(marks) < _MAX_INTERVALS:
                seen.add(index)
                marks.append((index, now))

    def snapshot(
        self,
        handler_start: Optional[float] = None,
        handler_end: Optional[float] = None,
    ) -> Dict[str, int]:
        """Render ``stage_ms``."""
        with self._lock:
            totals = dict(self._totals)
            intervals = list(self._intervals)
            pre = dict(self._pre)
            phases = dict(self._phases)
            steps = {k: list(v) for k, v in self._steps.items()}
            start = handler_start if handler_start is not None else self._handler_start
            end = handler_end if handler_end is not None else self._handler_end
            truncated = self._truncated

        out: Dict[str, int] = {}
        for name, seconds in pre.items():
            out[name] = _ms(seconds)
        for (stage, phase), seconds in phases.items():
            out[stage + "." + phase] = _ms(seconds)
        if start is None:
            return out
        if end is None:
            end = time.monotonic()
        handler_total = max(0.0, end - start)

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
            out["resid.prep"] = _ms(
                prep - _clipped(intervals, start, denoise_start))
            out["resid.tail"] = _ms(
                tail - _clipped(intervals, denoise_end, end))

        residual = handler_total - measured
        out["resid.unattributed"] = _ms(max(0.0, residual))
        if residual < 0:
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


PRE_HANDLER_STAGES = frozenset(
    {"gpu_permit_wait", "input_fetch", "setup_wait", "instance_gate_wait",
     "gpu_idle_before"})


def reconciliation(stage_ms: Mapping[str, int]) -> Tuple[int, int]:
    """``(attributed_ms, runtime_ms)`` for a ``stage_ms`` map."""
    total = int(stage_ms.get("total.runtime", stage_ms.get("total.handler", 0)))
    attributed = int(stage_ms.get("resid.unattributed", 0))
    for key, value in stage_ms.items():
        if "." in key or key in PRE_HANDLER_STAGES:
            continue
        attributed += int(value)
    return attributed, total


def stage_ms_for_metrics(timer: Optional[StageTimer], runtime_ms: int) -> Dict[str, int]:
    """Render ``timer`` for ``JobMetrics.stage_ms``, closed against ``runtime_ms``."""
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
    """Bracket a stage on ``ctx``'s timer; a no-op for contexts that carry none (CLI dispatch, endpoint unit tests with a stub context)."""
    timer = getattr(ctx, "_stages", None)
    if not isinstance(timer, StageTimer):
        yield
        return
    with timer.stage(name):
        yield


def record_phase_of(ctx: object, stage: str, phase: str, seconds: float) -> None:
    """Record a sub-phase on ``ctx``'s timer; a no-op for contexts that carry none (CLI dispatch, endpoint unit tests with a stub context)."""
    timer = getattr(ctx, "_stages", None)
    if not isinstance(timer, StageTimer):
        return
    timer.record_phase(stage, phase, seconds)


def _stage_name(name: str) -> str:
    return str(name or "").strip().replace(".", "_")


def ms_from_seconds(seconds: float) -> int:
    """THE quantizer for every millisecond a request reports."""
    return int(round(max(0.0, float(seconds)) * 1000.0))


_ms = ms_from_seconds


__all__ = [
    "StageTimer",
    "ms_from_seconds",
    "stage_of",
    "record_phase_of",
    "stage_ms_for_metrics",
    "reconciliation",
    "PRE_HANDLER_STAGES",
    "stage_class",
    "GPU_BUSY",
    "GPU_IDLE",
    "SMALL_GPU",
]
