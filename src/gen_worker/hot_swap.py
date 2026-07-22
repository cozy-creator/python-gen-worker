"""Eager-while-compiling with hot-swap (pgw#622).

``torch.compile(dynamic=False)`` recompiles at every novel input signature,
which used to stall the first request at a new image shape behind a full
Dynamo+Inductor compile (30-60s, CPU-dominant). Consumer guards now route a
novel signature to the EAGER original immediately and warm the compiled
callable concurrently in one background thread with a zero-filled dummy
batch of the same signature (same weights in VRAM); a successful warm
atomically marks the signature warm so later calls take the compiled path.
The executor's ``on_warmed`` hook republishes the grown cell so the fleet
never compiles that (shape, GPU, lane) again.

Sequential (today's compile-then-serve) is kept when: concurrency is not
enabled (boot warmup window), the lane is mandatory-quantized (w8a8/w4a4 —
eager is not a production lane there), VRAM headroom is tight (degrade,
never OOM), or the dummy batch cannot be built. Regional-compiled targets
never consult the router (blocks are compiled in place; there is no
separable eager callable).
"""

from __future__ import annotations

import contextlib
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

logger = logging.getLogger(__name__)

EAGER = "eager"
COMPILED = "compiled"

# Concurrent warm transient ~= one extra batch of activations. Conservative
# free-VRAM floor; below it the request degrades to sequential (never OOM).
_BG_FLOOR_BYTES = 8 << 30
_QUEUE_MAX = 16
_SIG_DEPTH = 4
# A healthy shape-bucket vocabulary is dozens of signatures. An explosion
# means some per-request scalar leaks into the signature — stop concurrent
# routing for that router (back to today's behavior) instead of spamming
# warm jobs forever.
_MAX_SIGS = 256


# ---------------------------------------------------------------------------
# Input signatures
# ---------------------------------------------------------------------------


def _sig_value(value: Any, depth: int = 0) -> Any:
    if depth > _SIG_DEPTH:
        return type(value).__name__
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return ("T", tuple(value.shape), str(value.dtype), value.device.type)
    except Exception:
        pass
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_sig_value(v, depth + 1) for v in value)
    if isinstance(value, dict):
        return tuple(
            (str(k), _sig_value(v, depth + 1)) for k, v in sorted(
                value.items(), key=lambda kv: str(kv[0]))
        )
    return type(value).__name__


def signature(args: tuple, kwargs: dict) -> Tuple[Any, ...]:
    """Hashable identity of one call's guard-relevant inputs: tensor
    shapes/dtypes/devices plus scalar values (what dynamo specializes on)."""
    return (
        tuple(_sig_value(a) for a in args),
        tuple((str(k), _sig_value(v)) for k, v in sorted(kwargs.items())),
    )


def _dummy_value(value: Any, depth: int = 0) -> Any:
    if depth > _SIG_DEPTH:
        return value
    try:
        import torch

        if isinstance(value, torch.Tensor):
            # zeros: never retains request content; preserve_format keeps
            # channels_last strides (a guard axis for VAE targets).
            return torch.zeros_like(value)
    except Exception:
        pass
    if isinstance(value, tuple):
        return tuple(_dummy_value(v, depth + 1) for v in value)
    if isinstance(value, list):
        return [_dummy_value(v, depth + 1) for v in value]
    if isinstance(value, dict):
        return {k: _dummy_value(v, depth + 1) for k, v in value.items()}
    return value


def _first_cuda_device(args: tuple, kwargs: dict) -> Optional[int]:
    try:
        import torch
    except Exception:
        return None

    def scan(value: Any, depth: int = 0) -> Optional[int]:
        if isinstance(value, torch.Tensor) and value.is_cuda:
            return int(value.device.index or 0)
        if depth > _SIG_DEPTH:
            return None
        if isinstance(value, (list, tuple)):
            for v in value:
                found = scan(v, depth + 1)
                if found is not None:
                    return found
        if isinstance(value, dict):
            for v in value.values():
                found = scan(v, depth + 1)
                if found is not None:
                    return found
        return None

    for value in (*args, *kwargs.values()):
        found = scan(value)
        if found is not None:
            return found
    return None


def _headroom_ok(device: Optional[int]) -> bool:
    """True when a concurrent dummy-batch forward has honest VRAM room.
    Unknown/unprobeable state degrades to sequential (never OOM)."""
    if device is None:
        return True  # CPU-only call: no VRAM to protect
    try:
        import torch

        free, total = torch.cuda.mem_get_info(device)
        cached = max(
            0,
            torch.cuda.memory_reserved(device)
            - torch.cuda.memory_allocated(device),
        )
        return (free + cached) >= max(_BG_FLOOR_BYTES, total // 8)
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------


@dataclass
class _WarmJob:
    router: "Router"
    label: str
    sig: Tuple[Any, ...]
    compiled: Callable[..., Any]
    args: tuple
    kwargs: dict
    device: Optional[int]
    grad_mode: str  # "grad" | "no_grad" | "inference"
    autocast_dtype: Optional[Any]


class Router:
    """Per-pipeline signature routing shared by every whole-graph guard.

    Sequential until :meth:`enable`; the executor enables concurrency only
    AFTER the boot warmup proof, so the proof window keeps today's exact
    semantics. ``fail_closed`` lanes never enable — eager is not a W8A8/W4A4
    production lane, so their novel shapes keep the sequential inline
    compile."""

    def __init__(self, *, fail_closed: bool = False) -> None:
        self.lock = threading.Lock()
        self.fail_closed = bool(fail_closed)
        self.concurrent = False
        self.closed = False
        self.on_warmed: Optional[Callable[[], None]] = None
        self.warm: set = set()
        self.pending: set = set()
        self.bg_failed: set = set()

    def enable(self, on_warmed: Optional[Callable[[], None]] = None) -> bool:
        if self.fail_closed:
            return False
        with self.lock:
            self.concurrent = True
            self.on_warmed = on_warmed
        return True

    def close(self) -> None:
        with self.lock:
            self.closed = True
            self.concurrent = False
            self.on_warmed = None

    def route(
        self, label: str, compiled: Callable[..., Any],
        args: tuple, kwargs: dict,
    ) -> Tuple[str, Optional[Tuple[Any, ...]]]:
        """(verdict, sig): COMPILED routes through the compiled callable
        (sequential compile on a miss — today's behavior); EAGER serves the
        original while the background warm compiles this signature."""
        sig = (label, signature(args, kwargs))
        with self.lock:
            if not self.concurrent or self.closed:
                return COMPILED, sig
            if sig in self.warm:
                return COMPILED, sig
            if sig in self.pending or sig in self.bg_failed:
                return EAGER, sig
            if (len(self.warm) + len(self.pending)
                    + len(self.bg_failed)) >= _MAX_SIGS:
                logger.error(
                    "hot-swap: signature vocabulary exceeded %d — a "
                    "per-request scalar is leaking into signatures; "
                    "disabling concurrent routing for this pipeline",
                    _MAX_SIGS)
                self.concurrent = False
                return COMPILED, sig
            device = _first_cuda_device(args, kwargs)
            if not _headroom_ok(device):
                logger.warning(
                    "hot-swap: tight VRAM headroom for novel %s signature; "
                    "degrading to sequential compile-then-serve", label)
                return COMPILED, sig
            self.pending.add(sig)
        try:
            job = _WarmJob(
                router=self, label=label, sig=sig, compiled=compiled,
                args=_dummy_value(args), kwargs=_dummy_value(kwargs),
                device=device, grad_mode=_grad_mode(),
                autocast_dtype=_autocast_dtype(),
            )
        except Exception:
            logger.warning(
                "hot-swap: dummy batch for %s failed; sequential compile",
                label, exc_info=True)
            with self.lock:
                self.pending.discard(sig)
            return COMPILED, sig
        if not _submit(job):
            with self.lock:
                self.pending.discard(sig)
            logger.warning(
                "hot-swap: warm queue full; %s stays eager (retried on a "
                "later request)", label)
            return EAGER, sig
        logger.info(
            "hot-swap: novel input signature for %s — serving eager while "
            "the compiled path warms in the background", label)
        return EAGER, sig

    def mark_warm(self, sig: Optional[Tuple[Any, ...]]) -> None:
        """A successful compiled call at ``sig`` (inline or background)."""
        if sig is None:
            return
        with self.lock:
            self.pending.discard(sig)
            self.bg_failed.discard(sig)
            self.warm.add(sig)


# ---------------------------------------------------------------------------
# Background warm worker (one thread, one warm at a time)
# ---------------------------------------------------------------------------

_QUEUE: "queue.Queue[_WarmJob]" = queue.Queue(maxsize=_QUEUE_MAX)
_WORKER_LOCK = threading.Lock()
_WORKER: Optional[threading.Thread] = None


def _grad_mode() -> str:
    try:
        import torch

        if torch.is_inference_mode_enabled():
            return "inference"
        if not torch.is_grad_enabled():
            return "no_grad"
    except Exception:
        pass
    return "grad"


def _autocast_dtype() -> Optional[Any]:
    try:
        import torch

        if torch.is_autocast_enabled("cuda"):
            return torch.get_autocast_dtype("cuda")
    except Exception:
        pass
    return None


def _submit(job: _WarmJob) -> bool:
    global _WORKER
    with _WORKER_LOCK:
        if _WORKER is None or not _WORKER.is_alive():
            _WORKER = threading.Thread(
                target=_worker_loop, name="shape-warm", daemon=True)
            _WORKER.start()
    try:
        _QUEUE.put_nowait(job)
        return True
    except queue.Full:
        return False


def _worker_loop() -> None:
    try:  # background compile must never contend evenly with serving CPU
        os.setpriority(os.PRIO_PROCESS, threading.get_native_id(), 10)
    except Exception:
        pass
    while True:
        job = _QUEUE.get()
        try:
            _run_warm(job)
        except Exception:
            logger.warning("hot-swap: warm worker item crashed", exc_info=True)
        finally:
            _QUEUE.task_done()


def _run_warm(job: _WarmJob) -> None:
    router = job.router
    with router.lock:
        if router.closed:
            router.pending.discard(job.sig)
            return
    t0 = time.monotonic()
    try:
        with contextlib.ExitStack() as stack:
            import torch

            if job.grad_mode == "inference":
                stack.enter_context(torch.inference_mode())
            elif job.grad_mode == "no_grad":
                stack.enter_context(torch.no_grad())
            if job.device is not None:
                torch.cuda.set_device(job.device)
                if job.autocast_dtype is not None:
                    stack.enter_context(
                        torch.autocast("cuda", dtype=job.autocast_dtype))
                # A separate stream so warm kernels/autotune benchmarks
                # interleave with (never queue ahead of) a running
                # generation on the default stream.
                stream = torch.cuda.Stream(device=job.device)
                stack.enter_context(torch.cuda.stream(stream))
                job.compiled(*job.args, **job.kwargs)
                stream.synchronize()
            else:
                job.compiled(*job.args, **job.kwargs)
    except BaseException as exc:  # noqa: BLE001 — contained per-signature
        with router.lock:
            router.pending.discard(job.sig)
            router.bg_failed.add(job.sig)
        try:
            import torch

            if isinstance(exc, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
        except Exception:
            pass
        logger.warning(
            "hot-swap: background compile for %s failed (%s: %s); that "
            "signature stays eager for this process",
            job.label, type(exc).__name__, exc)
        return
    router.mark_warm(job.sig)
    with router.lock:
        callback = router.on_warmed
    logger.info(
        "hot-swap: compiled %s for novel signature in %.1fs; hot-swapped to "
        "the compiled path", job.label, time.monotonic() - t0)
    if callback is not None:
        try:
            callback()
        except Exception:
            logger.warning("hot-swap: on_warmed callback failed", exc_info=True)


# ---------------------------------------------------------------------------
# Pipeline-level wiring
# ---------------------------------------------------------------------------


def router_of(pipeline: Any) -> Optional[Router]:
    from . import compile_cache

    marker = getattr(pipeline, compile_cache._MARKER_ATTR, None) or {}
    signal = marker.get("failure_signal")
    if not isinstance(signal, dict):
        return None
    router = signal.get("router")
    return router if isinstance(router, Router) else None


def enable(
    pipeline: Any, on_warmed: Optional[Callable[[], None]] = None,
) -> bool:
    """Turn on eager-while-compiling for an armed pipeline's guards.

    Call AFTER the boot warmup proof (the proof window must keep sequential
    semantics). False when the pipeline has no router (eager-armed,
    producer arms, regional-only) or its lane is mandatory-quantized."""
    router = router_of(pipeline)
    if router is None:
        return False
    return router.enable(on_warmed)


@dataclass
class Debounce:
    """Coalesce ``on_warmed`` bursts into serialized runs of ``fn`` on a
    background thread: at most one in flight, one queued."""

    fn: Callable[[], None]
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _running: bool = False
    _dirty: bool = False

    def __call__(self) -> None:
        with self._lock:
            if self._running:
                self._dirty = True
                return
            self._running = True
        threading.Thread(
            target=self._run, name="cell-republish", daemon=True).start()

    def _run(self) -> None:
        while True:
            try:
                self.fn()
            except Exception:
                logger.warning("hot-swap: debounced callback failed", exc_info=True)
            with self._lock:
                if not self._dirty:
                    self._running = False
                    return
                self._dirty = False


__all__ = [
    "COMPILED",
    "Debounce",
    "EAGER",
    "Router",
    "enable",
    "router_of",
    "signature",
]
