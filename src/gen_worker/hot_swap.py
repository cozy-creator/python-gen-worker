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
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Callable, ContextManager, Iterator, Optional, Tuple
from . import activity as activity_mod
from . import compile_cache
from . import shape_growth
from .shape_growth import Debounce, TurnGateBusy, TurnGateClosed
from . import postmortem

logger = logging.getLogger(__name__)

EAGER = "eager"
COMPILED = "compiled"

# pgw#916: the GPU-turn admission types and the debounced republish are
# arm-agnostic and now live in `shape_growth`, which BOTH arms reach. They are
# re-exported here (same names, same objects) so the dynamo arm's call sites
# and its `except TurnGateBusy` handlers are untouched — one implementation,
# no second copy to drift.


# pgw#677: the background mint's seed forwards run inside this window. A
# novel signature seen here must NEVER compile inline — the seed holds the
# per-instance run gate, and an inline Dynamo+Inductor compile turns a
# ~seconds eager forward into a minutes-long gate hold (the measured
# 3.5-7 min warm units that starved every tenant request). Inside the
# window, route() forces EAGER + background enqueue regardless of the
# concurrent flag or VRAM headroom.
_MINT_SEED: ContextVar[bool] = ContextVar("gw_mint_seed_window", default=False)


@contextlib.contextmanager
def mint_seed_window() -> Iterator[None]:
    """Mark the current context (and its to_thread descendants) as a mint
    seed forward (pgw#677)."""
    token = _MINT_SEED.set(True)
    try:
        yield
    finally:
        _MINT_SEED.reset(token)


def in_mint_seed_window() -> bool:
    return _MINT_SEED.get()

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
# pgw#680: how many background heals one signature gets after serve-window
# guard misses. A signature that keeps missing AFTER its heals compiled is
# diverging on something dynamo guards but our signature does not capture
# per REQUEST (not per shape class) — healing cannot converge; route it
# eager permanently instead of thrashing compile churn.
_GUARD_MISS_HEAL_LIMIT = 2


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
    # The requesting thread's intra-op thread count. Dynamo's GLOBAL_STATE
    # guard snapshots torch.get_num_threads() on the COMPILING thread, and
    # the OpenMP ICV is per-thread once lazy-initialized — so an entry
    # compiled on the warm thread with a diverged value can never serve the
    # requesting thread (every heal would be dead, sigs would go volatile).
    # The warm compile imposes this value first, like grad/autocast.
    num_threads: Optional[int] = None
    # pgw#677: the executor's background GPU turn — the compile executes
    # ONLY inside it (yields to tenant demand; mutually exclusive with
    # tenant forwards on the owning instance). None = ungated legacy.
    turn: Optional[Callable[[str], ContextManager[None]]] = None


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
        # pgw#680 guard-miss heal state. ``healing``: sigs with an in-flight
        # serve-window heal — routed EAGER (even on non-concurrent routers)
        # until the warm thread marks them warm. ``volatile``: sigs past
        # _GUARD_MISS_HEAL_LIMIT — permanently eager-routed (per-request
        # guard variance; compiling cannot converge). Both are populated
        # ONLY from serve-window misses, so proof/warm windows — which rely
        # on sequential COMPILED verdicts — are never rerouted by them.
        self.healing: set = set()
        self.volatile: set = set()
        self.guard_miss_counts: dict = {}
        # pgw#677 reopen: seeds that could NOT enqueue their background
        # compile (vocabulary overflow / dummy failure). A nonzero count
        # means the mint's capture would be incomplete — the driver aborts
        # loudly instead of finalizing/publishing a partial cell.
        self.seed_dropped = 0
        # pgw#677: executor-provided background-turn factory. When set,
        # every warm job for this router executes inside a turn, and
        # route() stops degrading novel signatures to inline compiles on
        # tight VRAM headroom (the warm thread ensures headroom inside its
        # exclusive turn instead).
        self.turn_gate: Optional[Callable[[str], ContextManager[None]]] = None

    def set_turn_gate(
        self, turn_gate: Optional[Callable[[str], ContextManager[None]]],
    ) -> None:
        with self.lock:
            self.turn_gate = turn_gate

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

    def suspend(self) -> None:
        """Stop concurrent routing WITHOUT discarding warm/pending state
        (pgw#671): novel signatures go back to today's sequential
        compile-then-serve until :meth:`enable` is called again. Used
        around an adoption's proof warmup, whose sequential semantics an
        eager route would break."""
        with self.lock:
            self.concurrent = False
            self.on_warmed = None

    def stats(self) -> Tuple[int, int, int]:
        """(warm, pending, failed) signature counts — the eager-first boot
        mint driver's completion evidence (pgw#671)."""
        with self.lock:
            return len(self.warm), len(self.pending), len(self.bg_failed)

    def route(
        self, label: str, compiled: Callable[..., Any],
        args: tuple, kwargs: dict,
    ) -> Tuple[str, Optional[Tuple[Any, ...]]]:
        """(verdict, sig): COMPILED routes through the compiled callable
        (sequential compile on a miss — today's behavior); EAGER serves the
        original while the background warm compiles this signature."""
        sig = (label, signature(args, kwargs))
        seed = _MINT_SEED.get()
        with self.lock:
            # pgw#680: guard-miss verdicts outrank the concurrent gate — a
            # sig mid-heal (or proven per-request-volatile) serves eager on
            # EVERY router, including the never-concurrent mandatory lanes,
            # instead of re-raising the stance each request.
            if sig in self.volatile:
                return EAGER, sig
            if sig in self.healing:
                return EAGER, sig
            if self.closed:
                return COMPILED, sig
            # pgw#677: a mint seed forward holds the per-instance run gate —
            # it must never pay an inline compile there. The seed's only job
            # is vocabulary discovery: EAGER + background enqueue, even when
            # concurrent routing is off for ordinary requests.
            if not self.concurrent and not seed:
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
                # pgw#760: a permanent serving decision (every future novel
                # signature on this pipeline now pays the inline compile)
                # must not live only in pod logs.
                activity_mod.emit_event(
                    activity_mod.KIND_SERVE_DEGRADE,
                    f"target={label} warm={len(self.warm)} "
                    f"pending={len(self.pending)} failed={len(self.bg_failed)}"
                    f": signature vocabulary exceeded {_MAX_SIGS}; concurrent "
                    "routing disabled for this pipeline (a per-request scalar "
                    "is leaking into signatures)",
                    phase="sig_vocab_exceeded",
                )
                # pgw#677 reopen: NEVER inline-compile inside a seed — the
                # seed holds the run gate. The sig stays eager (unwarmed);
                # the mint driver's convergence loop fails LOUDLY instead.
                if seed:
                    self.seed_dropped += 1
                    return EAGER, sig
                return COMPILED, sig
            device = _first_cuda_device(args, kwargs)
            # pgw#677: with a turn gate the warm thread owns the device while
            # it compiles (no concurrent transient to protect against) and
            # ensures headroom itself; only ungated legacy routers keep the
            # degrade-to-inline-compile fallback — and never for seeds.
            if (self.turn_gate is None and not seed
                    and not _headroom_ok(device)):
                logger.warning(
                    "hot-swap: tight VRAM headroom for novel %s signature; "
                    "degrading to sequential compile-then-serve", label)
                return COMPILED, sig
            self.pending.add(sig)
            turn = self.turn_gate
        try:
            job = _WarmJob(
                router=self, label=label, sig=sig, compiled=compiled,
                args=_dummy_value(args), kwargs=_dummy_value(kwargs),
                device=device, grad_mode=_grad_mode(),
                autocast_dtype=_autocast_dtype(),
                num_threads=_num_threads(), turn=turn,
            )
        except Exception:
            with self.lock:
                self.pending.discard(sig)
            if seed:
                # pgw#677 reopen: a seed must never pay the inline compile,
                # even when its dummy cannot be built — the signature stays
                # eager and the mint's convergence loop reports it loudly.
                with self.lock:
                    self.seed_dropped += 1
                logger.warning(
                    "hot-swap: dummy batch for %s failed in a mint seed; "
                    "the signature stays eager", label, exc_info=True)
                return EAGER, sig
            logger.warning(
                "hot-swap: dummy batch for %s failed; sequential compile",
                label, exc_info=True)
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
            self.healing.discard(sig)
            self.warm.add(sig)

    def record_guard_miss(
        self, sig: Tuple[Any, ...], label: str,
        compiled: Callable[..., Any], args: tuple, kwargs: dict,
    ) -> str:
        """pgw#680 background heal: schedule the recompile for the exact
        input class that just guard-missed at serve time.

        Optimistic rules of the existing warm driver apply unchanged: the
        one shape-warm thread compiles at nice +10 on its own CUDA stream,
        no request ever waits, and the job carries a zero-filled dummy of
        the failing request's own args (never tenant content) so the heal
        targets the exact class. Dedup by signature; sigs past
        ``_GUARD_MISS_HEAL_LIMIT`` become permanently eager (``volatile``).
        Returns the verdict: healing | volatile | closed | queue_full |
        no_dummy."""
        with self.lock:
            if self.closed:
                return "closed"
            self.warm.discard(sig)
            count = self.guard_miss_counts.get(sig, 0) + 1
            self.guard_miss_counts[sig] = count
            if sig in self.volatile:
                return "volatile"
            if count > _GUARD_MISS_HEAL_LIMIT:
                self.volatile.add(sig)
                self.healing.discard(sig)
                self.pending.discard(sig)
                logger.error(
                    "hot-swap: %s guard-missed %d times despite heals — "
                    "per-request guard variance our signatures do not "
                    "capture; routing this signature EAGER from now on "
                    "(pgw#680)", label, count)
                return "volatile"
            if sig in self.healing:
                return "healing"
            self.healing.add(sig)
            self.pending.add(sig)
            turn = self.turn_gate
        try:
            job = _WarmJob(
                router=self, label=label, sig=sig, compiled=compiled,
                args=_dummy_value(args), kwargs=_dummy_value(kwargs),
                device=_first_cuda_device(args, kwargs),
                grad_mode=_grad_mode(), autocast_dtype=_autocast_dtype(),
                num_threads=_num_threads(), turn=turn,
            )
        except Exception:
            logger.warning(
                "hot-swap: guard-miss heal dummy for %s failed; the "
                "signature stays eager until its next miss", label,
                exc_info=True)
            with self.lock:
                self.pending.discard(sig)
                self.healing.discard(sig)
            return "no_dummy"
        if not _submit(job):
            with self.lock:
                self.pending.discard(sig)
                self.healing.discard(sig)
            logger.warning(
                "hot-swap: warm queue full; guard-missed %s heal retried on "
                "a later request", label)
            return "queue_full"
        logger.info(
            "hot-swap: background heal scheduled for guard-missed %s "
            "signature (pgw#680)", label)
        return "healing"


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


def _num_threads() -> Optional[int]:
    try:
        import torch

        return int(torch.get_num_threads())
    except Exception:
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
        except Exception as exc:
            logger.warning("hot-swap: warm worker item crashed", exc_info=True)
            # pgw#760: background-thread exception outside _run_warm_compile's
            # own catch — without this the crash has no channel at all.
            activity_mod.emit_event(
                activity_mod.KIND_SERVE_DEGRADE,
                f"target={job.label}: warm worker item crashed: "
                f"{type(exc).__name__}: {exc}",
                phase="warm_worker_crashed",
            )
        finally:
            _QUEUE.task_done()


def _ensure_headroom(device: Optional[int]) -> None:
    """Best-effort VRAM headroom for the warm forward: inside an exclusive
    background turn the only reclaimable pressure is allocator cache
    (pgw#677 — this replaces route()'s degrade-to-inline-compile). A real
    OOM is still caught per-signature by the caller."""
    if device is None:
        return
    try:
        import torch

        if not _headroom_ok(device):
            torch.cuda.empty_cache()
    except Exception:
        pass


def _run_warm(job: _WarmJob) -> None:
    router = job.router
    with router.lock:
        if router.closed:
            router.pending.discard(job.sig)
            return
    if job.turn is not None:
        # pgw#677: the compile + dummy forward execute the SAME modules the
        # serving path runs (and inductor benchmarks on the same device) —
        # ungated, that raced live tenant forwards: measured 8.6x tenant
        # latency during mints and the pgw#676 sm_86 SIGSEGV
        # (_forward_with_branch concurrent with compile_wrapper). The turn
        # yields to tenant demand and excludes tenant forwards for the
        # bounded duration of ONE compile.
        try:
            with job.turn("compile"):
                with router.lock:
                    if router.closed:
                        router.pending.discard(job.sig)
                        return
                _ensure_headroom(job.device)
                _run_warm_gated(job)
        except TurnGateBusy:
            # Live tenant demand: re-queue rather than block the one warm
            # thread — other routers' jobs keep flowing; this one retries
            # (and is eventually admitted by idle or the steal rule).
            if not _submit(job):
                with router.lock:
                    router.pending.discard(job.sig)
                    router.healing.discard(job.sig)
                logger.warning(
                    "hot-swap: warm queue full while yielding to tenant "
                    "demand; %s stays eager (retried on a later request)",
                    job.label)
        except TurnGateClosed:
            with router.lock:
                router.pending.discard(job.sig)
                router.healing.discard(job.sig)
            logger.info(
                "hot-swap: background turn gate closed; dropping warm job "
                "for %s (stays eager)", job.label)
        return
    _run_warm_gated(job)


def _run_warm_gated(job: _WarmJob) -> None:
    # pgw#714: name the background compile before it touches the GPU. A
    # signal death mid-compile then attributes to THIS compile marker, not
    # to whatever tenant request was in flight — the misattribution that
    # refused fn=generate and condemned (release, SKU) pairs for a software
    # race (th#1226/th#1236).

    token = postmortem.note_inflight(
        postmortem.COMPILE_KIND, postmortem.compile_marker(job.label))
    try:
        _run_warm_compile(job)
    finally:
        postmortem.clear_inflight(token)


def _run_warm_compile(job: _WarmJob) -> None:
    router = job.router
    t0 = time.monotonic()
    try:
        with contextlib.ExitStack() as stack:
            import torch

            # Align this thread's intra-op count with the requesting
            # thread's BEFORE compiling: the entry's GLOBAL_STATE guard
            # snapshots the compiling thread's value, and a mismatch makes
            # the entry unservable from every serving thread (the CI-only
            # heal-never-converges failure of test_guard_miss_pgw680).
            if (job.num_threads is not None
                    and job.num_threads != torch.get_num_threads()):
                torch.set_num_threads(job.num_threads)
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
            # pgw#680: a failed guard-miss heal keeps the signature eager
            # via bg_failed (concurrent routers) — the healing veto must
            # not outlive its job on the non-concurrent ones.
            router.healing.discard(job.sig)
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
        # pgw#760: the guard_miss event promised heal=healing; this is the
        # heal's (or first warm's) terminal outcome — the signature serves
        # eager for the life of the process. Name it on the wire.
        activity_mod.emit_event(
            activity_mod.KIND_SERVE_DEGRADE,
            f"target={job.label} sig={repr(job.sig)[:400]}: "
            f"{type(exc).__name__}: {exc}",
            phase="warm_compile_failed",
        )
        # pgw#916: and it is a permanent COVERAGE hole, which is the
        # arm-agnostic fact — the dynamo arm books it in the same ledger the
        # AOT arm books an uncovered declared class in, so the hub counts one
        # population instead of two half-populations.
        shape_growth.report(shape_growth.ShapeGap(
            arm=shape_growth.ARM_DYNAMO,
            family="",
            target=job.label,
            declared_class=f"sig:{repr(job.sig)[:200]}",
            reason=shape_growth.REASON_UNCOVERED,
            detail=f"background warm failed: {type(exc).__name__}: {exc}",
        ))
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
        except Exception as exc:
            logger.warning("hot-swap: on_warmed callback failed", exc_info=True)
            # pgw#760: on_warmed republishes the grown cell — a swallowed
            # failure means the fleet re-compiles this shape forever.
            activity_mod.emit_event(
                activity_mod.KIND_SERVE_DEGRADE,
                f"target={job.label}: on_warmed (cell republish) callback "
                f"failed: {type(exc).__name__}: {exc}",
                phase="republish_failed",
            )


# ---------------------------------------------------------------------------
# Pipeline-level wiring
# ---------------------------------------------------------------------------


def router_of(pipeline: Any) -> Optional[Router]:

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


__all__ = [
    "COMPILED",
    "Debounce",
    "EAGER",
    "Router",
    "enable",
    "router_of",
    "signature",
]
