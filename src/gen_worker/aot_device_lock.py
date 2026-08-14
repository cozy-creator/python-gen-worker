"""One GPU timing at a time, across the whole entry-compile pool.

THE HAZARD. An AOTI compile does not only generate code — it BENCHMARKS on the device.
On the pin (torch 2.13.0+cu130) the mint leaves
``triton.autotune_at_compile_time`` unset, and ``get_cpp_wrapper_config``
(``compile_fx.py:2237-2242``) resolves an unset value to ``has_triton() and
V.aot_compilation`` — i.e. **True for AOTI** ("Default to True for AOTI",
verbatim). So a mint's CUDA entries take the ONE-pass autotune-block path at
``graph.py:2549``, not the two-pass "run the whole model on real inputs" path
below it — that branch is reached only when the flag is explicitly False.
Either way, kernel configs are chosen by TIMING KERNELS ON THE CARD.

Which makes concurrency a CORRECTNESS problem, not a speed one: two entries
benchmarking at once measure each other's contention, pick worse configs, and
bake them into the artifact. Nothing downstream can see it — the cell key
digests the traced graph, the sm, the toolchain and the env seal, none of
which move when a kernel config changes, so a slower cell would publish under
the same key as a good one.

THE SEAM IS UPSTREAM'S, NOT OURS.
torch 2.13 ships ``set_gpu_benchmark_lock_context`` precisely for this, and
decorates EVERY ``benchmark_gpu`` / ``benchmark_gpu_with_cuda_graph``
implementation with ``@gpu_benchmark_lock``. Registering a cross-process file
lock as that context serializes every GPU timing the pool performs, whatever
path inductor takes, with no monkeypatching and no version-specific internals.
Upstream's own note — "harness contexts should support nested entry from the
same thread" — is why :class:`DeviceBenchmarkLock` is reentrant.

WHAT THIS DOES NOT FIX.
The serving tenant is running eager inference on the same card for the whole
mint — that is the mint contract, not an accident. So autotune timings are
ALREADY perturbed by live traffic at K=1, and this lock cannot exclude the
tenant. The pool is therefore held to a narrower claim: it must not perturb
ITSELF. Whether a mint taken beside live serving picks the same configs as a
mint on a quiet card is a pre-existing, unmeasured question this does not
close.
"""

from __future__ import annotations

import contextlib
import fcntl
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterator, Optional

logger = logging.getLogger(__name__)

#: Default lock file name inside the pool's workdir.
LOCK_NAME = "device-benchmark.lock"


class DeviceBenchmarkLock:
    """A reentrant, cross-process advisory lock over one file.

    ``flock`` is the right primitive here and a ``multiprocessing.Lock`` is
    not: the holders are spawned children of a spawned child, they come and
    go, and a child that is SIGKILLed mid-benchmark (the OOM killer is the
    expected way an entry dies) must release the lock by dying. The kernel
    drops an ``flock`` when the fd closes, so crash-release is free and there
    is no stale-lock state for anyone to reason about.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd: Optional[int] = None
        self._local = threading.local()
        self._guard = threading.Lock()
        self.waited_s = 0.0
        self.holds = 0

    def _depth(self) -> int:
        return int(getattr(self._local, "depth", 0))

    def _set_depth(self, value: int) -> None:
        self._local.depth = value

    @contextlib.contextmanager
    def hold(self) -> Iterator[None]:
        if self._depth() > 0:
            # Nested entry from the same thread: inductor's benchmark helpers
            # delegate to one another, and re-taking a held flock from the
            # same process is a no-op anyway. Counted, not re-acquired.
            self._set_depth(self._depth() + 1)
            try:
                yield
            finally:
                self._set_depth(self._depth() - 1)
            return
        with self._guard:
            if self._fd is None:
                self._fd = os.open(
                    self.path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o644)
        started = time.monotonic()
        fcntl.flock(self._fd, fcntl.LOCK_EX)
        self.waited_s += time.monotonic() - started
        self.holds += 1
        self._set_depth(1)
        try:
            yield
        finally:
            self._set_depth(0)
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            except OSError:
                pass

    def __call__(self) -> Any:
        """The context FACTORY torch's hook expects."""
        return self.hold()

    def close(self) -> None:
        with self._guard:
            if self._fd is not None:
                try:
                    os.close(self._fd)
                except OSError:
                    pass
                self._fd = None


_INSTALLED: Optional[DeviceBenchmarkLock] = None


def install(path: Path) -> Optional[DeviceBenchmarkLock]:
    """Route every inductor GPU benchmark in THIS process through ``path``.

    Returns the lock, or ``None`` when the running torch has no such hook —
    in which case the caller must NOT run a wide pool on a GPU cell, because
    nothing would keep two entries from benchmarking at once. Never raises:
    a mint that died installing a safety interlock would be a worse outcome
    than a mint that reports it could not install one.
    """
    global _INSTALLED
    if _INSTALLED is not None:
        return _INSTALLED
    try:
        from torch._inductor.runtime import benchmarking
    except Exception as exc:  # noqa: BLE001
        logger.warning("pgw#809: no inductor benchmarking module (%s)", exc)
        return None
    setter = getattr(benchmarking, "set_gpu_benchmark_lock_context", None)
    if setter is None:
        logger.warning(
            "pgw#809: this torch has no set_gpu_benchmark_lock_context — GPU "
            "autotune timings CANNOT be serialized across the entry pool, so "
            "a wide pool would silently benchmark against itself")
        return None
    lock = DeviceBenchmarkLock(Path(path))
    setter(lock)
    _INSTALLED = lock
    logger.info("pgw#809: GPU benchmark lock installed at %s", lock.path)
    return lock


def installed() -> Optional[DeviceBenchmarkLock]:
    return _INSTALLED


def supported() -> bool:
    """Whether this torch exposes the hook the pool's safety rests on."""
    try:
        from torch._inductor.runtime import benchmarking
    except Exception:  # noqa: BLE001
        return False
    return hasattr(benchmarking, "set_gpu_benchmark_lock_context")


__all__ = [
    "LOCK_NAME",
    "DeviceBenchmarkLock",
    "install",
    "installed",
    "supported",
]
