"""Stage 5 + Stage 3 concurrency model tests (gen-orchestrator #337).

Worker-side acceptance criteria:

- ``_function_needs_gpu(spec)`` returns True iff the spec's
  ``resources.accelerator`` starts with ``"cuda"`` (case-insensitive).
  Returns False for ``accelerator=none`` / unset / missing-resources.
- The ``_gpu_semaphore`` BoundedSemaphore is sized to ``max(1, gpu_count)``
  and serializes SerialWorker handler dispatches that need a GPU.
- ``accelerator=none`` endpoints (marco-polo) skip the semaphore entirely,
  scaling to ThreadPoolExecutor capacity for regular-Python-webserver
  throughput.
- ``_bind_gpu_for_request(ctx)`` writes ``CUDA_VISIBLE_DEVICES`` from
  ``ctx.compute.gpu_index`` and is robust to the proto field being
  absent (defaults to 0); returns -1 when accelerator!=cuda.

These tests exercise the helpers in isolation + drive a SerialWorker
dispatch directly to confirm the semaphore behavior under burst load.
"""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from gen_worker.api.decorators import Resources
from gen_worker._worker_support import _SerialWorkerSpec
from gen_worker.worker import Worker


class _ResourcesAcceleratorOnly:
    """Minimal duck-typed stand-in carrying a single ``accelerator`` field.

    We can't instantiate the real Resources struct with arbitrary
    accelerator strings (post-init normalizes "gpu"→"cuda", rejects
    unknowns). Tests for the bare helper want full control over the
    field. A plain attribute object suffices.
    """

    def __init__(self, accelerator: Any) -> None:
        self.accelerator = accelerator


class _SpecStub:
    """Minimal duck-typed spec carrying only ``resources``."""

    def __init__(self, resources: Any) -> None:
        self.resources = resources


class _ComputeStub:
    """Minimal duck-typed RequestContext.compute carrying accelerator + gpu_index."""

    def __init__(self, accelerator: str = "", gpu_index: Any = 0) -> None:
        self.accelerator = accelerator
        # gpu_index intentionally optional — older pb may not define it;
        # `_bind_gpu_for_request` reads via getattr with default.
        if gpu_index is not None:
            self.gpu_index = gpu_index


class _CtxStub:
    """Minimal duck-typed RequestContext carrying ``compute``."""

    def __init__(self, compute: Any = None) -> None:
        self.compute = compute


# ============================================================================
# _function_needs_gpu
# ============================================================================


@pytest.mark.parametrize(
    "accelerator, expected",
    [
        ("cuda", True),
        ("CUDA", True),
        ("cuda:0", True),
        ("Cuda", True),
        ("none", False),
        ("NONE", False),
        ("cpu", False),
        ("", False),
        (None, False),
    ],
)
def test_function_needs_gpu_classifies_accelerator_string(
    accelerator: Any, expected: bool,
) -> None:
    spec = _SpecStub(_ResourcesAcceleratorOnly(accelerator))
    assert Worker._function_needs_gpu(spec) is expected


def test_function_needs_gpu_handles_missing_resources() -> None:
    """Defensive: spec without resources attribute → False (don't serialize)."""
    class _Empty:
        pass

    assert Worker._function_needs_gpu(_Empty()) is False


def test_function_needs_gpu_handles_none_spec() -> None:
    assert Worker._function_needs_gpu(None) is False


def test_function_needs_gpu_with_real_resources_struct() -> None:
    """End-to-end: the real Resources struct round-trips through the helper."""
    res_cuda = Resources(accelerator="cuda", requires_gpu=True)
    res_none = Resources(accelerator="none")
    res_unset = Resources()
    assert Worker._function_needs_gpu(_SpecStub(res_cuda)) is True
    assert Worker._function_needs_gpu(_SpecStub(res_none)) is False
    assert Worker._function_needs_gpu(_SpecStub(res_unset)) is False


# ============================================================================
# _bind_gpu_for_request
# ============================================================================


def test_bind_gpu_for_request_returns_neg1_for_no_compute() -> None:
    w = _semaphore_only_worker(gpu_count=1)
    assert w._bind_gpu_for_request(_CtxStub(compute=None)) == -1


def test_bind_gpu_for_request_returns_neg1_for_non_cuda_accelerator() -> None:
    w = _semaphore_only_worker(gpu_count=1)
    ctx = _CtxStub(compute=_ComputeStub(accelerator="none", gpu_index=0))
    assert w._bind_gpu_for_request(ctx) == -1


def test_bind_gpu_for_request_sets_env_var_for_cuda(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    w = _semaphore_only_worker(gpu_count=1)
    ctx = _CtxStub(compute=_ComputeStub(accelerator="cuda", gpu_index=3))
    assert w._bind_gpu_for_request(ctx) == 3
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "3"


def test_bind_gpu_for_request_defaults_gpu_index_to_zero_when_absent(monkeypatch) -> None:
    """Older pb without ``gpu_index = 9`` → getattr defaults to 0.

    The proto regeneration is a separate concern; the worker must still
    boot cleanly against an older pb. Verify by constructing a compute
    object that lacks the field entirely.
    """
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    w = _semaphore_only_worker(gpu_count=1)
    compute = _ComputeStub(accelerator="cuda", gpu_index=None)
    # Strip the gpu_index attribute to simulate older-pb behavior.
    if hasattr(compute, "gpu_index"):
        delattr(compute, "gpu_index")
    ctx = _CtxStub(compute=compute)
    assert w._bind_gpu_for_request(ctx) == 0
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"


# ============================================================================
# Semaphore — directly assert size + behavior on the Worker instance.
# ============================================================================


def _semaphore_only_worker(gpu_count: int) -> Worker:
    """Build a Worker with just enough state for the semaphore + helpers.

    Avoids the heavyweight gRPC init the real `__init__` does. Mirrors
    the `_bare_worker` pattern in test_serial_worker_dispatch.py.
    """
    w = Worker.__new__(Worker)
    w._gpu_count = gpu_count
    w._has_gpu = gpu_count > 0
    w._gpu_busy_lock = threading.Lock()
    w._gpu_busy_refcount = 0
    w._gpu_semaphore = threading.BoundedSemaphore(max(1, gpu_count))
    w._active_requests = {}
    w._active_requests_lock = threading.Lock()
    return w


def test_gpu_semaphore_sized_to_gpu_count_when_one() -> None:
    w = _semaphore_only_worker(gpu_count=1)
    # BoundedSemaphore exposes _value (CPython internal); not load-bearing
    # but useful for assertion. Documented in the spec as a debug surface.
    assert getattr(w._gpu_semaphore, "_value", None) == 1


def test_gpu_semaphore_sized_to_gpu_count_when_four() -> None:
    w = _semaphore_only_worker(gpu_count=4)
    assert getattr(w._gpu_semaphore, "_value", None) == 4


def test_gpu_semaphore_sized_to_one_when_zero_gpus() -> None:
    """No-GPU host gets a 1-slot semaphore (never blocks real workers since
    accelerator=cuda handlers skip if no GPU, but the primitive must exist)."""
    w = _semaphore_only_worker(gpu_count=0)
    assert getattr(w._gpu_semaphore, "_value", None) == 1


def test_eager_serial_setup_acquires_gpu_semaphore_for_gpu_record() -> None:
    w = _semaphore_only_worker(gpu_count=1)
    w._serial_class_specs = {}
    w._conversion_class_specs = {}
    w._micro_batch_aggregators = {}
    w._configure_torchinductor_cache_dir = lambda: None
    w._resolve_serial_model_paths = lambda _ep_spec: {}

    seen: list[tuple[Any, bool]] = []

    class Instance:
        def setup(self) -> None:
            seen.append((getattr(w._gpu_semaphore, "_value", None), w._get_gpu_busy_status()))

    rec = {
        "cls_name": "CompilingEndpoint",
        "instance": Instance(),
        "endpoint_spec": _SpecStub(Resources(accelerator="cuda", requires_gpu=True)),
        "started": False,
        "started_lock": threading.Lock(),
    }

    w._ensure_serial_class_started(rec, acquire_gpu_semaphore=True)

    assert seen == [(0, True)]
    assert rec["started"] is True
    assert w._get_gpu_busy_status() is False
    assert getattr(w._gpu_semaphore, "_value", None) == 1


def test_gpu_semaphore_serializes_two_acquires_when_size_one() -> None:
    """Direct semaphore behavior: with gpu_count=1, the second acquire blocks
    until the first releases. Simulates the SerialWorker dispatch pattern.
    """
    w = _semaphore_only_worker(gpu_count=1)
    holder_running = threading.Event()
    second_acquired = threading.Event()
    release_event = threading.Event()

    def holder() -> None:
        w._gpu_semaphore.acquire()
        try:
            holder_running.set()
            release_event.wait(timeout=2.0)
        finally:
            w._gpu_semaphore.release()

    def waiter() -> None:
        w._gpu_semaphore.acquire()
        try:
            second_acquired.set()
        finally:
            w._gpu_semaphore.release()

    t1 = threading.Thread(target=holder, daemon=True)
    t2 = threading.Thread(target=waiter, daemon=True)
    t1.start()
    holder_running.wait(timeout=1.0)
    t2.start()
    # Second acquire MUST be blocked while holder is in the section.
    assert not second_acquired.wait(timeout=0.1)
    release_event.set()
    t1.join(timeout=2.0)
    t2.join(timeout=2.0)
    assert second_acquired.is_set()


def test_gpu_semaphore_allows_four_concurrent_when_size_four() -> None:
    """With gpu_count=4, four concurrent acquires must succeed without
    blocking; the 5th blocks until one releases.
    """
    w = _semaphore_only_worker(gpu_count=4)
    n_inside = 0
    inside_lock = threading.Lock()
    all_in_event = threading.Event()
    release_event = threading.Event()

    def worker_fn() -> None:
        nonlocal n_inside
        w._gpu_semaphore.acquire()
        try:
            with inside_lock:
                n_inside += 1
                if n_inside == 4:
                    all_in_event.set()
            release_event.wait(timeout=2.0)
        finally:
            w._gpu_semaphore.release()

    threads = [threading.Thread(target=worker_fn, daemon=True) for _ in range(4)]
    for t in threads:
        t.start()
    assert all_in_event.wait(timeout=2.0), "expected 4 concurrent acquires under gpu_count=4"
    # 5th waiter blocks.
    fifth_acquired = threading.Event()

    def fifth() -> None:
        w._gpu_semaphore.acquire()
        try:
            fifth_acquired.set()
        finally:
            w._gpu_semaphore.release()

    t5 = threading.Thread(target=fifth, daemon=True)
    t5.start()
    assert not fifth_acquired.wait(timeout=0.1)
    release_event.set()
    for t in threads:
        t.join(timeout=2.0)
    t5.join(timeout=2.0)
    assert fifth_acquired.is_set()


# ============================================================================
# Conditional acquire — accelerator=none endpoints skip the semaphore entirely.
# ============================================================================


def test_accelerator_none_skips_semaphore_acquire() -> None:
    """100 concurrent ``accelerator=none`` dispatches must NOT contend on the
    semaphore — they all run in parallel on the ThreadPoolExecutor.

    Simulates the marco-polo no-GPU-webserver-scaling path.
    """
    w = _semaphore_only_worker(gpu_count=1)  # 1-slot sema
    spec_none = _SpecStub(_ResourcesAcceleratorOnly("none"))

    inside_lock = threading.Lock()
    peak_concurrent = 0
    current = 0
    start_event = threading.Event()
    all_done = threading.Event()
    completed = [0]
    total = 50

    def simulated_dispatch() -> None:
        nonlocal current, peak_concurrent
        # Mirror SerialWorker dispatch: check needs_gpu, conditionally
        # acquire. accelerator=none → no acquire.
        needs_gpu = Worker._function_needs_gpu(spec_none)
        if needs_gpu:
            w._gpu_semaphore.acquire()
        try:
            with inside_lock:
                current += 1
                if current > peak_concurrent:
                    peak_concurrent = current
            start_event.wait(timeout=2.0)
            with inside_lock:
                current -= 1
                completed[0] += 1
                if completed[0] == total:
                    all_done.set()
        finally:
            if needs_gpu:
                w._gpu_semaphore.release()

    pool = ThreadPoolExecutor(max_workers=total)
    try:
        futures = [pool.submit(simulated_dispatch) for _ in range(total)]
        # Wait for all 50 to enter the section concurrently.
        time.sleep(0.1)  # let threads queue + enter
        # Allow them to finish.
        start_event.set()
        all_done.wait(timeout=5.0)
        for f in futures:
            f.result(timeout=2.0)
    finally:
        pool.shutdown(wait=True)
    # With accelerator=none, the semaphore must NOT serialize — at least
    # some real overlap should have happened. We loosen this to >1 to
    # tolerate scheduler jitter; a serialized run would have peak == 1
    # all the way through.
    assert peak_concurrent > 1, (
        f"expected concurrent dispatch under accelerator=none, "
        f"got peak={peak_concurrent}"
    )


def test_accelerator_cuda_serializes_through_semaphore_gpu_count_one() -> None:
    """``accelerator=cuda`` dispatch with gpu_count=1 must serialize:
    peak concurrent == 1 across N parallel dispatch attempts.
    """
    w = _semaphore_only_worker(gpu_count=1)
    spec_cuda = _SpecStub(_ResourcesAcceleratorOnly("cuda"))

    inside_lock = threading.Lock()
    peak_concurrent = 0
    current = 0

    def simulated_dispatch() -> None:
        nonlocal current, peak_concurrent
        needs_gpu = Worker._function_needs_gpu(spec_cuda)
        if needs_gpu:
            w._gpu_semaphore.acquire()
        try:
            with inside_lock:
                current += 1
                if current > peak_concurrent:
                    peak_concurrent = current
            # Hold the section briefly so concurrent attempts have a
            # chance to overlap if the semaphore is broken.
            time.sleep(0.02)
            with inside_lock:
                current -= 1
        finally:
            if needs_gpu:
                w._gpu_semaphore.release()

    pool = ThreadPoolExecutor(max_workers=10)
    try:
        futures = [pool.submit(simulated_dispatch) for _ in range(10)]
        for f in futures:
            f.result(timeout=5.0)
    finally:
        pool.shutdown(wait=True)
    assert peak_concurrent == 1, (
        f"expected serialized dispatch under accelerator=cuda + gpu_count=1, "
        f"got peak={peak_concurrent}"
    )


def test_accelerator_cuda_allows_gpu_count_concurrent() -> None:
    """``accelerator=cuda`` with gpu_count=4 must allow up to 4 concurrent
    dispatches before blocking.
    """
    w = _semaphore_only_worker(gpu_count=4)
    spec_cuda = _SpecStub(_ResourcesAcceleratorOnly("cuda"))

    inside_lock = threading.Lock()
    peak_concurrent = 0
    current = 0

    def simulated_dispatch() -> None:
        nonlocal current, peak_concurrent
        needs_gpu = Worker._function_needs_gpu(spec_cuda)
        if needs_gpu:
            w._gpu_semaphore.acquire()
        try:
            with inside_lock:
                current += 1
                if current > peak_concurrent:
                    peak_concurrent = current
            time.sleep(0.05)
            with inside_lock:
                current -= 1
        finally:
            if needs_gpu:
                w._gpu_semaphore.release()

    pool = ThreadPoolExecutor(max_workers=10)
    try:
        futures = [pool.submit(simulated_dispatch) for _ in range(10)]
        for f in futures:
            f.result(timeout=5.0)
    finally:
        pool.shutdown(wait=True)
    assert peak_concurrent == 4, (
        f"expected up to 4 concurrent dispatches under gpu_count=4, "
        f"got peak={peak_concurrent}"
    )


# ============================================================================
# Stage 4 vestigial-cleanup check: max_concurrent_per_worker must be gone.
# ============================================================================


def test_max_concurrent_per_worker_kwarg_deleted_from_decorator() -> None:
    """Stage 4: the SDK has zero ways for endpoint authors to declare
    concurrency. Confirm the kwarg is rejected by the decorator.
    """
    from gen_worker import inference

    with pytest.raises(TypeError):
        # noinspection PyArgumentList
        @inference.function(max_concurrent_per_worker=4)
        def _bad(self, ctx, payload):
            pass


def test_function_spec_dataclass_has_no_max_concurrent_per_worker() -> None:
    """Stage 4: the dataclass field is gone too — not just the decorator."""
    from gen_worker.api.decorators import _FunctionSpec

    field_names = set(_FunctionSpec.__struct_fields__)  # msgspec.Struct API
    assert "max_concurrent_per_worker" not in field_names
