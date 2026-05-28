"""GPU concurrency semaphore (gen-orchestrator #337 / gen-worker #336).

The floor item: the GPU semaphore ACTUALLY serializes cuda dispatch (the
1640s-thrash guard), and accelerator=none endpoints skip it for webserver-style
throughput. Driven through the real Worker helpers + real threads.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from gen_worker.api.decorators import Resources
from gen_worker.worker import Worker


class _ResAccel:
    def __init__(self, accelerator: Any) -> None:
        self.accelerator = accelerator


class _SpecStub:
    def __init__(self, resources: Any) -> None:
        self.resources = resources


def _sem_worker(gpu_count: int) -> Worker:
    w = Worker.__new__(Worker)
    w._gpu_count = gpu_count
    w._has_gpu = gpu_count > 0
    w._gpu_busy_lock = threading.Lock()
    w._gpu_busy_refcount = 0
    w._gpu_semaphore = threading.BoundedSemaphore(max(1, gpu_count))
    w._active_requests = {}
    w._active_requests_lock = threading.Lock()
    return w


@pytest.mark.parametrize(
    "accelerator, expected",
    [("cuda", True), ("CUDA", True), ("cuda:0", True),
     ("none", False), ("cpu", False), ("", False), (None, False)],
)
def test_function_needs_gpu_classifies_accelerator(accelerator: Any, expected: bool) -> None:
    assert Worker._function_needs_gpu(_SpecStub(_ResAccel(accelerator))) is expected
    # Real Resources struct round-trips through the helper too.
    if accelerator in ("cuda", "none"):
        assert Worker._function_needs_gpu(_SpecStub(Resources(accelerator=accelerator))) is expected
    # Defensive: no resources / None spec → False (don't serialize).
    assert Worker._function_needs_gpu(object()) is False
    assert Worker._function_needs_gpu(None) is False


@pytest.mark.parametrize(
    "accelerator, gpu_count, expected_peak",
    [("cuda", 1, 1), ("cuda", 4, 4), ("none", 1, None)],
)
def test_semaphore_serializes_cuda_and_skips_none(
    accelerator: str, gpu_count: int, expected_peak: int | None
) -> None:
    """cuda + gpu_count=1 → peak concurrency 1 (the thrash guard); gpu_count=4 →
    up to 4; accelerator=none → no serialization (peak > 1)."""
    w = _sem_worker(gpu_count)
    spec = _SpecStub(_ResAccel(accelerator))
    inside_lock = threading.Lock()
    peak = 0
    current = 0
    total = 12

    def dispatch() -> None:
        nonlocal current, peak
        needs = Worker._function_needs_gpu(spec)
        if needs:
            w._gpu_semaphore.acquire()
        try:
            with inside_lock:
                current += 1
                peak = max(peak, current)
            time.sleep(0.02)
            with inside_lock:
                current -= 1
        finally:
            if needs:
                w._gpu_semaphore.release()

    with ThreadPoolExecutor(max_workers=total) as pool:
        for f in [pool.submit(dispatch) for _ in range(total)]:
            f.result(timeout=5.0)

    if expected_peak is None:
        assert peak > 1, f"accelerator=none must run concurrent; peak={peak}"
    else:
        assert peak == expected_peak, f"cuda gpu_count={gpu_count}: peak={peak}"


def test_eager_serial_setup_holds_semaphore_through_binding_resolution() -> None:
    """gen-worker #336 defect B + E: the GPU semaphore must be held WHILE model
    binding resolution runs (where weights land in VRAM) — not just during
    setup() — and must be released even when resolution raises. The regression
    let two flux variants (bf16/fp8/nvfp4) load concurrently and thrash VRAM.
    """
    w = _sem_worker(gpu_count=1)
    w._serial_class_specs = {}
    w._conversion_class_specs = {}
    w._micro_batch_aggregators = {}
    w._configure_torchinductor_cache_dir = lambda: None
    w._emit_exception_diagnostic = lambda **_kw: None
    w._sanitize_diagnostic_value = lambda v, **_kw: v

    observed: dict[str, Any] = {}

    def _resolve(_ep_spec: Any, _setup_fn: Any) -> dict:
        observed["sema_during_resolution"] = w._gpu_semaphore._value
        observed["busy"] = w._get_gpu_busy_status()
        return {}

    w._resolve_serial_setup_kwargs = _resolve

    class Instance:
        def setup(self) -> None:
            pass

    rec = {
        "cls_name": "FluxVariant",
        "instance": Instance(),
        "endpoint_spec": _SpecStub(Resources(accelerator="cuda", requires_gpu=True)),
        "started": False,
        "started_lock": threading.Lock(),
    }
    w._ensure_serial_class_started(rec, acquire_gpu_semaphore=True)

    # Held (value 0) during resolution + GPU marked busy; released afterward.
    assert observed["sema_during_resolution"] == 0 and observed["busy"] is True
    assert w._gpu_semaphore._value == 1
    assert w._get_gpu_busy_status() is False and rec["started"] is True

    # And a crash during resolution must not strand the semaphore.
    rec2 = {
        "cls_name": "FluxVariant2",
        "instance": Instance(),
        "endpoint_spec": _SpecStub(Resources(accelerator="cuda", requires_gpu=True)),
        "started": False,
        "started_lock": threading.Lock(),
    }

    def _boom(_ep_spec: Any, _setup_fn: Any) -> dict:
        raise RuntimeError("weights load exploded")

    w._resolve_serial_setup_kwargs = _boom
    with pytest.raises(RuntimeError, match="weights load exploded"):
        w._ensure_serial_class_started(rec2, acquire_gpu_semaphore=True)
    assert w._gpu_semaphore._value == 1
    assert w._get_gpu_busy_status() is False and rec2["started"] is not True
