"""Worker boot-time self-advertise: per-function Resources -> unavailable set.

`Worker._function_host_availability` checks a function's `Resources`
envelope against host hardware (cuda availability, SM, VRAM, libraries) and
returns ``(ok, status_dict)``. The worker's
`_refresh_worker_local_function_availability` walks every registered
function and builds the unavailable map at boot.

This test exercises the gating logic directly without spinning up gRPC.
"""

from __future__ import annotations

from gen_worker import Resources
from gen_worker.worker import Worker


def _bare_worker() -> Worker:
    w = Worker.__new__(Worker)
    return w


def test_resources_requiring_cuda_on_cpu_host_is_unavailable() -> None:
    w = _bare_worker()
    res = Resources(requires_gpu=True, min_vram_gb=22.0)
    gpu_info = {"gpu_count": 0, "gpu_sm": ""}
    ok, status = w._function_host_availability("generate-bf16", res, gpu_info)
    assert ok is False
    assert status["reason"] == "cuda_unavailable"


def test_resources_requiring_more_vram_than_host_has_is_unavailable() -> None:
    w = _bare_worker()
    res = Resources(requires_gpu=True, min_vram_gb=24.0)
    gpu_info = {"gpu_count": 1, "gpu_sm": "8.0", "gpu_total_mem": int(16 * 1024**3)}
    ok, status = w._function_host_availability("generate-bf16", res, gpu_info)
    assert ok is False
    assert status["reason"] == "insufficient_vram"


def test_resources_with_compute_capability_below_host_is_unavailable() -> None:
    w = _bare_worker()
    res = Resources(requires_gpu=True, min_vram_gb=4.0, min_compute_capability=10.0)
    gpu_info = {"gpu_count": 1, "gpu_sm": "8.0", "gpu_total_mem": int(80 * 1024**3)}
    ok, status = w._function_host_availability("blackwell-only", res, gpu_info)
    assert ok is False
    assert status["reason"] == "compute_capability_unmet"


def test_resources_satisfied_by_host_marks_available() -> None:
    w = _bare_worker()
    res = Resources(requires_gpu=True, min_vram_gb=8.0, min_compute_capability=8.0)
    gpu_info = {"gpu_count": 1, "gpu_sm": "8.6", "gpu_total_mem": int(24 * 1024**3)}
    ok, status = w._function_host_availability("generate-fp8", res, gpu_info)
    assert ok is True
    assert status == {}


def test_resources_cpu_function_on_cpu_host_is_available() -> None:
    w = _bare_worker()
    res = Resources()  # no GPU requirement
    gpu_info = {"gpu_count": 0, "gpu_sm": ""}
    ok, status = w._function_host_availability("cpu-only", res, gpu_info)
    assert ok is True
    assert status == {}


def test_per_function_resources_drive_per_function_unavailable_set() -> None:
    """The end-to-end check: declare three functions with different Resources
    on the same host, verify only the under-spec'd ones land in the
    unavailable set.
    """
    w = _bare_worker()
    w._discovered_resources = {
        "generate-bf16": Resources(requires_gpu=True, min_vram_gb=22.0),
        "generate-fp8": Resources(requires_gpu=True, min_vram_gb=12.0),
        "generate-nf4": Resources(requires_gpu=True, min_vram_gb=6.0),
    }
    gpu_info = {"gpu_count": 1, "gpu_sm": "8.0", "gpu_total_mem": int(16 * 1024**3)}

    # Drive the per-function loop and inspect the result.
    unavailable: dict[str, dict[str, str]] = {}
    for fn_name, req in w._discovered_resources.items():
        ok, status = w._function_host_availability(fn_name, req, gpu_info)
        if not ok:
            unavailable[fn_name] = status

    # 22 GB function is out; 12 GB and 6 GB fit on a 16 GB host.
    assert "generate-bf16" in unavailable
    assert "generate-fp8" not in unavailable
    assert "generate-nf4" not in unavailable
