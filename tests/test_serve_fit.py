"""th#683 P3 serve-time adaptive fit — plan_serve decision logic."""

from gen_worker.api.decorators import Resources
from gen_worker.models.hub_policy import TensorhubWorkerCapabilities
from gen_worker.models.serve_fit import (
    RUN_CPU,
    RUN_EMERGENCY,
    RUN_NATIVE,
    RUN_OFFLOAD,
    plan_serve,
)

CAPS_4090 = TensorhubWorkerCapabilities(cuda_version="12.8", gpu_sm=89, torch_version="2.8", installed_libs=[])
CAPS_SMALL = TensorhubWorkerCapabilities(cuda_version="12.8", gpu_sm=86, torch_version="2.8", installed_libs=[])  # e.g. an 8GB 3050
CAPS_CPU = TensorhubWorkerCapabilities(cuda_version="", gpu_sm=0, torch_version="2.8", installed_libs=[])


def test_native_fit_full_quality() -> None:
    plan = plan_serve(Resources(vram_gb=24.0), CAPS_4090, free_vram_gb=23.6, forbid_cpu_offload=False)
    assert plan.serveable and plan.run_mode == RUN_NATIVE and not plan.degraded
    assert not plan.warning


def test_small_card_uses_emergency_before_offload() -> None:
    # 24GB model, ~8GB card free: 24*0.45=10.8 > 8 -> not emergency -> offload.
    plan = plan_serve(Resources(vram_gb=24.0), CAPS_SMALL, free_vram_gb=8.0, forbid_cpu_offload=False)
    assert plan.serveable and plan.run_mode == RUN_OFFLOAD
    assert plan.est_latency_multiplier > 1.0 and plan.warning
    assert plan.recommended_vram_gb == 24.0

    # 12GB model, ~8GB card free: 12*0.45=5.4 <= 8 -> emergency 4-bit (on-GPU).
    plan = plan_serve(Resources(vram_gb=12.0), CAPS_SMALL, free_vram_gb=8.0, forbid_cpu_offload=False)
    assert plan.serveable and plan.run_mode == RUN_EMERGENCY


def test_offload_forbidden_here_is_unserveable_with_reason() -> None:
    plan = plan_serve(Resources(vram_gb=24.0), CAPS_SMALL, free_vram_gb=8.0, forbid_cpu_offload=True)
    assert not plan.serveable and plan.run_mode == RUN_OFFLOAD
    assert "forbidden" in plan.reason.lower()


def test_emergency_serves_even_when_offload_forbidden() -> None:
    # Emergency 4-bit is on-GPU (not CPU-touching), so the veto does not block it.
    plan = plan_serve(Resources(vram_gb=12.0), CAPS_SMALL, free_vram_gb=8.0, forbid_cpu_offload=True)
    assert plan.serveable and plan.run_mode == RUN_EMERGENCY


def test_cpu_only_offered_behind_warning() -> None:
    plan = plan_serve(Resources(vram_gb=24.0), CAPS_CPU, free_vram_gb=0.0, forbid_cpu_offload=False)
    assert plan.serveable and plan.run_mode == RUN_CPU
    assert plan.est_latency_multiplier >= 10.0
    assert "cpu" in plan.warning.lower() and plan.recommended_vram_gb == 24.0


def test_cpu_only_refused_when_forbidden() -> None:
    plan = plan_serve(Resources(vram_gb=24.0), CAPS_CPU, free_vram_gb=0.0, forbid_cpu_offload=True)
    assert not plan.serveable and plan.run_mode == RUN_CPU
    assert "no gpu" in plan.reason.lower()


def test_genuine_compute_capability_incompatibility_refused() -> None:
    # Requires Blackwell (cc=12.0) but on an SM89 card: no lever helps.
    plan = plan_serve(
        Resources(vram_gb=15.0, compute_capability=12.0),
        CAPS_4090, free_vram_gb=23.6, forbid_cpu_offload=False,
    )
    assert not plan.serveable and plan.run_mode == RUN_NATIVE
    assert "compute capability" in plan.reason.lower()


def test_never_refuses_on_hint_alone_in_production() -> None:
    """The core P3 invariant: with offload allowed (production/GPU-lane), no
    hardware-inadequacy case is ever a flat refusal — every GPU-present case
    is serveable by some lever."""
    for rec_gb, free in [(80.0, 8.0), (48.0, 16.0), (24.0, 6.0), (100.0, 23.6)]:
        plan = plan_serve(Resources(vram_gb=rec_gb), CAPS_SMALL, free_vram_gb=free, forbid_cpu_offload=False)
        assert plan.serveable, f"refused vram_gb={rec_gb} on free={free}"
