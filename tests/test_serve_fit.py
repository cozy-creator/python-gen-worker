"""th#683 P3 serve-time adaptive fit — plan_serve decision logic."""

import msgspec

from gen_worker.api.decorators import Resources
from gen_worker.models.hub_policy import (
    FIT_FP8,
    FIT_INCOMPATIBLE,
    FIT_NVFP4,
    TensorhubWorkerCapabilities,
    select_variant,
)
from gen_worker.models.serve_fit import (
    RUN_CPU,
    RUN_EMERGENCY,
    RUN_FP8_STORAGE,
    RUN_NATIVE,
    RUN_OFFLOAD,
    plan_serve,
)

CAPS_4090 = TensorhubWorkerCapabilities(cuda_version="12.8", gpu_sm=89, torch_version="2.8", installed_libs=[])
CAPS_SMALL = TensorhubWorkerCapabilities(cuda_version="12.8", gpu_sm=86, torch_version="2.8", installed_libs=[])  # e.g. an 8GB 3050
CAPS_BLACKWELL = TensorhubWorkerCapabilities(cuda_version="13.0", gpu_sm=120, torch_version="2.9", installed_libs=[])
CAPS_CPU = TensorhubWorkerCapabilities(cuda_version="", gpu_sm=0, torch_version="2.8", installed_libs=[])


class _Binding(msgspec.Struct):
    flavor: str = ""


def test_native_fit_full_quality() -> None:
    plan = plan_serve(Resources(vram_gb=24.0), CAPS_4090, free_vram_gb=23.6)
    assert plan.serveable and plan.run_mode == RUN_NATIVE and not plan.degraded
    assert not plan.warning
    assert plan.wanted == "bf16" and plan.ran == "bf16"


def test_small_card_ladder_fp8_then_nf4_then_offload() -> None:
    # 24GB model, ~8GB card free: 24*0.55=13.2 > 8 and 24*0.45=10.8 > 8
    # -> neither runtime quant rung fits -> offload.
    plan = plan_serve(Resources(vram_gb=24.0), CAPS_SMALL, free_vram_gb=8.0)
    assert plan.serveable and plan.run_mode == RUN_OFFLOAD
    assert plan.est_latency_multiplier > 1.0 and plan.warning
    assert plan.recommended_vram_gb == 24.0
    assert plan.ran == RUN_OFFLOAD

    # 12GB model, ~8GB card free: 12*0.55=6.6 <= 8 -> runtime fp8-E4M3
    # storage (near-native quality) engages BEFORE the nf4 rung.
    plan = plan_serve(Resources(vram_gb=12.0), CAPS_SMALL, free_vram_gb=8.0)
    assert plan.serveable and plan.run_mode == RUN_FP8_STORAGE
    assert plan.degraded and "#fp8" in plan.warning
    assert plan.est_latency_multiplier < 1.1

    # 16GB model, ~8GB card free: 16*0.55=8.8 > 8 but 16*0.45=7.2 <= 8
    # -> only the 4-bit emergency rung fits.
    plan = plan_serve(Resources(vram_gb=16.0), CAPS_SMALL, free_vram_gb=8.0)
    assert plan.serveable and plan.run_mode == RUN_EMERGENCY
    assert "4-bit" in plan.warning


def test_strict_vram_makes_offload_unserveable_with_reason() -> None:
    # strict_vram is the AUTHOR opt-out: refuse rather than serve via the
    # CPU-touching offload rung. (Paul's ruling 2026-07-10: the box/env veto
    # is gone; only the author may choose refuse-over-degrade.)
    plan = plan_serve(
        Resources(vram_gb=24.0, strict_vram=True), CAPS_SMALL, free_vram_gb=8.0)
    assert not plan.serveable and plan.run_mode == RUN_OFFLOAD
    assert "strict_vram" in plan.reason.lower()


def test_strict_vram_still_serves_on_gpu_runtime_quant_rungs() -> None:
    # fp8 storage / emergency 4-bit are on-GPU (not CPU-touching), so
    # strict_vram (which only bars CPU-resident weights) does not block them.
    plan = plan_serve(
        Resources(vram_gb=12.0, strict_vram=True), CAPS_SMALL, free_vram_gb=8.0)
    assert plan.serveable and plan.run_mode == RUN_FP8_STORAGE
    plan = plan_serve(
        Resources(vram_gb=16.0, strict_vram=True), CAPS_SMALL, free_vram_gb=8.0)
    assert plan.serveable and plan.run_mode == RUN_EMERGENCY


def test_offload_needed_serves_by_default_no_env_veto() -> None:
    # The core of Paul's ruling: a card too small for even the 4-bit rung
    # falls to offload and SERVES (degraded) by default — no env/box veto,
    # no forbid flag. Only strict_vram would refuse.
    plan = plan_serve(Resources(vram_gb=24.0), CAPS_SMALL, free_vram_gb=8.0)
    assert plan.serveable and plan.run_mode == RUN_OFFLOAD and plan.degraded


def test_cpu_only_offered_behind_warning() -> None:
    plan = plan_serve(Resources(vram_gb=24.0), CAPS_CPU, free_vram_gb=0.0)
    assert plan.serveable and plan.run_mode == RUN_CPU
    assert plan.est_latency_multiplier >= 10.0
    assert "cpu" in plan.warning.lower() and plan.recommended_vram_gb == 24.0
    assert plan.ran == RUN_CPU


def test_cpu_only_refused_under_strict_vram() -> None:
    # No GPU + strict_vram (author refuses CPU-resident weights) -> refuse.
    plan = plan_serve(
        Resources(vram_gb=24.0, strict_vram=True), CAPS_CPU, free_vram_gb=0.0)
    assert not plan.serveable and plan.run_mode == RUN_CPU
    assert "no gpu" in plan.reason.lower()


def test_genuine_compute_capability_incompatibility_refused() -> None:
    # Requires Blackwell (cc=12.0) but on an SM89 card: no lever helps.
    plan = plan_serve(
        Resources(vram_gb=15.0, compute_capability=12.0),
        CAPS_4090, free_vram_gb=23.6,
    )
    assert not plan.serveable and plan.run_mode == RUN_NATIVE
    assert "compute capability" in plan.reason.lower()


def test_never_refuses_on_hint_alone_in_production() -> None:
    """The core P3 invariant: with offload allowed (production/GPU-lane), no
    hardware-inadequacy case is ever a flat refusal — every GPU-present case
    is serveable by some lever."""
    for rec_gb, free in [(80.0, 8.0), (48.0, 16.0), (24.0, 6.0), (100.0, 23.6)]:
        plan = plan_serve(Resources(vram_gb=rec_gb), CAPS_SMALL, free_vram_gb=free)
        assert plan.serveable, f"refused vram_gb={rec_gb} on free={free}"


# --------------------------------------------------------------------------
# Stored-flavor rungs (fp8 / nvfp4): HW-window gating + native runs
# --------------------------------------------------------------------------


def test_fp8_capable_card_serves_stored_fp8_natively() -> None:
    plan = plan_serve(
        Resources(vram_gb=12.0), CAPS_4090, free_vram_gb=23.6,
        binding=_Binding(flavor="fp8"),
    )
    assert plan.serveable and plan.run_mode == RUN_NATIVE and not plan.degraded
    assert plan.fit == FIT_FP8
    assert plan.wanted == "fp8" and plan.ran == "fp8"


def test_blackwell_serves_stored_nvfp4_natively() -> None:
    plan = plan_serve(
        Resources(vram_gb=8.0), CAPS_BLACKWELL, free_vram_gb=23.6,
        binding=_Binding(flavor="nvfp4"),
    )
    assert plan.serveable and plan.run_mode == RUN_NATIVE
    assert plan.fit == FIT_NVFP4


def test_pre_ada_card_serves_stored_fp8_flavor() -> None:
    # The refuse-bug fix: a stored #fp8 flavor upcasts to bf16 at compute
    # (loading.apply_fp8_storage), so it SERVES natively on a pre-Ada card
    # (SM86) — never refused. fp8 storage is universal; only the fp8-over-bf16
    # PREFERENCE is SM-gated.
    plan = plan_serve(
        Resources(vram_gb=12.0), CAPS_SMALL, free_vram_gb=23.6,
        binding=_Binding(flavor="fp8"),
    )
    assert plan.serveable and plan.run_mode == RUN_NATIVE and not plan.degraded
    assert plan.fit == FIT_FP8
    assert plan.wanted == "fp8" and plan.ran == "fp8"


def test_non_blackwell_card_refuses_stored_nvfp4_flavor() -> None:
    plan = plan_serve(
        Resources(vram_gb=8.0), CAPS_4090, free_vram_gb=23.6,
        binding=_Binding(flavor="nvfp4"),
    )
    assert not plan.serveable and plan.fit == FIT_INCOMPATIBLE
    assert "nvfp4" in plan.reason and "Blackwell" in plan.reason


def test_stored_flavor_that_does_not_fit_offloads_not_requants() -> None:
    # An already-quantized flavor can't be halved again: no fp8/nf4 rung.
    plan = plan_serve(
        Resources(vram_gb=40.0), CAPS_4090, free_vram_gb=23.6,
        binding=_Binding(flavor="fp8"),
    )
    assert plan.serveable and plan.run_mode == RUN_OFFLOAD


# --------------------------------------------------------------------------
# Cross-flavor selection (cozy-local `--variant auto`): the ladder over the
# pre-expanded flavor rows — bf16 -> fp8 -> nvfp4 -> 4-bit.
# --------------------------------------------------------------------------

_ROWS = [
    ("gen@bf16", Resources(vram_gb=24.0), _Binding(flavor="")),
    ("gen@fp8", Resources(vram_gb=13.0), _Binding(flavor="fp8")),
    ("gen@nvfp4", Resources(vram_gb=8.0), _Binding(flavor="nvfp4")),
]


def test_select_variant_prefers_fp8_on_fp8_compute_card() -> None:
    # SM89 (Ada) has fp8 tensor cores: with both bf16 and fp8 fitting, fp8
    # wins (faster AND smaller) — Paul's "run fp8 wherever we can".
    choice = select_variant(_ROWS, CAPS_4090, free_vram_gb=23.6)
    assert choice is not None and choice.name == "gen@fp8"
    assert choice.fit == FIT_FP8


def test_select_variant_prefers_bf16_below_fp8_compute() -> None:
    # SM86 (Ampere) has no fp8 tensor cores — fp8 storage upcasts to bf16 at
    # the same speed, so bf16 is preferred when it fits; fp8 is a fit fallback.
    choice = select_variant(_ROWS, CAPS_SMALL, free_vram_gb=23.6)
    assert choice is not None and choice.name == "gen@bf16"


def test_select_variant_picks_fp8_flavor_on_fp8_capable_card() -> None:
    # 16GB free: bf16 (24 GB card hint) doesn't fit; the stored fp8 row does
    # and this SM89 card is fp8-compatible.
    choice = select_variant(_ROWS, CAPS_4090, free_vram_gb=16.0)
    assert choice is not None and choice.name == "gen@fp8"
    assert choice.fit == FIT_FP8


def test_select_variant_picks_nvfp4_on_blackwell_when_only_it_fits() -> None:
    # 10GB free: bf16 + fp8 don't fit; nvfp4 fits and Blackwell runs it.
    choice = select_variant(_ROWS, CAPS_BLACKWELL, free_vram_gb=10.0)
    assert choice is not None and choice.name == "gen@nvfp4"
    assert choice.fit == FIT_NVFP4


def test_select_variant_sub_sm89_uses_fp8_fallback_when_bf16_too_big() -> None:
    # SM86: nvfp4 stays HW-incompatible (Blackwell-only) and is dropped, but
    # the stored fp8 row is NOT dropped (universal storage). 12GB free: bf16
    # (24-1=23) doesn't fit, the fp8 row (13-1=12 <= 12) does and serves
    # natively — fp8 is the fit fallback below fp8-compute silicon.
    choice = select_variant(_ROWS, CAPS_SMALL, free_vram_gb=12.0)
    assert choice is not None and choice.name == "gen@fp8"
    assert choice.fit == FIT_FP8
