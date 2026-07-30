"""th#1043: joint precision fit for shared-component multi-lane records.

Real qwen-image evidence: two lanes (t2i/edit) share a ~15.5GB text
encoder+VAE, each carries an exclusive ~40GB bf16 transformer. Loading each
lane's precision reactively — one at a time, against whatever free VRAM
happens to be left at that moment — let the FIRST lane consume all headroom
at native precision, starving the second into an offload placement the
shared-component invariant then refuses outright (RetryableError,
"shared-component lanes require resident placement"). The fix decides
precision for the WHOLE group up front, against its COMBINED footprint.
"""

from __future__ import annotations

from gen_worker.executor import _shared_lanes_need_fp8

_GiB = 1024 ** 3

# Real th#1043 shapes: shared text_encoder+vae ~15.5GB, each transformer
# ~40GB bf16 (no fp8 checkpoint exists yet for this family).
_QWEN_SIZES = {
    "t2i": {"text_encoder": 15 * _GiB, "vae": int(0.5 * _GiB), "transformer": 40 * _GiB},
    "edit": {"text_encoder": 15 * _GiB, "vae": int(0.5 * _GiB), "transformer": 40 * _GiB},
}
_SHARED = ["text_encoder", "vae"]


def test_starved_group_forces_fp8_when_it_fits():
    # A100-80GB: ~79GiB usable. Native (15.5 + 40 + 40 = 95.5GiB) doesn't
    # fit; fp8'd transformers (15.5 + 20 + 20 = 55.5GiB) do.
    free = 79 * _GiB
    assert _shared_lanes_need_fp8(_QWEN_SIZES, _SHARED, free) is True


def test_group_that_fits_natively_is_left_alone():
    # A card with enough headroom for both lanes at native precision needs
    # no forcing.
    free = 200 * _GiB
    assert _shared_lanes_need_fp8(_QWEN_SIZES, _SHARED, free) is False


def test_group_that_cannot_fit_even_at_fp8_is_left_to_the_ladder():
    # Too small even halved: forcing fp8 here would just relocate the
    # failure, not fix it — leave it to the existing per-lane ladder.
    free = 30 * _GiB
    assert _shared_lanes_need_fp8(_QWEN_SIZES, _SHARED, free) is False


def test_single_lane_group_never_forces():
    assert _shared_lanes_need_fp8({"t2i": _QWEN_SIZES["t2i"]}, _SHARED, 1 * _GiB) is False


def test_no_shared_components_never_forces():
    assert _shared_lanes_need_fp8(_QWEN_SIZES, [], 1 * _GiB) is False


# ---------------------------------------------------------------------------
# th#1043 second layer (found live, pod 4xh4m999n26u5f): the gw#534
# bf16-resident upcast made a SINGLE-lane fit call against current free VRAM
# and silently un-forced the group's fp8 decision. pgw#772 removed that
# voluntary upcast entirely (it also forked the ck5 `lane` axis per-card);
# both the unforced and the forced load now stay on the fp8 storage lane.
# ---------------------------------------------------------------------------


def test_fp8_lane_holds_with_and_without_group_forcing(tmp_path):
    import pytest

    pytest.importorskip("torch")
    pytest.importorskip("diffusers")
    pytest.importorskip("accelerate")
    from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

    from gen_worker.models.provision import load_slot

    root = tmp_path / "src"
    unet = UNet2DModel(
        sample_size=8, in_channels=3, out_channels=3,
        block_out_channels=(32, 32), layers_per_block=1,
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"), norm_num_groups=8,
    )
    DDPMPipeline(unet=unet, scheduler=DDPMScheduler()).save_pretrained(str(root))

    # Unforced declared fp8: served as fp8 storage regardless of headroom
    # (pgw#772 — the lane is declared, never free-VRAM-probed).
    sl = load_slot(DDPMPipeline, str(root), slot="t2i", device="cpu",
                   binding=type("B", (), {"dtype": "", "storage_dtype": "fp8"})())
    assert getattr(sl.obj, "_cozy_weight_lane", "") == "fp8-hooks"
    assert getattr(sl.obj, "_cozy_fp8_storage_requested", False) is True

    # Forced group fp8 (th#1043): identical lane outcome, reported
    # structurally as the group-fit downgrade it is.
    sl = load_slot(DDPMPipeline, str(root), slot="t2i", device="cpu",
                   force_storage_dtype="fp8")
    assert getattr(sl.obj, "_cozy_weight_lane", "") == "fp8-hooks"
    assert getattr(sl.obj, "_cozy_fp8_storage_requested", False) is True
    # th#1043 (0.48.2): the forced downgrade is reported structurally, not
    # served silently as a native bf16 plan.
    assert sl.rung == "fp8"
    assert "shared-lane" in sl.rung_detail
