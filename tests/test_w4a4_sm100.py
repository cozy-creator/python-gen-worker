"""W4A4 fp4 scaled_mm lane on real Blackwell silicon (sm_100+) — skip-clean
everywhere else. Mirrors the w8a8 sm_89 GPU suite: probe verdict, module
numerics vs the dequant reference, and a real pipeline on the fp4 lane."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("accelerate")

from gen_worker.models import w4a4
from gen_worker.models.loading import pipeline_weight_lane
from gen_worker.models.w4a4 import (
    detect_w4a4_artifact,
    load_w4a4_pipeline,
    quantize_nvfp4_tensor,
    quantize_tree_w4a4,
    to_blocked_scales,
    w4a4_linear_class,
)


def _cuda_sm100() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor >= 100


gpu = pytest.mark.skipif(
    not _cuda_sm100(), reason="needs CUDA sm_100+ (Blackwell fp4 tensor cores)")


@pytest.fixture(scope="module")
def w4a4_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

    root = tmp_path_factory.mktemp("w4a4gpu") / "src"
    unet = UNet2DModel(
        sample_size=8, in_channels=3, out_channels=3,
        block_out_channels=(32, 32), layers_per_block=1,
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        norm_num_groups=8,
    )
    DDPMPipeline(unet=unet, scheduler=DDPMScheduler()).save_pretrained(str(root))
    return quantize_tree_w4a4(root, root.parent / "w4a4")


@gpu
def test_gemm_mode_probe_arms_blockwise_on_blackwell() -> None:
    """On real fp4 silicon the blockwise lane must arm — '' would mean the
    numerics self-check or micro-benchmark wrongly demoted a capable card
    to the dequant lane."""
    w4a4.w4a4_gemm_mode.cache_clear()
    assert w4a4.w4a4_gemm_mode() == "blockwise"


@gpu
def test_w4a4_linear_matches_dequant_reference() -> None:
    torch.manual_seed(0)
    lin_cls = w4a4_linear_class()
    M, K, N = 64, 128, 96
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    packed, bs, ws2 = quantize_nvfp4_tensor(w)

    mod = lin_cls(K, N, bias=True, compute_dtype=torch.bfloat16,
                  static_input_scale=False)
    mod.load_state_dict({
        "weight": packed,
        "weight_scale": to_blocked_scales(bs),
        "weight_scale_2": ws2.reshape(1, 1),
        "bias": bias,
    }, assign=True)
    y = mod(x)
    from gen_worker.models.w4a4 import dequantize_nvfp4_tensor

    w_deq = dequantize_nvfp4_tensor(packed.cpu(), bs.cpu(), ws2.cpu()).to(
        "cuda", torch.bfloat16)
    ref = torch.nn.functional.linear(x, w_deq, bias)
    rel = ((y - ref).norm() / ref.norm().clamp(min=1e-9)).item()
    # weight quant identical on both sides; activation fp4 quant error only
    assert rel < 0.25


@gpu
def test_static_input_scale_branch_matches() -> None:
    torch.manual_seed(1)
    lin_cls = w4a4_linear_class()
    M, K, N = 32, 64, 32
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    packed, bs, ws2 = quantize_nvfp4_tensor(w)
    s2 = (x.abs().amax().float() / (6.0 * 448.0)).reshape(1, 1)

    mod = lin_cls(K, N, bias=False, compute_dtype=torch.bfloat16,
                  static_input_scale=True)
    mod.load_state_dict({
        "weight": packed,
        "weight_scale": to_blocked_scales(bs),
        "weight_scale_2": ws2.reshape(1, 1),
        "input_scale": s2.cpu(),
    }, assign=True)
    mod.to("cuda")
    dyn = lin_cls(K, N, bias=False, compute_dtype=torch.bfloat16,
                  static_input_scale=False)
    dyn.load_state_dict({
        "weight": packed,
        "weight_scale": to_blocked_scales(bs),
        "weight_scale_2": ws2.reshape(1, 1),
    }, assign=True)
    # static scale == the amax the dynamic path derives for this x
    y_static, y_dyn = mod(x), dyn(x)
    rel = ((y_static - y_dyn).norm() / y_dyn.norm().clamp(min=1e-9)).item()
    assert rel < 1e-3


@gpu
def test_w4a4_pipeline_serves_on_fp4_lane(w4a4_tree: Path) -> None:
    from diffusers import DDPMPipeline

    w4a4.w4a4_gemm_mode.cache_clear()
    art = detect_w4a4_artifact(w4a4_tree)
    assert art is not None
    pipe = load_w4a4_pipeline(DDPMPipeline, w4a4_tree, art,
                              compute_dtype=torch.float16)
    assert pipe._cozy_weight_lane == "w4a4"
    assert pipeline_weight_lane(pipe) == "w4a4"
    lin_cls = w4a4_linear_class()
    swapped = [m for m in pipe.unet.modules() if isinstance(m, lin_cls)]
    assert swapped, "no W4A4Linear modules in the denoiser"
    assert all(m.weight.dtype == torch.uint8 for m in swapped)
    pipe.to("cuda")
    out = pipe(batch_size=1, num_inference_steps=2, output_type="np")
    assert out.images.shape[-3:] == (8, 8, 3)
