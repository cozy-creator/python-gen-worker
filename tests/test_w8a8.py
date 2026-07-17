"""W8A8 fp8-GEMM loader mode (gw#534) — contract round-trip against a REAL
tiny diffusers pipeline (no network), detection negatives, lane stamps, and
scaled_mm numerics on capable GPUs.

CPU lane: the data-free producer writes the exact artifact contract, the
loader dequants to bf16-resident and must reproduce the source weights to
fp8 rounding. GPU lane (sm_89+): Fp8ScaledLinear vs the dequant reference,
and a full pipeline load on the scaled_mm lane.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("accelerate")

from gen_worker.models import w8a8
from gen_worker.models.loading import load_from_pretrained, pipeline_weight_lane
from gen_worker.models.w8a8 import (
    W8A8_FLAVOR,
    detect_w8a8_artifact,
    fp8_scaled_linear_class,
    load_w8a8_pipeline,
    quantize_tree_w8a8,
)


def _cuda_sm89() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor >= 89


@pytest.fixture(scope="module")
def tiny_ddpm(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

    root = tmp_path_factory.mktemp("w8a8") / "src"
    unet = UNet2DModel(
        sample_size=8, in_channels=3, out_channels=3,
        block_out_channels=(32, 32), layers_per_block=1,
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        norm_num_groups=8,
    )
    DDPMPipeline(unet=unet, scheduler=DDPMScheduler()).save_pretrained(str(root))
    return root


@pytest.fixture(scope="module")
def w8a8_tree(tiny_ddpm: Path) -> Path:
    return quantize_tree_w8a8(tiny_ddpm, tiny_ddpm.parent / "w8a8")


def test_detects_contract_artifact(w8a8_tree: Path) -> None:
    art = detect_w8a8_artifact(w8a8_tree)
    assert art is not None
    assert art.component == "unet"
    assert len(art.quantized) > 0
    assert all("norm" not in n and "embed" not in n for n in art.quantized)
    # config corroboration written by the producer
    cfg = json.loads((w8a8_tree / "unet" / "config.json").read_text())
    assert cfg["quantization_config"]["quant_algo"] == "FP8"


def test_scale_free_fp8_tree_never_detects(tiny_ddpm: Path, tmp_path: Path) -> None:
    """The storage-cast #fp8 flavor (fp8 bytes, NO scales) must not take the
    w8a8 lane — the scales are the distinguisher."""
    import shutil

    from safetensors.torch import load_file, save_file

    cast = tmp_path / "cast"
    shutil.copytree(tiny_ddpm, cast)
    for f in (cast / "unet").glob("*.safetensors"):
        tensors = {
            k: (v.to(torch.float8_e4m3fn) if v.ndim == 2 and v.is_floating_point() else v)
            for k, v in load_file(str(f)).items()
        }
        save_file(tensors, str(f))
    assert detect_w8a8_artifact(cast) is None
    assert detect_w8a8_artifact(tiny_ddpm) is None  # plain bf16/fp32 tree


def test_dequant_lane_round_trips_weights(
    tiny_ddpm: Path, w8a8_tree: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dequant lane (host without scaled_mm): loaded weights reproduce the
    source to fp8 rounding, lane stamps bf16-resident."""
    from diffusers import DDPMPipeline, UNet2DModel

    monkeypatch.setattr(w8a8, "w8a8_gemm_mode", lambda: "")
    pipe = load_from_pretrained(DDPMPipeline, w8a8_tree)
    assert pipe._cozy_weight_lane == "bf16-resident"
    assert pipeline_weight_lane(pipe) == ""
    assert pipe.unet._cozy_w8a8_mode == "dequant"

    ref = UNet2DModel.from_pretrained(str(tiny_ddpm / "unet"))
    art = detect_w8a8_artifact(w8a8_tree)
    assert art is not None
    ref_sd = ref.state_dict()
    got_sd = pipe.unet.state_dict()
    name = art.quantized[0] + ".weight"
    a, b = ref_sd[name].float(), got_sd[name].float()
    rel = ((a - b).abs() / a.abs().clamp(min=1e-3)).max().item()
    assert rel < 0.13  # e4m3 rounding
    assert got_sd[name].dtype == torch.bfloat16


def test_full_dequant_pipeline_runs_on_cpu(
    w8a8_tree: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from diffusers import DDPMPipeline

    monkeypatch.setattr(w8a8, "w8a8_gemm_mode", lambda: "")
    # fp32 compute: DDPM's output path numpy-converts (no bf16 support there).
    pipe = load_from_pretrained(DDPMPipeline, w8a8_tree, dtype="fp32")
    out = pipe(batch_size=1, num_inference_steps=2, output_type="np")
    assert out.images.shape[-3:] == (8, 8, 3)


def test_ladder_classifies_w8a8_as_universal_fp8() -> None:
    from gen_worker.models.ladder import (
        CLASS_FP8,
        classify_flavor_token,
        placement_for_flavor,
    )

    assert classify_flavor_token(W8A8_FLAVOR) == CLASS_FP8
    p = placement_for_flavor(W8A8_FLAVOR)
    assert p is not None
    assert p.admits_sm(75) and p.admits_sm(89) and p.admits_sm(120)


def test_variant_fit_never_refuses_w8a8_on_sm() -> None:
    from gen_worker.api import Resources
    from gen_worker.models.hub_policy import (
        FIT_FP8,
        TensorhubWorkerCapabilities,
        variant_fit,
    )

    class _B:
        flavor = W8A8_FLAVOR

    caps = TensorhubWorkerCapabilities(
        cuda_version="12.9", gpu_sm=75, torch_version="2.13", installed_libs=[])
    fit, _ = variant_fit(Resources(vram_gb=1.0), caps, 20.0, binding=_B())
    assert fit == FIT_FP8


def test_compile_lane_key_separates_w8a8() -> None:
    from gen_worker.compile_cache import artifact_metadata, lane_drift

    class _P:
        pass

    w8a8_pipe = _P()
    w8a8_pipe._cozy_weight_lane = "w8a8"
    plain = _P()
    assert lane_drift(artifact_metadata(family="f", weight_lane="w8a8"), w8a8_pipe) == ""
    assert "weight_lane" in lane_drift(artifact_metadata(family="f"), w8a8_pipe)
    assert "weight_lane" in lane_drift(
        artifact_metadata(family="f", weight_lane="w8a8"), plain)


# ---------------------------------------------------------------------------
# GPU lane (sm_89+): numerics + the scaled_mm serve path. Tiny tensors only.
# ---------------------------------------------------------------------------

gpu = pytest.mark.skipif(not _cuda_sm89(), reason="needs CUDA sm_89+ (fp8 tensor cores)")


@gpu
def test_gemm_mode_probe_arms_a_branch_on_capable_gpu() -> None:
    """On real fp8 silicon one branch must win the micro-benchmark gate:
    rowwise on sm_90+, pertensor on sm_89 (Ada) — '' would mean the gate
    wrongly demoted a capable card to the dequant lane."""
    w8a8.w8a8_gemm_mode.cache_clear()
    mode = w8a8.w8a8_gemm_mode()
    major, minor = torch.cuda.get_device_capability()
    if major * 10 + minor >= w8a8.W8A8_ROWWISE_MIN_SM:
        assert mode == "rowwise"
    else:
        assert mode == "pertensor"


@gpu
def test_fp8_scaled_linear_matches_dequant_reference() -> None:
    torch.manual_seed(0)
    lin_cls = fp8_scaled_linear_class()
    M, K, N = 64, 128, 96
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    scale = (w.float().abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1e-12)
    wq = (w.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)

    mod = lin_cls(K, N, bias=True, compute_dtype=torch.bfloat16,
                  static_input_scale=False)
    mod.load_state_dict(
        {"weight": wq, "weight_scale": scale, "bias": bias}, assign=True)
    y = mod(x)
    ref = torch.nn.functional.linear(x, (wq.float() * scale).to(torch.bfloat16), bias)
    rel = ((y - ref).abs().max() / ref.abs().max()).item()
    assert rel < 0.06  # dynamic per-row activation quant error only


@gpu
def test_w8a8_pipeline_serves_on_scaled_mm_lane(w8a8_tree: Path) -> None:
    from diffusers import DDPMPipeline

    w8a8.w8a8_gemm_mode.cache_clear()
    art = detect_w8a8_artifact(w8a8_tree)
    assert art is not None
    # fp16 compute: DDPM's output path numpy-converts (no bf16 support there).
    pipe = load_w8a8_pipeline(DDPMPipeline, w8a8_tree, art,
                              compute_dtype=torch.float16)
    assert pipe._cozy_weight_lane == "w8a8"
    assert pipeline_weight_lane(pipe) == "w8a8"
    lin_cls = fp8_scaled_linear_class()
    swapped = [m for m in pipe.unet.modules() if isinstance(m, lin_cls)]
    assert swapped, "no Fp8ScaledLinear modules in the denoiser"
    assert all(m.weight.dtype == torch.float8_e4m3fn for m in swapped)
    pipe.to("cuda")
    out = pipe(batch_size=1, num_inference_steps=2, output_type="np")
    assert out.images.shape[-3:] == (8, 8, 3)
