"""W4A4 nvfp4 loader mode (gw#540) — contract round-trip against a REAL
tiny diffusers pipeline (no network), detection negatives (incl. cross-lane
w8a8 disambiguation), e2m1 helper exactness, lane stamps, and ladder/compile
classification. The fp4 scaled_mm lane itself is GPU-gated in
test_w4a4_sm100.py (Blackwell only).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("accelerate")

from gen_worker.models import w4a4
from gen_worker.models.loading import load_from_pretrained, pipeline_weight_lane
from gen_worker.models.w4a4 import (
    W4A4_FLAVOR,
    cast_e2m1,
    dequantize_nvfp4_tensor,
    detect_w4a4_artifact,
    pack_e2m1,
    quantize_nvfp4_tensor,
    quantize_tree_w4a4,
    sanitize_w4a4_state_dict,
    unpack_e2m1,
)


@pytest.fixture(scope="module")
def tiny_ddpm(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

    root = tmp_path_factory.mktemp("w4a4") / "src"
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
def w4a4_tree(tiny_ddpm: Path) -> Path:
    return quantize_tree_w4a4(tiny_ddpm, tiny_ddpm.parent / "w4a4")


# ---------------------------------------------------------------------------
# e2m1 helpers: modelopt-identical math.
# ---------------------------------------------------------------------------


def test_e2m1_cast_is_exact_on_grid_and_rounds_ties_up() -> None:
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    codes = cast_e2m1(torch.cat([grid, -grid]))
    vals = unpack_e2m1(pack_e2m1(codes))
    assert torch.equal(vals, torch.cat([grid, -grid]))
    # modelopt tie behavior: 0.75/1.75/2.5 round UP; 5.0 rounds DOWN to 4
    # (round-half-to-even on the e2m1 grid — byte-identical to _cast_fp4)
    ties = cast_e2m1(torch.tensor([0.75, 1.75, 2.5, 5.0]))
    assert torch.equal(unpack_e2m1(pack_e2m1(ties)),
                       torch.tensor([1.0, 2.0, 3.0, 4.0]))
    # clamp behavior beyond the grid
    big = cast_e2m1(torch.tensor([7.0, -9.0]))
    assert torch.equal(unpack_e2m1(pack_e2m1(big)),
                       torch.tensor([6.0, -6.0]))


def test_pack_unpack_nibble_order() -> None:
    codes = torch.tensor([[1, 2, 3, 4]], dtype=torch.uint8)
    packed = pack_e2m1(codes)
    # element 0 in the LOW nibble (torch.float4_e2m1fn_x2 convention)
    assert packed.tolist() == [[0x21, 0x43]]
    vals = unpack_e2m1(packed)
    assert vals.tolist() == [[0.5, 1.0, 1.5, 2.0]]


def test_quantize_dequantize_round_trip_snr() -> None:
    torch.manual_seed(0)
    w = torch.randn(64, 128)
    packed, bs, ws2 = quantize_nvfp4_tensor(w)
    assert packed.shape == (64, 64) and packed.dtype == torch.uint8
    assert bs.shape == (64, 8) and bs.dtype == torch.float8_e4m3fn
    assert ws2.numel() == 1 and ws2.dtype == torch.float32
    got = dequantize_nvfp4_tensor(packed, bs, ws2)
    rel = ((w - got).norm() / w.norm()).item()
    assert rel < 0.25  # e2m1 + two-level-scale rounding on gaussian weights


# ---------------------------------------------------------------------------
# Detection: the contract triple, and nothing else.
# ---------------------------------------------------------------------------


def test_detects_contract_artifact(w4a4_tree: Path) -> None:
    art = detect_w4a4_artifact(w4a4_tree)
    assert art is not None
    assert art.component == "unet"
    assert len(art.quantized) > 0
    assert not art.static_input_scales  # data-free producer = dynamic
    assert all("norm" not in n and "embed" not in n for n in art.quantized)
    cfg = json.loads((w4a4_tree / "unet" / "config.json").read_text())
    assert cfg["quantization_config"]["quant_algo"] == "NVFP4"


def test_plain_and_w8a8_trees_never_detect(
    tiny_ddpm: Path, tmp_path: Path,
) -> None:
    """Cross-lane disambiguation: a w8a8 tree (e4m3 weights + scales) and a
    plain tree must not take the w4a4 lane, and a w4a4 tree must not take
    the w8a8 lane — the weight dtype in the triple is the distinguisher."""
    from gen_worker.models.w8a8 import detect_w8a8_artifact, quantize_tree_w8a8

    assert detect_w4a4_artifact(tiny_ddpm) is None
    w8a8_tree = quantize_tree_w8a8(tiny_ddpm, tmp_path / "w8a8")
    assert detect_w4a4_artifact(w8a8_tree) is None
    w4a4_tree = quantize_tree_w4a4(tiny_ddpm, tmp_path / "w4a4")
    assert detect_w8a8_artifact(w4a4_tree) is None
    assert detect_w4a4_artifact(w4a4_tree) is not None


def test_static_input_scale_and_pre_quant_scale_detection(
    tiny_ddpm: Path, tmp_path: Path,
) -> None:
    from safetensors.torch import load_file, save_file

    tree = quantize_tree_w4a4(tiny_ddpm, tmp_path / "calibrated")
    art = detect_w4a4_artifact(tree)
    assert art is not None
    f = art.files[0]
    tensors = load_file(str(f))
    layer = art.quantized[0]
    in_f = tensors[f"{layer}.weight"].shape[1] * 2
    tensors[f"{layer}.input_scale"] = torch.tensor(0.01)
    tensors[f"{layer}.pre_quant_scale"] = torch.ones(in_f)
    save_file(tensors, str(f))
    art2 = detect_w4a4_artifact(tree)
    assert art2 is not None and art2.static_input_scales


# ---------------------------------------------------------------------------
# Dequant lane (CPU): numerics + lane stamps + a real pipeline call.
# ---------------------------------------------------------------------------


def test_dequant_lane_round_trips_weights(
    tiny_ddpm: Path, w4a4_tree: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from diffusers import DDPMPipeline, UNet2DModel

    monkeypatch.setattr(w4a4, "w4a4_gemm_mode", lambda: "")
    pipe = load_from_pretrained(DDPMPipeline, w4a4_tree)
    assert pipe._cozy_weight_lane == "bf16-resident"
    assert pipeline_weight_lane(pipe) == ""
    assert pipe.unet._cozy_w4a4_mode == "dequant"

    ref = UNet2DModel.from_pretrained(str(tiny_ddpm / "unet"))
    art = detect_w4a4_artifact(w4a4_tree)
    assert art is not None
    name = art.quantized[0] + ".weight"
    a = ref.state_dict()[name].float()
    b = pipe.unet.state_dict()[name].float()
    rel = ((a - b).norm() / a.norm()).item()
    assert rel < 0.25  # e2m1 rounding, norm-level (per-element is coarse)
    assert pipe.unet.state_dict()[name].dtype == torch.bfloat16


def test_full_dequant_pipeline_runs_on_cpu(
    w4a4_tree: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from diffusers import DDPMPipeline

    monkeypatch.setattr(w4a4, "w4a4_gemm_mode", lambda: "")
    pipe = load_from_pretrained(DDPMPipeline, w4a4_tree, dtype="fp32")
    out = pipe(batch_size=1, num_inference_steps=2, output_type="np")
    assert out.images.shape[-3:] == (8, 8, 3)


def test_sanitize_state_dict_dequants_and_folds_pre_quant_scale() -> None:
    torch.manual_seed(1)
    w = torch.randn(32, 64)
    pqs = torch.rand(64) + 0.5
    packed, bs, ws2 = quantize_nvfp4_tensor(w / pqs.reshape(1, -1))
    sd = {
        "lin.weight": packed,
        "lin.weight_scale": bs,
        "lin.weight_scale_2": ws2,
        "lin.input_scale": torch.tensor(0.02),
        "lin.pre_quant_scale": pqs,
        "other.weight": torch.randn(4, 4),
    }
    out = sanitize_w4a4_state_dict(sd, compute_dtype=torch.float32)
    assert set(out) == {"lin.weight", "other.weight"}
    rel = ((out["lin.weight"] - w).norm() / w.norm()).item()
    assert rel < 0.25  # smoothing folded back, quant rounding only
    # a non-matching dict passes through untouched
    plain = {"a.weight": torch.randn(2, 2)}
    assert sanitize_w4a4_state_dict(plain) == plain


# ---------------------------------------------------------------------------
# Ladder + compile-cache classification.
# ---------------------------------------------------------------------------


def test_ladder_classifies_w4a4_blackwell_only() -> None:
    from gen_worker.models.ladder import (
        CLASS_NVFP4,
        CLASS_NVFP4_W4A4,
        classify_flavor_token,
        placement_for_flavor,
    )

    assert classify_flavor_token(W4A4_FLAVOR) == CLASS_NVFP4_W4A4
    assert classify_flavor_token("nvfp4") == CLASS_NVFP4  # TRT lane unchanged
    p = placement_for_flavor(W4A4_FLAVOR)
    assert p is not None and p.sm_min == 100
    assert p.admits_sm(100) and p.admits_sm(103) and p.admits_sm(120)
    assert not p.admits_sm(89) and not p.admits_sm(90)


def test_variant_fit_gates_w4a4_on_blackwell() -> None:
    from gen_worker.api import Resources
    from gen_worker.models.hub_policy import (
        FIT_INCOMPATIBLE,
        FIT_NVFP4,
        TensorhubWorkerCapabilities,
        variant_fit,
    )

    class _B:
        flavor = W4A4_FLAVOR

    hopper = TensorhubWorkerCapabilities(
        cuda_version="12.9", gpu_sm=90, torch_version="2.13", installed_libs=[])
    fit, _ = variant_fit(Resources(vram_gb=1.0), hopper, 20.0, binding=_B())
    assert fit == FIT_INCOMPATIBLE
    blackwell = TensorhubWorkerCapabilities(
        cuda_version="12.9", gpu_sm=120, torch_version="2.13", installed_libs=[])
    fit, _ = variant_fit(Resources(vram_gb=1.0), blackwell, 20.0, binding=_B())
    assert fit == FIT_NVFP4


def test_compile_lane_key_separates_w4a4() -> None:
    from gen_worker.compile_cache import (
        artifact_metadata,
        compile_target_lane_error,
        flavor_label,
        lane_drift,
        lane_token,
    )

    class _P:
        pass

    w4a4_pipe = _P()
    w4a4_pipe._cozy_weight_lane = "w4a4"
    w8a8_pipe = _P()
    w8a8_pipe._cozy_weight_lane = "w8a8"
    assert lane_drift(
        artifact_metadata(family="f", weight_lane="w4a4"), w4a4_pipe) == ""
    assert "weight_lane" in lane_drift(
        artifact_metadata(family="f"), w4a4_pipe)
    assert "weight_lane" in lane_drift(
        artifact_metadata(family="f", weight_lane="w4a4"), w8a8_pipe)
    assert lane_token("w4a4") == "w4a4"
    assert flavor_label("rtx-5090", "2.13.0+cu130", "w4a4").endswith("-w4a4")
    assert compile_target_lane_error("w4a4", 0) == ""


def test_executor_ref_mandatory_lane() -> None:
    from gen_worker.executor import _ref_mandatory_lane

    assert _ref_mandatory_lane("tensorhub/qwen-image:prod#fp8-w8a8") == "w8a8"
    assert _ref_mandatory_lane("tensorhub/klein-4b:prod#nvfp4-w4a4") == "w4a4"
    assert _ref_mandatory_lane("tensorhub/klein-4b:prod") == ""
    assert _ref_mandatory_lane("tensorhub/klein-4b:prod#fp8") == ""
    assert _ref_mandatory_lane("tensorhub/klein-4b:prod#nvfp4") == ""
    assert _ref_mandatory_lane("not a ref !!") == ""
