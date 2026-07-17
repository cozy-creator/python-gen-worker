"""gw#557 / ie#494: streaming per-channel-scaled w8a8 producer + byte-gate.

Contract under test: ``streaming_w8a8_snapshot`` requantizes ONLY
repeated-block denoiser Linears from the bf16 source into fp8-E4M3 weights
with per-output-channel ``weight_scale`` twins (the gw#534 artifact the
loader detects by scales presence), leaves everything else at source
precision, and ``verify_w8a8_snapshot`` byte-gates the result. Plus the
w8a8-lane fp8+te TE wiring and the gw#547 LoRA-branch composability
assertion, all CPU-safe.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("accelerate")

from gen_worker.convert.writer import (  # noqa: E402
    ConversionImplementationError,
    W8A8_QUANT_SCHEME,
    W8A8_SKIP_TENSOR_PATTERNS,
    streaming_w8a8_cast,
    streaming_w8a8_snapshot,
    verify_w8a8_snapshot,
    w8a8_cast_eligible,
)
from gen_worker.models import w8a8  # noqa: E402
from gen_worker.models.w8a8 import (  # noqa: E402
    W8A8_FLAVOR,
    detect_w8a8_artifact,
    fp8_scaled_linear_class,
    load_w8a8_denoiser,
)


def test_flavor_constants_do_not_drift() -> None:
    assert W8A8_QUANT_SCHEME == W8A8_FLAVOR
    assert "gate_logits" in W8A8_SKIP_TENSOR_PATTERNS


@pytest.mark.parametrize(
    ("name", "st", "shape", "want"),
    [
        # the flip list: block Linears, 16-aligned, float
        ("transformer_blocks.0.attn1.to_q.weight", "BF16", [32, 32], True),
        ("transformer_blocks.47.audio_ff.net.2.weight", "F32", [64, 256], True),
        # the probe's explicit skip: gate logits stay full precision
        ("transformer_blocks.3.attn1.to_gate_logits.weight", "BF16", [32, 32], False),
        # outside a repeated-block container
        ("proj_in.weight", "BF16", [32, 32], False),
        ("caption_projection.linear_1.weight", "BF16", [64, 64], False),
        # misaligned dims / not 2-D / not a .weight / norms / already fp8
        ("transformer_blocks.0.ff.net.0.proj.weight", "BF16", [30, 32], False),
        ("transformer_blocks.0.norm1.weight", "BF16", [32], False),
        ("transformer_blocks.0.scale_shift_table", "BF16", [6, 32], False),
        ("transformer_blocks.0.norm_out.weight", "BF16", [32, 32], False),
        ("transformer_blocks.0.attn1.to_q.weight", "F8_E4M3", [32, 32], False),
    ],
)
def test_w8a8_cast_eligibility(name: str, st: str, shape: list, want: bool) -> None:
    assert w8a8_cast_eligible(name, st, shape) is want


# ---------------------------------------------------------------------------
# Streaming snapshot on a REAL tiny DiT (Transformer2DModel: transformer_blocks
# with 16-aligned Linears — the LTX shape class at toy size).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tiny_dit(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from diffusers import Transformer2DModel

    root = tmp_path_factory.mktemp("w8a8p") / "src"
    model = Transformer2DModel(
        num_attention_heads=2, attention_head_dim=8, in_channels=4,
        num_layers=2, sample_size=8, norm_num_groups=4,
    ).to(torch.bfloat16)
    model.save_pretrained(str(root / "transformer"), safe_serialization=True)
    (root / "scheduler").mkdir(parents=True)
    (root / "scheduler" / "scheduler_config.json").write_text("{}")
    (root / "model_index.json").write_text(json.dumps({
        "_class_name": "DiTPipeline",
        "_diffusers_version": "0.39.0",
        "transformer": ["diffusers", "Transformer2DModel"],
        "scheduler": ["diffusers", "DDIMScheduler"],
    }))
    return root


def _expected_quantized(root: Path) -> set[str]:
    """The spec: every 16-aligned nn.Linear inside transformer_blocks."""
    import torch.nn as nn

    from diffusers import Transformer2DModel

    model = Transformer2DModel.from_pretrained(str(root / "transformer"))
    want: set[str] = set()
    for bi, block in enumerate(model.transformer_blocks):
        for name, mod in block.named_modules():
            if (isinstance(mod, nn.Linear)
                    and mod.in_features % 16 == 0
                    and mod.out_features % 16 == 0):
                want.add(f"transformer_blocks.{bi}.{name}")
    assert want, "fixture must have eligible block Linears"
    return want


def _load_all(component_dir: Path) -> dict[str, "torch.Tensor"]:
    from safetensors import safe_open

    out: dict[str, torch.Tensor] = {}
    for f in sorted(component_dir.glob("*.safetensors")):
        with safe_open(str(f), framework="pt") as fh:
            for k in fh.keys():
                out[k] = fh.get_tensor(k)
    return out


@pytest.fixture(scope="module")
def produced(tiny_dit: Path) -> Path:
    out = tiny_dit.parent / "w8a8"
    result = streaming_w8a8_snapshot(tiny_dit, out, file_layout="diffusers")
    assert result["converted_count"] > 0
    return out


def test_snapshot_quantizes_exactly_the_block_linears(
    tiny_dit: Path, produced: Path,
) -> None:
    art = detect_w8a8_artifact(produced)
    assert art is not None
    assert art.component == "transformer"
    assert not art.static_input_scales  # dynamic activation scales only
    assert set(art.quantized) == _expected_quantized(tiny_dit)

    src = _load_all(tiny_dit / "transformer")
    got = _load_all(produced / "transformer")
    for name in art.quantized:
        w, s = got[f"{name}.weight"], got[f"{name}.weight_scale"]
        assert w.dtype == torch.float8_e4m3fn
        assert s.dtype == torch.float32 and s.shape == (w.shape[0],)
    # every non-quantized tensor passes through byte-identically
    quantized_weights = {f"{n}.weight" for n in art.quantized}
    for name, t in src.items():
        if name in quantized_weights:
            continue
        assert torch.equal(t, got[name]), name
    assert set(got) == set(src) | {f"{n}.weight_scale" for n in art.quantized}


def test_snapshot_stamps_config_without_touching_source(
    tiny_dit: Path, produced: Path,
) -> None:
    out_cfg = json.loads((produced / "transformer" / "config.json").read_text())
    assert out_cfg["quantization_config"]["quant_algo"] == "FP8"
    src_cfg = json.loads((tiny_dit / "transformer" / "config.json").read_text())
    assert "quantization_config" not in src_cfg  # hardlink never written through
    # passthrough files intact
    assert (produced / "scheduler" / "scheduler_config.json").exists()
    assert (produced / "model_index.json").exists()


def test_snapshot_refuses_requantizing_w8a8_source(produced: Path, tmp_path: Path) -> None:
    with pytest.raises(ConversionImplementationError, match="re-quantize"):
        streaming_w8a8_snapshot(produced, tmp_path / "again", file_layout="diffusers")


def test_byte_gate_passes_then_catches_tampering(
    tiny_dit: Path, produced: Path, tmp_path: Path,
) -> None:
    report = verify_w8a8_snapshot(tiny_dit, produced, sample=4)
    assert report["byte_exact"] and report["sampled"] == 4
    assert report["max_rel_err"] <= 2 ** -4 + 2 ** -9
    assert report["source_compute_dtype"] == "storage"
    assert report["source_storage_dtypes"] == ["bfloat16"]

    import shutil

    from safetensors.torch import save_file

    bad = tmp_path / "tampered"
    shutil.copytree(produced, bad)
    f = sorted((bad / "transformer").glob("*.safetensors"))[0]
    tensors = _load_all(bad / "transformer")
    art = detect_w8a8_artifact(produced)
    assert art is not None
    scale_name = f"{art.quantized[0]}.weight_scale"
    tensors[scale_name] = tensors[scale_name] * 2.0
    for extra in sorted((bad / "transformer").glob("*.safetensors"))[1:]:
        extra.unlink()
    f.unlink()
    save_file(tensors, str(f))
    with pytest.raises(ConversionImplementationError, match="byte-gate"):
        verify_w8a8_snapshot(tiny_dit, bad, sample=len(art.quantized))


def _candidate_with_shifted_scales(
    source: Path, candidate: Path, destination: Path, *, ulps: int,
) -> Path:
    """Model a valid CUDA export whose /448 result neighbors CPU's value."""
    import shutil

    from safetensors.torch import save_file

    shutil.copytree(candidate, destination)
    art = detect_w8a8_artifact(candidate)
    assert art is not None
    source_tensors = _load_all(source / art.component)
    candidate_tensors = _load_all(destination / art.component)
    for layer in art.quantized:
        weight_name = f"{layer}.weight"
        scale_name = f"{layer}.weight_scale"
        source_weight = source_tensors[weight_name].float()
        scale = source_weight.abs().amax(dim=1) / 448.0
        for _ in range(ulps):
            scale = torch.nextafter(scale, torch.full_like(scale, torch.inf))
        candidate_tensors[scale_name] = scale
        candidate_tensors[weight_name] = (
            (source_weight / scale.reshape(-1, 1))
            .clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn)
        )

    component = destination / art.component
    for path in component.glob("*.safetensors"):
        path.unlink()
    index = component / "diffusion_pytorch_model.safetensors.index.json"
    if index.exists():
        index.unlink()
    save_file(candidate_tensors, str(component / "diffusion_pytorch_model.safetensors"))
    return destination


def test_byte_gate_accepts_only_cpu_cuda_one_ulp_scale_envelope(
    tiny_dit: Path, produced: Path, tmp_path: Path,
) -> None:
    adjacent = _candidate_with_shifted_scales(
        tiny_dit, produced, tmp_path / "adjacent", ulps=1,
    )
    report = verify_w8a8_snapshot(tiny_dit, adjacent, sample=1000)
    assert report["byte_exact"] is True
    assert report["max_scale_ulp"] == 1

    outside = _candidate_with_shifted_scales(
        tiny_dit, produced, tmp_path / "outside", ulps=2,
    )
    with pytest.raises(ConversionImplementationError, match="maximum 1"):
        verify_w8a8_snapshot(tiny_dit, outside, sample=1000)


def _rewrite_component_float_dtype(
    root: Path, dtype: "torch.dtype", *, inject_fp16_detail: bool = False,
) -> None:
    from safetensors.torch import save_file

    component = root / "transformer"
    files = sorted(component.glob("*.safetensors"))
    tensors = _load_all(component)
    for name, value in tensors.items():
        if value.is_floating_point():
            converted = value.to(dtype=dtype)
            if inject_fp16_detail and converted.ndim == 2 and converted.numel():
                converted = converted.clone()
                # Exactly representable in FP16, rounded to 1.0 in BF16. Put
                # it at a row maximum so the declared compute cast changes
                # both the exact scale and (where applicable) fp8 bytes.
                converted.reshape(-1)[0] = 1.0009765625
            tensors[name] = converted
    for path in files:
        path.unlink()
    save_file(tensors, str(component / "diffusion_pytorch_model.safetensors"))
    index = component / "diffusion_pytorch_model.safetensors.index.json"
    if index.exists():
        index.unlink()


def test_byte_gate_uses_declared_bf16_quantization_input(
    tiny_dit: Path, tmp_path: Path,
) -> None:
    """A prod#fp16 snapshot is quantized from its production BF16 compute
    view; exact verification must reproduce that cast, not raw FP16 bytes."""
    import shutil

    fp16_source = tmp_path / "source-fp16"
    shutil.copytree(tiny_dit, fp16_source)
    _rewrite_component_float_dtype(
        fp16_source, torch.float16, inject_fp16_detail=True,
    )

    bf16_input = tmp_path / "quant-input-bf16"
    shutil.copytree(fp16_source, bf16_input)
    _rewrite_component_float_dtype(bf16_input, torch.bfloat16)
    candidate = tmp_path / "candidate"
    streaming_w8a8_snapshot(bf16_input, candidate, file_layout="diffusers")

    with pytest.raises(ConversionImplementationError, match="byte-gate"):
        verify_w8a8_snapshot(fp16_source, candidate, sample=4)

    report = verify_w8a8_snapshot(
        fp16_source, candidate, sample=4, source_compute_dtype="bf16",
    )
    assert report["byte_exact"] is True
    assert report["source_storage_dtypes"] == ["float16"]
    assert report["source_compute_dtype"] == "bfloat16"

    with pytest.raises(ConversionImplementationError, match="source_compute_dtype"):
        verify_w8a8_snapshot(
            fp16_source, candidate, source_compute_dtype="float8-e4m3",
        )


def test_dequant_round_trip_reproduces_source(tiny_dit: Path, produced: Path) -> None:
    art = detect_w8a8_artifact(produced)
    assert art is not None
    model = load_w8a8_denoiser(produced, art, mode="dequant",
                               compute_dtype=torch.bfloat16)
    src = _load_all(tiny_dit / "transformer")
    got = dict(model.state_dict())
    for name in art.quantized:
        a = src[f"{name}.weight"].float()
        b = got[f"{name}.weight"].float()
        rel = ((a - b).abs() / a.abs().clamp(min=1e-3)).max().item()
        assert rel < 0.13, name  # e4m3 rounding
        assert got[f"{name}.weight"].dtype == torch.bfloat16


def test_streaming_shards_keep_scale_with_weight(tiny_dit: Path, tmp_path: Path) -> None:
    """Force multi-shard output: every weight_scale must land (possibly in a
    later shard) and the index must resolve."""
    entry = sorted((tiny_dit / "transformer").glob("*.safetensors"))[0]
    out = tmp_path / "sharded"
    result = streaming_w8a8_cast(entry, out, shard_threshold=4096)
    assert len(result["output_paths"]) > 1
    assert result["index_path"] is not None
    tensors = _load_all(out)
    scales = {k for k in tensors if k.endswith(".weight_scale")}
    assert len(scales) == result["converted_count"] > 0
    for s in scales:
        w = tensors[s[: -len(".weight_scale")] + ".weight"]
        assert w.dtype == torch.float8_e4m3fn
        assert tensors[s].shape == (w.shape[0],)


# ---------------------------------------------------------------------------
# ie#494(c): LoRA composability — a wrapped Linear carries the additive bf16
# low-rank branch without breaking the scaled path (gw#547 consumes this).
# CPU-safe: torch._scaled_mm is patched with its dequant reference.
# ---------------------------------------------------------------------------


def test_wrapped_linear_carries_additive_branch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _ref_scaled_mm(a, b, *, scale_a, scale_b, bias=None, out_dtype=None):
        y = (a.float() * scale_a) @ (b.float() * scale_b)
        if bias is not None:
            y = y + bias.float()
        return y.to(out_dtype or torch.float32)

    monkeypatch.setattr(torch, "_scaled_mm", _ref_scaled_mm)
    torch.manual_seed(7)
    K, N, rank = 32, 48, 4
    lin_cls = fp8_scaled_linear_class()
    mod = lin_cls(K, N, bias=True, compute_dtype=torch.bfloat16,
                  static_input_scale=False)
    w = torch.randn(N, K)
    scale = (w.abs().amax(dim=1, keepdim=True) / 448.0).clamp(min=1e-12)
    mod.load_state_dict({
        "weight": (w / scale).clamp(-448, 448).to(torch.float8_e4m3fn),
        "weight_scale": scale,
        "bias": torch.randn(N, dtype=torch.bfloat16),
    }, assign=True)

    x = torch.randn(3, K, dtype=torch.bfloat16)
    base = mod(x)

    a = torch.randn(rank, K, dtype=torch.bfloat16)
    b = torch.randn(N, rank, dtype=torch.bfloat16)
    mod.lora_a, mod.lora_b = a, b
    with_branch = mod(x)
    addend = (x.reshape(-1, K) @ a.t()) @ b.t()
    assert torch.allclose(with_branch, base + addend, atol=2e-2, rtol=2e-2)

    # branch removal restores the branchless scaled path bit-exactly
    mod.lora_a = mod.lora_b = None
    assert torch.equal(mod(x), base)


# ---------------------------------------------------------------------------
# gw#557 TE wiring: storage_dtype="fp8+te" on the w8a8 lane casts ONLY the
# text encoders; the scaled-mm denoiser is never touched by cast hooks.
# ---------------------------------------------------------------------------


def test_apply_fp8_storage_component_override_scopes_to_te() -> None:
    transformers = pytest.importorskip("transformers")

    from gen_worker.models.loading import apply_fp8_storage

    cfg = transformers.T5Config(
        d_model=32, d_ff=64, d_kv=8, num_heads=4,
        num_layers=1, vocab_size=64, decoder_start_token_id=0,
    )
    te = transformers.T5EncoderModel(cfg).to(torch.bfloat16)
    denoiser = torch.nn.Linear(8, 8)
    pipe = SimpleNamespace(text_encoder=te, transformer=denoiser)

    assert apply_fp8_storage(pipe, compute_dtype=torch.bfloat16,
                             components=("text_encoder",)) is True
    assert te._cozy_fp8_storage_applied is True
    assert any(p.dtype == torch.float8_e4m3fn for p in te.parameters())
    assert not getattr(denoiser, "_cozy_fp8_storage_applied", False)
    assert all(p.dtype != torch.float8_e4m3fn for p in denoiser.parameters())


def test_load_from_pretrained_threads_te_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gen_worker.models import loading

    from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel

    root = tmp_path / "src"
    unet = UNet2DModel(
        sample_size=8, in_channels=3, out_channels=3,
        block_out_channels=(32, 32), layers_per_block=1,
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "UpBlock2D"),
        norm_num_groups=8,
    )
    DDPMPipeline(unet=unet, scheduler=DDPMScheduler()).save_pretrained(str(root))
    tree = w8a8.quantize_tree_w8a8(root, tmp_path / "w8a8")

    seen: dict = {}
    real = w8a8.load_w8a8_pipeline

    def spy(cls, path, art, **kwargs):
        seen.update(kwargs)
        return real(cls, path, art, **kwargs)

    monkeypatch.setattr(w8a8, "w8a8_gemm_mode", lambda: "")
    monkeypatch.setattr(w8a8, "load_w8a8_pipeline", spy)
    pipe = loading.load_from_pretrained(DDPMPipeline, tree, storage_dtype="fp8+te")
    assert seen.get("fp8_text_encoders") is True
    # no TEs on this pipeline: loud warning path, lane still stamped
    assert pipe._cozy_weight_lane == "bf16-resident"

    seen.clear()
    loading.load_from_pretrained(DDPMPipeline, tree)
    assert seen.get("fp8_text_encoders") is False
