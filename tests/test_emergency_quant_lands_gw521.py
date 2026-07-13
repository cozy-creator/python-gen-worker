"""gw#521 integration: the emergency nf4 rung ACTUALLY LANDS on real tiny
pipelines of both archetypes (unet: SDXL-shaped; transformer: Flux-shaped).

The 4070 dogfood proved the failure mode is silent: a quant config whose
component names miss the pipeline is ignored by diffusers, "EMERGENCY 4-bit
quantization engaged" logs anyway, and the worker falls to CPU offload. These
tests make that no-op impossible to reintroduce: they assert the quantized
module CLASS is present on the correct component (bnb Linear4bit) and that
the rung stamp is CLEARED when a bad config no-ops.

CUDA + bitsandbytes only (skipped in CPU CI; runs on dev GPUs + the nightly
GPU lane). Tiny models only — a few MB of VRAM.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
diffusers = pytest.importorskip("diffusers")

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(
        importlib.util.find_spec("bitsandbytes") is None,
        reason="needs bitsandbytes"),
]

from gen_worker.models import loading as loading_mod  # noqa: E402

_GiB = float(1 << 30)


def _tiny_sdxl(dtype: torch.dtype = torch.float32):
    from diffusers import (
        AutoencoderKL,
        EulerDiscreteScheduler,
        StableDiffusionXLPipeline,
        UNet2DConditionModel,
    )
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

    unet = UNet2DConditionModel(
        block_out_channels=(32, 64),
        layers_per_block=2,
        sample_size=32,
        in_channels=4,
        out_channels=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        attention_head_dim=(2, 4),
        use_linear_projection=True,
        addition_embed_type="text_time",
        addition_time_embed_dim=8,
        transformer_layers_per_block=(1, 2),
        projection_class_embeddings_input_dim=80,
        cross_attention_dim=64,
    )
    vae = AutoencoderKL(
        block_out_channels=[32, 64],
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D", "DownEncoderBlock2D"],
        up_block_types=["UpDecoderBlock2D", "UpDecoderBlock2D"],
        latent_channels=4,
        sample_size=64,
        force_upcast=True,  # the SDXL dtype-fragile VAE (gw#441/gw#463)
    )
    cfg = CLIPTextConfig(
        bos_token_id=0, eos_token_id=2, hidden_size=32, intermediate_size=37,
        layer_norm_eps=1e-05, num_attention_heads=4, num_hidden_layers=5,
        pad_token_id=1, vocab_size=1000, hidden_act="gelu", projection_dim=32,
    )
    pipe = StableDiffusionXLPipeline(
        unet=unet,
        vae=vae,
        text_encoder=CLIPTextModel(cfg),
        text_encoder_2=CLIPTextModelWithProjection(cfg),
        tokenizer=None,
        tokenizer_2=None,
        scheduler=EulerDiscreteScheduler(
            beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"),
    )
    return pipe.to(dtype)


def _tiny_flux(dtype: torch.dtype = torch.float32):
    from diffusers import (
        AutoencoderKL,
        FlowMatchEulerDiscreteScheduler,
        FluxPipeline,
        FluxTransformer2DModel,
    )
    from transformers import (
        CLIPTextConfig,
        CLIPTextModel,
        T5Config,
        T5EncoderModel,
    )

    transformer = FluxTransformer2DModel(
        patch_size=1,
        in_channels=4,
        num_layers=1,
        num_single_layers=1,
        attention_head_dim=16,
        num_attention_heads=2,
        joint_attention_dim=32,
        pooled_projection_dim=32,
        axes_dims_rope=[4, 4, 8],
    )
    vae = AutoencoderKL(
        block_out_channels=[32],
        in_channels=3,
        out_channels=3,
        down_block_types=["DownEncoderBlock2D"],
        up_block_types=["UpDecoderBlock2D"],
        latent_channels=1,
        sample_size=32,
    )
    clip_cfg = CLIPTextConfig(
        bos_token_id=0, eos_token_id=2, hidden_size=32, intermediate_size=37,
        layer_norm_eps=1e-05, num_attention_heads=4, num_hidden_layers=2,
        pad_token_id=1, vocab_size=1000, hidden_act="gelu", projection_dim=32,
    )
    t5_cfg = T5Config(
        vocab_size=100, d_model=32, d_ff=37, num_layers=2, num_heads=4, d_kv=8,
    )
    pipe = FluxPipeline(
        scheduler=FlowMatchEulerDiscreteScheduler(),
        vae=vae,
        text_encoder=CLIPTextModel(clip_cfg),
        tokenizer=_tiny_tokenizer(),
        text_encoder_2=T5EncoderModel(t5_cfg),
        tokenizer_2=_tiny_tokenizer(),
        transformer=transformer,
    )
    return pipe.to(dtype)


def _tiny_tokenizer():
    """Offline stand-in tokenizer (never invoked — loading-path tests only)."""
    from tokenizers import Tokenizer, models
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(models.WordLevel({"<pad>": 0, "a": 1}, unk_token="<pad>"))
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>")


def _force_nf4_budget(monkeypatch: pytest.MonkeyPatch, tree: Path, denoiser: str) -> None:
    """Free-VRAM probe tuned so the stored tree does NOT fit but its
    nf4(denoiser) estimate does — the rung must engage and must be enough."""
    comp = loading_mod.snapshot_component_weight_bytes(tree)
    total = float(sum(comp.values()))
    est = total - comp[denoiser] * (1.0 - loading_mod.NF4_WEIGHT_BYTES_FACTOR)
    budget = (est + total) / 2.0
    free_gb = budget / _GiB + loading_mod._EMERGENCY_MARGIN_GB
    from gen_worker.models import memory

    monkeypatch.setattr(memory, "get_available_vram_gb", lambda *a, **k: free_gb)


def _bnb_linear_count(mod) -> int:
    return sum(1 for m in mod.modules() if type(m).__name__ == "Linear4bit")


@pytest.mark.parametrize("archetype", ["unet", "transformer"])
def test_emergency_nf4_lands_on_real_pipelines(
    archetype: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    build, pipe_cls_name = {
        "unet": (_tiny_sdxl, "StableDiffusionXLPipeline"),
        "transformer": (_tiny_flux, "FluxPipeline"),
    }[archetype]
    pipe = build()
    pipe.save_pretrained(tmp_path, safe_serialization=True)
    del pipe

    _force_nf4_budget(monkeypatch, tmp_path, archetype)
    pipe_cls = getattr(diffusers, pipe_cls_name)
    loaded = loading_mod.load_from_pretrained(pipe_cls, tmp_path)

    # The rung engaged, the stamp is honest, and the quant LANDED: the
    # denoiser's Linear layers are bnb Linear4bit with uint8 storage — a
    # silent no-op (the gw#521 bug class) cannot pass this.
    assert getattr(loaded, "_cozy_adaptive_rung", "") == "nf4"
    denoiser = getattr(loaded, archetype)
    assert _bnb_linear_count(denoiser) > 0
    quantized = [
        p for m in denoiser.modules() if type(m).__name__ == "Linear4bit"
        for p in m.parameters(recurse=False)
    ]
    assert any(p.dtype == torch.uint8 for p in quantized)


def test_wrong_component_config_is_detected_not_silent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The exact 4070 failure shape: a config naming 'transformer' on a unet
    pipeline. diffusers ignores it — the loader must detect the no-op, log
    loudly, and CLEAR the rung stamp instead of lying."""
    pipe = _tiny_sdxl()
    pipe.save_pretrained(tmp_path, safe_serialization=True)
    del pipe

    _force_nf4_budget(monkeypatch, tmp_path, "unet")
    real = loading_mod.emergency_quantization_config

    def _inverted(cls, *, components=None, compute_dtype=None):
        return real(cls, components=["transformer"], compute_dtype=compute_dtype)

    monkeypatch.setattr(loading_mod, "emergency_quantization_config", _inverted)
    from diffusers import StableDiffusionXLPipeline

    with caplog.at_level("ERROR"):
        loaded = loading_mod.load_from_pretrained(StableDiffusionXLPipeline, tmp_path)
    assert getattr(loaded, "_cozy_adaptive_rung", "") == ""
    assert _bnb_linear_count(loaded.unet) == 0
    assert "did NOT land" in caplog.text


def test_fragile_vae_stays_resident_and_decode_is_dtype_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The other half of the 4070 crash: a group-offloaded force_upcast VAE
    decodes fp32 latents against hook-restored fp16 weights ("Input type
    (float) and bias type (c10::Half)"). The fragile VAE must stay resident
    under EVERY offload rung and the SDXL upcast+decode path must work."""
    monkeypatch.setenv("GEN_WORKER_FORBID_CPU_OFFLOAD", "0")
    from gen_worker.models.memory import apply_low_vram_config

    pipe = _tiny_sdxl(torch.float16)
    applied = apply_low_vram_config(pipe, mode="group_offload")
    assert applied["group_offload"] or applied["sequential_offload"]
    assert applied["vae_resident"]

    # The SDXL __call__ decode sequence under force_upcast.
    assert pipe.vae.config.force_upcast
    pipe.upcast_vae()
    latents = torch.randn(
        1, 4, 16, 16, device="cuda",
        dtype=next(iter(pipe.vae.post_quant_conv.parameters())).dtype,
    )
    image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
    assert image.shape[-1] == 32  # 16x16 latents, one 2x upsample
    assert torch.isfinite(image).all()


def test_quantized_pipeline_places_resident_not_offloaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """select_auto_mode (gw#521): a pipeline the nf4 rung just shrank to fit
    must place resident — the old absolute low-free-VRAM rule group-offloaded
    it (or hit the FORBID guard), making the salvation rung pointless."""
    pipe = _tiny_sdxl()
    pipe.save_pretrained(tmp_path, safe_serialization=True)
    del pipe

    _force_nf4_budget(monkeypatch, tmp_path, "unet")
    from diffusers import StableDiffusionXLPipeline

    loaded = loading_mod.load_from_pretrained(StableDiffusionXLPipeline, tmp_path)
    assert getattr(loaded, "_cozy_adaptive_rung", "") == "nf4"

    # 4.4GB free (the live 4070 number): the quantized tiny pipe FITS, so
    # placement must not pick a CPU-offload rung (FORBID=1 stays enforced —
    # resident placement never trips it).
    monkeypatch.setenv("GEN_WORKER_FORBID_CPU_OFFLOAD", "1")
    from gen_worker.models import memory

    monkeypatch.setattr(memory, "get_available_vram_gb", lambda *a, **k: 4.4)
    placed = memory.place_pipeline(loaded, ref="tiny/sdxl")
    assert placed["mode"] in ("off", "vae_only")
