"""Transformers-aware fp8 storage for text encoders (gw#460).

Root cause of the ie#371 fp8-TE break: a naive cast leaves fp8 weights where
un-hooked transformers ops read them raw — Gemma3's embed-scale multiply
(``super().forward(input_ids) * self.embed_scale``) is the first to explode
(``mul_cuda not implemented for Float8_e4m3fn``). The fix is weight-only fp8:
Linear/conv weights fp8-stored with per-module upcast at forward time;
embeddings, norms, and anything weight-tied to them stay at compute dtype.

CPU-safe: correctness of the dequant path needs no GPU. Real tiny transformers
models (Gemma3/T5/CLIP — the TE family) + real diffusers hooks."""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
transformers = pytest.importorskip("transformers")

from gen_worker.models.loading import apply_fp8_storage  # noqa: E402

FP8 = torch.float8_e4m3fn


def _tiny_gemma3() -> Any:
    from transformers import Gemma3Config, Gemma3ForConditionalGeneration

    cfg = Gemma3Config(
        text_config=dict(
            vocab_size=256, hidden_size=64, intermediate_size=128,
            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
            head_dim=16, max_position_embeddings=128,
        ),
        vision_config=dict(
            hidden_size=32, intermediate_size=64, num_hidden_layers=1,
            num_attention_heads=2, image_size=28, patch_size=14,
        ),
        mm_tokens_per_image=4, image_token_index=255,
        boi_token_index=253, eoi_token_index=254,
    )
    return Gemma3ForConditionalGeneration(cfg).to(torch.bfloat16).eval()


def _tiny_t5() -> Any:
    from transformers import T5Config, T5EncoderModel

    cfg = T5Config(
        vocab_size=256, d_model=64, d_kv=16, d_ff=128,
        num_layers=2, num_heads=4,
    )
    return T5EncoderModel(cfg).to(torch.bfloat16).eval()


def _tiny_clip() -> Any:
    from transformers import CLIPTextConfig, CLIPTextModel

    cfg = CLIPTextConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4,
        max_position_embeddings=64,
    )
    return CLIPTextModel(cfg).to(torch.bfloat16).eval()


def _forward(model: Any, ids: Any) -> Any:
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                    output_hidden_states=True)
    return torch.stack(out.hidden_states, dim=-1).float()


def _cosine(a: Any, b: Any) -> float:
    return torch.nn.functional.cosine_similarity(
        a.flatten(1), b.flatten(1), dim=-1
    ).min().item()


@pytest.mark.parametrize("factory", [_tiny_gemma3, _tiny_t5, _tiny_clip],
                         ids=["gemma3", "t5", "clip"])
def test_fp8_storage_forward_parity(factory) -> None:
    """fp8-storage forward stays close to bf16 — the dequant path works for
    the whole TE family (Gemma3 broke pre-fix; fp8 mul is unimplemented on
    CPU too, so any raw-fp8 leak fails loudly here)."""
    torch.manual_seed(0)
    model = factory()
    ids = torch.randint(0, 250, (1, 16))
    ref = _forward(model, ids)

    assert apply_fp8_storage(model, compute_dtype=torch.bfloat16) is True
    fp8_params = [n for n, p in model.named_parameters() if p.dtype == FP8]
    assert fp8_params, "no weights were actually cast to fp8"

    out = _forward(model, ids)
    assert _cosine(out, ref) > 0.98
    # weights are re-stored fp8 after forward (window closed), and the
    # model-level dtype read (transformers get_parameter_dtype — what
    # pipelines consult for activation dtypes) still reports compute dtype.
    assert any(p.dtype == FP8 for p in model.parameters())
    assert model.dtype == torch.bfloat16


def test_t5_wo_dtype_read_is_defended() -> None:
    """T5DenseActDense casts ACTIVATIONS to ``self.wo.weight.dtype`` before
    calling wo — outside wo's own forward. Per-leaf hooks cannot defend this
    (fp8 activations leak); the block window upcasts wo for the whole block
    forward. wo must be fp8 at rest AND the forward must succeed."""
    torch.manual_seed(0)
    model = _tiny_t5()
    apply_fp8_storage(model, compute_dtype=torch.bfloat16)
    wo = model.encoder.block[0].layer[1].DenseReluDense.wo
    assert wo.weight.dtype == FP8
    ids = torch.randint(0, 250, (1, 16))
    out = _forward(model, ids)
    assert out.isfinite().all()
    assert wo.weight.dtype == FP8  # window recloses after forward


def test_gemma3_weight_only_policy() -> None:
    """Embeddings, norms, and the tied lm_head stay at compute dtype; linear
    weights go fp8. lm_head shares its weight with the (skipped) token
    embedding — casting it through the hooked side would feed raw fp8 into
    the embedding forward (the gw#460 failure class)."""
    model = _tiny_gemma3()
    apply_fp8_storage(model, compute_dtype=torch.bfloat16)

    embed = model.get_input_embeddings()
    assert embed.weight.dtype == torch.bfloat16
    assert model.lm_head.weight.dtype == torch.bfloat16
    assert embed.weight is model.lm_head.weight  # tie preserved
    for name, p in model.named_parameters():
        if "norm" in name:
            assert p.dtype == torch.bfloat16, name
    lm = model.model.language_model
    assert lm.layers[0].self_attn.q_proj.weight.dtype == FP8
    assert lm.layers[0].mlp.gate_proj.weight.dtype == FP8


def test_pipeline_te_components_opt_in() -> None:
    """storage_dtype="fp8" touches only the denoiser; text_encoders=True (the
    "fp8+te" rung) extends to text encoder components."""

    class _Denoiser:
        def __init__(self) -> None:
            self.calls: list = []

        def parameters(self):
            return iter(())

        def enable_layerwise_casting(self, **kw) -> None:
            self.calls.append(kw)

    class _Pipe:
        def __init__(self) -> None:
            self.transformer = _Denoiser()
            self.text_encoder = _tiny_t5()

    pipe = _Pipe()
    assert apply_fp8_storage(pipe, compute_dtype=torch.bfloat16) is True
    assert len(pipe.transformer.calls) == 1
    assert all(p.dtype == torch.bfloat16 for p in pipe.text_encoder.parameters())

    pipe = _Pipe()
    assert apply_fp8_storage(pipe, compute_dtype=torch.bfloat16,
                             text_encoders=True) is True
    assert len(pipe.transformer.calls) == 1
    assert any(p.dtype == FP8 for p in pipe.text_encoder.parameters())


def test_binding_accepts_fp8_te() -> None:
    from gen_worker.api.binding import HF, Hub

    assert Hub("o/r", storage_dtype="fp8+te").storage_dtype == "fp8+te"
    assert HF("o/r", storage_dtype="fp8+te").storage_dtype == "fp8+te"
    with pytest.raises(ValueError):
        Hub("o/r", storage_dtype="fp8+vae")


def test_load_from_pretrained_fp8_te(tmp_path, monkeypatch) -> None:
    """The executor path: storage_dtype="fp8+te" reaches the TE component
    (gw#534: pinned to the involuntary path — a roomy card upgrades to
    bf16-resident instead)."""
    import json as _json
    import struct as _struct

    monkeypatch.setattr(
        "gen_worker.models.loading.bf16_resident_fits", lambda *a, **k: False)

    from gen_worker.models.loading import load_from_pretrained

    header = _json.dumps(
        {"w": {"dtype": "BF16", "shape": [64], "data_offsets": [0, 64]}}
    ).encode()
    (tmp_path / "model_index.json").write_text('{"_class_name": "P"}')
    with open(tmp_path / "diffusion_pytorch_model.safetensors", "wb") as f:
        f.write(_struct.pack("<Q", len(header)))
        f.write(header)

    class _Denoiser:
        def __init__(self) -> None:
            self.calls: list = []

        def parameters(self):
            return iter(())

        def enable_layerwise_casting(self, **kw) -> None:
            self.calls.append(kw)

    class _P:
        def __init__(self) -> None:
            self.transformer = _Denoiser()
            self.text_encoder = _tiny_clip()

        @classmethod
        def from_pretrained(cls, path: str, **kwargs: Any) -> "_P":
            return cls()

    pipe = load_from_pretrained(_P, tmp_path, dtype="bf16", storage_dtype="fp8+te")
    assert len(pipe.transformer.calls) == 1
    assert any(p.dtype == FP8 for p in pipe.text_encoder.parameters())
