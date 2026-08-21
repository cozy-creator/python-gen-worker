"""The consumer half of the tensor-layout gate.

Real tree, real transformers loader, real forward — no mocks. A tiny Llama is
quantized to ``hf.fp8-blockwise@1`` (128x128 block scales in
``weight_scale_inv``), written as safetensors with the same
``quantization_config`` shape `tensorhub/minimax-h3`'s fp8 conditioner
carries, then loaded back through ``load_hf_fp8_blockwise`` and DECODED.

What is proven:

1. the tree verifies from headers alone, with the block grid derived per unit;
2. the loaded weights equal the contract's reference dequant BIT-FOR-BIT, and
   land within fp8 error of the pre-quantization weights — which is what
   distinguishes the multiply convention from its reciprocal (a divide is off
   by ~1e5, not by 4%);
3. a forward pass produces finite logits (decode, at unit scale);
4. a ``cozy.fp8-rowwise@1`` tree is REFUSED by name, never adapted — the
   silent-wrong-numbers defect the rule exists to prevent;
5. a transposed scale grid is refused;
6. the image's derived declaration carries the QUANT RULE, so the hub's
   `Satisfies` comparison can succeed for the first time.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from safetensors.torch import save_file  # noqa: E402
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM  # noqa: E402

from gen_worker.models.hf_fp8_blockwise import (  # noqa: E402
    HfFp8BlockwiseLayoutError,
    dequantize_block_scaled,
    inspect_hf_fp8_blockwise,
    load_hf_fp8_blockwise,
)
from gen_worker.models.tensor_layout_contract import (  # noqa: E402
    quant_rule_decoders_of,
)

#: pgw#1621: the ratified v2 quant-rule handle, which is now the WHOLE
#: declaration — a rule's conventions (128x128 block scales in
#: `weight_scale_inv`) are its identity, so there is no `decodes=` axis beside
#: it any more. `CONTRACT_HF_FP8_BLOCKWISE` is deleted with the v1 vocabulary.
HF_FP8_BLOCKWISE = "hf.fp8-blockwise@1"

BLOCK = (128, 128)
FP8_MAX = 448.0


def _quantize_block(w, block=BLOCK):
    out_f, in_f = w.shape
    srows, scols = math.ceil(out_f / block[0]), math.ceil(in_f / block[1])
    scale = torch.zeros(srows, scols, dtype=torch.float32)
    q = torch.zeros(out_f, in_f, dtype=torch.float32)
    for i in range(srows):
        for j in range(scols):
            sl = (slice(i * block[0], (i + 1) * block[0]),
                  slice(j * block[1], (j + 1) * block[1]))
            blk = w[sl].float()
            s = (blk.abs().max().clamp(min=1e-12) / FP8_MAX)
            scale[i, j] = s
            q[sl] = (blk / s).clamp(-FP8_MAX, FP8_MAX)
    return q.to(torch.float8_e4m3fn), scale


def _tiny_llama():
    cfg = LlamaConfig(
        vocab_size=256, hidden_size=128, intermediate_size=256,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, tie_word_embeddings=False)
    torch.manual_seed(1803)
    return cfg, LlamaForCausalLM(cfg).to(torch.bfloat16)


def _write_tree(d: Path, *, rowwise: bool = False, transpose_scale: bool = False):
    cfg, model = _tiny_llama()
    sd = model.state_dict()
    out, quantized = {}, []
    for key, value in sd.items():
        module = key[: -len(".weight")] if key.endswith(".weight") else None
        eligible = (module is not None and value.ndim == 2
                    and "embed_tokens" not in key and "lm_head" not in key)
        if not eligible:
            out[key] = value
            continue
        if rowwise:
            amax = value.float().abs().amax(dim=1).clamp(min=1e-12)
            s = amax / FP8_MAX
            out[key] = (value.float() / s[:, None]).clamp(
                -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
            out[f"{module}.weight_scale"] = s
        else:
            q, s = _quantize_block(value)
            out[key] = q
            out[f"{module}.weight_scale_inv"] = s.T.contiguous() if transpose_scale else s
        quantized.append(module)

    save_file(out, str(d / "model.safetensors"), metadata={"format": "pt"})
    cfgd = json.loads(cfg.to_json_string())
    cfgd["quantization_config"] = {
        "quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
        "weight_block_size": [1, 1] if rowwise else list(BLOCK),
        "modules_to_not_convert": ["lm_head"],
    }
    (d / "config.json").write_text(json.dumps(cfgd), "utf-8")
    return sd, tuple(quantized)


def test_a_blockwise_tree_verifies_and_decodes(tmp_path: Path):
    original, quantized = _write_tree(tmp_path)

    tree = inspect_hf_fp8_blockwise(tmp_path)
    assert tree.block_size == BLOCK
    assert tree.activation_scheme == "dynamic"
    assert tree.modules_to_not_convert == ("lm_head",)
    assert sorted(u.module for u in tree.units) == sorted(quantized)
    down = next(u for u in tree.units if u.module.endswith("down_proj"))
    assert (down.out_features, down.in_features) == (128, 256)
    assert (down.scale_rows, down.scale_cols) == (1, 2)
    assert down.block == BLOCK

    model = load_hf_fp8_blockwise(
        tmp_path, cls=AutoModelForCausalLM, resident=False)

    from safetensors.torch import load_file
    raw = load_file(str(tmp_path / "model.safetensors"))
    for name in ("model.layers.0.self_attn.q_proj",
                 "model.layers.0.mlp.down_proj"):
        got = model.get_submodule(name).weight
        want = dequantize_block_scaled(
            raw[f"{name}.weight"], raw[f"{name}.weight_scale_inv"],
            out_dtype=torch.bfloat16)
        assert torch.equal(got.float(), want.float()), name
        ref = original[f"{name}.weight"].float()
        assert (got.float() - ref).abs().max() <= 0.08 * ref.abs().max()

    logits = model(torch.randint(0, 256, (1, 8))).logits
    assert torch.isfinite(logits).all()
    assert logits.shape == (1, 8, 256)


def test_a_rowwise_tree_is_refused_by_name(tmp_path: Path):
    _write_tree(tmp_path, rowwise=True)
    with pytest.raises(HfFp8BlockwiseLayoutError) as excinfo:
        inspect_hf_fp8_blockwise(tmp_path)
    message = str(excinfo.value)
    assert "cozy.fp8-rowwise@1" in message
    assert "weight_scale" in message
    assert "PRODUCIBLE" in message


def test_a_transposed_scale_grid_is_refused(tmp_path: Path):
    _write_tree(tmp_path, transpose_scale=True)
    with pytest.raises(HfFp8BlockwiseLayoutError) as excinfo:
        inspect_hf_fp8_blockwise(tmp_path)
    assert "scale grid" in str(excinfo.value)


def test_a_dense_tree_is_not_this_contract(tmp_path: Path):
    cfg, model = _tiny_llama()
    save_file(model.state_dict(), str(tmp_path / "model.safetensors"),
              metadata={"format": "pt"})
    (tmp_path / "config.json").write_text(cfg.to_json_string(), "utf-8")
    with pytest.raises(HfFp8BlockwiseLayoutError) as excinfo:
        inspect_hf_fp8_blockwise(tmp_path)
    assert "plain.bf16@1" in str(excinfo.value)


def test_the_image_declares_the_quant_rule():
    """The declaration is a property of the decoder, and the build derivation harvests it — this is what the hub's gate reads."""
    from gen_worker.discovery.execution_lanes import derive_execution_lanes

    decls = quant_rule_decoders_of(load_hf_fp8_blockwise)
    assert [d.rule for d in decls] == [HF_FP8_BLOCKWISE]
    assert decls[0].composes_lora is False

    derived = derive_execution_lanes()
    mine = [c for c in derived.contracts if c.rule == HF_FP8_BLOCKWISE]
    assert mine, [c.rule for c in derived.contracts]
    assert mine[0].decoder.endswith(":load_hf_fp8_blockwise")
    assert "fp8-w8a8-dynamic+compiled" in mine[0].execution_lanes
