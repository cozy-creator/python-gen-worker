"""pgw#1253: the declared ``hf.fp8-blockwise@1`` decoder must RUN.

An EXECUTION assertion, not a registration one. Asserting that a decoder is
registered proves nothing about whether it runs, and that was literally the
defect: ``load_hf_fp8_blockwise`` carried the decode marker
(``@implements_quant_rule`` since pgw#1621, ``@implements_contract`` then), so
the image's derived ``[decode_set]`` told the hub it decodes
``hf.fp8-blockwise@1`` — and no production call site reached it. transformers
reads ``quantization_config`` out of ``config.json`` by itself, so the tree
loaded anyway, through the GENERIC arm that declares ``plain.bf16@1``, and
nothing failed.

Measured on the pre-fix tree by driving :func:`load_component` with a real
blockwise fixture: ``LlamaForCausalLM.from_pretrained`` ran,
``load_hf_fp8_blockwise`` did not, the conditioner materialized at **float32**
(the declaration's compute dtype is bf16), and a TRANSPOSED scale grid — the
grid this rule's verifier exists to refuse — loaded without complaint.

So these tests drive the production dispatch and name the function that ran.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from safetensors.torch import save_file  # noqa: E402
from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from gen_worker.models import loading  # noqa: E402
from gen_worker.models.hf_fp8_blockwise import (  # noqa: E402
    HfFp8BlockwiseLayoutError,
    detect_hf_fp8_blockwise,
    load_hf_fp8_blockwise,
)
from gen_worker.models.loading import load_component  # noqa: E402

BLOCK = (128, 128)
FP8_MAX = 448.0
COMPONENT = "text_encoder"


def _quantize_block(w: Any) -> tuple[Any, Any]:
    out_f, in_f = w.shape
    srows, scols = math.ceil(out_f / BLOCK[0]), math.ceil(in_f / BLOCK[1])
    scale = torch.zeros(srows, scols, dtype=torch.float32)
    q = torch.zeros(out_f, in_f, dtype=torch.float32)
    for i in range(srows):
        for j in range(scols):
            sl = (slice(i * BLOCK[0], (i + 1) * BLOCK[0]),
                  slice(j * BLOCK[1], (j + 1) * BLOCK[1]))
            blk = w[sl].float()
            s = blk.abs().max().clamp(min=1e-12) / FP8_MAX
            scale[i, j] = s
            q[sl] = (blk / s).clamp(-FP8_MAX, FP8_MAX)
    return q.to(torch.float8_e4m3fn), scale


def _tiny_llama(hidden: int = 128, inter: int = 256) -> tuple[Any, Any]:
    cfg = LlamaConfig(
        vocab_size=256, hidden_size=hidden, intermediate_size=inter,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, tie_word_embeddings=False)
    torch.manual_seed(1253)
    return cfg, LlamaForCausalLM(cfg).to(torch.bfloat16)


def _write_component(
    d: Path, *, kind: str = "blockwise", hidden: int = 128, inter: int = 256,
) -> None:
    """A real transformers component tree.

    ``blockwise`` is ``hf.fp8-blockwise@1``; ``transposed`` writes the same
    tree with the scale grid transposed (the silently-wrong-numbers case);
    ``rowwise`` writes ``cozy.fp8-rowwise@1``'s per-row leaf and declares no
    ``weight_block_size``; ``dense`` writes plain bf16.
    """
    cfg, model = _tiny_llama(hidden, inter)
    out: dict[str, Any] = {}
    for key, value in model.state_dict().items():
        module = key[: -len(".weight")] if key.endswith(".weight") else None
        eligible = (module is not None and value.ndim == 2
                    and "embed_tokens" not in key and "lm_head" not in key)
        if not eligible or kind == "dense":
            out[key] = value
            continue
        if kind == "rowwise":
            out[key] = value.to(torch.float8_e4m3fn)
            out[f"{module}.weight_scale"] = torch.ones(
                value.shape[0], dtype=torch.float32)
            continue
        q, s = _quantize_block(value)
        out[key] = q
        out[f"{module}.weight_scale_inv"] = s.T.contiguous() \
            if kind == "transposed" else s
    d.mkdir(parents=True, exist_ok=True)
    save_file(out, str(d / "model.safetensors"))

    conf = cfg.to_dict()
    if kind == "blockwise" or kind == "transposed":
        conf["quantization_config"] = {
            "quant_method": "fp8",
            "activation_scheme": "dynamic",
            "weight_block_size": list(BLOCK),
            "fmt": "e4m3",
            "modules_to_not_convert": ["lm_head"],
        }
    elif kind == "rowwise":
        conf["quantization_config"] = {
            "quant_method": "fp8", "activation_scheme": "dynamic"}
    (d / "config.json").write_text(json.dumps(conf), encoding="utf-8")


def _tree(tmp_path: Path, kind: str, **kw: Any) -> Path:
    root = tmp_path / kind
    _write_component(root / COMPONENT, kind=kind, **kw)
    (root / "model_index.json").write_text(json.dumps({
        "_class_name": "StableDiffusionPipeline",
        COMPONENT: ["transformers", "LlamaForCausalLM"],
    }), encoding="utf-8")
    return root


@pytest.fixture()
def ran(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Record WHICH decoder the production dispatch drove.

    Both spies delegate to the real function — this instruments a real
    end-to-end load, it does not stand in for one.
    """
    seen: list[str] = []
    explicit = load_hf_fp8_blockwise
    generic = LlamaForCausalLM.from_pretrained

    def spy_explicit(*a: Any, **kw: Any) -> Any:
        seen.append("gen_worker.models.hf_fp8_blockwise:load_hf_fp8_blockwise")
        return explicit(*a, **kw)

    def spy_generic(*a: Any, **kw: Any) -> Any:
        seen.append("transformers:LlamaForCausalLM.from_pretrained")
        return generic(*a, **kw)

    monkeypatch.setattr(loading, "load_hf_fp8_blockwise", spy_explicit)
    monkeypatch.setattr(
        LlamaForCausalLM, "from_pretrained", staticmethod(spy_generic))
    return seen


def test_production_dispatch_runs_the_declared_blockwise_decoder(
    tmp_path: Path, ran: list[str],
) -> None:
    """RED before pgw#1253: ``ran`` held only the generic from_pretrained."""
    obj = load_component(_tree(tmp_path, "blockwise"), COMPONENT)

    assert ran[0] == (
        "gen_worker.models.hf_fp8_blockwise:load_hf_fp8_blockwise"), (
        f"the declared hf.fp8-blockwise@1 decoder did not run; ran={ran}")
    # and it produced a materialized module, not a shell.
    weight = obj.model.layers[0].self_attn.q_proj.weight
    assert not weight.is_meta and torch.isfinite(weight.float()).all()


def test_transposed_scale_grid_refuses_through_the_production_dispatch(
    tmp_path: Path,
) -> None:
    """The refusal that only exists on the explicit arm. No spies: the
    dispatch either reaches the verifier or it does not.

    RED before pgw#1253: this tree loaded, and every block scale was applied
    to the wrong span of the wrong row.
    """
    root = _tree(tmp_path, "transposed", hidden=128, inter=384)
    with pytest.raises(HfFp8BlockwiseLayoutError) as excinfo:
        load_component(root, COMPONENT)
    assert "mis-blocked" in str(excinfo.value)


def test_rowwise_tree_is_not_claimed_by_the_blockwise_arm(
    tmp_path: Path, ran: list[str],
) -> None:
    """Detection is the config's own claim: no ``weight_block_size``, no
    claim. Over-claiming here would route ``cozy.fp8-rowwise@1`` bytes into a
    128x128 reader, which is the conflation the two handles exist to stop."""
    root = _tree(tmp_path, "rowwise")
    assert detect_hf_fp8_blockwise(root / COMPONENT) is None
    load_component(root, COMPONENT)
    assert ran == ["transformers:LlamaForCausalLM.from_pretrained"]


def test_dense_component_still_takes_the_generic_path(
    tmp_path: Path, ran: list[str],
) -> None:
    root = _tree(tmp_path, "dense")
    assert detect_hf_fp8_blockwise(root / COMPONENT) is None
    load_component(root, COMPONENT)
    assert ran == ["transformers:LlamaForCausalLM.from_pretrained"]
