"""pgw#1638: the streaming loader runs the config's QUANTIZER, then asserts.

The incident: `minimax-h3` 0.12.8 (release `03be533251f8b9b093adf845`, pick
`h200-sxm:quant=hf.fp8-blockwise@1`, 2026-08-21) died on every invoke —
three attempts, `compute 1.703s` each, deterministic — with

    NameMismatch: text_encoder/model.safetensors: 357 tensor(s) name nothing
    in component 'text_encoder' — model.language_model.layers.0.mlp
    .down_proj.weight_scale_inv, …

against a checkpoint that was CORRECT: `text_encoder=hf.fp8-blockwise@1`,
`weight_block_size [128,128]`, `modules_to_not_convert ["model.visual",
"lm_head"]`, `num_hidden_layers 51`. 51 x 7 quantized linears = 357.

`skeleton._build_on_meta` built the component with the bare `cls(config)`.
`HfQuantizer.preprocess_model` — the step that swaps `nn.Linear -> FP8Linear`
and registers `weight_scale_inv` — lives only inside `from_pretrained`, so the
skeleton held plain linears and every scale tensor in the container named
nothing. Same family as pgw#1626 (`tie_weights()`, also skipped, also paid for
on a pod), one member later.

No mocks: a real bf16 pipeline whose `text_encoder` is REALLY quantized to
`hf.fp8-blockwise@1` on disk, ingested into a real chunked `LocalCAS`,
projected to a stub-only tree, and streamed by the real engine on CPU — the
same article shape as `test_tied_weights_streaming_pgw1626.py`.

The second half of the fix is here too, and it is not the swap. H3's declared
lane is the DiT's, which is **bf16**, while its conditioner is fp8: the
pgw#1623 lane cast sees an F32 `[out/128, in/128]` scale grid as an off-lane
wide float and would round all 357 of them to bf16 — after which
`_assert_lane_dtype` passes and the pipeline serves plausible, wrong numbers.
`test_the_lane_cast_would_have_rounded_the_scale_grid` is that arm, red.
"""

from __future__ import annotations

import json
import math
import shutil
import struct
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")
pytest.importorskip("transformers")
pytest.importorskip("safetensors")

from safetensors.torch import load_file, save_file  # noqa: E402

from cas_fixture import ingest_repository  # noqa: E402
from gen_worker._vendor.tensorfs import LocalCAS, project_snapshot  # noqa: E402
from gen_worker.models.hf_fp8_blockwise import (  # noqa: E402
    QUANT_RULE,
    SCALE_LEAF,
    dequantize_block_scaled,
)
from gen_worker.models.projection import REF_PREFIX, SNAPSHOTS_DIR  # noqa: E402
from gen_worker.serving.streaming import NameMismatch, engine_for  # noqa: E402
from gen_worker.serving.streaming import engine as engine_mod  # noqa: E402
from gen_worker.serving.streaming import census as census_mod  # noqa: E402
from gen_worker.serving.streaming import skeleton as skeleton_mod  # noqa: E402
from streaming_fixture import Lane, scramble_offsets  # noqa: E402

BLOCK = (128, 128)
FP8_MAX = 448.0
LAYERS = 2
#: q,k,v,o + gate,up,down — the seven per layer that made H3's 51 x 7 = 357.
PROJECTIONS = 7
SCALES = LAYERS * PROJECTIONS

ALIAS = "lm_head.weight"
SOURCE = "model.embed_tokens.weight"


# -- the article ------------------------------------------------------------


def _quantize_block(w: Any) -> Tuple[Any, Any]:
    """Producer side, test-only: what a conversion endpoint emits.

    Lifted from `test_hf_fp8_blockwise_th1803.py`, which measured it against
    the contract's reference dequant bit-for-bit.
    """
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


def _quantized_leaves() -> Tuple[str, ...]:
    return tuple(
        f"model.layers.{layer}.{stem}.weight"
        for layer in range(LAYERS)
        for stem in (
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
            "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        )
    )


def _pipeline_class() -> type:
    from diffusers import DiffusionPipeline

    class TinyQuantizedPipeline(DiffusionPipeline):
        """A real pipeline whose `text_encoder` is a QUANTIZED tree."""

        def __init__(self, unet: Any, vae: Any, text_encoder: Any,
                     scheduler: Any) -> None:
            super().__init__()
            self.register_modules(  # type: ignore[attr-defined]
                unet=unet, vae=vae, text_encoder=text_encoder,
                scheduler=scheduler,
            )

    return TinyQuantizedPipeline


def _build_source(target: Path) -> type:
    """A real bf16 pipeline whose `text_encoder` is really fp8-blockwise.

    The model is saved by transformers first and the conditioner's container
    is then REWRITTEN as the conversion endpoint would emit it — fp8 weights
    with an F32 block-scale twin beside each — rather than asking a quantizer
    to produce it, because what is under test is reading such a tree.
    """
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(1638)
    unet = UNet2DConditionModel(
        sample_size=16, in_channels=4, out_channels=4, layers_per_block=1,
        block_out_channels=(32, 64), norm_num_groups=4, cross_attention_dim=256,
        attention_head_dim=4,
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
    )
    vae = AutoencoderKL(
        in_channels=3, out_channels=3, latent_channels=4, norm_num_groups=4,
        block_out_channels=(32,), down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
    )
    # `tie_word_embeddings=True` on purpose: H3's conditioner ties too, so the
    # pgw#1626 retie and the pgw#1638 swap have to hold on ONE module at once.
    text_encoder = LlamaForCausalLM(LlamaConfig(
        vocab_size=256, hidden_size=256, intermediate_size=512,
        num_hidden_layers=LAYERS, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, tie_word_embeddings=True,
    ))
    pipeline_cls = _pipeline_class()
    pipeline = pipeline_cls(
        unet=unet, vae=vae, text_encoder=text_encoder, scheduler=DDIMScheduler())
    pipeline.to(torch.bfloat16)
    pipeline.save_pretrained(str(target), safe_serialization=True)

    _requantize(target / "text_encoder")
    for container in sorted(target.rglob("*.safetensors")):
        scramble_offsets(container)
    return pipeline_cls


def _requantize(component: Path) -> None:
    """Rewrite the conditioner as `hf.fp8-blockwise@1` and say so in its
    config — the two facts a real artifact carries together."""
    container = component / "model.safetensors"
    tensors = load_file(str(container))
    for leaf in _quantized_leaves():
        weight, scale = _quantize_block(tensors[leaf])
        tensors[leaf] = weight
        tensors[leaf.removesuffix(".weight") + f".{SCALE_LEAF}"] = scale
    save_file(tensors, str(container))

    config_path = component / "config.json"
    config = json.loads(config_path.read_text())
    config["quantization_config"] = {
        "quant_method": "fp8",
        "fmt": "e4m3",
        "activation_scheme": "dynamic",
        "weight_block_size": list(BLOCK),
        "modules_to_not_convert": ["lm_head"],
    }
    config_path.write_text(json.dumps(config, indent=2))


def _project(base: Path, source: Path, key: str) -> Path:
    cas = LocalCAS(base)
    manifest = ingest_repository(cas, source)
    cas.compare_and_swap_ref(
        REF_PREFIX + key, cas.store_manifest(manifest), expected=None)
    tree = base / SNAPSHOTS_DIR / key
    project_snapshot(cas, manifest, tree)
    return tree


@pytest.fixture(scope="module")
def article(tmp_path_factory: pytest.TempPathFactory) -> Dict[str, Any]:
    base = tmp_path_factory.mktemp("pgw1638")
    source = base / "source-model"
    pipeline_cls = _build_source(source)
    tree = _project(base, source, key="a" * 64)
    return {"base": base, "source": source, "tree": tree,
            "pipeline_cls": pipeline_cls}


def _engine(tree: Path) -> Any:
    """The real streaming engine for a projected tree, on CPU.

    `engine_for` answers None for a tree with no chunk store behind it, and
    that answer would silently turn every assertion below into a test of the
    eager bridge — so it is a refusal here, not an `if`.
    """
    engine = engine_for(tree, device="cpu")
    assert engine is not None, f"{tree} bound no streaming engine"
    return engine


def _container(root: Path) -> Path:
    found = sorted((root / "text_encoder").glob("*.safetensors"))
    assert len(found) == 1, found
    return found[0]


def _header(path: Path) -> Dict[str, Any]:
    raw = path.read_bytes()
    (size,) = struct.unpack("<Q", raw[:8])
    return json.loads(raw[8: 8 + size])


# -- the fixture's own claim ------------------------------------------------


def test_the_article_really_is_a_blockwise_fp8_checkpoint(
    article: Dict[str, Any]
) -> None:
    """The guard on the guard. A container without real fp8 weights and real
    scale twins would make every assertion below pass on a lie — and a
    conditioner that carried `lm_head.weight` would erase the tie half."""
    header = _header(_container(article["source"]))
    scales = sorted(k for k in header if k.endswith(f".{SCALE_LEAF}"))
    assert len(scales) == SCALES, scales
    for name in scales:
        assert header[name]["dtype"] == "F32", header[name]
        assert header[name.removesuffix(f".{SCALE_LEAF}") + ".weight"]["dtype"] \
            == "F8_E4M3"
    assert SOURCE in header
    assert ALIAS not in header, (
        f"the fixture carries {ALIAS}; a tied checkpoint never does, so this "
        f"article cannot witness the swap and the retie at once"
    )

    from gen_worker.models.hf_fp8_blockwise import detect_hf_fp8_blockwise

    tree = detect_hf_fp8_blockwise(article["source"] / "text_encoder")
    assert tree is not None, "the article does not verify as the contract"
    assert tree.block_size == BLOCK
    assert len(tree.units) == SCALES


# -- 1. the fix -------------------------------------------------------------


def test_a_blockwise_fp8_checkpoint_loads_clean(article: Dict[str, Any]) -> None:
    """The incident's exact shape, green: the swap ran, the scale tensors have
    homes, nothing is left on meta, and the tie is still a tie."""
    engine = _engine(article["tree"])
    pipeline = engine.build(
        article["pipeline_cls"], checkpoint_dir=article["tree"], lane=Lane())

    text_encoder = pipeline.text_encoder
    assert census_mod.on_meta(text_encoder) == ()
    for component in ("unet", "vae"):
        assert census_mod.on_meta(getattr(pipeline, component)) == ()

    q_proj = text_encoder.model.layers[0].self_attn.q_proj
    assert type(q_proj).__name__ == "FP8Linear", type(q_proj).__name__
    assert q_proj.weight.dtype is torch.float8_e4m3fn
    assert getattr(q_proj, SCALE_LEAF).shape == (2, 2)
    # `lm_head` is in `modules_to_not_convert` and must NOT have been swapped.
    assert type(text_encoder.lm_head) is torch.nn.Linear
    assert text_encoder.lm_head.weight is text_encoder.model.embed_tokens.weight, (
        "the tie did not survive the quantizer swap (pgw#1626 on top of "
        "pgw#1638) — the alias holds its own tensor"
    )
    assert text_encoder.config.quantization_config is not None, (
        "postprocess_model never ran; the loaded model does not describe "
        "itself as quantized"
    )
    # The family's third member (pgw#1638's audit): `from_pretrained` ends
    # with `model.eval()` and this loader never did, so every component came
    # back with dropout armed. Asserted on the SERVED pipeline, including the
    # modules the quantizer built.
    for component in ("unet", "vae", "text_encoder"):
        module = getattr(pipeline, component)
        assert not module.training, f"{component} was served in train mode"
        assert all(not sub.training for sub in module.modules())


def test_the_scale_grid_keeps_the_rule_s_dtype_not_the_lane_s(
    article: Dict[str, Any]
) -> None:
    """The lane is bf16 and the block scales stay F32, byte-for-byte.

    This is the half a bare swap would have got wrong: an `hf.fp8-blockwise@1`
    scale grid is a wide float that is not the lane's, which is precisely what
    the pgw#1623 cast exists to "repair". Casting it is a numerics change with
    no exception and no red test — so the bytes are compared to the source.
    """
    engine = _engine(article["tree"])
    pipeline = engine.build(
        article["pipeline_cls"], checkpoint_dir=article["tree"], lane=Lane())
    stored = load_file(str(_container(article["source"])))

    checked = 0
    for name, want in stored.items():
        got = dict(pipeline.text_encoder.named_parameters(
            remove_duplicate=False))[name]
        assert got.dtype == want.dtype, f"{name}: {want.dtype} came back {got.dtype}"
        assert torch.equal(
            got.reshape(-1).view(torch.uint8), want.reshape(-1).view(torch.uint8)
        ), f"{name} is not byte-equal to the source"
        checked += 1
    assert checked == len(stored)
    assert engine.last_report.cast_to_lane == 0, (
        f"{engine.last_report.cast_to_lane} tensor(s) were cast to the lane; "
        f"this tree is the contract's own bytes and none of them are off-lane"
    )
    assert "F8_E4M3" in engine.last_report.dtypes, engine.last_report.dtypes

    # …and the grid still dequantizes to the weights it was made from.
    q_proj = pipeline.text_encoder.model.layers[0].self_attn.q_proj
    decoded = dequantize_block_scaled(
        q_proj.weight, getattr(q_proj, SCALE_LEAF), out_dtype=torch.float32)
    assert torch.isfinite(decoded).all()
    assert decoded.abs().max() > 0


# -- 2. the red arms --------------------------------------------------------


def test_without_the_quantizer_step_the_incident_reproduces(
    article: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Delete the fix and test 1 dies exactly as the H200 did: N orphans, all
    of them `weight_scale_inv`, blaming the checkpoint for a loader defect."""
    monkeypatch.setattr(
        skeleton_mod, "_prepare_quantized", lambda *a, **k: None)

    engine = _engine(article["tree"])
    with pytest.raises(NameMismatch) as caught:
        engine.build(
            article["pipeline_cls"], checkpoint_dir=article["tree"], lane=Lane())
    message = str(caught.value)
    assert f"{SCALES} tensor(s) name nothing" in message, message
    assert SCALE_LEAF in message, message
    assert "text_encoder" in message, message


def test_the_lane_cast_would_have_rounded_the_scale_grid(
    article: Dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Neuter the rule-owned exemption and the F32 block scales come back
    bf16 — silently, with the load green and `_assert_lane_dtype` satisfied.

    That is the failure mode this arm exists to keep visible: the swap alone
    turns 357 orphaned tensors into 357 quietly rounded ones.
    """
    monkeypatch.setattr(engine_mod, "_lane_exempt", lambda *a, **k: frozenset())

    engine = _engine(article["tree"])
    pipeline = engine.build(
        article["pipeline_cls"], checkpoint_dir=article["tree"], lane=Lane())
    scale = getattr(pipeline.text_encoder.model.layers[0].self_attn.q_proj,
                    SCALE_LEAF)
    assert scale.dtype is torch.bfloat16, (
        "the exemption is no longer what keeps the scale grid at its rule's "
        "dtype, so this arm proves nothing"
    )
    assert engine.last_report.cast_to_lane == SCALES


def test_an_absent_scale_tensor_is_still_refused(
    article: Dict[str, Any], tmp_path: Path
) -> None:
    """The survivor check keeps FULL teeth after the swap: dropping one
    `weight_scale_inv` from the container must still refuse, by name. The
    broad fix — exempting the quantizer's parameters from the meta scan —
    would have served an uninitialised scale grid here."""
    source = tmp_path / "source-model"
    shutil.copytree(article["source"], source)
    victim = f"model.layers.0.self_attn.q_proj.{SCALE_LEAF}"
    _drop_named_tensor(_container(source), victim)
    tree = _project(tmp_path, source, key="b" * 64)

    engine = _engine(tree)
    with pytest.raises(census_mod.CensusMismatch) as caught:
        engine.build(article["pipeline_cls"], checkpoint_dir=tree, lane=Lane())
    assert caught.value.invariant == census_mod.I4_PLACEMENT
    assert caught.value.tensor == victim
    message = str(caught.value)
    assert "STILL ON META" in message, message
    assert victim in message, message


def test_a_tree_whose_rule_this_image_cannot_decode_is_refused(
    article: Dict[str, Any], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The contract cross-check: the swap is only as honest as the image's
    DECLARATION behind it. pgw#1253's class was an arm the decode-set declared
    that nothing reached; the inverse — swapping modules for a rule this image
    never declared it can decode — must refuse before a byte moves."""
    from gen_worker.discovery import decode_set as decode_set_mod

    empty = decode_set_mod.DecodeSet(
        derivation="pgw#1638 test: an image declaring no decoder",
        entries=(), unregistered=(), excluded_modules=())
    monkeypatch.setattr(decode_set_mod, "runtime_decode_set", lambda: empty)

    engine = _engine(article["tree"])
    with pytest.raises(decode_set_mod.RuleNotDecodableError) as caught:
        engine.build(
            article["pipeline_cls"], checkpoint_dir=article["tree"], lane=Lane())
    assert QUANT_RULE in str(caught.value)


# -- fixture surgery --------------------------------------------------------


def _drop_named_tensor(path: Path, victim: str) -> None:
    """Rewrite the container WITHOUT ``victim``, reindexing every survivor."""
    raw = path.read_bytes()
    (size,) = struct.unpack("<Q", raw[:8])
    header = json.loads(raw[8: 8 + size])
    body = raw[8 + size:]
    assert victim in header, sorted(header)
    header.pop(victim)

    rebuilt = bytearray()
    for name in sorted(
        (key for key in header if key != "__metadata__"),
        key=lambda key: header[key]["data_offsets"][0],
    ):
        start, end = header[name]["data_offsets"]
        header[name]["data_offsets"] = [len(rebuilt), len(rebuilt) + (end - start)]
        rebuilt += body[start:end]
    blob = json.dumps(header, separators=(",", ":")).encode()
    path.write_bytes(struct.pack("<Q", len(blob)) + blob + bytes(rebuilt))
