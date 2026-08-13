"""te#148 — quantized low-rank branch decode (LoRaQ arXiv 2604.18117).

The rank-128 branch pair is bf16 today; te#148 lets the producer store it
int8 / fp8_e4m3 with per-block-32 scales along each factor's contraction dim
(LoRaQ quantizes by MX blocks of 32; at 8 bits MXFP8e4 edges MXINT8, Table 3).
Runtime v1 dequantizes ON LOAD — DecodedLinear stays bf16 logical, so
SvdqLinear / fold_to_dense / split_decoded are untouched downstream.

Accounting behind the "worth it?" call (qwen-image 20B, 60 blocks x 8 fused
W4A4 units, rank 128): branch params = 128 * sum(in+out) = 755M -> 1.51 GB
bf16, 11.5% of the 13.08 GB artifact. 8-bit + fp32 block-32 scales = 0.85 GB
-> saves ~0.66 GB of ARTIFACT/disk. v1 (dequant-on-load) saves ZERO VRAM;
even quantized-resident would save ~0.85 GB against a 65 GB bf16 pipeline
peak (~1.3%) — the honest VRAM story is "marginal", the artifact-size story
and the LoRaQ higher-rank-at-equal-memory lever are the real value.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from gen_worker.models import svdq_native as native  # noqa: E402
from gen_worker.models.nvfp4_quant import BLOCK, cast_e2m1  # noqa: E402
from gen_worker.models.svdq_layout import (  # noqa: E402
    LOWRANK_QUANT_BLOCK,
    LOWRANK_QUANT_KEY,
    LOWRANK_QUANT_SCHEMES,
    SvdqLayoutError,
    decode_linear,
    dequantize_decoded,
    dequantize_lowrank,
    pack_lowrank,
    pack_qweight,
    pack_vector,
    pack_wscales,
    quantize_lowrank,
    split_decoded,
)

E2M1_MAX, FP8_MAX = 6.0, 448.0

# (norm-relative) roundtrip bounds per scheme: int8 block-32 absmax on
# gaussian data lands ~0.007; e4m3's 3 mantissa bits land ~0.03-0.04.
_BOUNDS = {"int8": 0.02, "fp8_e4m3": 0.08}


def _factors(out_f: int, in_f: int, rank: int, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    down = (torch.randn(in_f, rank, generator=gen)
            * (in_f ** -0.5)).to(torch.bfloat16)     # logical [in, rank]
    up = (torch.randn(out_f, rank, generator=gen)
          * (rank ** -0.5)).to(torch.bfloat16)       # logical [out, rank]
    return down, up


def _synth_compiled_graph(out_f: int, in_f: int, *, second_key: str = "wtscale",
                 rank: int = 128, lowrank: str = "bf16", seed: int = 0,
                 smooth: bool = True) -> tuple[dict, Any, Any]:
    """A checkpoint compiled graph (nunchaku layout, branch per ``lowrank``) plus the
    logical bf16 branch factors it was built from."""
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(out_f, in_f, generator=gen)

    if second_key == "wcscales":
        second = (w.abs().amax(dim=1) / (E2M1_MAX * FP8_MAX)).clamp(min=1e-8)
        second_bcast = second.reshape(out_f, 1)
    else:
        second = (w.abs().amax() / (E2M1_MAX * FP8_MAX)).clamp(min=1e-8)
        second_bcast = second.reshape(1, 1)
    blocks = w.reshape(out_f, in_f // BLOCK, BLOCK)
    bs = (blocks.abs().amax(dim=-1) / (E2M1_MAX * second_bcast)).clamp(
        min=2.0 ** -9, max=FP8_MAX).to(torch.float8_e4m3fn)
    codes = cast_e2m1(
        (blocks / (bs.float().unsqueeze(-1) * second_bcast.unsqueeze(-1))
         ).reshape(out_f, in_f))

    tensors: dict[str, Any] = {
        "qweight": pack_qweight(codes),
        "wscales": pack_wscales(bs, out_f, in_f),
    }
    tensors[second_key] = (pack_vector(second.to(torch.bfloat16), out_f)
                           if second_key == "wcscales"
                           else second.to(torch.bfloat16))

    down, up = _factors(out_f, in_f, rank, seed=seed + 1)
    if lowrank == "bf16":
        tensors["proj_down"] = pack_lowrank(down.t().contiguous(), down=True)
        tensors["proj_up"] = pack_lowrank(up, down=False)
    else:
        qd, sd = quantize_lowrank(down, scheme=lowrank, block_dim=0)
        qu, su = quantize_lowrank(up, scheme=lowrank, block_dim=1)
        tensors["proj_down"], tensors["proj_down_scale"] = qd, sd
        tensors["proj_up"], tensors["proj_up_scale"] = qu, su

    if smooth:
        tensors["smooth_factor"] = pack_vector(
            (torch.rand(in_f, generator=gen) + 0.5).to(torch.bfloat16), in_f)
    tensors["bias"] = pack_vector(
        torch.randn(out_f, generator=gen).to(torch.bfloat16), out_f)
    return tensors, down, up


def _rel(a: Any, b: Any) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


# --- quantize -> dequantize roundtrip --------------------------------------


@pytest.mark.parametrize("scheme", LOWRANK_QUANT_SCHEMES)
@pytest.mark.parametrize("shape,block_dim", [((3072, 128), 0), ((9216, 128), 1)])
def test_roundtrip_error_is_bounded(scheme: str, shape: tuple, block_dim: int) -> None:
    """quantize -> dequantize on real factor shapes stays inside the scheme's
    error budget, and the scale tensor is exactly per-block-32 fp32."""
    w = (torch.randn(*shape, generator=torch.Generator().manual_seed(2))
         * (shape[0] ** -0.5)).to(torch.bfloat16)
    q, scale = quantize_lowrank(w, scheme=scheme, block_dim=block_dim)
    assert q.shape == w.shape
    assert q.dtype == (torch.int8 if scheme == "int8" else torch.float8_e4m3fn)
    nb = shape[block_dim] // LOWRANK_QUANT_BLOCK
    want_scale = ((nb, shape[1]) if block_dim == 0 else (shape[0], nb))
    assert tuple(scale.shape) == want_scale
    assert scale.dtype == torch.float32

    back = dequantize_lowrank(q, scale, block_dim=block_dim)
    assert back.dtype == torch.bfloat16
    rel = _rel(back, w)
    assert rel < _BOUNDS[scheme], (scheme, rel)


def test_quantize_and_dequantize_refuse_bad_inputs() -> None:
    w = torch.randn(64, 128).to(torch.bfloat16)
    with pytest.raises(SvdqLayoutError, match="unknown lowrank_quant scheme"):
        quantize_lowrank(w, scheme="int4", block_dim=0)
    with pytest.raises(SvdqLayoutError, match="block_dim"):
        quantize_lowrank(w, scheme="int8", block_dim=2)
    with pytest.raises(SvdqLayoutError, match="multiple of the quant block"):
        quantize_lowrank(torch.randn(50, 128), scheme="int8", block_dim=0)
    q, scale = quantize_lowrank(w, scheme="int8", block_dim=0)
    with pytest.raises(SvdqLayoutError, match="scale shape"):
        dequantize_lowrank(q, scale.t().contiguous(), block_dim=0)
    with pytest.raises(SvdqLayoutError, match="not a quantized"):
        dequantize_lowrank(w, scale, block_dim=0)


# --- decode equivalence vs the bf16 branch ---------------------------------


@pytest.mark.parametrize("scheme", LOWRANK_QUANT_SCHEMES)
def test_decode_quantized_branch_matches_bf16_branch(scheme: str) -> None:
    """Same residual, same factors, two on-disk branch schemes: the decoded
    logical branch and its forward must agree within the quant budget, and
    everything OUTSIDE the branch must be bit-identical."""
    out_f, in_f, rank = 3072, 3072, 128
    ref_t, down, up = _synth_compiled_graph(out_f, in_f, lowrank="bf16", seed=3)
    q_t, _, _ = _synth_compiled_graph(out_f, in_f, lowrank=scheme, seed=3)

    ref = decode_linear(ref_t, out_f, in_f)
    dec = decode_linear(q_t, out_f, in_f)
    assert ref.lowrank_quant == "bf16"
    assert dec.lowrank_quant == scheme
    assert dec.rank == rank

    # bf16 lane is bijective — its decode IS the factors.
    assert torch.equal(ref.proj_down, down)
    assert torch.equal(ref.proj_up, up)
    assert _rel(dec.proj_down, down) < _BOUNDS[scheme]
    assert _rel(dec.proj_up, up) < _BOUNDS[scheme]
    # The 4-bit residual is untouched by the branch scheme.
    assert torch.equal(dec.codes, ref.codes)
    assert torch.equal(dequantize_decoded(dec), dequantize_decoded(ref))

    x = torch.randn(8, in_f, generator=torch.Generator().manual_seed(4))
    want = (x @ down.float()) @ up.float().t()
    got = (x @ dec.proj_down.float()) @ dec.proj_up.float().t()
    rel = _rel(got, want)
    assert rel < 2 * _BOUNDS[scheme], rel


@pytest.mark.parametrize("scheme", LOWRANK_QUANT_SCHEMES)
def test_fold_to_dense_parity(scheme: str) -> None:
    """The any-hardware fold of a quantized-branch linear differs from the
    bf16-branch fold ONLY by the branch quantization error."""
    out_f, in_f = 3072, 3072
    ref_t, down, up = _synth_compiled_graph(out_f, in_f, lowrank="bf16", seed=5)
    q_t, _, _ = _synth_compiled_graph(out_f, in_f, lowrank=scheme, seed=5)
    ref = native.fold_to_dense(decode_linear(ref_t, out_f, in_f),
                               compute_dtype=torch.float32)
    got = native.fold_to_dense(decode_linear(q_t, out_f, in_f),
                               compute_dtype=torch.float32)
    branch = up.float() @ down.float().t()
    diff = (got.weight - ref.weight).norm().item()
    assert diff / branch.norm().item() < 2 * _BOUNDS[scheme]
    assert torch.equal(got.bias, ref.bias)


def test_split_fused_qkv_carries_the_quantized_branch() -> None:
    out_f, in_f = 9216, 3072
    q_t, down, _ = _synth_compiled_graph(out_f, in_f, second_key="wcscales",
                                lowrank="int8", seed=6)
    dec = decode_linear(q_t, out_f, in_f)
    parts = split_decoded(dec, (3072, 3072, 3072))
    for i, part in enumerate(parts):
        assert part.lowrank_quant == "int8"
        assert part.proj_down is dec.proj_down          # shared, decoded once
        assert torch.equal(part.proj_up,
                           dec.proj_up[i * 3072:(i + 1) * 3072])
    assert _rel(dec.proj_down, down) < _BOUNDS["int8"]


# --- refusals --------------------------------------------------------------


def test_decode_refuses_malformed_quantized_branches() -> None:
    out_f, in_f = 3072, 3072
    q_t, _, _ = _synth_compiled_graph(out_f, in_f, lowrank="int8", seed=7)

    half = dict(q_t)
    half.pop("proj_up_scale")
    with pytest.raises(SvdqLayoutError, match="missing proj_up_scale"):
        decode_linear(half, out_f, in_f)

    mixed = dict(q_t)
    mixed["proj_up"] = q_t["proj_up"].to(torch.float8_e4m3fn)
    with pytest.raises(SvdqLayoutError, match="mixes schemes"):
        decode_linear(mixed, out_f, in_f)

    bf16_t, _, _ = _synth_compiled_graph(out_f, in_f, lowrank="bf16", seed=7)
    stray = dict(bf16_t)
    stray["proj_down_scale"] = q_t["proj_down_scale"]
    with pytest.raises(SvdqLayoutError, match="not.*quantized dtype"):
        decode_linear(stray, out_f, in_f)

    wrong = dict(q_t)
    wrong["proj_down_scale"] = q_t["proj_down_scale"].t().contiguous()
    with pytest.raises(SvdqLayoutError, match="scale shape"):
        decode_linear(wrong, out_f, in_f)


def test_declaration_and_bytes_must_agree() -> None:
    """The __metadata__ flag is the contract: bytes in any other scheme
    refuse, in BOTH directions — no silent fallback."""
    out_f, in_f = 3072, 3072
    q_t, _, _ = _synth_compiled_graph(out_f, in_f, lowrank="int8", seed=8)
    bf16_t, _, _ = _synth_compiled_graph(out_f, in_f, lowrank="bf16", seed=8)

    with pytest.raises(SvdqLayoutError, match="declares lowrank_quant='bf16'"):
        decode_linear(q_t, out_f, in_f, lowrank_quant="bf16")
    with pytest.raises(SvdqLayoutError, match="declares lowrank_quant='int8'"):
        decode_linear(bf16_t, out_f, in_f, lowrank_quant="int8")
    with pytest.raises(SvdqLayoutError,
                       match="declares lowrank_quant='fp8_e4m3'"):
        decode_linear(q_t, out_f, in_f, lowrank_quant="fp8_e4m3")
    with pytest.raises(SvdqLayoutError, match="unknown lowrank_quant"):
        decode_linear(q_t, out_f, in_f, lowrank_quant="int4")
    # Matching declarations decode.
    assert decode_linear(q_t, out_f, in_f,
                         lowrank_quant="int8").lowrank_quant == "int8"
    assert decode_linear(bf16_t, out_f, in_f,
                         lowrank_quant="bf16").lowrank_quant == "bf16"


def test_bf16_execution_lane_is_byte_identical_to_before() -> None:
    """Backward compatibility: a bf16-branch checkpoint decodes exactly as it
    did before te#148, with or without the declaration."""
    out_f, in_f = 3072, 3072
    t, down, up = _synth_compiled_graph(out_f, in_f, lowrank="bf16", seed=9)
    a = decode_linear(t, out_f, in_f)
    b = decode_linear(t, out_f, in_f, lowrank_quant="bf16")
    assert a.lowrank_quant == b.lowrank_quant == "bf16"
    for attr in ("codes", "scales", "proj_down", "proj_up", "smooth_factor",
                 "bias"):
        av, bv = getattr(a, attr), getattr(b, attr)
        assert torch.equal(av.contiguous().view(torch.uint8),
                           bv.contiguous().view(torch.uint8)), attr
    assert torch.equal(a.proj_down, down)
    assert torch.equal(a.proj_up, up)


# --- whole-file loader (integration, tiny REAL qwen) -----------------------


def _tiny_qwen():
    diffusers = pytest.importorskip("diffusers")
    return diffusers.QwenImageTransformer2DModel(
        patch_size=2, in_channels=16, out_channels=16, num_layers=1,
        attention_head_dim=32, num_attention_heads=4,
        joint_attention_dim=128, axes_dims_rope=(8, 12, 12))


def _write_checkpoint(tmp_path, *, lowrank: str, flag: bool):
    """A full tiny-qwen checkpoint: to_qkv svdq-encoded (branch per
    ``lowrank``), every other tensor verbatim bf16."""
    import json

    from safetensors.torch import save_file

    model = _tiny_qwen()
    dim = 128
    prefix = "transformer_blocks.0.attn"
    compiled_graph, down, up = _synth_compiled_graph(3 * dim, dim, second_key="wcscales",
                                   lowrank=lowrank, seed=10)
    state: dict[str, Any] = {}
    for key, val in model.state_dict().items():
        if any(key.startswith(f"{prefix}.to_{p}.") for p in ("q", "k", "v")):
            continue
        state[key] = (val.to(torch.bfloat16).contiguous()
                      if val.is_floating_point() else val.contiguous())
    for leaf, val in compiled_graph.items():
        state[f"{prefix}.to_qkv.{leaf}"] = val.contiguous()

    cfg = {k: v for k, v in dict(model.config).items()
           if not k.startswith("_")}
    meta = {"model_class": "QwenImageTransformer2DModel",
            "config": json.dumps(cfg)}
    if flag and lowrank != "bf16":
        meta[LOWRANK_QUANT_KEY] = lowrank
    path = tmp_path / f"tiny-qwen-{lowrank}-{flag}.safetensors"
    save_file(state, str(path), metadata=meta)
    return path, down, up


class _Art:
    component = "transformer"

    def __init__(self, file) -> None:
        self.file = file


@pytest.mark.parametrize("scheme", LOWRANK_QUANT_SCHEMES)
def test_loader_decodes_a_quantized_branch_file(tmp_path, scheme: str) -> None:
    """load_svdq_native_denoiser end to end: metadata flag read, branch
    dequantized on load, folded weights match the bf16-branch file within the
    quant budget."""
    ref_path, _, _ = _write_checkpoint(tmp_path, lowrank="bf16", flag=False)
    q_path, _, _ = _write_checkpoint(tmp_path, lowrank=scheme, flag=True)

    ref = native.load_svdq_native_denoiser(_Art(ref_path), mode="dense")
    got = native.load_svdq_native_denoiser(_Art(q_path), mode="dense")
    for name in ("to_q", "to_k", "to_v"):
        rw = ref.transformer_blocks[0].attn.get_submodule(name).weight
        gw = got.transformer_blocks[0].attn.get_submodule(name).weight
        assert _rel(gw, rw) < 2 * _BOUNDS[scheme], name
    assert got._cozy_svdq_engine == "native"


def test_loader_refuses_quantized_bytes_without_the_flag(tmp_path) -> None:
    """A quantized branch with no __metadata__ declaration is a malformed
    artifact, not a guessing opportunity."""
    path, _, _ = _write_checkpoint(tmp_path, lowrank="int8", flag=False)
    with pytest.raises(SvdqLayoutError, match="declares lowrank_quant='bf16'"):
        native.load_svdq_native_denoiser(_Art(path), mode="dense")


def test_loader_refuses_an_unknown_scheme_declaration(tmp_path) -> None:
    import json

    from safetensors.torch import save_file

    model = _tiny_qwen()
    cfg = {k: v for k, v in dict(model.config).items()
           if not k.startswith("_")}
    state = {k: (v.to(torch.bfloat16) if v.is_floating_point() else v)
             for k, v in model.state_dict().items()}
    path = tmp_path / "bad-scheme.safetensors"
    save_file(state, str(path),
              metadata={"model_class": "QwenImageTransformer2DModel",
                        "config": json.dumps(cfg),
                        LOWRANK_QUANT_KEY: "int4"})
    with pytest.raises(native.SvdqNativeError, match="int4"):
        native.load_svdq_native_denoiser(_Art(path), mode="dense")
