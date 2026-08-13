"""pgw#770 — our svdq decoders must invert the UPSTREAM packers, not our own.

pgw#685's suite synthesized checkpoints with ``gen_worker``'s own
``pack_qweight``/``pack_wscales`` and asserted the decoders inverted them. That
proves bijectivity and nothing else: ``pack_wscales`` and ``unpack_wscales``
shared one wrong row split (4, 8, 4) instead of deepcompressor's (4, 4, 8), and
``proj_down``/``proj_up``/``smooth_factor``/``bias``/``wcscales`` were read
verbatim although every one of them is fragment-packed on disk. The official
qwen-image artifact therefore rendered noise (te#137 Run 1b: lpips 0.8215,
psnr 4.64 dB, clip 15.26 vs 37.19) through both the blockwise and the dense
lane, while our own artifact — written with the same wrong convention —
round-tripped and scored 0.5079.

So the reference here is UPSTREAM CODE, transcribed verbatim below from
``nunchaku-ai/deepcompressor`` @ main (Apache-2.0, attribution in-place), never
our own encode side. Every test drives the upstream packer forward and asserts
our decoder recovers the input exactly.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gen_worker.models.nvfp4_quant import BLOCK, cast_e2m1  # noqa: E402
from gen_worker.models.svdq_layout import (  # noqa: E402
    decode_linear,
    dequantize_decoded,
    pack_lowrank,
    pack_qweight,
    pack_vector,
    pack_wscales,
    split_decoded,
    unpack_lowrank,
    unpack_qweight,
    unpack_vector,
    unpack_wscales,
)
from gen_worker.models.svdq_native import fold_to_dense  # noqa: E402

E2M1_MAX, FP8_MAX = 6.0, 448.0

# ---------------------------------------------------------------------------
# VENDORED UPSTREAM — deepcompressor (Apache-2.0, nunchaku-ai/deepcompressor
# @ main). backend/utils.py MmaWeightPackerBase.__init__ L91-141 resolved for
# bits=4 / warp_n=128; backend/nunchaku/utils.py pack_weight L21-60,
# pack_scale L62-107, pack_micro_scale L109-151, pack_lowrank_weight L153-182.
# Transcribed, not adapted: this file's whole job is to be the other side of
# the contract, so it must not share a line with our implementation.
# ---------------------------------------------------------------------------

MEM_N, MEM_K = 128, 64
NUM_N_PACKS, N_PACK_SIZE, NUM_N_EXECUTION_LANES, REG_N = 8, 2, 8, 1
NUM_K_PACKS, K_PACK_SIZE, NUM_K_EXECUTION_LANES, REG_K = 1, 2, 4, 8
WARP_N, NUM_EXECUTION_LANES, INSN_K = 128, 32, 64


def dc_pack_weight(weight: torch.Tensor) -> torch.Tensor:
    """deepcompressor NunchakuWeightPacker.pack_weight, bits=4."""
    n, k = weight.shape
    n_tiles, k_tiles = n // MEM_N, k // MEM_K
    weight = weight.to(torch.int32).reshape(
        n_tiles, NUM_N_PACKS, N_PACK_SIZE, NUM_N_EXECUTION_LANES, REG_N,
        k_tiles, NUM_K_PACKS, K_PACK_SIZE, NUM_K_EXECUTION_LANES, REG_K)
    weight = weight.permute(0, 5, 6, 1, 3, 8, 2, 7, 4, 9).contiguous()
    assert weight.shape[4:-2] == (8, 4, 2, 2)
    weight = weight.bitwise_and_(0xF)
    shift = torch.arange(0, 32, 4, dtype=torch.int32)
    weight = weight.bitwise_left_shift_(shift)
    weight = weight.sum(dim=-1, dtype=torch.int32)
    return weight.view(dtype=torch.int8).view(n, -1)


def dc_pack_micro_scale(scale: torch.Tensor, group_size: int = 16,
                        *, cast: bool = True) -> torch.Tensor:
    """deepcompressor NunchakuWeightPacker.pack_micro_scale.

    ``cast=False`` skips only the e4m3 cast, so the identical view/permute can
    be replayed on an index tensor to read the store order off."""
    if cast:
        scale = scale.to(dtype=torch.float8_e4m3fn)
    n = scale.shape[0]
    s_pack_size = min(max(WARP_N // NUM_EXECUTION_LANES, 1), 4)
    num_s_lanes = 4 * 8
    num_s_packs = -(-WARP_N // (s_pack_size * num_s_lanes))
    warp_s = num_s_packs * num_s_lanes * s_pack_size
    assert warp_s == WARP_N
    scale = scale.view(n // warp_s, num_s_packs, s_pack_size, 4, 8, -1,
                       INSN_K // group_size)
    scale = scale.permute(0, 5, 1, 4, 3, 2, 6).contiguous()
    return scale.view(-1, n)


def dc_pack_scale(scale: torch.Tensor) -> torch.Tensor:
    """deepcompressor NunchakuWeightPacker.pack_scale, group_size=-1."""
    n = scale.shape[0]
    s_pack_size = min(max(WARP_N // NUM_EXECUTION_LANES, 2), 8)
    num_s_lanes = min(NUM_EXECUTION_LANES, WARP_N // s_pack_size)
    num_s_packs = WARP_N // (s_pack_size * num_s_lanes)
    warp_s = num_s_packs * num_s_lanes * s_pack_size
    assert warp_s == WARP_N
    scale = scale.reshape(n // warp_s, num_s_packs, num_s_lanes // 4,
                          s_pack_size // 2, 4, 2, -1)
    scale = scale.permute(0, 6, 1, 2, 4, 3, 5).contiguous()
    return scale.view(-1)


def dc_pack_lowrank_weight(weight: torch.Tensor, down: bool) -> torch.Tensor:
    """deepcompressor NunchakuWeightPacker.pack_lowrank_weight."""
    reg_n, reg_k = 1, 2
    pack_n = N_PACK_SIZE * NUM_N_EXECUTION_LANES * reg_n
    pack_k = K_PACK_SIZE * NUM_K_EXECUTION_LANES * reg_k
    if down:
        r, c = weight.shape
        r_packs, c_packs = r // pack_n, c // pack_k
        weight = weight.view(r_packs, pack_n, c_packs,
                             pack_k).permute(2, 0, 1, 3)
    else:
        c, r = weight.shape
        c_packs, r_packs = c // pack_n, r // pack_k
        weight = weight.view(c_packs, pack_n, r_packs,
                             pack_k).permute(0, 2, 1, 3)
    weight = weight.reshape(c_packs, r_packs, N_PACK_SIZE, NUM_N_EXECUTION_LANES, reg_n,
                            K_PACK_SIZE, NUM_K_EXECUTION_LANES, reg_k)
    weight = weight.permute(0, 1, 3, 6, 2, 5, 4, 7).contiguous()
    return weight.reshape(c, r)


# ---------------------------------------------------------------------------
# The four inverses, each against the upstream forward.
# ---------------------------------------------------------------------------

# Real qwen-image svdq shapes from the official artifact's header
# (nunchaku-ai/nunchaku-qwen-image, svdq-fp4_r128-qwen-image.safetensors).
SHAPES = (
    ("attn.to_qkv", 9216, 3072),
    ("attn.to_out.0", 3072, 3072),
    ("img_mlp.net.0.proj", 12288, 3072),
    ("img_mlp.net.2", 3072, 12288),
)


@pytest.mark.parametrize("name,out_f,in_f", SHAPES)
def test_unpack_qweight_inverts_upstream_pack_weight(name, out_f, in_f):
    gen = torch.Generator().manual_seed(hash(name) & 0xFFFF)
    codes = torch.randint(0, 16, (out_f, in_f), generator=gen,
                          dtype=torch.uint8)
    stored = dc_pack_weight(codes)
    assert tuple(stored.shape) == (out_f, in_f // 2)
    assert stored.dtype == torch.int8
    assert torch.equal(unpack_qweight(stored, out_f, in_f), codes)
    # and our encode side must produce the SAME bytes upstream does
    assert torch.equal(pack_qweight(codes), stored)


@pytest.mark.parametrize("name,out_f,in_f", SHAPES)
def test_unpack_wscales_inverts_upstream_pack_micro_scale(name, out_f, in_f):
    ng = in_f // BLOCK
    gen = torch.Generator().manual_seed(hash(name) & 0xFFFF)
    flat = (torch.rand(out_f, ng, generator=gen) * 400 + 1).to(
        torch.float8_e4m3fn)
    stored = dc_pack_micro_scale(flat)
    assert tuple(stored.shape) == (ng, out_f)
    back = unpack_wscales(stored, out_f, in_f)
    assert torch.equal(back.view(torch.uint8), flat.view(torch.uint8))
    assert torch.equal(pack_wscales(flat, out_f, in_f).view(torch.uint8),
                       stored.view(torch.uint8))


def test_wscales_row_split_is_4_4_8_not_4_8_4():
    """The pgw#770 regression guard, stated as a difference rather than a
    round trip — (4, 8, 4) reshapes without error, so only comparing against
    the upstream store order catches it."""
    out_f, in_f = 256, 512
    ng = in_f // BLOCK
    flat = (torch.rand(out_f, ng, generator=torch.Generator().manual_seed(3))
            * 400 + 1).to(torch.float8_e4m3fn)
    stored = dc_pack_micro_scale(flat)
    wrong = stored.reshape(out_f // 128, ng // 4, 4, 8, 4, 4).permute(
        0, 4, 3, 2, 1, 5).contiguous().reshape(out_f, ng)
    right = unpack_wscales(stored, out_f, in_f)
    assert torch.equal(right.view(torch.uint8), flat.view(torch.uint8))
    assert not torch.equal(wrong.view(torch.uint8), right.view(torch.uint8))

    # upstream's documented store order (pack_micro_scale L136-148): 32-bit
    # word j of a tile holds output channel (j%4)*32 + ((j//4)%4)*8 + (j//16).
    # Replayed on a channel-index tensor so no dtype rounding is involved.
    idx = torch.arange(out_f, dtype=torch.int64).reshape(out_f, 1).expand(
        out_f, ng).contiguous()
    order = dc_pack_micro_scale(idx, cast=False).reshape(-1)[:512:4]
    want = [(j % 4) * 32 + ((j // 4) % 4) * 8 + (j // 16) for j in range(128)]
    assert order.tolist() == want


@pytest.mark.parametrize("n", (128, 3072, 9216, 12288))
def test_unpack_vector_inverts_upstream_pack_scale(n):
    v = torch.arange(n, dtype=torch.float32)
    stored = dc_pack_scale(v.reshape(n, 1))
    assert tuple(stored.shape) == (n,)
    assert torch.equal(unpack_vector(stored, n), v)
    assert torch.equal(pack_vector(v, n), stored)
    # non-trivial: the swizzle must actually move channels
    assert int((stored != v).sum()) > n // 2
    # upstream's table (pack_scale L93-104): 0 1 8 9 / 2 3 10 11 / ...
    assert stored[:8].tolist() == [0.0, 1.0, 8.0, 9.0, 2.0, 3.0, 10.0, 11.0]


@pytest.mark.parametrize("out_f,in_f,rank", ((9216, 3072, 128),
                                            (3072, 12288, 128),
                                            (3072, 3072, 32)))
def test_unpack_lowrank_inverts_upstream_pack_lowrank(out_f, in_f, rank):
    gen = torch.Generator().manual_seed(7)
    down = torch.randn(rank, in_f, generator=gen).to(torch.bfloat16)
    up = torch.randn(out_f, rank, generator=gen).to(torch.bfloat16)
    sd, su = (dc_pack_lowrank_weight(down, down=True),
              dc_pack_lowrank_weight(up, down=False))
    assert tuple(sd.shape) == (in_f, rank)
    assert tuple(su.shape) == (out_f, rank)
    assert torch.equal(unpack_lowrank(sd, down=True), down)
    assert torch.equal(unpack_lowrank(su, down=False), up)
    assert torch.equal(pack_lowrank(down, down=True), sd)
    assert torch.equal(pack_lowrank(up, down=False), su)
    # a plain read is NOT the identity — the pgw#770 defect
    assert not torch.equal(sd.float(), down.float().t())


# ---------------------------------------------------------------------------
# End to end: an UPSTREAM-packed linear must fold back to the weight it came
# from. This is the assertion pgw#685 was missing.
# ---------------------------------------------------------------------------


def _upstream_linear(out_f: int, in_f: int, *, rank: int, per_channel: bool,
                     smooth: bool, seed: int = 0):
    """A checkpoint compiled graph written the way deepcompressor writes one, plus the
    float weight it encodes. Mirrors convert_to_nunchaku_w4x4y16_linear_weight:
    W_smoothed = W * diag(s), lora_down pre-divided by s, two-level nvfp4."""
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(out_f, in_f, generator=gen)
    s = ((torch.rand(in_f, generator=gen) + 0.5) if smooth
         else torch.ones(in_f)).to(torch.bfloat16)
    ws = w * s.float().reshape(1, -1)          # smoothed weight

    down = (torch.randn(rank, in_f, generator=gen) * in_f ** -0.5
            ).to(torch.bfloat16)
    up = (torch.randn(out_f, rank, generator=gen) * rank ** -0.5
          ).to(torch.bfloat16)
    resid = ws - (up.float() @ down.float())   # the 4-bit residual

    if per_channel:
        second = (resid.abs().amax(dim=1) / (E2M1_MAX * FP8_MAX)).clamp(min=1e-8)
        second = second.to(torch.bfloat16).float().reshape(out_f, 1)
    else:
        second = (resid.abs().amax() / (E2M1_MAX * FP8_MAX)).clamp(min=1e-8)
        second = second.to(torch.bfloat16).float().reshape(1, 1)
    blocks = resid.reshape(out_f, in_f // BLOCK, BLOCK)
    bs = (blocks.abs().amax(dim=-1) / (E2M1_MAX * second)).clamp(
        min=2.0 ** -9, max=FP8_MAX).to(torch.float8_e4m3fn)
    codes = cast_e2m1((blocks / (bs.float().unsqueeze(-1)
                                 * second.unsqueeze(-1))).reshape(out_f, in_f))
    bias = torch.randn(out_f, generator=gen).to(torch.bfloat16)

    tensors = {
        "qweight": dc_pack_weight(codes),
        "wscales": dc_pack_micro_scale(bs),
        "proj_down": dc_pack_lowrank_weight(
            (down.float() / s.float().reshape(1, -1)).to(torch.bfloat16),
            down=True),
        "proj_up": dc_pack_lowrank_weight(up, down=False),
        "smooth_factor": dc_pack_scale(s.reshape(in_f, 1)),
        "smooth_factor_orig": dc_pack_scale(s.reshape(in_f, 1)),
        "bias": dc_pack_scale(bias.reshape(out_f, 1)),
    }
    if per_channel:
        tensors["wcscales"] = dc_pack_scale(
            second.reshape(out_f, 1).to(torch.bfloat16))
    else:
        tensors["wtscale"] = second.reshape(1).to(torch.bfloat16)
    # what the runtime should reproduce
    return tensors, w, bias


@pytest.mark.parametrize("out_f,in_f,per_channel,smooth", (
    (3072, 3072, False, False),   # the official artifact's shape: smooth == 1
    (3072, 3072, False, True),    # int4-style non-trivial smoothing
    (9216, 3072, True, True),     # fused qkv: per-channel second level
    (1536, 6144, False, True),
))
def test_dense_fold_recovers_the_upstream_weight(out_f, in_f, per_channel,
                                                smooth):
    tensors, want, want_bias = _upstream_linear(
        out_f, in_f, rank=128, per_channel=per_channel, smooth=smooth,
        seed=out_f + in_f)
    dec = decode_linear(tensors, out_f, in_f)

    assert dec.second_kind == ("per_channel" if per_channel else "per_tensor")
    torch.testing.assert_close(dec.bias.float(), want_bias.float())
    if smooth:
        assert dec.smooth_factor.float().max() > 1.0

    lin = fold_to_dense(dec, compute_dtype=torch.float32)
    got = lin.weight.detach().float()
    err = ((got - want).norm() / want.norm()).item()
    # 4-bit residual on top of a rank-128 branch over Gaussian data; the point
    # is that it is a QUANTIZATION error, not a permutation (which lands >1.0).
    assert err < 0.35, f"dense fold rel-err {err:.4f}"
    torch.testing.assert_close(lin.bias.detach().float(), want_bias.float())


def test_split_decoded_keeps_the_unswizzled_order():
    """A fused qkv must split into three logical projections AFTER the
    per-channel unswizzle, or each third gets another third's scales."""
    tensors, want, want_bias = _upstream_linear(
        9216, 3072, rank=128, per_channel=True, smooth=True, seed=11)
    dec = decode_linear(tensors, 9216, 3072)
    parts = split_decoded(dec, (3072, 3072, 3072))
    assert len(parts) == 3
    full = dequantize_decoded(dec)
    for i, part in enumerate(parts):
        assert part.proj_down is dec.proj_down
        assert part.smooth_factor is dec.smooth_factor
        torch.testing.assert_close(dequantize_decoded(part),
                                   full[i * 3072:(i + 1) * 3072])
        torch.testing.assert_close(part.bias.float(),
                                   want_bias.float()[i * 3072:(i + 1) * 3072])
        lin = fold_to_dense(part, compute_dtype=torch.float32)
        err = ((lin.weight.detach().float()
                - want[i * 3072:(i + 1) * 3072]).norm()
               / want[i * 3072:(i + 1) * 3072].norm()).item()
        assert err < 0.35, f"qkv part {i} rel-err {err:.4f}"
