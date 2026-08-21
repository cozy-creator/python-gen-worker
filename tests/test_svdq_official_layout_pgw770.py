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
    """deepcompressor NunchakuWeightPacker.pack_micro_scale."""
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
    assert int((stored != v).sum()) > n // 2
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
    assert not torch.equal(sd.float(), down.float().t())


def _upstream_linear(out_f: int, in_f: int, *, rank: int, per_channel: bool,
                     smooth: bool, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(out_f, in_f, generator=gen)
    s = ((torch.rand(in_f, generator=gen) + 0.5) if smooth
         else torch.ones(in_f)).to(torch.bfloat16)
    ws = w * s.float().reshape(1, -1)

    down = (torch.randn(rank, in_f, generator=gen) * in_f ** -0.5
            ).to(torch.bfloat16)
    up = (torch.randn(out_f, rank, generator=gen) * rank ** -0.5
          ).to(torch.bfloat16)
    resid = ws - (up.float() @ down.float())

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
    return tensors, w, bias


@pytest.mark.parametrize("out_f,in_f,per_channel,smooth", (
    (3072, 3072, False, False),
    (3072, 3072, False, True),
    (9216, 3072, True, True),
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
    assert err < 0.35, f"dense fold rel-err {err:.4f}"
    torch.testing.assert_close(lin.bias.detach().float(), want_bias.float())


def test_split_decoded_keeps_the_unswizzled_order():
    """A fused qkv must split into three logical projections AFTER the per-channel unswizzle, or each third gets another third's scales."""
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
