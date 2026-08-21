from __future__ import annotations

from typing import Any, Optional

import pytest

torch = pytest.importorskip("torch")

from gen_worker.models.svdq_awq import (  # noqa: E402
    decode_awq_linear,
    dequantize_w4x16,
    is_awq_linear,
    num_scale_rows,
    undo_adanorm_splits,
    unpack_w4x16,
)
from gen_worker.models.svdq_layout import SvdqLayoutError  # noqa: E402


def _up_ceil_num_groups(in_features: int, group_size: int) -> int:
    num_groups = in_features // group_size
    pack_size = 32 // 4
    num_packs = -(-num_groups // pack_size)
    factor = {32: 4, 64: 2}.get(group_size, 1)
    num_packs = -(-num_packs // factor) * factor
    return num_packs * pack_size


def _up_pack_w4(weight: Any) -> Any:
    oc, ic = weight.shape
    weight = weight.view(-1, 4, 8)
    weight = (weight[:, 0] | (weight[:, 1] << 4) | (weight[:, 2] << 8)
              | (weight[:, 3] << 12))
    weight = weight.view(oc // 4, 4, ic // 64, 16).permute(
        0, 2, 1, 3).reshape(oc // 4, ic)
    return weight.to(torch.int16)


def _up_convert_w4x16(weight: Any, scale: Any, zero: Any) -> tuple:
    dtype = torch.bfloat16
    weight = weight.to(torch.float32)
    scale = scale.to(torch.float32)
    zero = zero.to(torch.float32) * scale
    oc, ic = weight.shape
    ng = scale.numel() // oc
    gs = ic // ng
    scale = scale.reshape(oc, ng).contiguous().view(oc, ng, 1)
    zero = zero.reshape(oc, ng).contiguous().view(oc, ng, 1)
    q = weight.view(oc, ng, -1).add(zero).div(scale).round().view(oc, ic)
    assert q.min() >= 0 and q.max() <= 15
    packed = _up_pack_w4(q.to(torch.int32))
    _ng = _up_ceil_num_groups(ic, gs)
    _scale = torch.zeros((_ng, oc), dtype=dtype)
    _zero = torch.zeros((_ng, oc), dtype=dtype)
    _scale[:ng] = scale.view(oc, ng).t().to(dtype)
    _zero[:ng] = zero.view(oc, ng).t().to(dtype).neg_()
    return packed.view(torch.int32), _scale, _zero


def _up_adanorm(weight: Any, bias: Any, splits: int) -> tuple:
    oc, ic = weight.shape
    weight = weight.view(splits, oc // splits, ic).transpose(0, 1).reshape(oc, ic)
    bias = bias.reshape(splits, oc // splits).transpose(0, 1)
    delta = [0] * splits
    delta[1] = delta[-2] = 1
    bias = bias.add(torch.tensor(delta, dtype=bias.dtype))
    return weight, bias.reshape(oc)


def _awq_params(w: Any, group_size: int) -> tuple[Any, Any]:
    oc, ic = w.shape
    g = w.reshape(oc, ic // group_size, group_size)
    lo, hi = g.amin(dim=-1), g.amax(dim=-1)
    scale = ((hi - lo) / 15.0).clamp(min=1e-6)
    return scale, -lo / scale


def _synth_awq(oc: int, ic: int, *, group_size: int = 64, splits: int = 1,
               seed: int = 0) -> tuple[dict, Any, Any]:
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(oc, ic, generator=gen).to(torch.bfloat16).float()
    b = torch.randn(oc, generator=gen).to(torch.bfloat16).float()
    w_t, b_t = (_up_adanorm(w, b, splits) if splits > 1 else (w, b))
    scale, zero = _awq_params(w_t, group_size)
    qweight, wscales, wzeros = _up_convert_w4x16(w_t, scale, zero)
    return ({"qweight": qweight, "wscales": wscales, "wzeros": wzeros,
             "bias": b_t.to(torch.bfloat16)}, w, b)


@pytest.mark.parametrize("oc,ic", [
    (18432, 3072),
    (64, 128), (256, 192),
])
def test_unpack_w4x16_inverts_the_upstream_packer(oc: int, ic: int) -> None:
    gen = torch.Generator().manual_seed(2)
    q = torch.randint(0, 16, (oc, ic), dtype=torch.int32, generator=gen)
    packed = _up_pack_w4(q).view(torch.int32)
    assert tuple(packed.shape) == (oc // 4, ic // 2)
    back = unpack_w4x16(packed, oc, ic)
    assert back.dtype == torch.uint8
    assert torch.equal(back.to(torch.int32), q)


def test_real_qwen_modulation_geometry() -> None:
    """Pins the geometry read off the REAL artifact header by range request (nunchaku-ai/nunchaku-qwen-image, svdq-fp4_r128, transformer_blocks.0."""
    oc, ic, gs = 18432, 3072, 64
    assert (oc // 4, ic // 2) == (4608, 1536)
    assert num_scale_rows(ic, gs) == 48
    assert oc // ic == 6


def test_unpack_refuses_a_shape_that_is_not_this_layout() -> None:
    packed = torch.zeros(4608, 1536, dtype=torch.int32)
    with pytest.raises(SvdqLayoutError, match="!= expected"):
        unpack_w4x16(packed, 3072, 3072)
    with pytest.raises(SvdqLayoutError, match="oc%4"):
        unpack_w4x16(packed, 18434, 3072)


def test_dequantize_is_bit_exact_against_the_exporters_own_dequant() -> None:
    """The decisive check."""
    oc, ic, gs = 3072, 3072, 64
    gen = torch.Generator().manual_seed(3)
    w = torch.randn(oc, ic, generator=gen).to(torch.bfloat16).float()
    scale, zero = _awq_params(w, gs)
    ng = ic // gs
    zs = zero * scale
    q = w.view(oc, ng, -1).add(zs.view(oc, ng, 1)).div(
        scale.view(oc, ng, 1)).round().view(oc, ic)
    packed = _up_pack_w4(q.to(torch.int32)).view(torch.int32)
    rows = _up_ceil_num_groups(ic, gs)
    S = torch.zeros(rows, oc, dtype=torch.bfloat16)
    Z = torch.zeros(rows, oc, dtype=torch.bfloat16)
    S[:ng] = scale.view(oc, ng).t().to(torch.bfloat16)
    Z[:ng] = zs.view(oc, ng).t().to(torch.bfloat16).neg_()

    mine = dequantize_w4x16(packed, S, Z, oc, ic)
    ref = (q.view(oc, ng, gs) * S[:ng].t().float().unsqueeze(-1)
           + Z[:ng].t().float().unsqueeze(-1)).reshape(oc, ic)
    assert torch.equal(mine, ref)


def test_padded_scale_rows_are_recognized_not_misread_as_groups() -> None:
    """ceil_num_groups pads: ic=128 at group 64 is 2 real groups but 16 stored rows."""
    oc, ic = 128, 128
    assert num_scale_rows(ic, 64) == 16
    tensors, w, _ = _synth_awq(oc, ic, seed=4)
    assert tensors["wscales"].shape == (16, oc)
    assert bool((tensors["wscales"][2:] == 0).all()), "rows 2..15 are padding"
    got = dequantize_w4x16(tensors["qweight"], tensors["wscales"],
                           tensors["wzeros"], oc, ic)
    assert ((got - w).norm() / w.norm()).item() < 0.15


def test_dequantize_refuses_mismatched_grids() -> None:
    tensors, _, _ = _synth_awq(3072, 3072, seed=5)
    with pytest.raises(SvdqLayoutError, match="do not match out_features"):
        dequantize_w4x16(tensors["qweight"], tensors["wscales"][:, :16],
                         tensors["wzeros"][:, :16], 3072, 3072)


@pytest.mark.parametrize("splits", [2, 3, 6])
def test_undo_adanorm_splits_round_trips_exactly(splits: int) -> None:
    """Both halves: the output-channel interleave AND the +1 on splits 1 and splits-2."""
    oc, ic = 3072 * splits, 128
    gen = torch.Generator().manual_seed(6)
    w = torch.randn(oc, ic, generator=gen)
    b = torch.randn(oc, generator=gen)
    w_t, b_t = _up_adanorm(w, b, splits)
    assert not torch.equal(w_t, w), "the transform must actually permute"
    w_back, b_back = undo_adanorm_splits(w_t, b_t, splits)
    assert torch.equal(w_back, w)
    assert torch.allclose(b_back, b, atol=1e-6)


def test_adanorm_bias_delta_is_on_splits_one_and_minus_two() -> None:
    """Nails WHICH splits carry the +1 — a decoder that subtracts it from the wrong pair is off by exactly 1.0 on two adaLN channels."""
    splits, per, ic = 6, 4, 64
    w = torch.zeros(splits * per, ic)
    b = torch.zeros(splits * per)
    _, b_t = _up_adanorm(w, b, splits)
    got = b_t.view(per, splits).transpose(0, 1)[:, 0]
    assert got.tolist() == [0.0, 1.0, 0.0, 0.0, 1.0, 0.0]


def test_undo_adanorm_is_a_noop_for_a_plain_layer() -> None:
    w = torch.randn(64, 32)
    b = torch.randn(64)
    w2, b2 = undo_adanorm_splits(w, b, 1)
    assert w2 is w and b2 is b


def test_decode_awq_linear_reproduces_the_original_modulation_linear() -> None:
    """The whole S2b path at the real qwen modulation shape: exporter forward (adaLN transform + AWQ quantize + TinyChat pack) -> our decode -> a plain Linear whose forward matches the ORIGINAL module."""
    oc, ic, splits = 18432, 3072, 6
    tensors, w, b = _synth_awq(oc, ic, splits=splits, seed=7)
    lin = decode_awq_linear(tensors, oc, ic, adanorm_splits=splits,
                            compute_dtype=torch.float32)
    torch.manual_seed(8)
    x = torch.randn(2, ic)
    want = x @ w.t() + b
    rel = ((lin(x) - want).norm() / want.norm()).item()
    assert rel < 0.15, rel


def test_decoding_with_the_wrong_split_count_is_visibly_wrong() -> None:
    """Documents WHY adanorm_splits is required rather than inferred: the wrong count still produces a full-rank plausible weight."""
    oc, ic, splits = 3072 * 6, 3072, 6
    tensors, w, b = _synth_awq(oc, ic, splits=splits, seed=9)
    torch.manual_seed(10)
    x = torch.randn(2, ic)
    want = x @ w.t() + b
    good = decode_awq_linear(tensors, oc, ic, adanorm_splits=splits,
                             compute_dtype=torch.float32)
    bad = decode_awq_linear(tensors, oc, ic, adanorm_splits=1,
                            compute_dtype=torch.float32)
    assert ((good(x) - want).norm() / want.norm()).item() < 0.15
    assert ((bad(x) - want).norm() / want.norm()).item() > 0.5


def test_is_awq_linear_discriminates_on_wzeros() -> None:
    tensors, _, _ = _synth_awq(128, 64, seed=11)
    assert is_awq_linear(tensors)
    assert not is_awq_linear({"qweight": tensors["qweight"],
                              "wscales": tensors["wscales"]})
    with pytest.raises(SvdqLayoutError, match="missing"):
        decode_awq_linear({"qweight": tensors["qweight"]}, 128, 64)


from gen_worker.models.svdq_awq import (  # noqa: E402
    apply_adanorm_splits,
    encode_awq_linear,
    pack_w4x16,
)


@pytest.mark.parametrize("oc,ic", [(18432, 3072), (128, 64), (256, 192)])
def test_pack_w4x16_matches_the_vendored_upstream_packer(oc: int, ic: int) -> None:
    gen = torch.Generator().manual_seed(20)
    q = torch.randint(0, 16, (oc, ic), dtype=torch.int32, generator=gen)
    assert torch.equal(pack_w4x16(q.to(torch.uint8)),
                       _up_pack_w4(q).view(torch.int32))


@pytest.mark.parametrize("splits", [2, 6])
def test_apply_adanorm_matches_upstream_and_round_trips(splits: int) -> None:
    oc, ic = 384 * splits, 64
    gen = torch.Generator().manual_seed(21)
    w = torch.randn(oc, ic, generator=gen)
    b = torch.randn(oc, generator=gen)
    w_up, b_up = _up_adanorm(w, b, splits)
    w_us, b_us = apply_adanorm_splits(w, b, splits)
    assert torch.equal(w_us, w_up) and torch.equal(b_us, b_up)
    w_back, b_back = undo_adanorm_splits(w_us, b_us, splits)
    assert torch.equal(w_back, w)
    assert torch.allclose(b_back, b, atol=1e-6)


@pytest.mark.parametrize("oc,ic,splits", [
    (18432, 3072, 6),
    (3072, 3072, 1),
    (128, 128, 1),
])
def test_encode_awq_linear_is_bit_exact_vs_the_vendored_exporter(
    oc: int, ic: int, splits: int,
) -> None:
    """The production encoder must emit the same BYTES the upstream exporter chain emits for the same weight — packed codes, padded bf16 grids, bias."""
    tensors, w, b = _synth_awq(oc, ic, splits=splits, seed=22)
    got = encode_awq_linear(w.clone(), b.clone(),
                            group_size=64, adanorm_splits=splits)
    assert torch.equal(got["qweight"], tensors["qweight"])
    assert torch.equal(got["wscales"], tensors["wscales"])
    assert torch.equal(got["wzeros"], tensors["wzeros"])
    assert torch.equal(got["bias"], tensors["bias"])


def test_encode_refuses_adanorm_without_bias() -> None:
    with pytest.raises(SvdqLayoutError, match="requires a bias"):
        encode_awq_linear(torch.randn(384, 64), None, adanorm_splits=6)
