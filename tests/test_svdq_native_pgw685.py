"""pgw#685 — the NATIVE svdq engine: layout converter, SvdqLinear, engine choice.

Integration-style over the real code paths: the fixtures synthesize a checkpoint
in nunchaku's ACTUAL v1 layout (via the same fragment/lane packers the decoder
inverts), then drive `decode_linear` -> `split_decoded` -> `swap_svdq_linears`
into a module with diffusers-style names. No nunchaku wheel, no 13 GB artifact,
no GPU: the fp4 GEMM is the only GPU-only piece and it is gated.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from gen_worker.models import svdq as svdq_mod  # noqa: E402
from gen_worker.models import svdq_native as native  # noqa: E402
from gen_worker.models.nvfp4_quant import BLOCK, cast_e2m1  # noqa: E402
from gen_worker.models.svdq_layout import (  # noqa: E402
    SvdqLayoutError,
    decode_linear,
    dequantize_decoded,
    pack_lowrank,
    pack_qweight,
    pack_vector,
    pack_wscales,
    split_decoded,
    unpack_qweight,
    unpack_wscales,
)

E2M1_MAX, FP8_MAX = 6.0, 448.0

# Real qwen-image svdq shapes (out, in, second-level key, rank) from the
# official artifact's header (pgw#682 G-C).
QWEN_CASES = (
    ("attn.to_qkv", 9216, 3072, "wcscales", 128),
    ("attn.to_out.0", 3072, 3072, "wtscale", 128),
    ("img_mlp.net.0.proj", 12288, 3072, "wtscale", 128),
    ("img_mlp.net.2", 3072, 12288, "wtscale", 128),
)


def _synth_nunchaku_linear(out_f: int, in_f: int, *, second_key: str,
                           rank: int, seed: int = 0,
                           smooth: bool = True) -> tuple[dict, torch.Tensor]:
    """A checkpoint entry in nunchaku's layout + the float weight behind it."""
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(out_f, in_f, generator=gen)

    if second_key == "wcscales":
        second = (w.abs().amax(dim=1) / (E2M1_MAX * FP8_MAX)).clamp(min=1e-8)
        second_bcast = second.reshape(out_f, 1)
    else:
        second = (w.abs().amax() / (E2M1_MAX * FP8_MAX)).clamp(min=1e-8)
        second_bcast = second.reshape(1, 1)

    blocks = w.reshape(out_f, in_f // BLOCK, BLOCK)
    bmax = blocks.abs().amax(dim=-1)
    bs = (bmax / (E2M1_MAX * second_bcast)).clamp(min=2.0 ** -9, max=FP8_MAX)
    bs = bs.to(torch.float8_e4m3fn)
    q = blocks / (bs.float().unsqueeze(-1) * second_bcast.unsqueeze(-1))
    codes = cast_e2m1(q.reshape(out_f, in_f))

    # pgw#770: EVERY tensor is fragment-packed on disk, so the fixture packs
    # every one of them. wtscale is the sole exception (it is a scalar).
    tensors: dict[str, Any] = {
        "qweight": pack_qweight(codes),
        "wscales": pack_wscales(bs, out_f, in_f),
    }
    tensors[second_key] = (pack_vector(second.to(torch.bfloat16), out_f)
                           if second_key == "wcscales"
                           else second.to(torch.bfloat16))
    if rank:
        tensors["proj_down"] = pack_lowrank(
            (torch.randn(rank, in_f, generator=gen)
             * (in_f ** -0.5)).to(torch.bfloat16), down=True)
        tensors["proj_up"] = pack_lowrank(
            (torch.randn(out_f, rank, generator=gen)
             * (rank ** -0.5)).to(torch.bfloat16), down=False)
    if smooth:
        tensors["smooth_factor"] = pack_vector(
            (torch.rand(in_f, generator=gen) + 0.5).to(torch.bfloat16), in_f)
        # Provenance only — the decoder must never read this.
        tensors["smooth_factor_orig"] = torch.zeros(in_f, dtype=torch.bfloat16)
    tensors["bias"] = pack_vector(
        torch.randn(out_f, generator=gen).to(torch.bfloat16), out_f)
    return tensors, w


# --- layout inverses -------------------------------------------------------


@pytest.mark.parametrize("name,out_f,in_f,second_key,rank", QWEN_CASES)
def test_qweight_and_wscale_layouts_round_trip(
    name: str, out_f: int, in_f: int, second_key: str, rank: int,
) -> None:
    """The fragment interleave and the lane swizzle are both bijective — the
    decoder replays the packer backwards rather than re-deriving index math."""
    gen = torch.Generator().manual_seed(1)
    codes = torch.randint(0, 16, (out_f, in_f), dtype=torch.uint8, generator=gen)
    assert torch.equal(unpack_qweight(pack_qweight(codes), out_f, in_f), codes)

    # Build scales as real values then cast — random e4m3 BYTES would include
    # the 0x7F/0xFF NaN patterns.
    flat = (torch.rand(out_f, in_f // BLOCK, generator=gen) * 8 + 0.05).to(
        torch.float8_e4m3fn)
    back = unpack_wscales(pack_wscales(flat, out_f, in_f), out_f, in_f)
    assert torch.equal(back.view(torch.uint8), flat.view(torch.uint8))


def test_qweight_row_is_not_the_output_channel() -> None:
    """Guards the finding that makes a naive 'just transpose it' converter
    silently wrong: the packed row index is NOT the logical output channel.

    At the real qwen shape, ``packed[1, 0]``'s low nibble is logical
    ``(o=48, k=0)`` — and ``o=1`` lives out at column 64. (The mapping is
    geometry-dependent: at a degenerate single-k-tile shape the same probe
    lands elsewhere, so this is asserted at a shape the artifact actually
    uses.)"""
    out_f, in_f = 3072, 3072
    codes = torch.zeros(out_f, in_f, dtype=torch.uint8)
    codes[48, 0] = 7
    packed = pack_qweight(codes).view(torch.uint8)
    assert int(packed[1, 0]) & 0x0F == 7
    assert int(packed[0, 0]) & 0x0F == 0

    codes = torch.zeros(out_f, in_f, dtype=torch.uint8)
    codes[1, 0] = 7
    packed = pack_qweight(codes).view(torch.uint8)
    assert int(packed[1, 0]) & 0x0F == 0
    assert int(packed[0, 64]) & 0x0F == 7


@pytest.mark.parametrize("name,out_f,in_f,second_key,rank", QWEN_CASES)
def test_decode_recovers_the_weight_within_block_quant_error(
    name: str, out_f: int, in_f: int, second_key: str, rank: int,
) -> None:
    tensors, w = _synth_nunchaku_linear(out_f, in_f, second_key=second_key,
                                       rank=rank)
    dec = decode_linear(tensors, out_f, in_f)
    assert dec.rank == rank
    assert dec.second_kind == ("per_channel" if second_key == "wcscales"
                              else "per_tensor")
    assert dec.smooth_factor is not None
    deq = dequantize_decoded(dec)
    assert deq.shape == (out_f, in_f)
    rel = ((deq - w).norm() / w.norm()).item()
    assert rel < 0.15, rel


def test_decode_refuses_layouts_it_does_not_understand() -> None:
    tensors, _ = _synth_nunchaku_linear(3072, 3072, second_key="wtscale",
                                        rank=32)
    with pytest.raises(SvdqLayoutError, match="missing"):
        decode_linear({k: v for k, v in tensors.items() if k != "qweight"},
                      3072, 3072)
    with pytest.raises(SvdqLayoutError, match="wcscales nor wtscale"):
        decode_linear({k: v for k, v in tensors.items() if k != "wtscale"},
                      3072, 3072)
    half = dict(tensors)
    half.pop("proj_up")
    with pytest.raises(SvdqLayoutError, match="one half of the low-rank"):
        decode_linear(half, 3072, 3072)


def test_split_fused_qkv_is_exact() -> None:
    """nunchaku fuses q/k/v; diffusers keeps them separate. Splitting in the
    logical domain must reproduce the fused dequant row-block for row-block,
    and partition proj_up while SHARING proj_down."""
    out_f, in_f, rank = 9216, 3072, 128
    tensors, _ = _synth_nunchaku_linear(out_f, in_f, second_key="wcscales",
                                        rank=rank)
    dec = decode_linear(tensors, out_f, in_f)
    fused = dequantize_decoded(dec)
    parts = split_decoded(dec, (3072, 3072, 3072))
    assert len(parts) == 3
    for i, part in enumerate(parts):
        assert part.out_features == 3072
        assert part.in_features == in_f
        assert torch.equal(dequantize_decoded(part),
                           fused[i * 3072:(i + 1) * 3072])
        # proj_down is shared verbatim; proj_up is partitioned by rows.
        assert part.proj_down is dec.proj_down
        assert torch.equal(part.proj_up, dec.proj_up[i * 3072:(i + 1) * 3072])
        assert part.smooth_factor is dec.smooth_factor
    with pytest.raises(SvdqLayoutError, match="sum to"):
        split_decoded(dec, (3072, 3072))


# --- the folded bf16 fallback ---------------------------------------------


def test_fold_to_dense_matches_the_two_branch_reference() -> None:
    """The any-hardware fallback must be the WHOLE linear: the smoothing divide
    and the low-rank branch folded in, not just the 4-bit weight. Reference is
    the forward it replaces, computed branch by branch."""
    out_f, in_f, rank = 3072, 3072, 128
    tensors, _ = _synth_nunchaku_linear(out_f, in_f, second_key="wtscale",
                                        rank=rank, seed=3)
    dec = decode_linear(tensors, out_f, in_f)
    lin = native.fold_to_dense(dec, compute_dtype=torch.float32)

    torch.manual_seed(4)
    x = torch.randn(8, in_f)
    # y = (x / smooth) @ dequant(W).T + (x @ down) @ up.T + bias
    xs = x / dec.smooth_factor.float()
    want = (xs @ dequantize_decoded(dec).t()
            + (x @ dec.proj_down.float()) @ dec.proj_up.float().t()
            + dec.bias.float())
    got = lin(x)
    rel = ((got - want).norm() / want.norm()).item()
    assert rel < 1e-5, rel


def test_fold_to_dense_without_smoothing_or_low_rank() -> None:
    tensors, _ = _synth_nunchaku_linear(3072, 3072, second_key="wtscale",
                                        rank=0, smooth=False, seed=5)
    dec = decode_linear(tensors, 3072, 3072)
    assert dec.rank == 0 and dec.smooth_factor is None
    lin = native.fold_to_dense(dec, compute_dtype=torch.float32)
    torch.manual_seed(6)
    x = torch.randn(4, 3072)
    want = x @ dequantize_decoded(dec).t() + dec.bias.float()
    assert ((lin(x) - want).norm() / want.norm()).item() < 1e-5


# --- swapping into a real module graph ------------------------------------


class _Attn(torch.nn.Module):
    """diffusers-style attention naming: separate q/k/v, fused in nunchaku."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.to_q = torch.nn.Linear(dim, dim)
        self.to_k = torch.nn.Linear(dim, dim)
        self.to_v = torch.nn.Linear(dim, dim)
        self.to_out = torch.nn.ModuleList([torch.nn.Linear(dim, dim)])


class _Block(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.attn = _Attn(dim)


def test_plan_targets_resolves_direct_and_fused_names() -> None:
    model = _Block(3072)
    assert native.plan_targets(model, "attn.to_out.0") == (("attn.to_out.0", 3072),)
    assert native.plan_targets(model, "attn.to_qkv") == (
        ("attn.to_q", 3072), ("attn.to_k", 3072), ("attn.to_v", 3072))
    with pytest.raises(SvdqLayoutError, match="neither a Linear"):
        native.plan_targets(model, "attn.nonesuch")


def test_swap_splits_a_fused_qkv_into_three_dense_linears() -> None:
    """The full native path in the any-hardware mode: one fused nunchaku
    ``to_qkv`` becomes three working diffusers projections."""
    dim = 3072
    model = _Block(dim)
    tensors, _ = _synth_nunchaku_linear(dim * 3, dim, second_key="wcscales",
                                        rank=128, seed=7)
    dec = decode_linear(tensors, dim * 3, dim)

    counts = native.swap_svdq_linears(
        model, {"attn.to_qkv": dec}, compute_dtype=torch.float32, mode="dense")
    assert counts == {"blockwise": 0, "dense": 3, "prefixes": 1, "linears": 3}

    torch.manual_seed(8)
    x = torch.randn(2, dim)
    fused_ref = (
        (x / dec.smooth_factor.float()) @ dequantize_decoded(dec).t()
        + (x @ dec.proj_down.float()) @ dec.proj_up.float().t()
        + dec.bias.float())
    got = torch.cat([model.attn.to_q(x), model.attn.to_k(x),
                     model.attn.to_v(x)], dim=-1)
    rel = ((got - fused_ref).norm() / fused_ref.norm()).item()
    assert rel < 1e-5, rel


def test_swap_refuses_a_shape_mismatch_rather_than_corrupting() -> None:
    model = _Block(3072)
    tensors, _ = _synth_nunchaku_linear(3072, 3072, second_key="wtscale",
                                        rank=0, seed=9)
    dec = decode_linear(tensors, 3072, 3072)
    # A single 3072-row linear cannot fill a fused q/k/v triple.
    with pytest.raises(SvdqLayoutError, match="sum to"):
        native.swap_svdq_linears(model, {"attn.to_qkv": dec}, mode="dense")


# --- engine selection -----------------------------------------------------


def test_native_sm_window_is_blackwell_only() -> None:
    """torch's own nvfp4 gate admits sm_89/sm_90, but neither has fp4 tensor
    cores. The engine window is Blackwell, and nothing emulates."""
    assert native.SVDQ_NATIVE_FP4_SMS == (100, 103, 120, 121)
    for sm in (75, 80, 86, 89, 90, 99):
        assert not native.svdq_native_sm_supported(sm)
    for sm in (100, 103, 120, 121):
        assert native.svdq_native_sm_supported(sm)


def test_int4_has_no_native_engine() -> None:
    assert svdq_mod.svdq_engine_candidates("int4") == ("nunchaku",)
    assert svdq_mod.svdq_engine_candidates("fp4") == ("native", "nunchaku")
    engine, reasons = svdq_mod.select_svdq_engine("int4")
    assert engine != "native"
    engine, reasons = svdq_mod.select_svdq_engine("int4", override="native")
    assert engine == "" and "nvfp4 only" in reasons["native"]


def test_native_is_preferred_for_fp4_when_it_can_serve(monkeypatch) -> None:
    """Native first: no nunchaku wheel, no diffusers window, no pin matrix."""
    monkeypatch.setattr(native, "svdq_native_reason", lambda: None)
    engine, reasons = svdq_mod.select_svdq_engine("fp4")
    assert engine == "native", reasons


def test_nunchaku_is_used_when_native_cannot_serve(monkeypatch) -> None:
    monkeypatch.setattr(native, "svdq_native_reason", lambda: "sm_89, no fp4")
    monkeypatch.setattr(svdq_mod, "svdq_stack_reason", lambda: None)
    engine, reasons = svdq_mod.select_svdq_engine("fp4")
    assert engine == "nunchaku"
    assert reasons["native"] == "sm_89, no fp4"


def test_no_engine_reports_every_closed_door(monkeypatch) -> None:
    monkeypatch.setattr(native, "svdq_native_reason", lambda: "no fp4 silicon")
    monkeypatch.setattr(svdq_mod, "svdq_stack_reason",
                        lambda: "nunchaku is not installed")
    engine, reasons = svdq_mod.select_svdq_engine("fp4")
    assert engine == ""
    assert reasons == {"native": "no fp4 silicon",
                       "nunchaku": "nunchaku is not installed"}


def test_explicit_override_is_strict_never_substituted(monkeypatch) -> None:
    """An operator who pins nunchaku gets nunchaku or an error — never a
    silent switch to the other engine."""
    monkeypatch.setattr(native, "svdq_native_reason", lambda: None)
    monkeypatch.setattr(svdq_mod, "svdq_stack_reason",
                        lambda: "nunchaku is not installed")
    engine, reasons = svdq_mod.select_svdq_engine("fp4", override="nunchaku")
    assert engine == ""
    assert reasons == {"nunchaku": "nunchaku is not installed"}
    with pytest.raises(svdq_mod.SvdqError, match="unknown svdq engine"):
        svdq_mod.select_svdq_engine("fp4", override="cutlass")


def test_engine_env_override_is_validated(monkeypatch) -> None:
    monkeypatch.setenv("GEN_WORKER_SVDQ_ENGINE", "native")
    assert svdq_mod.svdq_engine_override() == "native"
    monkeypatch.setenv("GEN_WORKER_SVDQ_ENGINE", "nunchaku")
    assert svdq_mod.svdq_engine_override() == "nunchaku"
    monkeypatch.setenv("GEN_WORKER_SVDQ_ENGINE", "trt")
    with pytest.raises(svdq_mod.SvdqError, match="not a known svdq engine"):
        svdq_mod.svdq_engine_override()
    monkeypatch.delenv("GEN_WORKER_SVDQ_ENGINE")
    assert svdq_mod.svdq_engine_override() == ""


# --- the fp4 module (CUDA) ------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_svdq_linear_forward_matches_the_folded_reference() -> None:
    """SvdqLinear's two-branch fp4 forward vs the dense fold of the same
    checkpoint — the fp4 path may only differ by activation-quant error."""
    if not native.svdq_native_available():
        pytest.skip("no fp4 tensor cores / lane not armed here")
    out_f, in_f, rank = 3072, 3072, 128
    tensors, _ = _synth_nunchaku_linear(out_f, in_f, second_key="wcscales",
                                       rank=rank, seed=11)
    dec = decode_linear(tensors, out_f, in_f)
    from gen_worker.models.svdq_layout import to_buffers

    mod = native.build_svdq_linear(to_buffers(dec), device="cuda")
    ref = native.fold_to_dense(dec, compute_dtype=torch.float32).cuda()

    torch.manual_seed(12)
    x = torch.randn(256, in_f, device="cuda", dtype=torch.bfloat16)
    got = mod(x)
    want = ref(x.float())
    rel = ((got.float() - want).norm() / want.norm()).item()
    # 4-bit ACTIVATION quant on top of the (already-folded) 4-bit weights.
    assert rel < 0.2, rel
