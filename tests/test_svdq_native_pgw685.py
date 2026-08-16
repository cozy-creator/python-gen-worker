"""pgw#685 — the NATIVE svdq engine: layout converter, SvdqLinear, engine choice.

Integration-style over the real code paths: the fixtures synthesize a checkpoint
in nunchaku's ACTUAL v1 layout (via the same fragment/lane packers the decoder
inverts), then drive `decode_linear` -> `split_decoded` -> `swap_svdq_linears`
into a module with diffusers-style names. No nunchaku wheel, no 13 GB artifact,
no GPU: the fp4 GEMM is the only GPU-only piece and it is gated.
"""

from __future__ import annotations

from pathlib import Path
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

    # EVERY tensor is fragment-packed on disk, so the fixture packs
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
    assert counts == {"blockwise": 0, "dense": 3, "fused": 0, "prefixes": 1,
                      "linears": 3}

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


# pgw#1298 deleted the engine LADDER along with the nunchaku engine, so the
# five selection tests that used to live here (candidate order, native
# preference, nunchaku fallback, every-closed-door reasons, strict `override=`)
# are gone with their subject. What replaces them is the one behaviour a
# one-engine module still has: which artifacts it refuses, and how.
#
# The th#1887 `GEN_WORKER_SVDQ_ENGINE`-is-inert test also died here — its
# subject was `select_svdq_engine`. The env is still fenced repo-wide by
# `tests/test_behaviour_gate_visitor_th1887.py:133`, which asserts the name
# appears in no behaviour gate anywhere; that fence is stronger than the one
# deleted and needs no svdq-specific companion.


def _art(precision: str) -> Any:
    from gen_worker.models.svdq import SvdqArtifact

    return SvdqArtifact(component="transformer", file=Path("m.safetensors"),
                        model_class="NunchakuQwenImageTransformer2DModel",
                        precision=precision, rank=128)


def test_int4_is_a_typed_refusal_on_the_detected_artifact() -> None:
    """int4 has no native engine and no other engine is installed, so it is
    refused BY NAME on the artifact — not by failing to find a loader.

    Behaviourally this is what already happened (the nunchaku lane died on
    "nunchaku is not installed"); what changed is that the message now names
    the real reason, and the type is specific enough to catch."""
    with pytest.raises(svdq_mod.SvdqInt4Unsupported) as exc:
        svdq_mod.check_svdq_servable(_art("int4"), "some/flavor")
    msg = str(exc.value)
    assert "svdq-int4" in msg
    assert "no native implementation" in msg
    assert "some/flavor" in msg
    # The refusal must not depend on the host: no CUDA probe, no importlib
    # metadata, nothing that could make it pass on a different machine.
    assert issubclass(svdq_mod.SvdqInt4Unsupported, svdq_mod.SvdqError)


def test_int4_is_refused_before_any_native_probe(monkeypatch) -> None:
    """The int4 arm must precede the silicon check, so an int4 artifact on a
    Blackwell card still gets the int4 message rather than a hardware one."""
    monkeypatch.setattr(native, "svdq_native_reason", lambda: None)
    with pytest.raises(svdq_mod.SvdqInt4Unsupported):
        svdq_mod.check_svdq_servable(_art("int4"))


def test_fp4_passes_when_the_native_engine_can_serve(monkeypatch) -> None:
    monkeypatch.setattr(native, "svdq_native_reason", lambda: None)
    assert svdq_mod.check_svdq_servable(_art("fp4")) is None


def test_fp4_carries_the_native_engine_reason_when_it_cannot(monkeypatch) -> None:
    """One engine, one reason — and it is the NATIVE one, named verbatim, so
    an operator reads about silicon rather than about a missing wheel."""
    monkeypatch.setattr(native, "svdq_native_reason", lambda: "this GPU is sm_89")
    with pytest.raises(svdq_mod.SvdqHardwareError, match="this GPU is sm_89"):
        svdq_mod.check_svdq_servable(_art("fp4"))


def test_the_nunchaku_engine_is_gone() -> None:
    """pgw#1298: the deletion, asserted rather than described. Nothing may
    reintroduce a second engine here without this test noticing."""
    for gone in ("load_svdq_nunchaku_pipeline", "check_svdq_stack_versions",
                 "svdq_stack_reason", "svdq_precision_for_sm", "SvdqPin",
                 "SvdqStackError", "SVDQ_ENGINES", "svdq_engine_candidates",
                 "select_svdq_engine", "_PIN_MATRIX"):
        assert not hasattr(svdq_mod, gone), f"{gone} came back"
    # `load_svdq_pipeline` takes no engine pin: there is nothing to pin.
    import inspect

    params = inspect.signature(svdq_mod.load_svdq_pipeline).parameters
    assert "engine" not in params


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
