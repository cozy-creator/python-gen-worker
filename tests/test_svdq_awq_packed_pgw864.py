"""pgw#864 — packed-resident AWQ W4A16 modulation, box-side proofs.

The resident buffers must dequantize to EXACTLY the weight the decode path
builds (same bytes, same adanorm undo, same bias delta); the kernel itself is
GPU-gated and rides the pgw#865 harness + the arming self-check."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from gen_worker.models import svdq_awq_packed as pk  # noqa: E402
from gen_worker.models.svdq_awq import decode_awq_linear  # noqa: E402
from tests.harness.svdq_awq_encode import encode_awq_linear  # noqa: E402


def _synth(oc: int, ic: int, splits: int, seed: int = 0):
    gen = torch.Generator().manual_seed(seed)
    w = torch.randn(oc, ic, generator=gen) * 0.05
    b = torch.randn(oc, generator=gen)
    return encode_awq_linear(w, b, adanorm_splits=splits), w, b


@pytest.mark.parametrize("oc,ic,splits", [
    (512, 256, 1),
    (768, 256, 6),      # qwen img_mod/txt_mod shape class (adanorm)
    (18432, 3072, 6),   # the real qwen modulation shape
])
def test_packed_buffers_dequantize_to_the_decoded_weight(
    oc: int, ic: int, splits: int,
) -> None:
    if pk.awq_op() is None:
        pytest.skip("triton unavailable")
    tensors, _w, _b = _synth(oc, ic, splits)
    ref = decode_awq_linear(tensors, oc, ic, adanorm_splits=splits)
    mod = pk.build_awq_packed_linear(tensors, oc, ic, adanorm_splits=splits)

    # Reconstruct the weight the kernel computes: nibble unpack -> per-group
    # dequant -> bf16 round.
    codes = torch.empty(oc, ic, dtype=torch.uint8)
    codes[:, 0::2] = mod.weight & 0xF
    codes[:, 1::2] = mod.weight >> 4
    ng = ic // pk.GROUP_SIZE
    w = (codes.float().reshape(oc, ng, pk.GROUP_SIZE)
         * mod.wscales.float().unsqueeze(-1)
         + mod.wzeros.float().unsqueeze(-1)).reshape(oc, ic)
    assert torch.equal(w.to(torch.bfloat16), ref.weight.detach())
    assert torch.equal(mod.bias.detach(), ref.bias.detach())


def test_packed_resident_bytes_are_quarter_of_bf16() -> None:
    if pk.awq_op() is None:
        pytest.skip("triton unavailable")
    oc, ic, splits = 18432, 3072, 6
    tensors, _w, _b = _synth(oc, ic, splits)
    mod = pk.build_awq_packed_linear(tensors, oc, ic, adanorm_splits=splits)
    resident = sum(t.numel() * t.element_size()
                   for t in (mod.weight, mod.wscales, mod.wzeros,
                             mod.bias.data))
    dense = oc * ic * 2 + oc * 2
    assert resident < dense * 0.30  # 4-bit + group scales ≈ 28% of bf16


def test_group_size_mismatch_refuses_typed() -> None:
    if pk.awq_op() is None:
        pytest.skip("triton unavailable")
    tensors, _w, _b = _synth(512, 256, 1)
    t32 = encode_awq_linear(
        torch.randn(512, 256) * 0.05, torch.randn(512), group_size=32)
    with pytest.raises(pk.AwqPackedError):
        pk.build_awq_packed_linear(t32, 512, 256)
    assert not pk.awq_packed_supported(512, 200)  # ic % 128
    assert not pk.awq_packed_supported(500, 256)  # oc % 16


def test_awq_kernel_compiles_for_blackwell() -> None:
    pytest.importorskip("triton")
    import gc
    import re

    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.compiler import compile as tt_compile
    from triton.runtime.autotuner import Autotuner

    assert pk.awq_op() is not None
    fn = None
    for obj in gc.get_objects():
        try:
            if not isinstance(obj, Autotuner):
                continue
            candidate = obj.fn
            if getattr(candidate, "__name__", "") == "_awq_mm_kernel":
                fn = candidate
                break
        except ReferenceError:
            # gc.get_objects() can contain an expired weak proxy. It is not a
            # kernel candidate, and dereferencing it must not flake this proof.
            continue
    assert fn is not None
    src = ASTSource(
        fn=fn,
        signature={"x_ptr": "*bf16", "w_ptr": "*u8", "s_ptr": "*bf16",
                   "z_ptr": "*bf16", "b_ptr": "*bf16", "out_ptr": "*bf16",
                   "M": "i32", "N": "i32", "K": "i32", "NG": "i32",
                   "GS": "constexpr", "HAS_BIAS": "constexpr",
                   "BN": "constexpr", "BK": "constexpr"},
        constexprs={"GS": 64, "HAS_BIAS": True, "BN": 128, "BK": 128},
    )
    for cap in (120, 100):
        ptx = tt_compile(src, target=GPUTarget("cuda", cap, 32)).asm["ptx"]
        assert re.search(r"\.target sm_1[02]0a", ptx)
