"""pgw#685 — the fused triton nvfp4 activation quantizer in the w4a4 lane.

The CPU tests pin the reference chain, the blocked-layout geometry, and the
fallback contract (no CUDA / no triton => the pure-torch chain still serves).
The CUDA tests are the ones that matter for the kernel itself: BIT-IDENTITY
against the reference chain (a tolerance check cannot see the ``div_rn`` class
of bug — pgw#682 G-A measured a 1-ulp divide drift flipping 0.16% of nibbles
outright) and survival under ``torch.compile``. They run in the GPU lane.
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")

from gen_worker.models.nvfp4_quant import (  # noqa: E402
    BLOCK,
    E2M1_MAX,
    FP8_MAX,
    blocked_scale_numel,
    nvfp4_quantizer_mode,
    quantize_activation,
    quantize_activation_torch,
    reset_nvfp4_quantizer_arming,
    to_blocked_scales,
    unpack_e2m1,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device")


def _second_level(x: Any) -> Any:
    return (x.abs().amax().float() / (E2M1_MAX * FP8_MAX)).clamp(min=1e-12)


# --- geometry / reference chain (CPU) --------------------------------------


@pytest.mark.parametrize("rows,k_blocks", [
    (128, 4), (128, 32), (333, 192), (1, 1), (4096, 768), (129, 5)])
def test_blocked_scale_numel_agrees_with_the_swizzle(rows: int,
                                                     k_blocks: int) -> None:
    """The fused kernel allocates its scale buffer from
    :func:`blocked_scale_numel`; it must match what the pure-torch swizzle
    produces for the same grid, padding included."""
    flat = torch.zeros(rows, k_blocks).to(torch.float8_e4m3fn)
    assert to_blocked_scales(flat).numel() == blocked_scale_numel(rows, k_blocks)


def test_reference_chain_shapes_and_dequant_error() -> None:
    """The reference chain's packed weights + blocked scales round-trip back
    to the input within e2m1 block-quantization error."""
    torch.manual_seed(0)
    m, k = 333, 3072
    x = torch.randn(m, k, dtype=torch.bfloat16)
    s2 = _second_level(x)
    packed, blocked = quantize_activation_torch(x, s2)

    assert packed.shape == (m, k // 2)
    assert packed.dtype == torch.uint8
    assert blocked.numel() == blocked_scale_numel(m, k // BLOCK)
    assert blocked.dtype == torch.float8_e4m3fn

    # Recover per-block scales from the un-swizzled chain to dequantize.
    xb = x.reshape(-1, k // BLOCK, BLOCK).float()
    sa = (xb.abs().amax(dim=-1) / (E2M1_MAX * s2)).clamp(
        min=2.0 ** -9, max=FP8_MAX).to(torch.float8_e4m3fn)
    deq = (unpack_e2m1(packed).reshape(m, k // BLOCK, BLOCK)
           * sa.float().unsqueeze(-1) * s2).reshape(m, k)
    rel = ((deq - x.float()).norm() / x.float().norm()).item()
    # e2m1 has 3 mantissa levels per binade — ~10% relative is the format's
    # own floor, not slack in the implementation.
    assert rel < 0.15, rel


def test_low_nibble_holds_the_even_element() -> None:
    """The packed convention (element 2j in the LOW nibble) is what
    ``torch.float4_e2m1fn_x2`` and the tensor-layout contract assume; getting it
    backwards silently transposes every pair."""
    x = torch.zeros(1, 32)
    x[0, 0] = 6.0   # even position -> low nibble, code 7
    x[0, 1] = 0.0   # odd position -> high nibble, code 0
    packed, _ = quantize_activation_torch(x, torch.tensor(6.0 / (E2M1_MAX * 1.0)))
    assert int(packed[0, 0]) & 0x0F == 7
    assert int(packed[0, 0]) >> 4 == 0


def test_quantizer_mode_is_torch_without_cuda(monkeypatch) -> None:
    """No CUDA (CI, CPU boxes) => the reference chain, never a hard failure."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    reset_nvfp4_quantizer_arming()
    try:
        assert nvfp4_quantizer_mode() == "torch"
    finally:
        reset_nvfp4_quantizer_arming()


def test_quantize_activation_dispatches_to_the_reference_chain_on_cpu(
    monkeypatch,
) -> None:
    """The dispatcher's fallback is the reference chain BYTE for byte."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    reset_nvfp4_quantizer_arming()
    try:
        torch.manual_seed(0)
        x = torch.randn(64, 512, dtype=torch.bfloat16)
        s2 = _second_level(x)
        got_q, got_s = quantize_activation(x, s2)
        want_q, want_s = quantize_activation_torch(x, s2)
        assert torch.equal(got_q, want_q)
        assert torch.equal(got_s.view(torch.uint8), want_s.view(torch.uint8))
    finally:
        reset_nvfp4_quantizer_arming()


# --- the fused kernel (CUDA) ----------------------------------------------


@requires_cuda
@pytest.mark.parametrize("m,k", [(128, 512), (333, 3072), (1024, 12288)])
def test_fused_quantizer_is_bit_identical_to_the_reference_chain(
    m: int, k: int,
) -> None:
    """pgw#682 G-A, as a standing test: packed nibbles AND scale bytes must
    match exactly. Skipped (not failed) where triton cannot arm."""
    from gen_worker.models.nvfp4_quant import _build_fused_op

    op = _build_fused_op()
    if op is None:
        pytest.skip("triton unavailable — the reference chain serves")
    torch.manual_seed(0)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    s2 = _second_level(x)
    got_q, got_s = op(x, s2)
    want_q, want_s = quantize_activation_torch(x, s2)
    mismatched = int((got_q != want_q).sum())
    assert mismatched == 0, f"{mismatched}/{want_q.numel()} packed bytes differ"
    assert torch.equal(got_s.view(torch.uint8), want_s.view(torch.uint8))


@requires_cuda
def test_fused_quantizer_survives_torch_compile() -> None:
    """The custom op must be traced as an opaque call, not graph-break and not
    get decomposed — that is the whole reason it is a ``custom_op`` with a
    ``register_fake`` rather than a raw triton call."""
    from gen_worker.models.nvfp4_quant import _build_fused_op

    op = _build_fused_op()
    if op is None:
        pytest.skip("triton unavailable — the reference chain serves")

    def f(x: Any, s2: Any) -> Any:
        q, s = quantize_activation(x, s2)
        return q.int().sum() + s.view(torch.uint8).int().sum()

    torch.manual_seed(0)
    x = torch.randn(256, 3072, device="cuda", dtype=torch.bfloat16)
    s2 = _second_level(x)
    eager = f(x, s2)
    compiled = torch.compile(f, fullgraph=True)(x, s2)
    assert torch.equal(eager, compiled)
