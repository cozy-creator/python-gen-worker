from __future__ import annotations

import re

import pytest

torch = pytest.importorskip("torch")

from gen_worker.models import svdq_fused  # noqa: E402
from gen_worker.models.nvfp4_quant import (  # noqa: E402
    E2M1_MAX,
    FP8_MAX,
    quantize_activation_torch,
    to_blocked_scales,
)


def _require_triton():
    return pytest.importorskip("triton")


def test_reference_flat_scales_agree_with_pgw685_chain() -> None:
    """The fused lane's FLAT scales are the same numbers the proven chain swizzles into the cuBLAS blocked layout — one quantization, two layouts."""
    torch.manual_seed(0)
    for m, k in ((32, 256), (77, 512)):
        xs = torch.randn(m, k, dtype=torch.bfloat16)
        s2 = (xs.abs().amax().float() / (E2M1_MAX * FP8_MAX)).clamp(min=1e-12)
        q_flat, s_flat = svdq_fused._reference_quant_flat(xs, s2)
        q_ref, s_blocked = quantize_activation_torch(xs, s2)
        assert torch.equal(q_flat, q_ref)
        assert torch.equal(to_blocked_scales(s_flat).view(torch.uint8),
                           s_blocked.view(torch.uint8))


def test_dyn_s2_matches_baseline_formula() -> None:
    """Column-amax-then-divide == divide-then-global-amax under bf16 rounding (monotone, sign-symmetric) — the fused lane's s2 is the baseline lane's s2, bit for bit."""
    torch.manual_seed(1)
    for m, k in ((64, 256), (333, 1024)):
        x = torch.randn(m, k, dtype=torch.bfloat16) * 3
        smooth = (torch.rand(k, dtype=torch.bfloat16) + 0.5)
        want = ((x / smooth).abs().amax().float()
                / (E2M1_MAX * FP8_MAX)).clamp(min=1e-12)
        got = svdq_fused._dyn_s2(x, smooth)
        assert torch.equal(got, want)
        want_ns = (x.abs().amax().float()
                   / (E2M1_MAX * FP8_MAX)).clamp(min=1e-12)
        assert torch.equal(svdq_fused._dyn_s2(x, None), want_ns)


def _get_jit_fns() -> dict:
    import gc

    from triton.runtime.autotuner import Autotuner
    from triton.runtime.jit import JITFunction

    assert svdq_fused.fused_ops() is not None
    out = {}
    for obj in gc.get_objects():
        try:
            fn = obj.fn if isinstance(obj, Autotuner) else (
                obj if isinstance(obj, JITFunction) else None)
            if fn is not None and getattr(fn, "__name__", "") in (
                    "_quant_smooth_kernel", "_gemm_lora_kernel"):
                out[fn.__name__] = fn
        except ReferenceError:
            continue
    return out


@pytest.mark.parametrize("cap,native_marker", [
    (120, r"kind::mxf4nvf4"),
    (100, r"tcgen05"),
])
def test_fused_gemm_compiles_to_native_block_scaled_mma(
    cap: int, native_marker: str,
) -> None:
    _require_triton()
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.compiler import compile as tt_compile

    fns = _get_jit_fns()
    src = ASTSource(
        fn=fns["_gemm_lora_kernel"],
        signature={"qa_ptr": "*u8", "sa_ptr": "*fp8e4nv", "qb_ptr": "*u8",
                   "sb_ptr": "*fp8e4nv", "s2_ptr": "*fp32",
                   "sec_ptr": "*fp32", "la_ptr": "*bf16", "up_ptr": "*bf16",
                   "bias_ptr": "*bf16", "out_ptr": "*bf16",
                   "M": "i32", "N": "i32", "K": "i32", "R": "constexpr",
                   "PER_CHANNEL": "constexpr", "HAS_BIAS": "constexpr",
                   "BM": "constexpr", "BN": "constexpr", "BK": "constexpr",
                   "GROUP_M": "constexpr"},
        constexprs={"R": 128, "PER_CHANNEL": True, "HAS_BIAS": True,
                    "BM": 128, "BN": 128, "BK": 128, "GROUP_M": 8},
    )
    ptx = tt_compile(src, target=GPUTarget("cuda", cap, 32)).asm["ptx"]
    assert len(re.findall(native_marker, ptx)) > 0, (
        f"sm_{cap}: no native block-scaled MMA in PTX — dot_scaled fell back")


@pytest.mark.parametrize("cap", [120, 100])
def test_fused_quant_kernel_compiles(cap: int) -> None:
    _require_triton()
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource
    from triton.compiler import compile as tt_compile

    fns = _get_jit_fns()
    src = ASTSource(
        fn=fns["_quant_smooth_kernel"],
        signature={"x_ptr": "*bf16", "sm_ptr": "*bf16", "s2_ptr": "*fp32",
                   "q_ptr": "*u8", "s_ptr": "*fp8e4nv",
                   "K": "i32", "KB": "i32", "NCB": "i32",
                   "HAS_SMOOTH": "constexpr", "BLOCKED": "constexpr",
                   "BPP": "constexpr"},
        constexprs={"HAS_SMOOTH": True, "BLOCKED": True, "BPP": 128},
    )
    ptx = tt_compile(src, target=GPUTarget("cuda", cap, 32)).asm["ptx"]
    assert "e4m3" in ptx
