"""nvfp4 format primitives + the fused triton activation quantizer.

The pure-torch quantize chain (:func:`quantize_activation_torch`) is ~8 passes
over the activation: fp32 upcast, per-16-block amax, e4m3 block scale, divide,
``searchsorted`` e2m1 cast, nibble pack, then a pad/permute swizzle of the
scales into the cuBLAS blocked layout. That chain is the whole reason the w4a4
lane LOST to bf16 (0.807x eager on sm_120) — the fp4 GEMM was
never the problem. :func:`quantize_activation` runs all of it in ONE triton
kernel, writing each block scale directly at its blocked-layout address.

Measured (RTX 5090 sm_120 + B200 sm_100, torch 2.13/triton
3.7.1): the quant step alone 6-36x faster; the stacked-block lane 0.807x ->
2.043x bf16 eager and 2.045x -> 2.472x compiled on sm_120, 1.03x -> 1.24x on
sm_100. Triton JITs per arch, so one source covers sm_100/103 and sm_120/121
with no nvcc extension build and no per-family port.

Two hazards carried deliberately:

- ``tl.math.div_rn``. Triton's ``/`` lowers to an APPROXIMATE divide; a 1-ulp
  drift lands a value exactly on the 1.75 e2m1 tie boundary, and because ties
  round UP a whole 4-bit code flips (measured: 0.16% of nibbles, 0.6% output
  drift). w4a4's load-time numerics self-check cannot see this — its threshold
  is 2e-2.
- Arming is gated on BIT-IDENTITY against the torch chain, never a tolerance.
  A triton release that changes rounding, or an arch we have not measured,
  falls back to the torch chain instead of serving drifted numerics.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Optional
from ..hostfacts import cuda_ready

logger = logging.getLogger(__name__)

E2M1_MAX = 6.0
FP8_MAX = 448.0
SCALE_MIN = 2.0 ** -9  # e4m3 underflow guard (modelopt clamp)
BLOCK = 16

# Work-per-program for the fused kernel. Both arches want the SAME shape —
# 128 blocks (2048 elements) per program; only the warp count differs
# (sweeping this fixed a false "sm_100 is bad at 4-bit" reading: at
# 8 blocks/program the B200 ran 21x off its own bandwidth floor). Halved
# automatically when a shape has fewer k-blocks than this.
_BLOCKS_PER_PROG = 128
_WARPS_BY_SM = {100: 4, 103: 4, 120: 8, 121: 8}
_WARPS_DEFAULT = 4

# Probe shapes for the bit-identity gate: a small one, a row count that is not
# a multiple of 128 (exercises blocked-layout row padding), and a real
# qwen-image FFN width.
_IDENTITY_PROBES = ((128, 512), (333, 3072), (1024, 12288))


# ---------------------------------------------------------------------------
# Format primitives (pure torch) — shared by the producer, the dequant lane,
# and the reference activation-quant chain.
# ---------------------------------------------------------------------------


def _e2m1_lut(device: Any) -> Any:
    import torch

    return torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device=device, dtype=torch.float32)


def _e2m1_bounds(device: Any) -> Any:
    import torch

    return torch.tensor(
        [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
        device=device, dtype=torch.float32)


def cast_e2m1(t: Any) -> Any:
    """Float tensor -> e2m1 nibble codes (uint8, values 0-15). Round-to-
    nearest with ties at the odd bounds (0.75/1.75/2.5) rounding UP —
    byte-identical to modelopt's NVFP4QTensor._cast_fp4."""
    import torch

    v = t.float()
    sign = (v < 0).to(torch.uint8)
    a = v.abs()
    ord_ = torch.searchsorted(
        _e2m1_bounds(v.device), a.contiguous(), out_int32=True).to(torch.uint8)
    tie = ((a == 0.75) | (a == 1.75) | (a == 2.5)).to(torch.uint8)
    return (sign << 3) + ord_ + tie


def pack_e2m1(codes: Any) -> Any:
    """[..., K] nibble codes -> [..., K/2] packed uint8 (element 2j in the
    LOW nibble — torch.float4_e2m1fn_x2 / modelopt convention). Explicitly
    contiguous: cuBLASLt refuses strided fp4 operands
    (CUBLAS_STATUS_NOT_SUPPORTED, found live on B200), and TensorIterator
    may propagate the slice strides into the packed output."""
    return ((codes[..., 1::2] << 4) | codes[..., 0::2]).contiguous()


def unpack_e2m1(packed: Any) -> Any:
    """[..., K/2] packed uint8 -> [..., K] float32 e2m1 values."""
    import torch

    lut = _e2m1_lut(packed.device)
    shape = list(packed.shape)
    shape[-1] = shape[-1] * 2
    out = torch.empty(shape, dtype=torch.float32, device=packed.device)
    out[..., 0::2] = lut[(packed & 0x0F).long()]
    out[..., 1::2] = lut[(packed >> 4).long()]
    return out


def to_blocked_scales(scales: Any) -> Any:
    """Flat per-block scales [rows, k_blocks] (e4m3) -> the cuBLAS 2-D tiled
    ("blocked") layout torch's blockwise scaled_mm consumes: rows padded to
    128, block columns padded to 4, 128x4 tiles rearranged (32, 4, 4) —
    returned as a contiguous 1-D e4m3 tensor (the kernel checks numel +
    contiguity only). Same rearrangement as modelopt's swizzle_nvfp4_scales /
    torchao's to_blocked, and the layout the fused kernel writes directly."""
    import torch

    rows, cols = scales.shape
    nrb = (rows + 127) // 128
    ncb = (cols + 3) // 4
    # fp8 has no pad/zeros kernels everywhere — swizzle a uint8 view.
    raw = scales.view(torch.uint8)
    padded = raw.new_zeros(nrb * 128, ncb * 4)
    padded[:rows, :cols] = raw
    blocks = padded.view(nrb, 128, ncb, 4).permute(0, 2, 1, 3)
    out = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1)
    return out.contiguous().view(torch.float8_e4m3fn)


def blocked_scale_numel(rows: int, k_blocks: int) -> int:
    """Element count of :func:`to_blocked_scales`' output for a [rows,
    k_blocks] scale grid — the fused kernel's output buffer size."""
    return ((rows + 127) // 128) * 128 * ((k_blocks + 3) // 4) * 4


def quantize_activation_torch(x2: Any, s2: Any) -> tuple[Any, Any]:
    """Reference chain: 2-D activation [M, K] + per-tensor second-level scale
    -> (packed uint8 [M, K/2], e4m3 block scales in the cuBLAS blocked
    layout). The numerics the fused kernel must reproduce BIT-EXACTLY."""
    import torch

    in_f = int(x2.shape[1])
    xb = x2.reshape(-1, in_f // BLOCK, BLOCK).float()
    bmax = xb.abs().amax(dim=-1)
    sa = (bmax / (E2M1_MAX * s2)).clamp(
        min=SCALE_MIN, max=FP8_MAX).to(torch.float8_e4m3fn)
    q = xb / (sa.float().unsqueeze(-1) * s2)
    codes = cast_e2m1(q.reshape(-1, in_f))
    return pack_e2m1(codes), to_blocked_scales(sa)


# ---------------------------------------------------------------------------
# The fused kernel, behind a torch.library custom op so it survives
# torch.compile instead of graph-breaking.
# ---------------------------------------------------------------------------

_OP_NAME = "cozy_gen_worker::quant_nvfp4_act"


@functools.lru_cache(maxsize=1)
def _build_fused_op() -> Optional[Any]:
    """Compile the triton kernel and register the custom op. ``None`` when
    triton (or the fp8 cast it needs) is unavailable. Called at LOAD time so
    registration never happens mid-trace."""
    try:
        import torch
        import triton
        import triton.language as tl
    except Exception as exc:  # noqa: BLE001 — no triton => torch chain
        logger.info("nvfp4: fused quantizer unavailable (%s); torch chain", exc)
        return None

    @triton.jit
    def _e2m1_code(q):  # type: ignore[no-untyped-def]
        """RTN to e2m1 nibble codes — the searchsorted+tie path expressed as
        threshold counts (ties at 0.75/1.75/2.5 round UP, sign in bit 3)."""
        a = tl.abs(q)
        code = ((a > 0.25).to(tl.uint8) + (a > 0.75).to(tl.uint8)
                + (a > 1.25).to(tl.uint8) + (a > 1.75).to(tl.uint8)
                + (a > 2.5).to(tl.uint8) + (a > 3.5).to(tl.uint8)
                + (a > 5.0).to(tl.uint8))
        tie = ((a == 0.75) | (a == 1.75) | (a == 2.5)).to(tl.uint8)
        return code + tie + ((q < 0).to(tl.uint8) * 8)

    @triton.jit
    def _quant_nvfp4_kernel(  # type: ignore[no-untyped-def]
        x_ptr, q_ptr, s_ptr, s2_ptr,
        K, KB, NCB,
        stride_xm, stride_xk,
        BLOCKS_PER_PROG: tl.constexpr,
    ):
        """One program = one row x BLOCKS_PER_PROG groups of 16 elements.

        Writes packed nibbles (element 2j in the LOW nibble) and stores the
        e4m3 block scale DIRECTLY in the cuBLAS blocked layout:
          tile    = (row // 128) * NCB + (blk // 4)
          in-tile = (row % 32) * 16 + ((row % 128) // 32) * 4 + (blk % 4)
        """
        row = tl.program_id(0)
        blk0 = tl.program_id(1) * BLOCKS_PER_PROG

        s2 = tl.load(s2_ptr)

        offs_b = blk0 + tl.arange(0, BLOCKS_PER_PROG)
        blk_ok = offs_b < KB
        # Load each block's 16 elements as two [BPP, 8] tiles (even / odd
        # positions) — the nibble pairing the packed layout needs, with no
        # in-kernel reshape.
        offs_lo = offs_b[:, None] * 16 + tl.arange(0, 8)[None, :] * 2
        offs_hi = offs_lo + 1
        base = x_ptr + row * stride_xm
        m2 = blk_ok[:, None]
        xe = tl.load(base + offs_lo * stride_xk, mask=m2, other=0.0).to(tl.float32)
        xo = tl.load(base + offs_hi * stride_xk, mask=m2, other=0.0).to(tl.float32)

        amax = tl.maximum(tl.max(tl.abs(xe), axis=1), tl.max(tl.abs(xo), axis=1))
        scale = amax / (6.0 * s2)
        scale = tl.minimum(tl.maximum(scale, 0.001953125), 448.0)
        scale_f8 = scale.to(tl.float8e4nv)  # RTN to e4m3, as the torch chain does
        denom = (scale_f8.to(tl.float32) * s2)[:, None]

        # IEEE round-to-nearest divide. Triton's `/` is an approximate divide;
        # a 1-ulp drift lands values exactly ON the 1.75 tie boundary and the
        # ties-round-UP rule then flips a whole e2m1 code.
        lo = _e2m1_code(tl.math.div_rn(xe, denom))
        hi = _e2m1_code(tl.math.div_rn(xo, denom))
        packed = lo | (hi << 4)

        offs_p = offs_b[:, None] * 8 + tl.arange(0, 8)[None, :]
        tl.store(q_ptr + row * (K // 2) + offs_p, packed, mask=m2)

        rb = row // 128
        rr = row % 128
        tile = rb * NCB + (offs_b // 4)
        in_tile = (rr % 32) * 16 + (rr // 32) * 4 + (offs_b % 4)
        tl.store(s_ptr + tile * 512 + in_tile, scale_f8, mask=blk_ok)

    def _launch(x2: Any, s2: Any) -> tuple[Any, Any]:
        m, k = int(x2.shape[0]), int(x2.shape[1])
        kb = k // BLOCK
        ncb = (kb + 3) // 4
        q = torch.empty(m, k // 2, dtype=torch.uint8, device=x2.device)
        # Padding rows/columns are never written — the buffer must start zeroed.
        s = torch.zeros(blocked_scale_numel(m, kb),
                        dtype=torch.float8_e4m3fn, device=x2.device)
        bpp = min(_BLOCKS_PER_PROG, kb)
        while kb % bpp and bpp > 1:
            bpp //= 2
        _quant_nvfp4_kernel[(m, (kb + bpp - 1) // bpp)](
            x2, q, s, s2, k, kb, ncb,
            x2.stride(0), x2.stride(1), BLOCKS_PER_PROG=bpp,
            num_warps=_quant_warps())
        return q, s

    # Explicit schema: the op must not depend on annotation inference (torch
    # is imported lazily in this module, so string annotations would not
    # resolve). mutates_args=() — both outputs are freshly allocated.
    op = torch.library.custom_op(
        _OP_NAME, _launch, mutates_args=(),
        schema="(Tensor x2, Tensor s2) -> (Tensor, Tensor)")

    @op.register_fake
    def _(x2: Any, s2: Any) -> tuple[Any, Any]:
        m, k = int(x2.shape[0]), int(x2.shape[1])
        return (x2.new_empty(m, k // 2, dtype=torch.uint8),
                x2.new_empty(blocked_scale_numel(m, k // BLOCK),
                             dtype=torch.float8_e4m3fn))

    return op


def _quant_warps() -> int:
    try:
        import torch

        major, minor = torch.cuda.get_device_capability()
        return _WARPS_BY_SM.get(major * 10 + minor, _WARPS_DEFAULT)
    except Exception:  # noqa: BLE001
        return _WARPS_DEFAULT


def _bit_identical(op: Any) -> bool:
    """The arming gate: fused output must match the torch chain BYTE for byte
    on every probe. A tolerance check cannot see the div_rn class of bug — one
    flipped nibble in 600 is 0.6% output drift and reads as noise."""
    import torch

    for m, k in _IDENTITY_PROBES:
        torch.manual_seed(0)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        s2 = (x.abs().amax().float() / (E2M1_MAX * FP8_MAX)).clamp(min=1e-12)
        want_q, want_s = quantize_activation_torch(x, s2)
        got_q, got_s = op(x, s2)
        if not torch.equal(got_q, want_q):
            bad = int((got_q != want_q).sum())
            logger.warning(
                "nvfp4: fused quantizer differs from the torch chain at "
                "%dx%d (%d/%d packed bytes); torch chain",
                m, k, bad, want_q.numel())
            return False
        if not torch.equal(got_s.view(torch.uint8), want_s.view(torch.uint8)):
            logger.warning(
                "nvfp4: fused quantizer block scales differ from the torch "
                "chain at %dx%d; torch chain", m, k)
            return False
    return True


# Arming result, resolved once per process. Deliberately PLAIN globals rather
# than an lru_cache call: the hot path reads them under torch.compile, where a
# global str/callable lookup constant-folds but a C-implemented lru_cache
# wrapper can break the graph.
_QUANT_MODE: Optional[str] = None
_ARMED_OP: Any = None


def _arm() -> str:
    """Decide the quantizer for THIS device (idempotent). Fused arms only when
    triton compiles the kernel AND it is bit-identical to the reference chain
    on every probe shape."""
    global _QUANT_MODE, _ARMED_OP

    if _QUANT_MODE is not None:
        return _QUANT_MODE
    mode, op = "torch", None
    try:
        if cuda_ready():
            candidate = _build_fused_op()
            if candidate is not None and _bit_identical(candidate):
                mode, op = "fused", candidate
                logger.info(
                    "nvfp4: fused triton activation quantizer armed "
                    "(%d blocks/program, %d warps)",
                    _BLOCKS_PER_PROG, _quant_warps())
    except Exception as exc:  # noqa: BLE001 — any gap => reference chain
        logger.warning("nvfp4: fused quantizer probe failed (%s); torch chain",
                       exc)
        mode, op = "torch", None
    _QUANT_MODE, _ARMED_OP = mode, op
    return mode


def nvfp4_quantizer_mode() -> str:
    """``"fused"`` | ``"torch"`` for this device. Call at LOAD time — it
    compiles the kernel and registers the custom op, which must not happen
    mid-trace."""
    return _arm()


def reset_nvfp4_quantizer_arming() -> None:
    """Forget the arming decision (tests only)."""
    global _QUANT_MODE, _ARMED_OP

    _QUANT_MODE, _ARMED_OP = None, None


def quantize_activation(x2: Any, s2: Any) -> tuple[Any, Any]:
    """2-D activation + per-tensor second-level scale -> (packed fp4 uint8
    [M, K/2], e4m3 block scales in the cuBLAS blocked layout). Uses the fused
    triton kernel when it armed for this device, else the reference chain.
    Compile-safe: the fused path is a registered custom op.

    ``s2`` is coerced to a 0-dim fp32 tensor — the kernel loads it as fp32, so
    a bf16 scalar would be read as garbage bits rather than upcast."""
    s2 = s2.reshape(()).float()
    if _QUANT_MODE is None:
        _arm()
    op = _ARMED_OP
    if op is not None:
        return op(x2, s2)
    return quantize_activation_torch(x2, s2)


__all__ = [
    "BLOCK",
    "E2M1_MAX",
    "FP8_MAX",
    "SCALE_MIN",
    "blocked_scale_numel",
    "cast_e2m1",
    "nvfp4_quantizer_mode",
    "pack_e2m1",
    "quantize_activation",
    "quantize_activation_torch",
    "reset_nvfp4_quantizer_arming",
    "to_blocked_scales",
    "unpack_e2m1",
]
