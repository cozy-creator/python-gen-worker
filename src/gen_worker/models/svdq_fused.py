"""Fused native execution lane for SvdqLinear (pgw#862 phase B0).

The serving path is the HYBRID settled by the first 5090 iteration (evidence
in pgw#862): the baseline's ~8-op unfused chain per unit becomes

  1. ``svdq_quant`` — one pass over RAW x: smooth-divide fused in-register
     (bf16-rounded to match the aten reference) + per-16-block e4m3
     quantization BIT-IDENTICAL to the pgw#685 chain, storing scales in
     either the cuBLAS blocked layout (serving) or plain flat [M, K/16].
  2. the PROVEN cuBLAS block-scaled fp4 GEMM (``_gemm_w4a4`` — measured
     107us vs 265us for the best pure-triton ``tl.dot_scaled`` tile at qwen
     shapes on sm_120) with ``lora_down`` as a plain cuBLAS mm beside it.
  3. ``svdq_epilogue_lora`` — ONE pass over the GEMM output: second-level
     scale + ``lora_act @ proj_up.T`` + bias, fp32 accumulate, single bf16
     writeback (replaces the baseline's separate mul/mm/add/add chain).

``svdq_gemm_w4a4_lora`` (pure-triton block-scaled MMA GEMM with the low-rank
epilogue fused; native ``kind::mxf4nvf4`` on sm_120a, tcgen05 on sm_100a) is
kept registered as the measured alternative and the sm_100 seed for pgw#863,
but is NOT the serving path on sm_120.

Numerics vs the baseline lane: activation quantization is bit-identical by
construction (same formulas, ``div_rn``, ties-up e2m1, same s2 — the arming
self-check enforces it), and the GEMM is literally the same kernel; the
epilogue accumulates in fp32 with ONE final bf16 round where the baseline
rounds at every op boundary. Divergence is quantified per shape on the
pgw#865 harness, never assumed.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Optional

from .nvfp4_quant import BLOCK, E2M1_MAX, FP8_MAX, SCALE_MIN

logger = logging.getLogger(__name__)

_QUANT_OP = "cozy_gen_worker::svdq_quant"
_GEMM_OP = "cozy_gen_worker::svdq_gemm_w4a4_lora"
_EPI_OP = "cozy_gen_worker::svdq_epilogue_lora"

# The fused lane's operand contract. K alignment comes from the qweight
# fragment tile (every real checkpoint is in%128); rank from the mma tile.
_K_ALIGN = 128
_N_ALIGN = 16
_RANK_ALIGN = 16

_SELF_CHECK_PROBES = ((128, 512, 256), (77, 3072, 384))


class SvdqFusedError(RuntimeError):
    """Typed fused-lane failure (contract mismatch — never silent-wrong)."""


def fused_shape_supported(out_features: int, in_features: int,
                          rank: int) -> bool:
    return (in_features % _K_ALIGN == 0 and out_features % _N_ALIGN == 0
            and rank > 0 and rank % _RANK_ALIGN == 0)


# ---------------------------------------------------------------------------
# Kernels. Built lazily, registered as custom ops (compile-safe).
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _build_fused_ops() -> Optional[tuple[Any, Any, Any]]:
    """Compile-register both fused ops. ``None`` when triton is unavailable."""
    try:
        import torch
        import triton
        import triton.language as tl
    except Exception as exc:  # noqa: BLE001 — no triton => no fused lane
        logger.info("svdq_fused: triton unavailable (%s)", exc)
        return None

    @triton.jit
    def _e2m1_code(q):  # type: ignore[no-untyped-def]
        """RTN e2m1 nibble codes, ties at 0.75/1.75/2.5 round UP — identical
        to nvfp4_quant's kernel (modelopt convention)."""
        a = tl.abs(q)
        code = ((a > 0.25).to(tl.uint8) + (a > 0.75).to(tl.uint8)
                + (a > 1.25).to(tl.uint8) + (a > 1.75).to(tl.uint8)
                + (a > 2.5).to(tl.uint8) + (a > 3.5).to(tl.uint8)
                + (a > 5.0).to(tl.uint8))
        tie = ((a == 0.75) | (a == 1.75) | (a == 2.5)).to(tl.uint8)
        return code + tie + ((q < 0).to(tl.uint8) * 8)

    def _prune_k(configs, named_args, **kwargs):  # type: ignore[no-untyped-def]
        k = named_args["K"]
        keep = [c for c in configs if k % c.kwargs["BK"] == 0
                and c.kwargs["BK"] <= k]
        return keep or configs[:1]

    # Structure mirrors nvfp4_quant's proven fused kernel (27.6us at qwen
    # shapes): one row x BLOCKS_PER_PROG 16-elem groups per program, even/odd
    # strided loads. Two 5090-banked lessons (pgw#862): fusing the rank-R
    # lora_down here reloads its tile per M-tile (~16x slower — it stays a
    # cuBLAS mm); and a 2-D [BM, BK] tile with tl.split/reshape re-layout ran
    # 179us vs this structure. Differences vs pgw#685: optional in-register
    # smooth divide (bf16-rounded to match the aten reference) and FLAT
    # [M, K/16] scale stores (dot_scaled's native layout) instead of the
    # cuBLAS blocked swizzle.
    _QBPP = 128

    @triton.jit
    def _quant_smooth_kernel(  # type: ignore[no-untyped-def]
        x_ptr, sm_ptr, s2_ptr, q_ptr, s_ptr,
        K, KB, NCB,
        HAS_SMOOTH: tl.constexpr, BLOCKED: tl.constexpr, BPP: tl.constexpr,
    ):
        row = tl.program_id(0)
        blk0 = tl.program_id(1) * BPP
        s2 = tl.load(s2_ptr)

        offs_b = blk0 + tl.arange(0, BPP)
        blk_ok = offs_b < KB
        offs_lo = offs_b[:, None] * 16 + tl.arange(0, 8)[None, :] * 2
        offs_hi = offs_lo + 1
        base = x_ptr + row * K
        m2 = blk_ok[:, None]
        xe = tl.load(base + offs_lo, mask=m2, other=0.0).to(tl.float32)
        xo = tl.load(base + offs_hi, mask=m2, other=0.0).to(tl.float32)
        if HAS_SMOOTH:
            # bf16-rounded divide — matches the aten `x2 / smooth` the
            # baseline lane feeds its quantizer (opmath float, rn to bf16).
            sme = tl.load(sm_ptr + offs_lo, mask=m2, other=1.0).to(tl.float32)
            smo = tl.load(sm_ptr + offs_hi, mask=m2, other=1.0).to(tl.float32)
            xe = tl.math.div_rn(xe, sme).to(tl.bfloat16).to(tl.float32)
            xo = tl.math.div_rn(xo, smo).to(tl.bfloat16).to(tl.float32)

        amax = tl.maximum(tl.max(tl.abs(xe), axis=1),
                          tl.max(tl.abs(xo), axis=1))
        scale = tl.math.div_rn(amax, 6.0 * s2)
        scale = tl.minimum(tl.maximum(scale, 0.001953125), 448.0)
        scale_f8 = scale.to(tl.float8e4nv)
        denom = (scale_f8.to(tl.float32) * s2)[:, None]

        lo = _e2m1_code(tl.math.div_rn(xe, denom))
        hi = _e2m1_code(tl.math.div_rn(xo, denom))
        packed = lo | (hi << 4)

        offs_p = offs_b[:, None] * 8 + tl.arange(0, 8)[None, :]
        tl.store(q_ptr + row * (K // 2) + offs_p, packed, mask=m2)
        if BLOCKED:
            # cuBLAS blocked layout (pgw#685 addressing) — the scaled_mm path.
            rb = row // 128
            rr = row % 128
            tile = rb * NCB + (offs_b // 4)
            in_tile = (rr % 32) * 16 + (rr // 32) * 4 + (offs_b % 4)
            tl.store(s_ptr + tile * 512 + in_tile, scale_f8, mask=blk_ok)
        else:
            tl.store(s_ptr + row * KB + offs_b, scale_f8, mask=blk_ok)

    gemm_cfgs = [
        triton.Config({"BM": 128, "BN": 128, "BK": 128, "GROUP_M": 8},
                      num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 256, "GROUP_M": 8},
                      num_warps=8, num_stages=2),
        triton.Config({"BM": 128, "BN": 128, "BK": 64, "GROUP_M": 8},
                      num_warps=8, num_stages=4),
        triton.Config({"BM": 128, "BN": 64, "BK": 128, "GROUP_M": 8},
                      num_warps=8, num_stages=4),
        triton.Config({"BM": 64, "BN": 128, "BK": 128, "GROUP_M": 8},
                      num_warps=4, num_stages=4),
        triton.Config({"BM": 64, "BN": 128, "BK": 256, "GROUP_M": 8},
                      num_warps=4, num_stages=3),
        triton.Config({"BM": 64, "BN": 64, "BK": 256, "GROUP_M": 8},
                      num_warps=4, num_stages=4),
        triton.Config({"BM": 128, "BN": 256, "BK": 64, "GROUP_M": 8},
                      num_warps=8, num_stages=3),
        triton.Config({"BM": 64, "BN": 256, "BK": 128, "GROUP_M": 8},
                      num_warps=8, num_stages=3),
        triton.Config({"BM": 256, "BN": 128, "BK": 64, "GROUP_M": 8},
                      num_warps=8, num_stages=3),
    ]

    @triton.autotune(configs=gemm_cfgs, key=["M", "N", "K"],
                     prune_configs_by={"early_config_prune": _prune_k})
    @triton.jit
    def _gemm_lora_kernel(  # type: ignore[no-untyped-def]
        qa_ptr, sa_ptr, qb_ptr, sb_ptr, s2_ptr, sec_ptr, la_ptr, up_ptr,
        bias_ptr, out_ptr,
        M, N, K, R: tl.constexpr,
        PER_CHANNEL: tl.constexpr, HAS_BIAS: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
        GROUP_M: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_m = tl.cdiv(M, BM)
        num_pid_n = tl.cdiv(N, BN)
        num_in_group = GROUP_M * num_pid_n
        group_id = pid // num_in_group
        first_pid_m = group_id * GROUP_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
        pid_m = first_pid_m + (pid % num_in_group) % group_size_m
        pid_n = (pid % num_in_group) // group_size_m

        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        m_ok = offs_m < M
        n_ok = offs_n < N

        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k0 in range(0, K, BK):
            offs_kp = k0 // 2 + tl.arange(0, BK // 2)
            offs_ks = k0 // 16 + tl.arange(0, BK // 16)
            qa = tl.load(qa_ptr + offs_m[:, None] * (K // 2)
                         + offs_kp[None, :], mask=m_ok[:, None], other=0)
            sa = tl.load(sa_ptr + offs_m[:, None] * (K // 16)
                         + offs_ks[None, :], mask=m_ok[:, None], other=0.0)
            qb = tl.load(qb_ptr + offs_kp[:, None] * N + offs_n[None, :],
                         mask=n_ok[None, :], other=0)
            sb = tl.load(sb_ptr + offs_n[:, None] * (K // 16)
                         + offs_ks[None, :], mask=n_ok[:, None], other=0.0)
            acc = tl.dot_scaled(qa, sa, "e2m1", qb, sb, "e2m1", acc)

        s2 = tl.load(s2_ptr)
        if PER_CHANNEL:
            sec = tl.load(sec_ptr + offs_n, mask=n_ok, other=0.0)
            y = acc * (s2 * sec)[None, :]
        else:
            y = acc * (s2 * tl.load(sec_ptr))

        offs_r = tl.arange(0, R)
        la = tl.load(la_ptr + offs_m[:, None] * R + offs_r[None, :],
                     mask=m_ok[:, None], other=0.0)
        up = tl.load(up_ptr + offs_n[:, None] * R + offs_r[None, :],
                     mask=n_ok[:, None], other=0.0)
        y += tl.dot(la, tl.trans(up))

        if HAS_BIAS:
            y += tl.load(bias_ptr + offs_n, mask=n_ok,
                         other=0.0).to(tl.float32)[None, :]
        tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :],
                 y.to(tl.bfloat16), mask=m_ok[:, None] & n_ok[None, :])

    def _quant_warps() -> int:
        try:
            major, minor = torch.cuda.get_device_capability()
            return {100: 4, 103: 4, 120: 8, 121: 8}.get(
                major * 10 + minor, 4)
        except Exception:  # noqa: BLE001
            return 4

    def _quant_launch(x2: Any, smooth: Optional[Any], s2: Any,
                      blocked: bool) -> tuple[Any, Any]:
        from .nvfp4_quant import blocked_scale_numel

        m, k = int(x2.shape[0]), int(x2.shape[1])
        kb = k // BLOCK
        ncb = (kb + 3) // 4
        q = torch.empty(m, k // 2, dtype=torch.uint8, device=x2.device)
        if blocked:
            # Padding rows/cols are never written — must start zeroed.
            s = torch.zeros(blocked_scale_numel(m, kb),
                            dtype=torch.float8_e4m3fn, device=x2.device)
        else:
            s = torch.empty(m, kb, dtype=torch.float8_e4m3fn,
                            device=x2.device)
        bpp = min(_QBPP, kb)
        while kb % bpp and bpp > 1:
            bpp //= 2
        _quant_smooth_kernel[(m, (kb + bpp - 1) // bpp)](
            x2, smooth if smooth is not None else x2, s2, q, s, k, kb, ncb,
            HAS_SMOOTH=smooth is not None, BLOCKED=blocked, BPP=bpp,
            num_warps=_quant_warps())
        return q, s

    def _gemm_launch(qa: Any, sa: Any, wq_kn: Any, ws: Any, s2: Any,
                     second: Any, lora_act: Any, up: Any,
                     bias: Optional[Any]) -> Any:
        m, k2 = int(qa.shape[0]), int(qa.shape[1])
        k = k2 * 2
        n = int(wq_kn.shape[1])
        r = int(up.shape[1])
        per_channel = second.numel() > 1
        out = torch.empty(m, n, dtype=torch.bfloat16, device=qa.device)
        grid = lambda meta: (triton.cdiv(m, meta["BM"])  # noqa: E731
                             * triton.cdiv(n, meta["BN"]),)
        _gemm_lora_kernel[grid](
            qa, sa, wq_kn, ws, s2, second, lora_act, up,
            bias if bias is not None else second, out,
            m, n, k, R=r, PER_CHANNEL=per_channel, HAS_BIAS=bias is not None)
        return out

    quant_op = torch.library.custom_op(
        _QUANT_OP, _quant_launch, mutates_args=(),
        schema="(Tensor x2, Tensor? smooth, Tensor s2, bool blocked) "
               "-> (Tensor, Tensor)")

    @quant_op.register_fake
    def _(x2: Any, smooth: Optional[Any], s2: Any,
          blocked: bool) -> tuple[Any, Any]:
        from .nvfp4_quant import blocked_scale_numel

        m, k = int(x2.shape[0]), int(x2.shape[1])
        s_shape = ((blocked_scale_numel(m, k // BLOCK),) if blocked
                   else (m, k // BLOCK))
        return (x2.new_empty(m, k // 2, dtype=torch.uint8),
                x2.new_empty(*s_shape, dtype=torch.float8_e4m3fn))

    # Fused epilogue for the HYBRID path (cuBLAS block-scaled GEMM upstream):
    # y = y0 * (s2 * second) + la @ up.T + bias — replaces the baseline's
    # separate [M,N] scale-mul, lora-up mm, add and bias passes.
    epi_cfgs = [
        triton.Config({"BM": 64, "BN": 128}, num_warps=4, num_stages=3),
        triton.Config({"BM": 128, "BN": 128}, num_warps=8, num_stages=3),
        triton.Config({"BM": 64, "BN": 256}, num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 64}, num_warps=4, num_stages=3),
    ]

    @triton.autotune(configs=epi_cfgs, key=["N", "R"])
    @triton.jit
    def _epilogue_lora_kernel(  # type: ignore[no-untyped-def]
        y0_ptr, la_ptr, up_ptr, s2_ptr, sec_ptr, bias_ptr, out_ptr,
        M, N, R: tl.constexpr,
        PER_CHANNEL: tl.constexpr, HAS_BIAS: tl.constexpr,
        BM: tl.constexpr, BN: tl.constexpr,
    ):
        pid = tl.program_id(0)
        num_pid_n = tl.cdiv(N, BN)
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        m_ok = offs_m < M
        n_ok = offs_n < N
        mask = m_ok[:, None] & n_ok[None, :]

        y0 = tl.load(y0_ptr + offs_m[:, None] * N + offs_n[None, :],
                     mask=mask, other=0.0).to(tl.float32)
        s2 = tl.load(s2_ptr)
        if PER_CHANNEL:
            sec = tl.load(sec_ptr + offs_n, mask=n_ok, other=0.0)
            y = y0 * (s2 * sec)[None, :]
        else:
            y = y0 * (s2 * tl.load(sec_ptr))

        offs_r = tl.arange(0, R)
        la = tl.load(la_ptr + offs_m[:, None] * R + offs_r[None, :],
                     mask=m_ok[:, None], other=0.0)
        up = tl.load(up_ptr + offs_n[:, None] * R + offs_r[None, :],
                     mask=n_ok[:, None], other=0.0)
        y += tl.dot(la, tl.trans(up))
        if HAS_BIAS:
            y += tl.load(bias_ptr + offs_n, mask=n_ok,
                         other=0.0).to(tl.float32)[None, :]
        tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :],
                 y.to(tl.bfloat16), mask=mask)

    def _epilogue_launch(y0: Any, la: Any, up: Any, s2: Any, second: Any,
                         bias: Optional[Any]) -> Any:
        m, n = int(y0.shape[0]), int(y0.shape[1])
        r = int(up.shape[1])
        per_channel = second.numel() > 1
        out = torch.empty(m, n, dtype=torch.bfloat16, device=y0.device)
        grid = lambda meta: (triton.cdiv(m, meta["BM"])  # noqa: E731
                             * triton.cdiv(n, meta["BN"]),)
        _epilogue_lora_kernel[grid](
            y0, la, up, s2, second, bias if bias is not None else second,
            out, m, n, R=r, PER_CHANNEL=per_channel,
            HAS_BIAS=bias is not None)
        return out

    epi_op = torch.library.custom_op(
        _EPI_OP, _epilogue_launch, mutates_args=(),
        schema="(Tensor y0, Tensor la, Tensor up, Tensor s2, Tensor second, "
               "Tensor? bias) -> Tensor")

    @epi_op.register_fake
    def _(y0: Any, la: Any, up: Any, s2: Any, second: Any,
          bias: Optional[Any]) -> Any:
        return y0.new_empty(int(y0.shape[0]), int(y0.shape[1]),
                            dtype=torch.bfloat16)

    gemm_op = torch.library.custom_op(
        _GEMM_OP, _gemm_launch, mutates_args=(),
        schema="(Tensor qa, Tensor sa, Tensor wq_kn, Tensor ws, Tensor s2, "
               "Tensor second, Tensor lora_act, Tensor up, Tensor? bias) "
               "-> Tensor")

    @gemm_op.register_fake
    def _(qa: Any, sa: Any, wq_kn: Any, ws: Any, s2: Any, second: Any,
          lora_act: Any, up: Any, bias: Optional[Any]) -> Any:
        return qa.new_empty(int(qa.shape[0]), int(wq_kn.shape[1]),
                            dtype=torch.bfloat16)

    return quant_op, gemm_op, epi_op


def fused_ops() -> Optional[tuple[Any, Any, Any]]:
    return _build_fused_ops()


# ---------------------------------------------------------------------------
# Self-check — the arming gate the dispatch probe calls (pgw#860).
# ---------------------------------------------------------------------------


def _reference_quant_flat(xs: Any, s2: Any) -> tuple[Any, Any]:
    """The pgw#685 reference chain with FLAT [M, K/16] scales (the fused lane
    consumes plain scales; flat->blocked bijectivity is proven separately)."""
    import torch

    from .nvfp4_quant import cast_e2m1, pack_e2m1

    in_f = int(xs.shape[1])
    xb = xs.reshape(-1, in_f // BLOCK, BLOCK).float()
    bmax = xb.abs().amax(dim=-1)
    sa = (bmax / (E2M1_MAX * s2)).clamp(
        min=SCALE_MIN, max=FP8_MAX).to(torch.float8_e4m3fn)
    q = xb / (sa.float().unsqueeze(-1) * s2)
    codes = cast_e2m1(q.reshape(-1, in_f))
    return pack_e2m1(codes), sa


def _dyn_s2(x2: Any, smooth: Optional[Any]) -> Any:
    """Per-tensor second-level scale from column-amax — bit-identical to the
    baseline lane's ``(x2 / smooth).abs().amax()`` (rounding is monotonic and
    sign-symmetric, so amax commutes with the bf16 divide; max(|min|,|max|)
    is exactly amax(|x|) — and aminmax skips the [M, K] abs temporary)."""
    import torch

    mn, mx = x2.aminmax(dim=0)
    col = torch.maximum(mn.abs(), mx.abs())
    if smooth is not None:
        col = col / smooth
    return (col.abs().amax().float() / (E2M1_MAX * FP8_MAX)).clamp(min=1e-12)


def fused_self_check() -> Optional[str]:
    """Arm gate for the serving (hybrid) path: quantization BIT-IDENTICAL to
    the pgw#685 reference chain in BOTH scale layouts, and the fused epilogue
    within tolerance of its torch reference. Returns None when armed."""
    import torch

    ops = fused_ops()
    if ops is None:
        return "triton unavailable"
    quant_op, _gemm_op, epi_op = ops
    from .nvfp4_quant import quantize_activation_torch

    rank = 128
    for m, k, n in _SELF_CHECK_PROBES:
        torch.manual_seed(0)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        smooth = (torch.rand(k, device="cuda", dtype=torch.bfloat16) + 0.5)
        up = torch.randn(n, rank, device="cuda", dtype=torch.bfloat16) * 0.02
        bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
        second = (torch.rand(n, device="cuda") + 0.1)

        s2 = _dyn_s2(x, smooth)
        xs_ref = x / smooth
        want_q, want_s = _reference_quant_flat(xs_ref, s2)
        got_q, got_s = quant_op(x, smooth, s2, False)
        if not torch.equal(got_q, want_q):
            bad = int((got_q != want_q).sum())
            return (f"quant not bit-identical at {m}x{k} "
                    f"({bad}/{want_q.numel()} bytes differ)")
        if not torch.equal(got_s.view(torch.uint8),
                           want_s.view(torch.uint8)):
            return f"quant scales not bit-identical at {m}x{k}"
        _bq, want_sb = quantize_activation_torch(xs_ref, s2)
        got_qb, got_sb = quant_op(x, smooth, s2, True)
        if not torch.equal(got_qb, want_q) or not torch.equal(
                got_sb.view(torch.uint8), want_sb.view(torch.uint8)):
            return f"blocked-scale quant not bit-identical at {m}x{k}"

        y0 = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        la = torch.randn(m, rank, device="cuda", dtype=torch.bfloat16)
        y = epi_op(y0, la, up, s2, second, bias)
        ref = (y0.float() * (s2 * second).reshape(1, n)
               + la.float() @ up.float().t() + bias.float())
        rel = ((y.float() - ref).norm() / ref.norm().clamp(min=1e-9)).item()
        if rel > 5e-3:
            return f"fused epilogue rel err {rel:.4f} at {m}x{n}"
    return None


# ---------------------------------------------------------------------------
# The module + builder (twin of svdq_native.build_svdq_linear).
# ---------------------------------------------------------------------------


def _build_fused_linear_class() -> type:
    import torch
    import torch.nn as nn

    class _SvdqFusedLinear(nn.Module):
        """SvdqLinear on the fused lane: quant kernel + cuBLAS lora_down +
        fused GEMM/epilogue kernel per forward."""

        _cozy_svdq_linear = True
        _cozy_svdq_fused = True

        def __init__(self, in_features: int, out_features: int, *,
                     rank: int, bias: bool, compute_dtype: Any,
                     per_channel_scale: bool, smooth: bool) -> None:
            super().__init__()
            self.in_features = int(in_features)
            self.out_features = int(out_features)
            self.rank = int(rank)
            self.per_channel_scale = bool(per_channel_scale)
            if not fused_shape_supported(out_features, in_features, rank):
                raise SvdqFusedError(
                    f"SvdqFusedLinear [{out_features}, {in_features}] r{rank} "
                    f"breaks the fused contract (in%{_K_ALIGN}, "
                    f"out%{_N_ALIGN}, rank%{_RANK_ALIGN})")
            meta = torch.device("meta")
            # Hybrid serving operands: cuBLAS block-scaled GEMM consumes the
            # SAME weight layout as the baseline SvdqLinear.
            self.register_buffer("weight", torch.empty(
                out_features, in_features // 2, dtype=torch.uint8,
                device=meta))
            nrb = (out_features + 127) // 128
            ncb = (in_features // BLOCK + 3) // 4
            self.register_buffer("weight_scale", torch.empty(
                nrb * 128 * ncb * 4, dtype=torch.float8_e4m3fn, device=meta))
            self.register_buffer("second", torch.empty(
                out_features if per_channel_scale else 1,
                dtype=torch.float32, device=meta))
            if smooth:
                self.register_buffer("smooth_factor", torch.empty(
                    in_features, dtype=compute_dtype, device=meta))
            else:
                self.smooth_factor = None
            self.register_buffer("proj_down", torch.empty(
                in_features, self.rank, dtype=compute_dtype, device=meta))
            self.register_buffer("proj_up", torch.empty(
                out_features, self.rank, dtype=compute_dtype, device=meta))
            if bias:
                self.bias: Optional[nn.Parameter] = nn.Parameter(torch.empty(
                    out_features, dtype=compute_dtype, device=meta))
            else:
                self.bias = None

        def forward(self, x: Any) -> Any:
            from .w4a4 import _gemm_w4a4

            shape = x.shape
            x2 = x.reshape(-1, self.in_features).contiguous()
            s2 = _dyn_s2(x2, self.smooth_factor)
            qa, sa = torch.ops.cozy_gen_worker.svdq_quant(
                x2, self.smooth_factor, s2, True)
            y0 = _gemm_w4a4(qa, self.weight, sa, self.weight_scale, x.dtype)
            la = x2 @ self.proj_down
            y = torch.ops.cozy_gen_worker.svdq_epilogue_lora(
                y0, la, self.proj_up, s2, self.second, self.bias)
            return y.reshape(*shape[:-1], self.out_features)

        def extra_repr(self) -> str:
            return (f"in_features={self.in_features}, "
                    f"out_features={self.out_features}, rank={self.rank}, "
                    f"bias={self.bias is not None}, "
                    f"per_channel_scale={self.per_channel_scale}, "
                    f"smooth={self.smooth_factor is not None}, lane=fused")

    return _SvdqFusedLinear


@functools.lru_cache(maxsize=1)
def svdq_fused_linear_class() -> type:
    return _build_fused_linear_class()


def build_svdq_fused_linear(dec: Any, *, compute_dtype: Any = None,
                            device: Any = None) -> Any:
    """A device-resident fused-lane module from a ``DecodedLinear``. The
    resident swizzle (pack + transpose) runs once, here, at load."""
    import torch
    import torch.nn as nn

    from .nvfp4_quant import pack_e2m1, to_blocked_scales

    if fused_ops() is None:
        raise SvdqFusedError("fused ops unavailable (no triton)")
    compute = compute_dtype or torch.bfloat16
    cls = svdq_fused_linear_class()
    mod = cls(dec.in_features, dec.out_features, rank=dec.rank,
              bias=dec.bias is not None, compute_dtype=compute,
              per_channel_scale=dec.second_kind == "per_channel",
              smooth=dec.smooth_factor is not None)
    dev = device or "cpu"
    mod.weight = pack_e2m1(dec.codes).contiguous().to(dev)
    mod.weight_scale = to_blocked_scales(dec.scales).to(dev)
    mod.second = dec.second.float().reshape(-1).to(dev)
    if dec.smooth_factor is not None:
        mod.smooth_factor = (dec.smooth_factor.to(compute)
                             .reshape(-1).contiguous().to(dev))
    mod.proj_down = dec.proj_down.to(compute).contiguous().to(dev)
    mod.proj_up = dec.proj_up.to(compute).contiguous().to(dev)
    if dec.bias is not None:
        mod.bias = nn.Parameter(
            dec.bias.detach().to(compute).reshape(-1).to(dev),
            requires_grad=False)
    return mod


__all__ = [
    "SvdqFusedError",
    "build_svdq_fused_linear",
    "fused_ops",
    "fused_self_check",
    "fused_shape_supported",
    "svdq_fused_linear_class",
]
