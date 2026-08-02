"""Fused native execution lane for SvdqLinear (pgw#862 phase B0).

Replaces the unfused per-unit chain (triton act-quant launch + blockwise
``torch._scaled_mm`` + full-[M,N] epilogue multiply + two dense bf16 low-rank
GEMMs + bias) with TWO triton kernels per unit:

  1. ``svdq_quant_lora`` — one pass over x: smooth-divide (bf16-rounded to
     match the aten reference), per-16-block e4m3 quantization BIT-IDENTICAL
     to the pgw#685 reference chain, and the rank-R ``x @ proj_down`` GEMM on
     RAW x fused into the same pass.
  2. ``svdq_gemm_w4a4_lora`` — block-scaled fp4 ``tl.dot_scaled`` GEMM (native
     ``kind::mxf4nvf4`` MMA on sm_120a, tcgen05 on sm_100a — PTX census banked
     in pgw#862) + second-level scale + ``lora_act @ proj_up.T`` + bias, one
     fp32 accumulator, single bf16 writeback.

Weight-side operands are PLAIN row-major (scales [N, K/16], packed weight
transposed to [K/2, N]) — a resident swizzle computed once at load (pgw#861);
the on-disk format and the baseline lane's buffers are untouched.

Numerics vs the baseline lane: activation quantization is bit-identical by
construction (same formulas, ``div_rn``, ties-up e2m1, same s2 — the arming
self-check enforces it). The GEMM + epilogue accumulate in fp32 with ONE final
bf16 round, where the baseline lane rounds to bf16 at every op boundary; the
divergence is quantified per shape on the pgw#865 harness, never assumed.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Optional

from .nvfp4_quant import BLOCK, E2M1_MAX, FP8_MAX, SCALE_MIN

logger = logging.getLogger(__name__)

_QUANT_OP = "cozy_gen_worker::svdq_quant_lora"
_GEMM_OP = "cozy_gen_worker::svdq_gemm_w4a4_lora"

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
def _build_fused_ops() -> Optional[tuple[Any, Any]]:
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

    quant_cfgs = [
        triton.Config({"BM": 32, "BK": 256}, num_warps=4, num_stages=3),
        triton.Config({"BM": 64, "BK": 256}, num_warps=8, num_stages=3),
        triton.Config({"BM": 64, "BK": 128}, num_warps=4, num_stages=4),
        triton.Config({"BM": 128, "BK": 128}, num_warps=8, num_stages=3),
    ]

    @triton.autotune(configs=quant_cfgs, key=["K", "R"],
                     prune_configs_by={"early_config_prune": _prune_k})
    @triton.jit
    def _quant_lora_kernel(  # type: ignore[no-untyped-def]
        x_ptr, sm_ptr, s2_ptr, down_ptr, q_ptr, s_ptr, la_ptr,
        M, K, R: tl.constexpr, HAS_SMOOTH: tl.constexpr,
        BM: tl.constexpr, BK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs_m = pid * BM + tl.arange(0, BM)
        m_ok = offs_m < M
        s2 = tl.load(s2_ptr)
        offs_r = tl.arange(0, R)
        acc = tl.zeros((BM, R), dtype=tl.float32)
        for k0 in range(0, K, BK):
            offs_k = k0 + tl.arange(0, BK)
            x_full = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :],
                             mask=m_ok[:, None], other=0.0)
            # Low-rank branch consumes RAW x (proj_down is pre-divided by the
            # smooth vector at export).
            dwn = tl.load(down_ptr + offs_k[:, None] * R + offs_r[None, :])
            acc = tl.dot(x_full, dwn, acc)
            if HAS_SMOOTH:
                # bf16-rounded divide — matches the aten `x2 / smooth` the
                # baseline lane feeds its quantizer (opmath float, rn to bf16).
                sm = tl.load(sm_ptr + offs_k).to(tl.float32)
                xs = tl.math.div_rn(x_full.to(tl.float32),
                                    sm[None, :]).to(tl.bfloat16)
            else:
                xs = x_full
            xs3 = tl.reshape(xs, (BM, BK // 16, 16)).to(tl.float32)
            amax = tl.max(tl.abs(xs3), axis=2)
            scale = tl.math.div_rn(amax, 6.0 * s2)
            scale = tl.minimum(tl.maximum(scale, 0.001953125), 448.0)
            scale_f8 = scale.to(tl.float8e4nv)
            denom = (scale_f8.to(tl.float32) * s2)[:, :, None]
            xse, xso = tl.split(tl.reshape(xs, (BM, BK // 2, 2)))
            qe = _e2m1_code(tl.math.div_rn(
                tl.reshape(xse, (BM, BK // 16, 8)).to(tl.float32), denom))
            qo = _e2m1_code(tl.math.div_rn(
                tl.reshape(xso, (BM, BK // 16, 8)).to(tl.float32), denom))
            packed = tl.reshape(qe | (qo << 4), (BM, BK // 2))
            tl.store(q_ptr + offs_m[:, None] * (K // 2)
                     + (k0 // 2 + tl.arange(0, BK // 2))[None, :],
                     packed, mask=m_ok[:, None])
            tl.store(s_ptr + offs_m[:, None] * (K // 16)
                     + (k0 // 16 + tl.arange(0, BK // 16))[None, :],
                     scale_f8, mask=m_ok[:, None])
        tl.store(la_ptr + offs_m[:, None] * R + offs_r[None, :],
                 acc.to(tl.bfloat16), mask=m_ok[:, None])

    gemm_cfgs = [
        triton.Config({"BM": 128, "BN": 128, "BK": 128, "GROUP_M": 8},
                      num_warps=8, num_stages=3),
        triton.Config({"BM": 128, "BN": 128, "BK": 64, "GROUP_M": 8},
                      num_warps=8, num_stages=4),
        triton.Config({"BM": 128, "BN": 64, "BK": 128, "GROUP_M": 8},
                      num_warps=8, num_stages=4),
        triton.Config({"BM": 64, "BN": 128, "BK": 128, "GROUP_M": 8},
                      num_warps=4, num_stages=4),
        triton.Config({"BM": 64, "BN": 64, "BK": 128, "GROUP_M": 8},
                      num_warps=4, num_stages=4),
        triton.Config({"BM": 128, "BN": 256, "BK": 64, "GROUP_M": 8},
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

    def _quant_launch(x2: Any, smooth: Optional[Any], s2: Any,
                      down: Any) -> tuple[Any, Any, Any]:
        m, k = int(x2.shape[0]), int(x2.shape[1])
        r = int(down.shape[1])
        q = torch.empty(m, k // 2, dtype=torch.uint8, device=x2.device)
        s = torch.empty(m, k // BLOCK, dtype=torch.float8_e4m3fn,
                        device=x2.device)
        la = torch.empty(m, r, dtype=torch.bfloat16, device=x2.device)
        grid = lambda meta: (triton.cdiv(m, meta["BM"]),)  # noqa: E731
        _quant_lora_kernel[grid](
            x2, smooth if smooth is not None else x2, s2, down, q, s, la,
            m, k, R=r, HAS_SMOOTH=smooth is not None)
        return q, s, la

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
        schema="(Tensor x2, Tensor? smooth, Tensor s2, Tensor down) "
               "-> (Tensor, Tensor, Tensor)")

    @quant_op.register_fake
    def _(x2: Any, smooth: Optional[Any], s2: Any,
          down: Any) -> tuple[Any, Any, Any]:
        m, k = int(x2.shape[0]), int(x2.shape[1])
        return (x2.new_empty(m, k // 2, dtype=torch.uint8),
                x2.new_empty(m, k // BLOCK, dtype=torch.float8_e4m3fn),
                x2.new_empty(m, int(down.shape[1]), dtype=torch.bfloat16))

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

    return quant_op, gemm_op


def fused_ops() -> Optional[tuple[Any, Any]]:
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
    sign-symmetric, so amax commutes with the bf16 divide)."""
    col = x2.abs().amax(dim=0)
    if smooth is not None:
        col = col / smooth
    return (col.abs().amax().float() / (E2M1_MAX * FP8_MAX)).clamp(min=1e-12)


def fused_self_check() -> Optional[str]:
    """Arm gate: quantization BIT-IDENTICAL to the reference chain; fused GEMM
    within the w4a4 lane's numerics bar vs the fp32 dequant reference.
    Returns None when armed, else the reason."""
    import torch

    ops = fused_ops()
    if ops is None:
        return "triton unavailable"
    quant_op, gemm_op = ops
    from .nvfp4_quant import unpack_e2m1

    rank = 128
    for m, k, n in _SELF_CHECK_PROBES:
        torch.manual_seed(0)
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        smooth = (torch.rand(k, device="cuda", dtype=torch.bfloat16) + 0.5)
        down = torch.randn(k, rank, device="cuda", dtype=torch.bfloat16) * 0.02
        up = torch.randn(n, rank, device="cuda", dtype=torch.bfloat16) * 0.02
        bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        ws2 = (w.abs().amax().float() / (E2M1_MAX * FP8_MAX)).clamp(min=1e-12)
        wq, wsf = _reference_quant_flat(w, ws2)
        second = torch.full((n,), float(ws2), device="cuda")

        s2 = _dyn_s2(x, smooth)
        xs_ref = x / smooth
        want_q, want_s = _reference_quant_flat(xs_ref, s2)
        got_q, got_s, got_la = quant_op(x, smooth, s2, down)
        if not torch.equal(got_q, want_q):
            bad = int((got_q != want_q).sum())
            return (f"quant not bit-identical at {m}x{k} "
                    f"({bad}/{want_q.numel()} bytes differ)")
        if not torch.equal(got_s.view(torch.uint8),
                           want_s.view(torch.uint8)):
            return f"quant scales not bit-identical at {m}x{k}"
        la_ref = x @ down
        la_rel = ((got_la.float() - la_ref.float()).norm()
                  / la_ref.float().norm().clamp(min=1e-9)).item()
        if la_rel > 1e-2:
            return f"lora_down drift {la_rel:.4f} at {m}x{k}"

        y = gemm_op(got_q, got_s, wq.t().contiguous(), wsf, s2, second,
                    got_la, up, bias)
        a_deq = (unpack_e2m1(got_q).reshape(m, k // BLOCK, BLOCK)
                 * got_s.float().unsqueeze(-1)).reshape(m, k) * s2
        b_deq = (unpack_e2m1(wq).reshape(n, k // BLOCK, BLOCK)
                 * wsf.float().unsqueeze(-1)).reshape(n, k)
        ref = (a_deq @ b_deq.t()) * second.reshape(1, n)
        ref = ref + (got_la.float() @ up.float().t()) + bias.float()
        rel = ((y.float() - ref).norm() / ref.norm().clamp(min=1e-9)).item()
        if rel > 2e-2:  # the w4a4 lane's numerics bar
            return f"fused gemm rel err {rel:.4f} at {m}x{k}x{n}"
    return None


# ---------------------------------------------------------------------------
# The module + builder (twin of svdq_native.build_svdq_linear).
# ---------------------------------------------------------------------------


def _build_fused_linear_class() -> type:
    import torch
    import torch.nn as nn

    class _SvdqFusedLinear(nn.Module):
        """SvdqLinear on the fused lane: two kernel launches per forward."""

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
            self.register_buffer("weight_kn", torch.empty(
                in_features // 2, out_features, dtype=torch.uint8,
                device=meta))
            self.register_buffer("wscales", torch.empty(
                out_features, in_features // BLOCK,
                dtype=torch.float8_e4m3fn, device=meta))
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
            shape = x.shape
            x2 = x.reshape(-1, self.in_features).contiguous()
            s2 = _dyn_s2(x2, self.smooth_factor)
            qa, sa, la = torch.ops.cozy_gen_worker.svdq_quant_lora(
                x2, self.smooth_factor, s2, self.proj_down)
            y = torch.ops.cozy_gen_worker.svdq_gemm_w4a4_lora(
                qa, sa, self.weight_kn, self.wscales, s2, self.second, la,
                self.proj_up, self.bias)
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

    from .nvfp4_quant import pack_e2m1

    if fused_ops() is None:
        raise SvdqFusedError("fused ops unavailable (no triton)")
    compute = compute_dtype or torch.bfloat16
    cls = svdq_fused_linear_class()
    mod = cls(dec.in_features, dec.out_features, rank=dec.rank,
              bias=dec.bias is not None, compute_dtype=compute,
              per_channel_scale=dec.second_kind == "per_channel",
              smooth=dec.smooth_factor is not None)
    dev = device or "cpu"
    mod.weight_kn = pack_e2m1(dec.codes).t().contiguous().to(dev)
    mod.wscales = dec.scales.contiguous().to(dev)
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
