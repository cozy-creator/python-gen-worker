"""Packed-resident AWQ W4A16 modulation serving.

The baseline decodes every ``img_mod``/``txt_mod`` layer to a dense bf16
``nn.Linear`` AT LOAD (`svdq_awq.decode_awq_linear`) — the entire measured
+6.8 GB peak-VRAM delta vs nunchaku. This module keeps the weights RESIDENT at
4 bits and dequantizes per-group in-kernel:

  ``y = x @ (codes * wscales + wzeros).T + bias`` — group-64 asymmetric int4,
  zeros pre-scaled AND pre-negated (the svdq_awq byte contract).

Resident buffers are a load-time swizzle of the on-disk tensors (the on-disk
format is untouched):

  weight   uint8 [oc, ic/2]  row-major nibble pairs (element 2j LOW nibble),
                             adanorm row-interleave UNDONE on the codes
                             (a row permutation — exact in the 4-bit domain)
  wscales  bf16  [oc, ng]    live rows only, transposed, row-permuted
  wzeros   bf16  [oc, ng]    same
  bias     bf16  [oc]        adanorm +1 delta subtracted (as the decode path)

Modulation GEMMs are skinny (x = timestep embedding, M = batch): this is a
VRAM feature first (tracker: ms priced by the pending per-op profile). The
kernel rounds the dequantized weight to bf16 before the multiply, so its
per-element math matches the baseline's bf16 ``nn.Linear`` weight exactly;
only fp32 accumulation order differs (quantified on the parity harness).
Gated by the same probe/env as the fused svdq lane.
"""

from __future__ import annotations

import functools
import logging
from typing import Any, Optional
from .svdq_awq import _scales_and_zeros, unpack_w4x16
from .svdq_awq import decode_awq_linear, encode_awq_linear

logger = logging.getLogger(__name__)

_OP_NAME = "cozy_gen_worker::awq_w4a16_mm"

GROUP_SIZE = 64


class AwqPackedError(RuntimeError):
    """Typed packed-AWQ failure (contract mismatch — never silent-wrong)."""


def awq_packed_supported(out_features: int, in_features: int) -> bool:
    return out_features % 16 == 0 and in_features % 128 == 0


@functools.lru_cache(maxsize=1)
def _build_awq_op() -> Optional[Any]:
    try:
        import torch
        import triton
        import triton.language as tl
    except Exception as exc:  # noqa: BLE001
        logger.info("awq_packed: triton unavailable (%s)", exc)
        return None

    cfgs = [
        triton.Config({"BN": 128, "BK": 128}, num_warps=4, num_stages=3),
        triton.Config({"BN": 64, "BK": 256}, num_warps=4, num_stages=3),
        triton.Config({"BN": 256, "BK": 128}, num_warps=8, num_stages=2),
        triton.Config({"BN": 128, "BK": 256}, num_warps=8, num_stages=2),
    ]

    def _prune(configs, named_args, **kwargs):  # type: ignore[no-untyped-def]
        k = named_args["K"]
        keep = [c for c in configs if k % c.kwargs["BK"] == 0]
        return keep or configs[:1]

    @triton.autotune(configs=cfgs, key=["N", "K"],
                     prune_configs_by={"early_config_prune": _prune})
    @triton.jit
    def _awq_mm_kernel(  # type: ignore[no-untyped-def]
        x_ptr, w_ptr, s_ptr, z_ptr, b_ptr, out_ptr,
        M, N, K, NG,
        GS: tl.constexpr, HAS_BIAS: tl.constexpr,
        BN: tl.constexpr, BK: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        m = tl.program_id(1)
        offs_n = pid_n * BN + tl.arange(0, BN)
        n_ok = offs_n < N

        acc = tl.zeros((BN,), dtype=tl.float32)
        for k0 in range(0, K, BK):
            xk = tl.load(x_ptr + m * K + k0 + tl.arange(0, BK)).to(tl.float32)
            xe, xo = tl.split(tl.reshape(xk, (BK // 2, 2)))
            xe3 = tl.reshape(xe, (BK // GS, GS // 2))
            xo3 = tl.reshape(xo, (BK // GS, GS // 2))

            qp = tl.load(w_ptr + offs_n[:, None] * (K // 2)
                         + (k0 // 2 + tl.arange(0, BK // 2))[None, :],
                         mask=n_ok[:, None], other=0)
            offs_g = k0 // GS + tl.arange(0, BK // GS)
            sg = tl.load(s_ptr + offs_n[:, None] * NG + offs_g[None, :],
                         mask=n_ok[:, None], other=0.0).to(tl.float32)
            zg = tl.load(z_ptr + offs_n[:, None] * NG + offs_g[None, :],
                         mask=n_ok[:, None], other=0.0).to(tl.float32)

            we = tl.reshape((qp & 0xF).to(tl.float32),
                            (BN, BK // GS, GS // 2))
            wo = tl.reshape((qp >> 4).to(tl.float32),
                            (BN, BK // GS, GS // 2))
            # Round to bf16 like the decoded nn.Linear weight — per-element
            # parity with the baseline; only accumulation order differs.
            we = (we * sg[:, :, None] + zg[:, :, None]).to(
                tl.bfloat16).to(tl.float32)
            wo = (wo * sg[:, :, None] + zg[:, :, None]).to(
                tl.bfloat16).to(tl.float32)
            acc += tl.sum(tl.sum(we * xe3[None, :, :], axis=2), axis=1)
            acc += tl.sum(tl.sum(wo * xo3[None, :, :], axis=2), axis=1)

        if HAS_BIAS:
            acc += tl.load(b_ptr + offs_n, mask=n_ok, other=0.0).to(tl.float32)
        tl.store(out_ptr + m * N + offs_n, acc.to(tl.bfloat16), mask=n_ok)

    def _launch(x2: Any, weight: Any, wscales: Any, wzeros: Any,
                bias: Optional[Any]) -> Any:
        m, k = int(x2.shape[0]), int(x2.shape[1])
        n = int(weight.shape[0])
        ng = int(wscales.shape[1])
        out = torch.empty(m, n, dtype=torch.bfloat16, device=x2.device)
        grid = lambda meta: (triton.cdiv(n, meta["BN"]), m)  # noqa: E731
        _awq_mm_kernel[grid](
            x2, weight, wscales, wzeros,
            bias if bias is not None else wscales, out,
            m, n, k, ng, GS=GROUP_SIZE, HAS_BIAS=bias is not None)
        return out

    op = torch.library.custom_op(
        _OP_NAME, _launch, mutates_args=(),
        schema="(Tensor x2, Tensor weight, Tensor wscales, Tensor wzeros, "
               "Tensor? bias) -> Tensor")

    @op.register_fake
    def _(x2: Any, weight: Any, wscales: Any, wzeros: Any,
          bias: Optional[Any]) -> Any:
        return x2.new_empty(int(x2.shape[0]), int(weight.shape[0]),
                            dtype=torch.bfloat16)

    return op


def awq_op() -> Optional[Any]:
    return _build_awq_op()


# ---------------------------------------------------------------------------
# Module + builder.
# ---------------------------------------------------------------------------


def _build_awq_linear_class() -> type:
    import torch
    import torch.nn as nn

    class _AwqPackedLinear(nn.Module):
        """AWQ W4A16 modulation layer served packed-resident."""

        _cozy_awq_packed = True

        def __init__(self, in_features: int, out_features: int, *,
                     bias: bool, compute_dtype: Any) -> None:
            super().__init__()
            # record it (twin of _SvdqLinear).
            self.compute_dtype = compute_dtype
            self.in_features = int(in_features)
            self.out_features = int(out_features)
            if not awq_packed_supported(out_features, in_features):
                raise AwqPackedError(
                    f"AwqPackedLinear [{out_features}, {in_features}] breaks "
                    f"the packed contract (out%16, in%128)")
            ng = in_features // GROUP_SIZE
            meta = torch.device("meta")
            self.register_buffer("weight", torch.empty(
                out_features, in_features // 2, dtype=torch.uint8,
                device=meta))
            self.register_buffer("wscales", torch.empty(
                out_features, ng, dtype=compute_dtype, device=meta))
            self.register_buffer("wzeros", torch.empty(
                out_features, ng, dtype=compute_dtype, device=meta))
            if bias:
                self.bias: Optional[nn.Parameter] = nn.Parameter(torch.empty(
                    out_features, dtype=compute_dtype, device=meta))
            else:
                self.bias = None

        def forward(self, x: Any) -> Any:
            shape = x.shape
            x2 = x.reshape(-1, self.in_features).contiguous()
            y = torch.ops.cozy_gen_worker.awq_w4a16_mm(
                x2, self.weight, self.wscales, self.wzeros, self.bias)
            return y.reshape(*shape[:-1], self.out_features)

        def extra_repr(self) -> str:
            return (f"in_features={self.in_features}, "
                    f"out_features={self.out_features}, "
                    f"bias={self.bias is not None}, lane=awq-packed")

    return _AwqPackedLinear


@functools.lru_cache(maxsize=1)
def awq_packed_linear_class() -> type:
    return _build_awq_linear_class()


def _undo_interleave_perm(oc: int, splits: int, device: Any) -> Any:
    """Row permutation that undoes the exporter's adaLN interleave: stored row
    ``j*splits + s`` is original row ``s*(oc/splits) + j``."""
    import torch

    per = oc // splits
    idx = torch.arange(oc, device=device)
    # original row r = s*per + j  ->  stored row j*splits + s
    s, j = idx // per, idx % per
    return j * splits + s


def build_awq_packed_linear(tensors: dict[str, Any], out_features: int,
                            in_features: int, *, adanorm_splits: int = 1,
                            compute_dtype: Any = None,
                            device: Any = None) -> Any:
    """Packed-resident module from the on-disk AWQ tensors. The load-time
    swizzle: unpack -> undo adanorm row interleave -> row-major nibble
    repack; scales/zeros transposed + permuted; bias per the decode path."""
    import torch
    import torch.nn as nn

    if awq_op() is None:
        raise AwqPackedError("awq packed op unavailable (no triton)")
    compute = compute_dtype or torch.bfloat16
    oc, ic = int(out_features), int(in_features)
    dev = device or "cpu"

    codes = unpack_w4x16(tensors["qweight"].to(dev), oc, ic)
    scale, zero, gs = _scales_and_zeros(
        tensors["wscales"].to(dev), tensors["wzeros"].to(dev), oc, ic)
    if gs != GROUP_SIZE:
        raise AwqPackedError(
            f"AWQ group size {gs} != {GROUP_SIZE} — packed lane only ships "
            f"the g{GROUP_SIZE} contract; decode path serves this layer")
    bias = tensors.get("bias")
    splits = int(adanorm_splits)
    if splits > 1:
        perm = _undo_interleave_perm(oc, splits, codes.device)
        codes = codes[perm]
        scale = scale[perm]
        zero = zero[perm]
        if bias is not None:
            per = oc // splits
            b = bias.float().to(codes.device).view(per, splits).transpose(
                0, 1).contiguous()
            delta = torch.zeros(splits, dtype=b.dtype, device=b.device)
            delta[1] = 1.0
            delta[splits - 2] = 1.0
            bias = (b - delta.reshape(splits, 1)).reshape(oc)

    mod = awq_packed_linear_class()(
        ic, oc, bias=bias is not None, compute_dtype=compute)
    mod.weight = ((codes[:, 1::2] << 4) | codes[:, 0::2]).contiguous().to(dev)
    mod.wscales = scale.to(compute).contiguous().to(dev)
    mod.wzeros = zero.to(compute).contiguous().to(dev)
    if bias is not None:
        mod.bias = nn.Parameter(
            bias.detach().to(compute).reshape(-1).to(dev),
            requires_grad=False)
    return mod


def awq_packed_self_check() -> Optional[str]:
    """Benchmark-side packed output check against the decoded bf16 linear."""
    import torch

    if awq_op() is None:
        return "triton unavailable"
    for oc, ic, splits in ((512, 256, 1), (768, 256, 6)):
        torch.manual_seed(0)
        w = torch.randn(oc, ic) * 0.05
        b = torch.randn(oc)
        tensors = encode_awq_linear(w, b, adanorm_splits=splits)
        ref = decode_awq_linear(
            tensors, oc, ic, adanorm_splits=splits, device="cuda",
        )
        mod = build_awq_packed_linear(
            tensors, oc, ic, adanorm_splits=splits, device="cuda",
        )
        x = torch.randn(3, ic, device="cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            got = mod(x)
            want = ref(x)
        rel = (
            (got.float() - want.float()).norm()
            / want.float().norm().clamp(min=1e-9)
        ).item()
        if rel > 5e-3:
            return (
                f"awq packed rel err {rel:.4f} vs decoded Linear at "
                f"[{oc},{ic}] splits={splits}"
            )
    return None


__all__ = [
    "AwqPackedError",
    "GROUP_SIZE",
    "awq_op",
    "awq_packed_linear_class",
    "awq_packed_self_check",
    "awq_packed_supported",
    "build_awq_packed_linear",
]
