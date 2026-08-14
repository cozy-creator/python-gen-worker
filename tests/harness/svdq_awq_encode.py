"""Test-only encoder for fabricating AWQ W4A16 checkpoint entries.

Production only decodes published checkpoint bytes. Keeping the inverse
exporter here prevents fixture construction from becoming an uncalled public
worker API.
"""

from __future__ import annotations

from typing import Any, Optional

from gen_worker.models.svdq_awq import num_scale_rows
from gen_worker.models.svdq_layout import SvdqLayoutError


def pack_w4x16(codes: Any) -> Any:
    """Pack uint8 codes [oc, ic] into TinyChat int32 [oc/4, ic/2]."""
    import torch

    oc, ic = int(codes.shape[0]), int(codes.shape[1])
    if oc % 4 or ic % 64:
        raise SvdqLayoutError(
            f"AWQ W4A16 [{oc}, {ic}] needs oc%4==0 and ic%64==0")
    w = codes.to(torch.int32).view(-1, 4, 8)
    w = w[:, 0] | (w[:, 1] << 4) | (w[:, 2] << 8) | (w[:, 3] << 12)
    w = w.view(oc // 4, 4, ic // 64, 16).permute(0, 2, 1, 3)
    return w.reshape(oc // 4, ic).to(torch.int16).view(torch.int32)


def apply_adanorm_splits(weight: Any, bias: Any, splits: int) -> tuple[Any, Any]:
    """Apply the exporter's adaLN row interleave and bias transform."""
    import torch

    if splits <= 1:
        return weight, bias
    oc = int(weight.shape[0])
    if oc % splits:
        raise SvdqLayoutError(
            f"out_features {oc} not divisible by adanorm_splits {splits}")
    w = weight.view(splits, oc // splits, -1).transpose(0, 1).reshape(oc, -1)
    if bias is None:
        raise SvdqLayoutError(
            "adanorm_splits > 1 requires a bias (adaLN folds 1+scale into it)")
    b = bias.reshape(splits, oc // splits).transpose(0, 1)
    delta = torch.zeros(splits, dtype=b.dtype, device=b.device)
    delta[1] = 1.0
    delta[splits - 2] = 1.0
    return w, (b + delta).reshape(oc)


def encode_awq_linear(
    weight: Any,
    bias: Optional[Any],
    *,
    group_size: int = 64,
    adanorm_splits: int = 1,
) -> dict[str, Any]:
    """Encode one synthetic Linear in the checkpoint's AWQ W4A16 layout."""
    import torch

    w = weight.detach().to(torch.float32)
    b = bias.detach().to(torch.float32) if bias is not None else None
    w, b = apply_adanorm_splits(w, b, int(adanorm_splits))
    oc, ic = int(w.shape[0]), int(w.shape[1])
    if ic % group_size:
        raise SvdqLayoutError(
            f"in_features {ic} not divisible by group size {group_size}")
    ng = ic // group_size
    grouped = w.reshape(oc, ng, group_size)
    lo, hi = grouped.amin(dim=-1), grouped.amax(dim=-1)
    scale = ((hi - lo) / 15.0).clamp(min=1e-6)
    zero_scaled = (-lo / scale) * scale
    codes = grouped.add(zero_scaled.unsqueeze(-1)).div(scale.unsqueeze(-1))
    packed = pack_w4x16(
        codes.round().clamp(0.0, 15.0).view(oc, ic).to(torch.uint8))
    rows = num_scale_rows(ic, group_size)
    wscales = torch.zeros((rows, oc), dtype=torch.bfloat16)
    wzeros = torch.zeros((rows, oc), dtype=torch.bfloat16)
    wscales[:ng] = scale.t().to(torch.bfloat16)
    wzeros[:ng] = zero_scaled.t().to(torch.bfloat16).neg_()
    out: dict[str, Any] = {
        "qweight": packed,
        "wscales": wscales,
        "wzeros": wzeros,
    }
    if b is not None:
        out["bias"] = b.to(torch.bfloat16)
    return out


__all__ = ["apply_adanorm_splits", "encode_awq_linear", "pack_w4x16"]
