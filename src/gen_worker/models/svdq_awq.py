"""AWQ W4A16 modulation layers in an svdq checkpoint (pgw#685 S2b).

An svdq artifact does not quantize everything the same way. The DiT's Linears are
W4A4 nvfp4 with a low-rank branch (``svdq_layout``), but the adaLN MODULATION
layers — ``img_mod``/``txt_mod``, which consume the timestep embedding rather
than the token stream — are AWQ **W4A16**: 4-bit weights, 16-bit activations,
group-64 scales. They are ~33% of a 20B artifact's parameters but a negligible
share of its FLOPs, so serving them dequantized to bf16 costs speed nothing and
keeps the native engine simple.

Layout is NOT the nvfp4 one and NOT a guess — it is the inverse of
deepcompressor's ``convert_to_nunchaku_w4x16_linear_weight`` ->
``convert_to_tinychat_w4x16y16_linear_weight`` -> ``pack_w4`` chain, and the test
suite inverts that upstream forward code bit-exactly:

  qweight  I32  [oc/4, ic/2]   TinyChat pack_w4: int16 pairs viewed as int32
  wscales  BF16 [ng, oc]       plain transpose of [oc, ng], zero-padded to
                               ``ceil_num_groups``
  wzeros   BF16 [ng, oc]       same, PRE-SCALED and PRE-NEGATED
  bias     BF16 [oc]

Dequant is therefore an ADD, not a subtract::

    W = codes * wscales + wzeros

``pack_w4``'s two stages, both inverted here: within each run of 32 input
elements, output nibble ``j`` packs elements ``{j, 8+j, 16+j, 24+j}``; then the
int16 grid is shuffled ``[oc/4, 4, ic/64, 16] -> permute(0, 2, 1, 3)``.

THE TRAP (``adanorm_splits``): for modulation layers the exporter also
(1) INTERLEAVES output channels — stored row ``j*splits + s`` is original row
``s*(oc/splits) + j`` — and (2) ADDS 1 to the bias of splits ``1`` and
``splits-2`` (adaLN's ``1 + scale`` folded into the artifact). A decoder that
skips either produces a plausible-looking weight and silently wrong images, so
:func:`decode_awq_linear` requires the split count explicitly and defaults to 1
(a plain W4A16 layer, no adaLN transform) rather than inferring it.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from .svdq_layout import SvdqLayoutError

logger = logging.getLogger(__name__)

# deepcompressor tinychat ceil_num_groups: pack_size = 32/4 = 8 int32-packed
# groups, and group_size 64 forces the pack count to be even.
_PACK_SIZE = 8
_NUM_PACKS_FACTOR = {32: 4, 64: 2}  # >=128 -> 1


def num_scale_rows(in_features: int, group_size: int) -> int:
    """``ceil_num_groups`` — the (padded) scale row count the exporter writes."""
    if in_features % group_size:
        raise SvdqLayoutError(
            f"in_features {in_features} not divisible by group size {group_size}")
    groups = in_features // group_size
    packs = -(-groups // _PACK_SIZE)
    factor = _NUM_PACKS_FACTOR.get(group_size, 1)
    packs = -(-packs // factor) * factor
    return packs * _PACK_SIZE


def unpack_w4x16(qweight: Any, out_features: int, in_features: int) -> Any:
    """TinyChat ``pack_w4`` inverse: packed int32 [oc/4, ic/2] -> 4-bit codes
    uint8 [oc, ic] (values 0-15)."""
    import torch

    oc, ic = int(out_features), int(in_features)
    if oc % 4 or ic % 64:
        raise SvdqLayoutError(
            f"AWQ W4A16 [{oc}, {ic}] needs oc%4==0 and ic%64==0")
    expected = (oc // 4, ic // 2)
    if tuple(qweight.shape) != expected:
        raise SvdqLayoutError(
            f"AWQ qweight shape {tuple(qweight.shape)} != expected {expected} "
            f"for [{oc}, {ic}]")
    # int32 storage is really pairs of int16 lanes; take them unsigned.
    x = qweight.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    x = x.view(oc // 4, ic // 64, 4, 16).permute(0, 2, 1, 3).reshape(-1, 8)
    nib = torch.stack([(x >> (4 * t)) & 0xF for t in range(4)], dim=1)
    return nib.reshape(oc, ic).to(torch.uint8)


def _scales_and_zeros(wscales: Any, wzeros: Any, out_features: int,
                      in_features: int) -> tuple[Any, Any, int]:
    """Stored [ng, oc] scale/zero grids -> [oc, ng_eff] plus the group size.

    Trailing rows can be ``ceil_num_groups`` PADDING (all zero). A real scale is
    never zero — the exporter divides by it — so the zero rows are the pad."""
    import torch

    oc, ic = int(out_features), int(in_features)
    if wscales.shape[1] != oc or wzeros.shape != wscales.shape:
        raise SvdqLayoutError(
            f"AWQ scale/zero grids {tuple(wscales.shape)}/"
            f"{tuple(wzeros.shape)} do not match out_features {oc}")
    s = wscales.float()
    live = (s != 0).any(dim=1)
    ng = int(torch.count_nonzero(live))
    if ng == 0 or not bool(live[:ng].all()):
        raise SvdqLayoutError(
            "AWQ wscales has zero rows before its last live row — not the "
            "ceil_num_groups padding this decoder understands")
    if ic % ng:
        raise SvdqLayoutError(
            f"in_features {ic} not divisible by {ng} live scale groups")
    gs = ic // ng
    if num_scale_rows(ic, gs) != int(wscales.shape[0]):
        raise SvdqLayoutError(
            f"AWQ scale rows {int(wscales.shape[0])} != ceil_num_groups("
            f"{ic}, {gs}) = {num_scale_rows(ic, gs)}")
    return s[:ng].t().contiguous(), wzeros.float()[:ng].t().contiguous(), gs


def dequantize_w4x16(qweight: Any, wscales: Any, wzeros: Any,
                     out_features: int, in_features: int) -> Any:
    """AWQ W4A16 tensors -> float32 weight [oc, ic].

    ``W = codes * wscales + wzeros`` — an ADD, because the exporter stored the
    zero point already scaled AND negated."""
    codes = unpack_w4x16(qweight, out_features, in_features)
    scale, zero, gs = _scales_and_zeros(wscales, wzeros, out_features,
                                        in_features)
    oc, ic = int(out_features), int(in_features)
    w = codes.float().reshape(oc, ic // gs, gs)
    return (w * scale.unsqueeze(-1) + zero.unsqueeze(-1)).reshape(oc, ic)


def undo_adanorm_splits(weight: Any, bias: Optional[Any], splits: int,
                        ) -> tuple[Any, Optional[Any]]:
    """Undo the exporter's adaLN transform.

    Forward was ``w.view(splits, oc//splits, ic).transpose(0, 1)`` on the rows
    plus ``+1`` on the bias of splits ``1`` and ``splits-2``. Both are inverted
    so the result matches the ORIGINAL diffusers module."""
    import torch

    if splits <= 1:
        return weight, bias
    oc = int(weight.shape[0])
    if oc % splits:
        raise SvdqLayoutError(
            f"out_features {oc} not divisible by adanorm_splits {splits}")
    per = oc // splits
    w = weight.view(per, splits, -1).transpose(0, 1).reshape(oc, -1)
    if bias is None:
        return w, None
    b = bias.float().view(per, splits).transpose(0, 1).contiguous()
    delta = torch.zeros(splits, dtype=b.dtype, device=b.device)
    delta[1] = 1.0
    delta[splits - 2] = 1.0
    b = (b - delta.reshape(splits, 1)).reshape(oc)
    return w, b


def decode_awq_linear(tensors: dict[str, Any], out_features: int,
                      in_features: int, *, adanorm_splits: int = 1,
                      compute_dtype: Any = None) -> Any:
    """One AWQ W4A16 layer -> a plain ``nn.Linear`` for the diffusers module.

    ``adanorm_splits`` MUST be given for a modulation layer (qwen ``img_mod``/
    ``txt_mod``: 6). It is never inferred — a wrong split count yields a
    plausible weight and silently wrong output."""
    import torch
    import torch.nn as nn

    missing = [k for k in ("qweight", "wscales", "wzeros") if k not in tensors]
    if missing:
        raise SvdqLayoutError(f"not an AWQ W4A16 linear (missing {missing})")
    compute = compute_dtype or torch.bfloat16
    w = dequantize_w4x16(tensors["qweight"], tensors["wscales"],
                         tensors["wzeros"], out_features, in_features)
    bias = tensors.get("bias")
    w, bias = undo_adanorm_splits(w, bias, int(adanorm_splits))
    lin = nn.Linear(int(in_features), int(out_features), bias=bias is not None,
                    dtype=compute)
    with torch.no_grad():
        lin.weight.copy_(w.to(compute))
        if bias is not None:
            lin.bias.copy_(bias.to(compute))
    lin.weight.requires_grad_(False)
    if lin.bias is not None:
        lin.bias.requires_grad_(False)
    return lin


def is_awq_linear(tensors: dict[str, Any]) -> bool:
    """``wzeros`` is the discriminator: the W4A4 svdq linears never carry it."""
    return "wzeros" in tensors and "qweight" in tensors


__all__ = [
    "decode_awq_linear",
    "dequantize_w4x16",
    "is_awq_linear",
    "num_scale_rows",
    "undo_adanorm_splits",
    "unpack_w4x16",
]
