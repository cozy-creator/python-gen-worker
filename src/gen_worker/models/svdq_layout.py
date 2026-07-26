"""nunchaku "v1 single-file" svdq layout -> our SvdqLinear buffers (pgw#685).

Converts an official SVDQuant checkpoint's per-Linear tensors into the layout
``torch._scaled_mm`` wants, so ONE module serves svdq-fp4 on every Blackwell
part instead of nunchaku's sm_120a-only kernels. Layout decoded and verified in
pgw#682 G-C against nunchaku + deepcompressor source (not inferred):

  qweight   I8   [out, in/2]    mma.sync m16n8k64 FRAGMENT interleave
  wscales   E4M3 [in/16, out]   transposed + 8-lane/stride-4/4-pack swizzle
  wcscales  BF16 [out]          per-OUTPUT-CHANNEL 2nd-level scale, or
  wtscale   BF16 [1]            per-TENSOR 2nd-level scale
  proj_down BF16 [in, rank]     low-rank branch, consumes RAW (unsmoothed) x
  proj_up   BF16 [out, rank]    stored un-transposed
  smooth_factor      BF16 [in]  DIVIDES the activation, 4-bit branch ONLY
  smooth_factor_orig BF16 [in]  provenance only — never read at runtime

Neither inverse is a plain transpose; both replay the packer's reshape/permute
backwards, which is guaranteed bijective, rather than re-deriving index math.

Decoding lands in a LOGICAL domain first (nibble codes [out, in] + flat scales
[out, in/16]) for a reason: nunchaku fuses q/k/v into one ``to_qkv`` while
diffusers keeps ``to_q``/``to_k``/``to_v`` separate, and splitting is exact in
logical space along the output dim but not in the packed/blocked layouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from .nvfp4_quant import BLOCK, pack_e2m1, to_blocked_scales

# deepcompressor MmaWeightPackerBase geometry, bits=4 / warp_n=128.
_FRAG = dict(mem_n=128, mem_k=64, num_n_packs=8, n_pack_size=2, num_n_lanes=8,
             reg_n=1, num_k_packs=1, k_pack_size=2, num_k_lanes=4, reg_k=8)
_FWD_PERM = (0, 5, 6, 1, 3, 8, 2, 7, 4, 9)
_SCALE_FWD_PERM = (0, 4, 3, 2, 1, 5)

E2M1_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


class SvdqLayoutError(ValueError):
    """A checkpoint layout we do not understand — never guessed around."""


def _inv_perm(perm: tuple[int, ...]) -> tuple[int, ...]:
    out = [0] * len(perm)
    for i, p in enumerate(perm):
        out[p] = i
    return tuple(out)


@dataclass(frozen=True)
class DecodedLinear:
    """One svdq Linear in the LOGICAL domain — row-major, un-swizzled."""

    out_features: int
    in_features: int
    codes: Any                    # uint8 [out, in], e2m1 nibble codes
    scales: Any                   # e4m3 [out, in/16], flat row-major
    second: Any                   # fp32 [out] (per_channel) | [1] (per_tensor)
    second_kind: str              # "per_channel" | "per_tensor"
    rank: int = 0
    proj_down: Optional[Any] = None   # [in, rank]
    proj_up: Optional[Any] = None     # [out, rank]
    smooth_factor: Optional[Any] = None  # [in]
    bias: Optional[Any] = None        # [out]


@dataclass(frozen=True)
class SvdqBuffers:
    """The device buffers an :class:`SvdqLinear` holds."""

    out_features: int
    in_features: int
    weight: Any            # uint8 [out, in/2] packed e2m1 (even elem LOW nibble)
    weight_scale: Any      # e4m3 1-D, cuBLAS blocked layout
    weight_scale_2: Any    # fp32 [1, out] (per_channel) | [1, 1] (per_tensor)
    second_kind: str
    rank: int = 0
    proj_down: Optional[Any] = None
    proj_up: Optional[Any] = None
    smooth_factor: Optional[Any] = None
    bias: Optional[Any] = None


# ---------------------------------------------------------------------------
# Fragment / swizzle inverses.
# ---------------------------------------------------------------------------


def unpack_qweight(qweight: Any, out_features: int, in_features: int) -> Any:
    """nunchaku ``qweight`` int8 [out, in/2] -> nibble codes uint8 [out, in].

    Row index in the packed tensor is NOT the output channel — the layout is an
    ``mma.sync m16n8k64`` fragment interleave (``packed[1,0]``'s low nibble is
    logical ``(o=48, k=0)``). Replays the packer backwards."""
    import torch

    g = _FRAG
    n, k = int(out_features), int(in_features)
    if n % g["mem_n"] or k % g["mem_k"]:
        raise SvdqLayoutError(
            f"svdq qweight [{n}, {k}] is not a multiple of the fragment tile "
            f"({g['mem_n']}x{g['mem_k']})")
    n_tiles, k_tiles = n // g["mem_n"], k // g["mem_k"]
    words = qweight.reshape(-1).view(torch.int32)
    nib = (words.unsqueeze(-1)
           >> torch.arange(0, 32, 4, dtype=torch.int32)) & 0xF
    logical = (n_tiles, g["num_n_packs"], g["n_pack_size"], g["num_n_lanes"],
               g["reg_n"], k_tiles, g["num_k_packs"], g["k_pack_size"],
               g["num_k_lanes"], g["reg_k"])
    nib = nib.reshape(tuple(logical[i] for i in _FWD_PERM))
    nib = nib.permute(*_inv_perm(_FWD_PERM)).contiguous()
    return nib.reshape(n, k).to(torch.uint8)


def unpack_wscales(wscales: Any, out_features: int, in_features: int) -> Any:
    """nunchaku ``wscales`` [in/16, out] -> flat row-major [out, in/16].
    Transposed AND lane-swizzled (8-lane group / stride-4 / 4-pack)."""
    ng = int(in_features) // BLOCK
    out_f = int(out_features)
    if out_f % 128 or ng % 4:
        raise SvdqLayoutError(
            f"svdq wscales for [{out_f}, {in_features}] break the lane swizzle "
            f"(out%128, (in/16)%4)")
    pre = (out_f // 128, 4, 8, 4, ng // 4, 4)
    s = wscales.reshape(tuple(pre[i] for i in _SCALE_FWD_PERM))
    s = s.permute(*_inv_perm(_SCALE_FWD_PERM)).contiguous()
    return s.reshape(out_f, ng)


# Forward packers — used to synthesize checkpoints in THEIR layout for the
# round-trip tests. Production only ever unpacks.


def pack_qweight(codes: Any) -> Any:
    """nibble codes [out, in] -> nunchaku ``qweight`` int8 [out, in/2]."""
    import torch

    g = _FRAG
    n, k = int(codes.shape[0]), int(codes.shape[1])
    n_tiles, k_tiles = n // g["mem_n"], k // g["mem_k"]
    w = codes.to(torch.int32).reshape(
        n_tiles, g["num_n_packs"], g["n_pack_size"], g["num_n_lanes"],
        g["reg_n"], k_tiles, g["num_k_packs"], g["k_pack_size"],
        g["num_k_lanes"], g["reg_k"])
    w = w.permute(*_FWD_PERM).contiguous().bitwise_and(0xF)
    w = w.bitwise_left_shift(torch.arange(0, 32, 4, dtype=torch.int32))
    return w.sum(dim=-1, dtype=torch.int32).view(torch.int8).view(n, -1)


def pack_wscales(scales: Any, out_features: int, in_features: int) -> Any:
    """flat [out, in/16] -> nunchaku ``wscales`` [in/16, out]."""
    ng = int(in_features) // BLOCK
    s = scales.reshape(int(out_features) // 128, 4, 8, 4, ng // 4, 4)
    return s.permute(*_SCALE_FWD_PERM).contiguous().reshape(ng, int(out_features))


# ---------------------------------------------------------------------------
# Decode / split / pack.
# ---------------------------------------------------------------------------


def decode_linear(tensors: dict[str, Any], out_features: int,
                  in_features: int) -> DecodedLinear:
    """One nunchaku svdq Linear's tensors -> :class:`DecodedLinear`. Raises on
    a layout we do not understand rather than guessing."""

    missing = [k for k in ("qweight", "wscales") if k not in tensors]
    if missing:
        raise SvdqLayoutError(f"not an svdq W4A4 linear (missing {missing})")
    codes = unpack_qweight(tensors["qweight"], out_features, in_features)
    scales = unpack_wscales(tensors["wscales"], out_features, in_features)

    if "wcscales" in tensors:
        second = tensors["wcscales"].float().reshape(-1)
        second_kind = "per_channel"
        if int(second.numel()) != int(out_features):
            raise SvdqLayoutError(
                f"wcscales numel {int(second.numel())} != out_features "
                f"{out_features}")
    elif "wtscale" in tensors:
        second = tensors["wtscale"].float().reshape(1)
        second_kind = "per_tensor"
    else:
        raise SvdqLayoutError("svdq linear has neither wcscales nor wtscale")

    down = tensors.get("proj_down")
    up = tensors.get("proj_up")
    if (down is None) != (up is None):
        raise SvdqLayoutError(
            "svdq linear has only one half of the low-rank branch")
    rank = int(down.shape[1]) if down is not None else 0
    if down is not None and up is not None:
        if int(down.shape[0]) != int(in_features):
            raise SvdqLayoutError(
                f"proj_down {tuple(down.shape)} does not start with "
                f"in_features {in_features}")
        if tuple(up.shape) != (int(out_features), rank):
            raise SvdqLayoutError(
                f"proj_up {tuple(up.shape)} != [out_features={out_features}, "
                f"rank={rank}]")
    # smooth_factor_orig is provenance and is deliberately NOT read.
    return DecodedLinear(
        out_features=int(out_features), in_features=int(in_features),
        codes=codes, scales=scales, second=second, second_kind=second_kind,
        rank=rank, proj_down=down, proj_up=up,
        smooth_factor=tensors.get("smooth_factor"),
        bias=tensors.get("bias"),
    )


def split_decoded(dec: DecodedLinear,
                  sections: Sequence[int]) -> tuple[DecodedLinear, ...]:
    """Split a FUSED linear (nunchaku's ``to_qkv`` / ``add_qkv_proj``) into the
    separate diffusers projections, along the output dim.

    Exact: every per-output-channel quantity slices by row, and the low-rank
    branch splits because ``y_i = (x @ proj_down) @ proj_up_i.T`` — the down
    projection is SHARED, only ``proj_up`` rows are partitioned. ``in``-sized
    quantities (``proj_down``, ``smooth_factor``) are shared verbatim."""
    total = sum(int(s) for s in sections)
    if total != dec.out_features:
        raise SvdqLayoutError(
            f"split sections {tuple(sections)} sum to {total}, not "
            f"out_features {dec.out_features}")
    out: list[DecodedLinear] = []
    start = 0
    for size in (int(s) for s in sections):
        stop = start + size
        out.append(DecodedLinear(
            out_features=size, in_features=dec.in_features,
            codes=dec.codes[start:stop].contiguous(),
            scales=dec.scales[start:stop].contiguous(),
            second=(dec.second[start:stop].contiguous()
                    if dec.second_kind == "per_channel" else dec.second),
            second_kind=dec.second_kind,
            rank=dec.rank,
            proj_down=dec.proj_down,
            proj_up=(dec.proj_up[start:stop].contiguous()
                     if dec.proj_up is not None else None),
            smooth_factor=dec.smooth_factor,
            bias=(dec.bias[start:stop].contiguous()
                  if dec.bias is not None else None),
        ))
        start = stop
    return tuple(out)


def to_buffers(dec: DecodedLinear) -> SvdqBuffers:
    """:class:`DecodedLinear` -> the device buffers SvdqLinear holds (our
    packed-nibble convention + the cuBLAS blocked scale layout)."""
    second = (dec.second.reshape(1, dec.out_features)
              if dec.second_kind == "per_channel" else dec.second.reshape(1, 1))
    return SvdqBuffers(
        out_features=dec.out_features, in_features=dec.in_features,
        weight=pack_e2m1(dec.codes), weight_scale=to_blocked_scales(dec.scales),
        weight_scale_2=second.float(), second_kind=dec.second_kind,
        rank=dec.rank, proj_down=dec.proj_down, proj_up=dec.proj_up,
        smooth_factor=dec.smooth_factor, bias=dec.bias,
    )


def convert_linear(tensors: dict[str, Any], out_features: int,
                   in_features: int) -> SvdqBuffers:
    """``decode_linear`` + ``to_buffers`` for the un-fused case."""
    return to_buffers(decode_linear(tensors, out_features, in_features))


def dequantize_decoded(dec: DecodedLinear) -> Any:
    """Reference dequant: ``W ~= e2m1(codes) * scales * (wcscales|wtscale)``.
    The bf16-fallback weight and the tests' parity reference."""
    import torch

    lut = torch.tensor(E2M1_LUT, dtype=torch.float32, device=dec.codes.device)
    vals = lut[dec.codes.long()]
    s = dec.scales.float().reshape(dec.out_features,
                                  dec.in_features // BLOCK, 1)
    w = (vals.reshape(dec.out_features, dec.in_features // BLOCK, BLOCK)
         * s).reshape(dec.out_features, dec.in_features)
    second = dec.second.float()
    return w * (second.reshape(-1, 1) if dec.second_kind == "per_channel"
                else second.reshape(()))


__all__ = [
    "DecodedLinear",
    "SvdqBuffers",
    "SvdqLayoutError",
    "convert_linear",
    "decode_linear",
    "dequantize_decoded",
    "pack_qweight",
    "pack_wscales",
    "split_decoded",
    "to_buffers",
    "unpack_qweight",
    "unpack_wscales",
]
