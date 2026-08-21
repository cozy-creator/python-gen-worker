"""nunchaku "v1 single-file" svdq layout -> our SvdqLinear buffers, each inverse replaying ONE named deepcompressor forward packer backwards. EVERY tensor in the file is swizzled — reading any verbatim renders noise: qweight I8 [out, in/2] is an mma.sync m16n8k64 fragment interleave; wscales E4M3 [in/16, out] is pack_micro_scale's (4, 4, 8) row split — NOT (4, 8, 4), which reshapes without error and silently permutes channels; wcscales/smooth_factor/bias are pack_scale-swizzled; proj_down/proj_up are pack_lowrank_weight-packed; wtscale is the one unswizzled scalar. smooth_factor DIVIDES the activation feeding the 4-bit branch ONLY — the low-rank branch consumes RAW x (proj_down is pre-divided by smooth at export). Quantized low-rank branch (metadata key lowrank_quant = "int8"|"fp8_e4m3"): those four tensors are PLAIN row-major with per-block-32 scales along each factor's contraction dim — the 16-bit fragment pack is not applied. Decode lands in a LOGICAL domain first because splitting nunchaku's fused to_qkv into diffusers' to_q/to_k/to_v is exact only there."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from .tensor_layout_contract import unregistered_decode_path
from .nvfp4_quant import BLOCK, pack_e2m1, to_blocked_scales

_FRAG = dict(mem_n=128, mem_k=64, num_n_packs=8, n_pack_size=2, num_n_lanes=8,
             reg_n=1, num_k_packs=1, k_pack_size=2, num_k_lanes=4, reg_k=8)
_FWD_PERM = (0, 5, 6, 1, 3, 8, 2, 7, 4, 9)

_WARP_N = 128
_INSN_K = 64
_NUM_K_UNROLLS = 2

_MICRO_FWD_PERM = (0, 5, 1, 4, 3, 2, 6)
_MICRO_ROW_DIMS = (1, 4, 4, 8)

_VEC_FWD_PERM = (0, 6, 1, 2, 4, 3, 5)
_VEC_ROW_DIMS = (1, 8, 2, 4, 2)

_LR_REG_N, _LR_REG_K = 1, 2
_LR_PACK_N = 16
_LR_PACK_K = 16
_LR_PACK_PERM = (0, 1, 3, 6, 2, 5, 4, 7)
_LR_UNPACK_PERM = (0, 1, 4, 2, 6, 5, 3, 7)

E2M1_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

LOWRANK_QUANT_KEY = "lowrank_quant"
LOWRANK_QUANT_SCHEMES = ("int8", "fp8_e4m3")
LOWRANK_QUANT_BLOCK = 32
_INT8_QMAX = 127.0
_E4M3_QMAX = 448.0


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
    codes: Any
    scales: Any
    second: Any
    second_kind: str
    rank: int = 0
    proj_down: Optional[Any] = None
    proj_up: Optional[Any] = None
    smooth_factor: Optional[Any] = None
    bias: Optional[Any] = None
    lowrank_quant: str = "bf16"


@dataclass(frozen=True)
class SvdqBuffers:
    """The device buffers an :class:`SvdqLinear` holds."""

    out_features: int
    in_features: int
    weight: Any
    weight_scale: Any
    weight_scale_2: Any
    second_kind: str
    rank: int = 0
    proj_down: Optional[Any] = None
    proj_up: Optional[Any] = None
    smooth_factor: Optional[Any] = None
    bias: Optional[Any] = None


def unpack_qweight(qweight: Any, out_features: int, in_features: int) -> Any:
    """nunchaku ``qweight`` int8 [out, in/2] -> nibble codes uint8 [out, in]."""
    import torch

    g = _FRAG
    n, k = int(out_features), int(in_features)
    if n % g["mem_n"] or k % (g["mem_k"] * _NUM_K_UNROLLS):
        raise SvdqLayoutError(
            f"svdq qweight [{n}, {k}] is not a multiple of the padded "
            f"fragment tile ({g['mem_n']}x{g['mem_k'] * _NUM_K_UNROLLS})")
    n_tiles, k_tiles = n // g["mem_n"], k // g["mem_k"]
    words = qweight.reshape(-1).view(torch.int32)
    nib = (words.unsqueeze(-1)
           >> torch.arange(0, 32, 4, dtype=torch.int32,
                           device=words.device)) & 0xF
    logical = (n_tiles, g["num_n_packs"], g["n_pack_size"], g["num_n_lanes"],
               g["reg_n"], k_tiles, g["num_k_packs"], g["k_pack_size"],
               g["num_k_lanes"], g["reg_k"])
    nib = nib.reshape(tuple(logical[i] for i in _FWD_PERM))
    nib = nib.permute(*_inv_perm(_FWD_PERM)).contiguous()
    return nib.reshape(n, k).to(torch.uint8)


def _micro_logical(out_f: int, ng: int) -> tuple[int, ...]:
    return (out_f // _WARP_N, *_MICRO_ROW_DIMS, ng // 4, _INSN_K // BLOCK)


def unpack_wscales(wscales: Any, out_features: int, in_features: int) -> Any:
    """nunchaku ``wscales`` [in/16, out] -> flat row-major [out, in/16]."""
    ng = int(in_features) // BLOCK
    out_f = int(out_features)
    if out_f % _WARP_N or ng % 4:
        raise SvdqLayoutError(
            f"svdq wscales for [{out_f}, {in_features}] break the lane swizzle "
            f"(out%{_WARP_N}, (in/16)%4)")
    pre = _micro_logical(out_f, ng)
    s = wscales.reshape(tuple(pre[i] for i in _MICRO_FWD_PERM))
    s = s.permute(*_inv_perm(_MICRO_FWD_PERM)).contiguous()
    return s.reshape(out_f, ng)


def unpack_vector(vec: Any, n: int) -> Any:
    """One nunchaku ``pack_scale(group_size=-1)`` vector -> row-major [n]."""
    n = int(n)
    if n % _WARP_N:
        raise SvdqLayoutError(
            f"svdq packed vector length {n} is not a multiple of "
            f"warp_n={_WARP_N}")
    pre = (n // _WARP_N, *_VEC_ROW_DIMS, 1)
    v = vec.reshape(tuple(pre[i] for i in _VEC_FWD_PERM))
    return v.permute(*_inv_perm(_VEC_FWD_PERM)).contiguous().reshape(n)


def unpack_lowrank(weight: Any, *, down: bool) -> Any:
    """nunchaku ``proj_down``/``proj_up`` -> the plain low-rank factor."""
    c, r = int(weight.shape[0]), int(weight.shape[1])
    if down:
        if r % _LR_PACK_N or c % _LR_PACK_K:
            raise SvdqLayoutError(
                f"svdq proj_down [{c}, {r}] breaks the low-rank pack "
                f"(rank%{_LR_PACK_N}, in%{_LR_PACK_K})")
        r_packs, c_packs = r // _LR_PACK_N, c // _LR_PACK_K
    else:
        if c % _LR_PACK_N or r % _LR_PACK_K:
            raise SvdqLayoutError(
                f"svdq proj_up [{c}, {r}] breaks the low-rank pack "
                f"(out%{_LR_PACK_N}, rank%{_LR_PACK_K})")
        c_packs, r_packs = c // _LR_PACK_N, r // _LR_PACK_K
    g = _FRAG
    w = weight.reshape(c_packs, r_packs, g["num_n_lanes"], g["num_k_lanes"],
                       g["n_pack_size"], g["k_pack_size"], _LR_REG_N, _LR_REG_K)
    w = w.permute(*_LR_UNPACK_PERM).contiguous()
    w = w.reshape(c_packs, r_packs, _LR_PACK_N, _LR_PACK_K)
    if down:
        return w.permute(1, 2, 0, 3).contiguous().reshape(r, c)
    return w.permute(0, 2, 1, 3).contiguous().reshape(c, r)


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
    w = w.bitwise_left_shift(torch.arange(0, 32, 4, dtype=torch.int32,
                                          device=w.device))
    return w.sum(dim=-1, dtype=torch.int32).view(torch.int8).view(n, -1)


def pack_wscales(scales: Any, out_features: int, in_features: int) -> Any:
    """flat [out, in/16] -> nunchaku ``wscales`` [in/16, out] (pack_micro_scale)."""
    out_f, ng = int(out_features), int(in_features) // BLOCK
    s = scales.reshape(*_micro_logical(out_f, ng))
    return s.permute(*_MICRO_FWD_PERM).contiguous().reshape(ng, out_f)


def pack_vector(vec: Any, n: int) -> Any:
    """row-major [n] -> nunchaku ``pack_scale(group_size=-1)`` order."""
    n = int(n)
    if n % _WARP_N:
        raise SvdqLayoutError(
            f"svdq vector length {n} is not a multiple of warp_n={_WARP_N}")
    v = vec.reshape(n // _WARP_N, *_VEC_ROW_DIMS, 1)
    return v.permute(*_VEC_FWD_PERM).contiguous().reshape(n)


def pack_lowrank(weight: Any, *, down: bool) -> Any:
    """``[rank, in]`` (down) / ``[out, rank]`` (up) -> nunchaku's stored form."""
    g = _FRAG
    if down:
        r, c = int(weight.shape[0]), int(weight.shape[1])
        r_packs, c_packs = r // _LR_PACK_N, c // _LR_PACK_K
        if r % _LR_PACK_N or c % _LR_PACK_K:
            raise SvdqLayoutError(
                f"svdq low-rank down [{r}, {c}] breaks the 16x16 pack")
        w = weight.reshape(r_packs, _LR_PACK_N, c_packs,
                           _LR_PACK_K).permute(2, 0, 1, 3)
    else:
        c, r = int(weight.shape[0]), int(weight.shape[1])
        c_packs, r_packs = c // _LR_PACK_N, r // _LR_PACK_K
        if c % _LR_PACK_N or r % _LR_PACK_K:
            raise SvdqLayoutError(
                f"svdq low-rank up [{c}, {r}] breaks the 16x16 pack")
        w = weight.reshape(c_packs, _LR_PACK_N, r_packs,
                           _LR_PACK_K).permute(0, 2, 1, 3)
    w = w.reshape(c_packs, r_packs, g["n_pack_size"], g["num_n_lanes"],
                  _LR_REG_N, g["k_pack_size"], g["num_k_lanes"], _LR_REG_K)
    w = w.permute(*_LR_PACK_PERM).contiguous()
    return w.reshape(c, r)


def _lowrank_qdtype(scheme: str) -> Any:
    import torch

    if scheme == "int8":
        return torch.int8
    if scheme == "fp8_e4m3":
        return torch.float8_e4m3fn
    raise SvdqLayoutError(
        f"unknown lowrank_quant scheme {scheme!r} "
        f"(known: {', '.join(LOWRANK_QUANT_SCHEMES)})")


def _lowrank_scheme_of(t: Any) -> str:
    import torch

    if t.dtype == torch.int8:
        return "int8"
    if t.dtype == torch.float8_e4m3fn:
        return "fp8_e4m3"
    return "bf16"


def _lowrank_block_absmax(w32: Any, block_dim: int) -> Any:
    if block_dim not in (0, 1):
        raise SvdqLayoutError(f"block_dim must be 0 or 1, got {block_dim}")
    n0, n1 = int(w32.shape[0]), int(w32.shape[1])
    n = n0 if block_dim == 0 else n1
    if n % LOWRANK_QUANT_BLOCK:
        raise SvdqLayoutError(
            f"low-rank factor dim {n} is not a multiple of the quant block "
            f"{LOWRANK_QUANT_BLOCK}")
    nb = n // LOWRANK_QUANT_BLOCK
    if block_dim == 0:
        return w32.reshape(nb, LOWRANK_QUANT_BLOCK, n1).abs().amax(dim=1)
    return w32.reshape(n0, nb, LOWRANK_QUANT_BLOCK).abs().amax(dim=2)


def quantize_lowrank(w: Any, *, scheme: str,
                     block_dim: int) -> tuple[Any, Any]:
    """One low-rank factor -> (quantized tensor, fp32 per-block scales)."""
    import torch

    qdtype = _lowrank_qdtype(scheme)
    w32 = w.float()
    qmax = _INT8_QMAX if scheme == "int8" else _E4M3_QMAX
    scale = (_lowrank_block_absmax(w32, block_dim) / qmax).clamp(min=1e-12)
    scaled = w32 / scale.repeat_interleave(LOWRANK_QUANT_BLOCK, dim=block_dim)
    if scheme == "int8":
        q = scaled.round().clamp(-_INT8_QMAX, _INT8_QMAX).to(torch.int8)
    else:
        q = scaled.clamp(-_E4M3_QMAX, _E4M3_QMAX).to(qdtype)
    return q, scale.float()


def dequantize_lowrank(q: Any, scale: Any, *, block_dim: int) -> Any:
    """(quantized factor, per-block scales) -> the bf16 logical factor."""
    import torch

    scheme = _lowrank_scheme_of(q)
    if scheme == "bf16":
        raise SvdqLayoutError(
            f"dequantize_lowrank got a {q.dtype} tensor — not a quantized "
            f"low-rank factor")
    n = int(q.shape[block_dim])
    nb = n // LOWRANK_QUANT_BLOCK
    if block_dim == 0:
        want = (nb, int(q.shape[1]))
    else:
        want = (int(q.shape[0]), nb)
    if tuple(scale.shape) != want:
        raise SvdqLayoutError(
            f"low-rank scale shape {tuple(scale.shape)} != {want} for a "
            f"{tuple(q.shape)} factor blocked along dim {block_dim}")
    s = scale.float().repeat_interleave(LOWRANK_QUANT_BLOCK, dim=block_dim)
    return (q.float() * s).to(torch.bfloat16)


def _decode_quantized_lowrank(
    tensors: dict[str, Any], out_features: int, in_features: int,
) -> tuple[Any, Any, int, str]:
    down, up = tensors["proj_down"], tensors["proj_up"]
    dscale = tensors.get("proj_down_scale")
    uscale = tensors.get("proj_up_scale")
    scheme = _lowrank_scheme_of(down)
    if _lowrank_scheme_of(up) != scheme:
        raise SvdqLayoutError(
            f"low-rank pair mixes schemes: proj_down {down.dtype} vs "
            f"proj_up {up.dtype}")
    if dscale is None or uscale is None:
        raise SvdqLayoutError(
            f"quantized ({scheme}) low-rank branch is missing "
            f"{'proj_down_scale' if dscale is None else 'proj_up_scale'}")
    rank = int(down.shape[1])
    if tuple(down.shape) != (int(in_features), rank):
        raise SvdqLayoutError(
            f"quantized proj_down {tuple(down.shape)} != "
            f"[in_features={in_features}, rank]")
    if tuple(up.shape) != (int(out_features), rank):
        raise SvdqLayoutError(
            f"quantized proj_up {tuple(up.shape)} != "
            f"[out_features={out_features}, rank={rank}]")
    return (dequantize_lowrank(down, dscale, block_dim=0),
            dequantize_lowrank(up, uscale, block_dim=1), rank, scheme)


# NO QUANT RULE NAMES THESE BYTES (pgw#1621). The two v1 handles this decoder
# used to declare — `nunchaku.v1@1` and `cozy.svdq-nvfp4-lr8@1` — died with the
# v1 corpus, and neither has a ratified v2 successor: the eight rules in
# tensorfs `spec/v2/rules/` are the three plain dtypes, three fp8 packagings
# and two nvfp4 packagings, and this is none of them.
#
# It is NOT either nvfp4 rule, and the temptation to say otherwise is exactly
# what those two rules' descriptions warn about. `cozy.nvfp4-flat@1` is LOW-
# nibble e2m1 with FLAT [out, in/16] block scales; `bfl.nvfp4-preswizzled@1` is
# HIGH-nibble with cuBLAS-tiled scales; reading one as the other measured LPIPS
# 1.11. What this module reads is a THIRD packaging again — I8 [out, in/2] in
# deepcompressor's mma.sync m16n8k64 FRAGMENT interleave, E4M3 [in/16, out]
# micro-scales in the (4, 4, 8) row split, every vector (`wcscales`,
# `smooth_factor`, `bias`) swizzled by `pack_scale`, and a BAKED low-rank
# branch that is part of the arithmetic rather than a side tensor.
@unregistered_decode_path(
    reason="nunchaku's 'v1 single-file' SVDQ packaging, and its cozy "
           "quantized-low-rank-branch extension (`__metadata__.lowrank_quant` "
           "= int8|fp8_e4m3, te#148). No ratified v2 quant rule names either: "
           "the weights are I8 [out, in/2] in deepcompressor's mma.sync "
           "m16n8k64 FRAGMENT interleave with E4M3 [in/16, out] micro-scales "
           "in the (4, 4, 8) row split, swizzled per-channel (`wcscales`) or "
           "per-tensor (`wtscale`) second-level scales, and a BAKED low-rank "
           "branch. That is a THIRD 4-bit packaging — not `cozy.nvfp4-flat@1` "
           "(LOW-nibble, flat [out, in/16] scales, no low-rank branch) and not "
           "`bfl.nvfp4-preswizzled@1` (HIGH-nibble, cuBLAS-tiled scales); "
           "aliasing either measured LPIPS 1.11. What closes this gap is a "
           "ratified `spec/v2/rules/` document in tensorfs whose conventions "
           "carry the fragment interleave, the micro-scale pack, the "
           "second-level rank and the low-rank-branch scheme as IDENTITY, plus "
           "a re-vendor per `_vendor/VENDORED.toml`. Until then this decoder "
           "contributes no rule and no execution lane: pgw#685's "
           "`svdq-fp4-w4a4` body is real code that the decode-set cannot name.",
)
def decode_linear(tensors: dict[str, Any], out_features: int,
                  in_features: int, *,
                  lowrank_quant: Optional[str] = None) -> DecodedLinear:
    """One nunchaku svdq Linear's tensors -> :class:`DecodedLinear`."""

    missing = [k for k in ("qweight", "wscales") if k not in tensors]
    if missing:
        raise SvdqLayoutError(f"not an svdq W4A4 linear (missing {missing})")
    codes = unpack_qweight(tensors["qweight"], out_features, in_features)
    scales = unpack_wscales(tensors["wscales"], out_features, in_features)

    if "wcscales" in tensors:
        if int(tensors["wcscales"].numel()) != int(out_features):
            raise SvdqLayoutError(
                f"wcscales numel {int(tensors['wcscales'].numel())} != "
                f"out_features {out_features}")
        second = unpack_vector(tensors["wcscales"].float().reshape(-1),
                              out_features)
        second_kind = "per_channel"
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
    if lowrank_quant is not None and lowrank_quant not in (
            ("bf16",) + LOWRANK_QUANT_SCHEMES):
        raise SvdqLayoutError(
            f"unknown lowrank_quant declaration {lowrank_quant!r} "
            f"(known: bf16, {', '.join(LOWRANK_QUANT_SCHEMES)})")
    rank = 0
    scheme = "bf16"
    if down is not None and up is not None:
        scheme = _lowrank_scheme_of(down)
        if scheme == "bf16" and ("proj_down_scale" in tensors
                                 or "proj_up_scale" in tensors):
            raise SvdqLayoutError(
                "low-rank scale tensors present but the branch pair is not "
                "stored in a quantized dtype")
        if lowrank_quant is not None and scheme != lowrank_quant:
            raise SvdqLayoutError(
                f"artifact declares lowrank_quant={lowrank_quant!r} but the "
                f"branch is stored as {scheme!r}")
        if scheme != "bf16":
            down, up, rank, scheme = _decode_quantized_lowrank(
                tensors, out_features, in_features)
        else:
            rank = int(down.shape[1])
            if int(down.shape[0]) != int(in_features):
                raise SvdqLayoutError(
                    f"proj_down {tuple(down.shape)} does not start with "
                    f"in_features {in_features}")
            if tuple(up.shape) != (int(out_features), rank):
                raise SvdqLayoutError(
                    f"proj_up {tuple(up.shape)} != "
                    f"[out_features={out_features}, rank={rank}]")
            down = unpack_lowrank(down, down=True).t().contiguous()
            up = unpack_lowrank(up, down=False)

    smooth = tensors.get("smooth_factor")
    if smooth is not None:
        smooth = unpack_vector(smooth.reshape(-1), in_features)
    bias = tensors.get("bias")
    if bias is not None:
        bias = unpack_vector(bias.reshape(-1), out_features)
    return DecodedLinear(
        out_features=int(out_features), in_features=int(in_features),
        codes=codes, scales=scales, second=second, second_kind=second_kind,
        rank=rank, proj_down=down, proj_up=up,
        smooth_factor=smooth, bias=bias, lowrank_quant=scheme,
    )


def split_decoded(dec: DecodedLinear,
                  sections: Sequence[int]) -> tuple[DecodedLinear, ...]:
    """Split a FUSED linear (nunchaku's ``to_qkv`` / ``add_qkv_proj``) into the separate diffusers projections, along the output dim."""
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
            lowrank_quant=dec.lowrank_quant,
        ))
        start = stop
    return tuple(out)


def to_buffers(dec: DecodedLinear) -> SvdqBuffers:
    """:class:`DecodedLinear` -> the device buffers SvdqLinear holds (our packed-nibble convention + the cuBLAS blocked scale layout)."""
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
    """Reference dequant: ``W ~= e2m1(codes) * scales * (wcscales|wtscale)``."""
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
    "LOWRANK_QUANT_BLOCK",
    "LOWRANK_QUANT_KEY",
    "LOWRANK_QUANT_SCHEMES",
    "SvdqBuffers",
    "SvdqLayoutError",
    "convert_linear",
    "decode_linear",
    "dequantize_decoded",
    "dequantize_lowrank",
    "pack_lowrank",
    "pack_qweight",
    "pack_vector",
    "pack_wscales",
    "quantize_lowrank",
    "split_decoded",
    "to_buffers",
    "unpack_lowrank",
    "unpack_qweight",
    "unpack_vector",
    "unpack_wscales",
]
