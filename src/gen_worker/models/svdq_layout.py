"""nunchaku "v1 single-file" svdq layout -> our SvdqLinear buffers.

Converts an official SVDQuant checkpoint's per-Linear tensors into the layout
``torch._scaled_mm`` wants, so ONE module serves svdq-fp4 on every Blackwell
part instead of nunchaku's sm_120a-only kernels.

EVERY tensor in the file is swizzled, not just ``qweight``/``wscales``; reading
any of them verbatim renders noise. Each inverse below replays ONE named
deepcompressor forward packer backwards; the citations are exact, at
``nunchaku-ai/deepcompressor@main``:

  deepcompressor/backend/utils.py
    MmaWeightPackerBase.__init__            L91-141  (all geometry constants)
  deepcompressor/backend/nunchaku/utils.py
    NunchakuWeightPacker.pack_weight        L21-60   -> unpack_qweight
    NunchakuWeightPacker.pack_micro_scale   L109-151 -> unpack_wscales
    NunchakuWeightPacker.pack_scale         L62-107  -> unpack_vector
    NunchakuWeightPacker.unpack_lowrank_weight L184-214 -> unpack_lowrank
  deepcompressor/backend/nunchaku/convert.py
    L42-48  lora_down /= smooth  (the runtime branch therefore eats RAW x)
    L62-71  which tensor gets which name

  qweight   I8   [out, in/2]    mma.sync m16n8k64 FRAGMENT interleave
  wscales   E4M3 [in/16, out]   pack_micro_scale: (4, 4, 8) row split, NOT
                                (4, 8, 4) — see unpack_wscales
  wcscales  BF16 [out]          per-OUTPUT-CHANNEL 2nd-level scale (pack_scale
                                swizzled), or
  wtscale   BF16 [1]            per-TENSOR 2nd-level scale (scalar, unswizzled)
  proj_down BF16 [in, rank]     pack_lowrank_weight(down=True) of [rank, in]
  proj_up   BF16 [out, rank]    pack_lowrank_weight(down=False) of [out, rank]
  smooth_factor      BF16 [in]  pack_scale swizzled; DIVIDES the activation,
                                4-bit branch ONLY
  smooth_factor_orig BF16 [in]  provenance only — never read at runtime
  bias      BF16 [out]          pack_scale swizzled

QUANTIZED LOW-RANK BRANCH (cozy extension, LoRaQ arXiv 2604.18117):
the ``__metadata__`` key ``lowrank_quant`` = ``"int8"`` | ``"fp8_e4m3"``
declares that the branch pair is stored quantized. Per-block scales, block=32
along each factor's CONTRACTION dim (LoRaQ §4.3 quantizes by MX blocks of 32;
at 8 bits their MXFP8e4 edges MXINT8 — Table 3 — so both lanes ship and the
gate picks). nunchaku cannot serve a quantized-branch file regardless, so
these four tensors are PLAIN row-major — the 16x16 lowrank fragment pack is
defined for 16-bit operands only and is not applied:

  proj_down        I8|E4M3 [in, rank]     logical, NOT fragment-packed
  proj_down_scale  F32 [in/32, rank]      per-block absmax scale
  proj_up          I8|E4M3 [out, rank]    logical, NOT fragment-packed
  proj_up_scale    F32 [out, rank/32]     per-block absmax scale

Everything else in the file keeps the nunchaku-v1 layout above; a bf16-branch
artifact (no ``lowrank_quant`` key) is byte-identical to the historical format.

Decoding lands in a LOGICAL domain first (nibble codes [out, in] + flat scales
[out, in/16] + row-major vectors) for a reason: nunchaku fuses q/k/v into one
``to_qkv`` while diffusers keeps ``to_q``/``to_k``/``to_v`` separate, and
splitting is exact in logical space along the output dim but not in any of the
packed layouts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from .file_layout import MULTI_FILE
from .tensor_layout_contract import (
    BAKE_LOW_RANK_BRANCH,
    CONTRACT_COZY_SVDQ_NVFP4_LR8,
    CONTRACT_NUNCHAKU_V1,
    ELEMENT_BF16,
    ELEMENT_FP8_E4M3,
    ELEMENT_INT4,
    ELEMENT_NVFP4,
    SCALE_GROUP_16,
    SCALE_PER_CHANNEL_OUT,
    SCALE_PER_TENSOR,
    DecodeDimensions,
    implements_contract,
)
from .nvfp4_quant import BLOCK, pack_e2m1, to_blocked_scales

# deepcompressor MmaWeightPackerBase geometry, bits=4 / warp_n=128
# (backend/utils.py L91-141: comp_k = insn_k = mem_k = 256//bits = 64,
# mem_n = warp_n = 128, reg_k = 32//bits = 8, k_pack_size = n_pack_size = 2,
# num_k_lanes = 4, num_n_lanes = 8, num_k_packs = 1, num_n_packs = 8).
_FRAG = dict(mem_n=128, mem_k=64, num_n_packs=8, n_pack_size=2, num_n_lanes=8,
             reg_n=1, num_k_packs=1, k_pack_size=2, num_k_lanes=4, reg_k=8)
_FWD_PERM = (0, 5, 6, 1, 3, 8, 2, 7, 4, 9)

_WARP_N = 128
_INSN_K = 64
_NUM_K_UNROLLS = 2   # NunchakuWeightPacker.__init__ (nunchaku/utils.py L19)

# pack_micro_scale (nunchaku/utils.py L149-151), warp_n=128 => s_pack_size=4,
# num_s_lanes=32, num_s_packs=1. The 128 output channels of a tile split
# (s_pack_size=4, 4, 8) with strides (32, 8, 1) — the third factor is 8, not 4.
_MICRO_FWD_PERM = (0, 5, 1, 4, 3, 2, 6)
_MICRO_ROW_DIMS = (1, 4, 4, 8)          # (num_s_packs, s_pack_size, 4, 8)

# pack_scale, group_size=-1 (nunchaku/utils.py L105-107): the per-vector
# swizzle applied to wcscales, smooth_factor AND bias.
_VEC_FWD_PERM = (0, 6, 1, 2, 4, 3, 5)
_VEC_ROW_DIMS = (1, 8, 2, 4, 2)         # (num_s_packs, num_s_lanes//4,
                                        #  s_pack_size//2, 4, 2)

# pack_lowrank_weight (nunchaku/utils.py L153-182): reg_k is 2 for 16-bit
# operands (32 bits / 16 bits), so both pack tiles are 16x16.
_LR_REG_N, _LR_REG_K = 1, 2
_LR_PACK_N = 16                         # n_pack_size * num_n_lanes * reg_n
_LR_PACK_K = 16                         # k_pack_size * num_k_lanes * reg_k
_LR_PACK_PERM = (0, 1, 3, 6, 2, 5, 4, 7)    # forward, L181
_LR_UNPACK_PERM = (0, 1, 4, 2, 6, 5, 3, 7)  # inverse, L208

E2M1_LUT = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)

# Quantized low-rank branch.
LOWRANK_QUANT_KEY = "lowrank_quant"          # __metadata__ flag
LOWRANK_QUANT_SCHEMES = ("int8", "fp8_e4m3")
LOWRANK_QUANT_BLOCK = 32                     # LoRaQ MX block size
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
    codes: Any                    # uint8 [out, in], e2m1 nibble codes
    scales: Any                   # e4m3 [out, in/16], flat row-major
    second: Any                   # fp32 [out] (per_channel) | [1] (per_tensor)
    second_kind: str              # "per_channel" | "per_tensor"
    rank: int = 0
    proj_down: Optional[Any] = None   # [in, rank]
    proj_up: Optional[Any] = None     # [out, rank]
    smooth_factor: Optional[Any] = None  # [in]
    bias: Optional[Any] = None        # [out]
    lowrank_quant: str = "bf16"       # on-disk branch scheme (decoded to bf16)


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
    # pack_weight asserts k % (mem_k * num_k_unrolls) with num_k_unrolls = 2
    # (nunchaku/utils.py L26-29), and pad_weight pads to that (L221), so a real
    # checkpoint is always [%128, %128]. A looser k % 64 would admit a shape the
    # exporter cannot emit and the vector swizzles cannot express.
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
    """nunchaku ``wscales`` [in/16, out] -> flat row-major [out, in/16].

    Inverse of ``pack_micro_scale`` (deepcompressor nunchaku/utils.py L149-151)::

        scale.view(n // warp_s, num_s_packs, s_pack_size, 4, 8, -1,
                   insn_k // group_size).permute(0, 5, 1, 4, 3, 2, 6)

    With warp_n=128 that is dims ``(out/128, 1, 4, 4, 8, in/64, 4)``, i.e. a
    tile's 128 output channels split (4, 4, 8) with strides (32, 8, 1). The
    upstream store-order table at L136-148 is the check: stored 32-bit word j
    holds channel ``(j%4)*32 + ((j//4)%4)*8 + (j//16)``.

    (4, 8, 4) reshapes without error and silently permutes output channels
    within every 128-row tile, giving each channel another channel's block
    scales; a round trip against our own ``pack_wscales`` cannot catch it."""
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
    """One nunchaku ``pack_scale(group_size=-1)`` vector -> row-major [n].

    Inverse of deepcompressor nunchaku/utils.py L105-107::

        scale.view(n // warp_s, num_s_packs, num_s_lanes // 4,
                   s_pack_size // 2, 4, 2, -1).permute(0, 6, 1, 2, 4, 3, 5)

    warp_n=128 => dims ``(n/128, 1, 8, 2, 4, 2, 1)``. Applies to ``wcscales``,
    ``smooth_factor`` and ``bias`` — every one of them goes through
    ``pack_scale`` in ``convert_to_nunchaku_w4x4y16_linear_weight`` (L297-308).
    Upstream's store-order table (L93-104) is the check: stored positions
    0,1,2,3,... hold logical channels 0,1,8,9,2,3,10,11,...

    ``wtscale`` is exempt: it is reduced to ``scale.view(-1)[0]`` at L313-314,
    so a scalar carries no swizzle."""
    n = int(n)
    if n % _WARP_N:
        raise SvdqLayoutError(
            f"svdq packed vector length {n} is not a multiple of "
            f"warp_n={_WARP_N}")
    pre = (n // _WARP_N, *_VEC_ROW_DIMS, 1)
    v = vec.reshape(tuple(pre[i] for i in _VEC_FWD_PERM))
    return v.permute(*_inv_perm(_VEC_FWD_PERM)).contiguous().reshape(n)


def unpack_lowrank(weight: Any, *, down: bool) -> Any:
    """nunchaku ``proj_down``/``proj_up`` -> the plain low-rank factor.

    Verbatim port of deepcompressor ``unpack_lowrank_weight``
    (nunchaku/utils.py L184-214). Returns ``[rank, in]`` for ``down=True`` and
    ``[out, rank]`` for ``down=False`` — the shapes the exporter was HANDED,
    before ``pack_lowrank_weight`` rearranged them into the mma fragment layout
    and re-viewed the result as ``[in, rank]`` / ``[out, rank]``.

    Reading the stored tensors as plain matrices scrambles the rank-128 branch
    that carries SVDQuant's entire outlier budget."""
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


# Forward packers — the encode side of the same contract. Used by the producer
# to emit nunchaku-readable artifacts, and by the tests to synthesize
# checkpoints in THEIR layout. Serving only ever unpacks.
#
# A round trip through these proves BIJECTIVITY, not agreement with nunchaku: a
# pack/unpack pair can agree with each other and with nothing else. Anything
# asserting the CONVENTION must check against real upstream bytes — see
# tests/test_svdq_official_layout_pgw770.py.


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
    """row-major [n] -> nunchaku ``pack_scale(group_size=-1)`` order. The
    encode side for ``wcscales``, ``smooth_factor`` and ``bias``."""
    n = int(n)
    if n % _WARP_N:
        raise SvdqLayoutError(
            f"svdq vector length {n} is not a multiple of warp_n={_WARP_N}")
    v = vec.reshape(n // _WARP_N, *_VEC_ROW_DIMS, 1)
    return v.permute(*_VEC_FWD_PERM).contiguous().reshape(n)


def pack_lowrank(weight: Any, *, down: bool) -> Any:
    """``[rank, in]`` (down) / ``[out, rank]`` (up) -> nunchaku's stored form.
    Port of deepcompressor ``pack_lowrank_weight`` (nunchaku/utils.py L153-182)."""
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
    return w.reshape(c, r)   # down -> [in, rank], up -> [out, rank]


# ---------------------------------------------------------------------------
# Quantized low-rank branch. Plain row-major on disk — see header.
# ---------------------------------------------------------------------------


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
    """The scheme a stored factor's dtype declares, or "bf16" for any float."""
    import torch

    if t.dtype == torch.int8:
        return "int8"
    if t.dtype == torch.float8_e4m3fn:
        return "fp8_e4m3"
    return "bf16"


def _lowrank_block_absmax(w32: Any, block_dim: int) -> Any:
    """absmax per block of LOWRANK_QUANT_BLOCK along ``block_dim``:
    [n0, n1] -> [n0/B, n1] (dim 0) | [n0, n1/B] (dim 1)."""
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
    """One low-rank factor -> (quantized tensor, fp32 per-block scales).

    Symmetric absmax per block of :data:`LOWRANK_QUANT_BLOCK` along
    ``block_dim`` (the factor's contraction dim: 0 for ``proj_down`` [in, rank],
    1 for ``proj_up`` [out, rank]). Encode side for the producer and the tests;
    serving only dequantizes."""
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
    """The quantized-branch leg of :func:`decode_linear`."""
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


# ---------------------------------------------------------------------------
# Decode / split / pack.
# ---------------------------------------------------------------------------


@implements_contract(
    contract=CONTRACT_NUNCHAKU_V1,
    serves=("svdq-fp4-w4a4",),
    composes_lora=False,
    decodes=DecodeDimensions(
        elements=(ELEMENT_NVFP4, ELEMENT_INT4, ELEMENT_BF16),
        # `wscales` are group-of-16; the second level is per-channel
        # (`wcscales`) or per-tensor (`wtscale`) and the decoder branches on
        # which is present, so both are decoded shapes of ONE handle.
        scales=(SCALE_GROUP_16, SCALE_PER_CHANNEL_OUT, SCALE_PER_TENSOR),
        # UNCONSTRAINED, and that is a statement: the nunchaku descriptor
        # fixes the keys, so the fact lives on the HANDLE and a synonym for it
        # here would be a second home (th#1937 declined `contract.native`).
        key_topologies=(),
        # MULTI-FILE ONLY, and it is the artifact that is constrained, not the
        # checkpoint: the nunchaku file is one flat namespace, but BOTH svdq
        # engines refuse an artifact that is only that file —
        # `load_svdq_nunchaku_pipeline` and `load_svdq_native_pipeline` each
        # raise on `not art.component` ("a servable flavor must be a full
        # diffusers tree with the checkpoint under its denoiser directory"),
        # and `convert/svdq.py` builds exactly that.
        file_layouts=(MULTI_FILE,),
        bakes=(BAKE_LOW_RANK_BRANCH,),
    ),
    why="pgw#685: this is THE decoder of the nunchaku v1 single-file layout. "
        "The decoded SvdqLinear has no additive adapter branch (w8a8_lora is "
        "branch-capable on w8a8 / fp8-storage / plain bf16 only).",
)
@implements_contract(
    contract=CONTRACT_COZY_SVDQ_NVFP4_LR8,
    serves=("svdq-fp4-w4a4",),
    composes_lora=False,
    decodes=DecodeDimensions(
        elements=(ELEMENT_NVFP4, ELEMENT_INT4, ELEMENT_FP8_E4M3, ELEMENT_BF16),
        scales=(SCALE_GROUP_16, SCALE_PER_CHANNEL_OUT, SCALE_PER_TENSOR),
        key_topologies=(),
        file_layouts=(MULTI_FILE,),   # same two engine refusals as above
        # The QUANTIZED low-rank branch is what separates this handle from
        # nunchaku.v1@1, and the handle is where that fact lives — the ruled
        # tensor-set registry names the SET, not the scheme it is stored in.
        bakes=(BAKE_LOW_RANK_BRANCH,),
    ),
    why="te#148's quantized low-rank branch is a distinct MAJOR and this "
        "decoder branches on it (`lowrank_quant`), which is what makes the "
        "claim true rather than inherited from nunchaku.v1@1.",
)
def decode_linear(tensors: dict[str, Any], out_features: int,
                  in_features: int, *,
                  lowrank_quant: Optional[str] = None) -> DecodedLinear:
    """One nunchaku svdq Linear's tensors -> :class:`DecodedLinear`. Raises on
    a layout we do not understand rather than guessing.

    ``lowrank_quant`` is the artifact-level ``__metadata__`` declaration
    ("bf16" | "int8" | "fp8_e4m3"): when given, a branch stored in any OTHER
    scheme refuses — the flag and the bytes must agree. ``None`` decodes
    whatever the tensors self-describe (dtype is unambiguous, not a guess)."""

    missing = [k for k in ("qweight", "wscales") if k not in tensors]
    if missing:
        raise SvdqLayoutError(f"not an svdq W4A4 linear (missing {missing})")
    codes = unpack_qweight(tensors["qweight"], out_features, in_features)
    scales = unpack_wscales(tensors["wscales"], out_features, in_features)

    # wcscales / smooth_factor / bias all leave the exporter through
    # pack_scale(group_size=-1) and are NOT in channel order on disk.
    # wtscale escapes it: it is reduced to a scalar (nunchaku/utils.py L313).
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
            # Quantized branch: plain row-major on disk (see header).
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
            # Fragment-packed on disk. unpack_lowrank returns [rank, in] for
            # the down leg; SvdqLinear/fold_to_dense want [in, rank] for
            # `x @ down`.
            down = unpack_lowrank(down, down=True).t().contiguous()
            up = unpack_lowrank(up, down=False)

    smooth = tensors.get("smooth_factor")
    if smooth is not None:
        smooth = unpack_vector(smooth.reshape(-1), in_features)
    bias = tensors.get("bias")
    if bias is not None:
        bias = unpack_vector(bias.reshape(-1), out_features)
    # smooth_factor_orig is provenance and is deliberately NOT read.
    return DecodedLinear(
        out_features=int(out_features), in_features=int(in_features),
        codes=codes, scales=scales, second=second, second_kind=second_kind,
        rank=rank, proj_down=down, proj_up=up,
        smooth_factor=smooth, bias=bias, lowrank_quant=scheme,
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
            lowrank_quant=dec.lowrank_quant,
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
