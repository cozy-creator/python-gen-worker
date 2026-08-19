"""GGML block-format dequantization as vectorized torch ops.

A GGML quantized tensor is a flat byte array cut into fixed-size BLOCKS. Each
block packs a scale (and for some types a min, or a second level of 6-bit
sub-scales) next to the packed weight nibbles/crumbs. Dequantizing is pure bit
arithmetic: split the block into its fields, shift, mask, scale. No kernel, no
vendor extension — every function here is a handful of batched torch ops that
run identically on CUDA, ROCm, MPS and CPU.

This is the half of the GGUF lane that lets quantized weights STAY quantized in
memory: :mod:`gen_worker.models.gguf_torch` holds the block bytes on the device
and calls in here once per forward, per layer, to materialize the weight for
one matmul and then drop it. Weight residency is the quantized size (2-4x under
bf16); the dequant is memory-bound and costs ~10-20% of step time.

Ported from city96's ComfyUI-GGUF ``dequant.py`` (Apache-2.0), whose numerics
are the de-facto reference for the diffusion GGUF ecosystem. The 13 types below
are the ones with a vectorized form; IQ1/IQ2/IQ3 (importance-matrix 2-3 bit)
have no batched decode and are deliberately NOT supported — a per-block numpy
fallback would be slower than not serving the rung at all, so an unsupported
qtype raises instead of silently degrading.

Correctness is pinned to the ``gguf`` package's own numpy implementation:
``tests/test_ggml_decode.py`` asserts BIT-EXACT equality against
``gguf.quants.dequantize`` over random block bytes for every type here.
"""

from __future__ import annotations

import functools
from typing import Any, Callable, Dict, Tuple

#: K-quant super-block: 256 weights, 8 or 16 sub-blocks with their own scales.
QK_K = 256

#: Bytes holding a K-quant super-block's packed 6-bit scale/min pairs.
K_SCALE_SIZE = 12

#: IQ4 codebook. The 4-bit index selects one of these 16 int8 levels, which are
#: spaced non-linearly to match a normal-ish weight distribution.
_IQ4_KVALUES: Tuple[int, ...] = (
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
)

#: Types torch holds natively — no block decode, just a dtype view.
_PASSTHROUGH_NAMES: Tuple[str, ...] = ("F32", "F16")


def _shifts(values: Tuple[int, ...], shape: Tuple[int, ...], device: Any) -> Any:
    import torch

    return torch.tensor(values, device=device, dtype=torch.uint8).reshape(shape)


def _split(blocks: Any, *widths: int) -> Tuple[Any, ...]:
    """Cut each block row into ``widths`` byte fields plus the remainder."""
    import torch

    dims = list(widths) + [blocks.shape[1] - sum(widths)]
    return tuple(torch.split(blocks, dims, dim=1))


def _to_uint32(x: Any) -> Any:
    """Little-endian u32 from 4 uint8 columns (torch has no uint32)."""
    import torch

    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8 | x[:, 2] << 16 | x[:, 3] << 24).unsqueeze(1)


def _to_uint16(x: Any) -> Any:
    import torch

    x = x.view(torch.uint8).to(torch.int32)
    return (x[:, 0] | x[:, 1] << 8).unsqueeze(1)


# --- full-precision -------------------------------------------------------

def _blocks_bf16(blocks: Any, block_size: int, type_size: int, dtype: Any) -> Any:
    import torch

    return (blocks.view(torch.int16).to(torch.int32) << 16).view(torch.float32)


# --- legacy quants (32-weight blocks) -------------------------------------

def _blocks_q8_0(blocks: Any, block_size: int, type_size: int, dtype: Any) -> Any:
    import torch

    d, x = _split(blocks, 2)
    return d.view(torch.float16).to(dtype) * x.view(torch.int8)


def _blocks_q5_1(blocks: Any, block_size: int, type_size: int, dtype: Any) -> Any:
    import torch

    n = blocks.shape[0]
    d, m, qh, qs = _split(blocks, 2, 2, 4)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)

    qh = _to_uint32(qh).reshape((n, 1)) >> torch.arange(
        32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape((n, -1, 1, block_size // 2)) >> _shifts((0, 4), (1, 1, 2, 1), d.device)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape((n, -1))
    return d * (ql | (qh << 4)) + m


def _blocks_q5_0(blocks: Any, block_size: int, type_size: int, dtype: Any) -> Any:
    import torch

    n = blocks.shape[0]
    d, qh, qs = _split(blocks, 2, 4)
    d = d.view(torch.float16).to(dtype)

    qh = _to_uint32(qh).reshape(n, 1) >> torch.arange(
        32, device=d.device, dtype=torch.int32).reshape(1, 32)
    ql = qs.reshape(n, -1, 1, block_size // 2) >> _shifts((0, 4), (1, 1, 2, 1), d.device)
    qh = (qh & 1).to(torch.uint8)
    ql = (ql & 0x0F).reshape(n, -1)
    return d * ((ql | (qh << 4)).to(torch.int8) - 16)


def _blocks_q4_1(blocks: Any, block_size: int, type_size: int, dtype: Any) -> Any:
    import torch

    n = blocks.shape[0]
    d, m, qs = _split(blocks, 2, 2)
    d = d.view(torch.float16).to(dtype)
    m = m.view(torch.float16).to(dtype)

    qs = qs.reshape((n, -1, 1, block_size // 2)) >> _shifts((0, 4), (1, 1, 2, 1), d.device)
    return d * (qs & 0x0F).reshape(n, -1) + m


def _blocks_q4_0(blocks: Any, block_size: int, type_size: int, dtype: Any) -> Any:
    import torch

    n = blocks.shape[0]
    d, qs = _split(blocks, 2)
    d = d.view(torch.float16).to(dtype)

    qs = qs.reshape((n, -1, 1, block_size // 2)) >> _shifts((0, 4), (1, 1, 2, 1), d.device)
    return d * ((qs & 0x0F).reshape((n, -1)).to(torch.int8) - 8)


# --- K quants (256-weight super-blocks) -----------------------------------

def _scale_min(scales: Any) -> Tuple[Any, Any]:
    """Unpack a K-quant's 12 scale bytes into 8 six-bit scales and 8 mins."""
    import torch

    n = scales.shape[0]
    scales = scales.view(torch.uint8).reshape((n, 3, 4))
    d, m, m_d = torch.split(scales, 1, dim=-2)
    sc = torch.cat([d & 0x3F, (m_d & 0x0F) | ((d >> 2) & 0x30)], dim=-1)
    mn = torch.cat([m & 0x3F, (m_d >> 4) | ((m >> 2) & 0x30)], dim=-1)
    return sc.reshape((n, 8)), mn.reshape((n, 8))


def _blocks_q6_k(blocks: Any, block_size: int, type_size: int, dtype: Any) -> Any:
    import torch

    n = blocks.shape[0]
    ql, qh, scales, d = _split(blocks, QK_K // 2, QK_K // 4, QK_K // 16)

    scales = scales.view(torch.int8).to(dtype)
    d = d.view(torch.float16).to(dtype)
    d = (d * scales).reshape((n, QK_K // 16, 1))

    ql = ql.reshape((n, -1, 1, 64)) >> _shifts((0, 4), (1, 1, 2, 1), d.device)
    ql = (ql & 0x0F).reshape((n, -1, 32))
    qh = qh.reshape((n, -1, 1, 32)) >> _shifts((0, 2, 4, 6), (1, 1, 4, 1), d.device)
    qh = (qh & 0x03).reshape((n, -1, 32))
    q = ((ql | (qh << 4)).to(torch.int8) - 32).reshape((n, QK_K // 16, -1))
    return (d * q).reshape((n, QK_K))


def _blocks_q5_k(blocks: Any, block_size: int, type_size: int, dtype: Any) -> Any:
    import torch

    n = blocks.shape[0]
    d, dmin, scales, qh, qs = _split(blocks, 2, 2, K_SCALE_SIZE, QK_K // 8)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    sc, mn = _scale_min(scales)
    d = (d * sc).reshape((n, -1, 1))
    dm = (dmin * mn).reshape((n, -1, 1))

    ql = qs.reshape((n, -1, 1, 32)) >> _shifts((0, 4), (1, 1, 2, 1), d.device)
    qh = qh.reshape((n, -1, 1, 32)) >> _shifts(tuple(range(8)), (1, 1, 8, 1), d.device)
    ql = (ql & 0x0F).reshape((n, -1, 32))
    qh = (qh & 0x01).reshape((n, -1, 32))
    return (d * (ql | (qh << 4)) - dm).reshape((n, QK_K))


def _blocks_q4_k(blocks: Any, block_size: int, type_size: int, dtype: Any) -> Any:
    import torch

    n = blocks.shape[0]
    d, dmin, scales, qs = _split(blocks, 2, 2, K_SCALE_SIZE)
    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    sc, mn = _scale_min(scales)
    d = (d * sc).reshape((n, -1, 1))
    dm = (dmin * mn).reshape((n, -1, 1))

    qs = qs.reshape((n, -1, 1, 32)) >> _shifts((0, 4), (1, 1, 2, 1), d.device)
    qs = (qs & 0x0F).reshape((n, -1, 32))
    return (d * qs - dm).reshape((n, QK_K))


def _blocks_q3_k(blocks: Any, block_size: int, type_size: int, dtype: Any) -> Any:
    import torch

    n = blocks.shape[0]
    hmask, qs, scales, d = _split(blocks, QK_K // 8, QK_K // 4, 12)
    d = d.view(torch.float16).to(dtype)

    lscales, hscales = scales[:, :8], scales[:, 8:]
    lscales = (lscales.reshape((n, 1, 8))
               >> _shifts((0, 4), (1, 2, 1), d.device)).reshape((n, 16))
    hscales = (hscales.reshape((n, 1, 4))
               >> _shifts((0, 2, 4, 6), (1, 4, 1), d.device)).reshape((n, 16))
    scales = (lscales & 0x0F) | ((hscales & 0x03) << 4)
    dl = (d * (scales.to(torch.int8) - 32)).reshape((n, 16, 1))

    ql = qs.reshape((n, -1, 1, 32)) >> _shifts((0, 2, 4, 6), (1, 1, 4, 1), d.device)
    qh = hmask.reshape((n, -1, 1, 32)) >> _shifts(tuple(range(8)), (1, 1, 8, 1), d.device)
    ql = ql.reshape((n, 16, QK_K // 16)) & 3
    qh = (qh.reshape((n, 16, QK_K // 16)) & 1) ^ 1
    q = ql.to(torch.int8) - (qh << 2).to(torch.int8)
    return (dl * q).reshape((n, QK_K))


def _blocks_q2_k(blocks: Any, block_size: int, type_size: int, dtype: Any) -> Any:
    n = blocks.shape[0]
    scales, qs, d, dmin = _split(blocks, QK_K // 16, QK_K // 4, 2)
    import torch

    d = d.view(torch.float16).to(dtype)
    dmin = dmin.view(torch.float16).to(dtype)

    dl = (d * (scales & 0x0F)).reshape((n, QK_K // 16, 1))
    ml = (dmin * (scales >> 4)).reshape((n, QK_K // 16, 1))

    qs = (qs.reshape((n, -1, 1, 32))
          >> _shifts((0, 2, 4, 6), (1, 1, 4, 1), d.device)) & 3
    qs = qs.reshape((n, QK_K // 16, 16))
    return (dl * qs - ml).reshape((n, -1))


# --- IQ4 (codebook) -------------------------------------------------------

def _kvalues(like: Any) -> Any:
    import torch

    return torch.tensor(_IQ4_KVALUES, device=like.device, dtype=torch.int8)


def _blocks_iq4_nl(blocks: Any, block_size: int, type_size: int, dtype: Any) -> Any:
    import torch

    n = blocks.shape[0]
    d, qs = _split(blocks, 2)
    d = d.view(torch.float16).to(dtype)

    qs = qs.reshape((n, -1, 1, block_size // 2)) >> _shifts((0, 4), (1, 1, 2, 1), d.device)
    qs = (qs & 0x0F).reshape((n, -1, 1)).to(torch.int64)
    table = _kvalues(qs).expand(*qs.shape[:-1], 16)
    qs = torch.gather(table, dim=-1, index=qs).reshape((n, -1))
    return d * qs


def _blocks_iq4_xs(blocks: Any, block_size: int, type_size: int, dtype: Any) -> Any:
    import torch

    n = blocks.shape[0]
    d, scales_h, scales_l, qs = _split(blocks, 2, 2, QK_K // 64)
    d = d.view(torch.float16).to(dtype)
    scales_h = _to_uint16(scales_h)

    shift_a = _shifts((0, 4), (1, 1, 2), d.device)
    shift_b = _shifts(tuple(2 * i for i in range(QK_K // 32)), (1, -1, 1), d.device)

    scales_l = (scales_l.reshape((n, -1, 1)) >> shift_a).reshape((n, -1)) & 0x0F
    scales_h = (scales_h.reshape((n, -1, 1)) >> shift_b).reshape(
        (n, -1)).to(torch.uint8) & 0x03
    scales = (scales_l | (scales_h << 4)).to(torch.int8) - 32
    dl = (d * scales.to(dtype)).reshape((n, -1, 1))

    qs = qs.reshape((n, -1, 1, 16)) >> shift_a.reshape((1, 1, 2, 1))
    qs = qs.reshape((n, -1, 32, 1)) & 0x0F
    table = _kvalues(qs).expand(*qs.shape[:-1], 16)
    qs = torch.gather(table, dim=-1, index=qs.to(torch.int64)).reshape((n, -1, 32))
    return (dl * qs).reshape((n, -1))


_Kernel = Callable[[Any, int, int, Any], Any]

#: qtype NAME -> block decoder. Keyed by name so the table needs no ``gguf``
#: import to exist; :func:`_kernels` resolves the names to the wire ids once.
_BY_NAME: Dict[str, _Kernel] = {
    "BF16": _blocks_bf16,
    "Q8_0": _blocks_q8_0,
    "Q5_1": _blocks_q5_1,
    "Q5_0": _blocks_q5_0,
    "Q4_1": _blocks_q4_1,
    "Q4_0": _blocks_q4_0,
    "Q6_K": _blocks_q6_k,
    "Q5_K": _blocks_q5_k,
    "Q4_K": _blocks_q4_k,
    "Q3_K": _blocks_q3_k,
    "Q2_K": _blocks_q2_k,
    "IQ4_NL": _blocks_iq4_nl,
    "IQ4_XS": _blocks_iq4_xs,
}


@functools.lru_cache(maxsize=1)
def _kernels() -> Dict[int, _Kernel]:
    import gguf

    out: Dict[int, _Kernel] = {}
    for name, fn in _BY_NAME.items():
        qtype = getattr(gguf.GGMLQuantizationType, name, None)
        if qtype is not None:
            out[int(qtype)] = fn
    return out


@functools.lru_cache(maxsize=1)
def passthrough_qtypes() -> frozenset[int]:
    """Types a torch tensor already holds natively (F32, F16)."""
    import gguf

    return frozenset(
        int(getattr(gguf.GGMLQuantizationType, n)) for n in _PASSTHROUGH_NAMES)


def supported_qtypes() -> frozenset[int]:
    """Every GGML type this lane can serve, block-decoded or native."""
    return frozenset(_kernels()) | passthrough_qtypes()


def is_supported(qtype: int) -> bool:
    return int(qtype) in supported_qtypes()


def qtype_name(qtype: int) -> str:
    import gguf

    try:
        return str(gguf.GGMLQuantizationType(int(qtype)).name)
    except ValueError:
        return f"<ggml type {int(qtype)}>"


def block_geometry(qtype: int) -> Tuple[int, int]:
    """``(weights per block, bytes per block)`` for one GGML type."""
    import gguf

    block_size, type_size = gguf.GGML_QUANT_SIZES[gguf.GGMLQuantizationType(int(qtype))]
    return int(block_size), int(type_size)


def dequantize(data: Any, qtype: int, shape: Any, *, dtype: Any = None) -> Any:
    """Decode packed GGML ``data`` into a dense tensor of logical ``shape``.

    ``data`` is the raw block bytes exactly as they sit in the store — any
    dtype whose bytes are the block stream; it is reinterpreted as uint8. The
    whole tensor decodes in ONE batched pass over ``(n_blocks, type_size)``:
    there is no python loop over blocks anywhere in this module.
    """
    import torch

    qtype = int(qtype)
    if qtype in passthrough_qtypes():
        return data.reshape(shape).to(dtype) if dtype is not None else data.reshape(shape)

    kernel = _kernels().get(qtype)
    if kernel is None:
        raise NotImplementedError(
            f"gguf-torch: no vectorized dequant for {qtype_name(qtype)}. "
            "IQ1/IQ2/IQ3 have no batched decode and are not served — "
            "pick a supported quant (Q2_K..Q8_0, IQ4_NL, IQ4_XS, BF16).")

    block_size, type_size = block_geometry(qtype)
    rows = data.reshape((-1, data.shape[-1])).view(torch.uint8)
    blocks = rows.reshape((rows.numel() // type_size, type_size))
    return kernel(blocks, block_size, type_size, dtype).reshape(shape)


__all__ = [
    "K_SCALE_SIZE",
    "QK_K",
    "block_geometry",
    "dequantize",
    "is_supported",
    "passthrough_qtypes",
    "qtype_name",
    "supported_qtypes",
]
