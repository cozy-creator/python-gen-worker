"""Block-sparse attention MECHANISM: turns block scores into a FlexAttention BlockMask and runs the attention; the per-model vocabulary lives in the endpoint, and it is not a selector. Two measured landmines: (1) there is NO eager mode — eager flex_attention silently IGNORES a manually built BlockMask's block structure and computes dense (measured: eager == dense to 1e-7 under a 54%-kept mask), and OOMs at video-DiT shapes, so every call goes through the compiled callable; (2) block_mask is a KEYWORD argument — flex_attention's fourth positional parameter is score_mod, so passing the mask positionally runs dense and announces itself only as a 299 GiB allocation failure."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from . import settings_authority
from .hostfacts import cuda_ready

logger = logging.getLogger(__name__)

DEFAULT_BLOCK = 128

_FLEX: dict = {}


class SparseUnavailable(RuntimeError):
    """The block-sparse path cannot run here."""


@dataclass(frozen=True)
class BlockGeometry:
    """Everything the mask builder needs about one packed sequence."""

    seq_len: int
    n_global: int = 0
    block: int = DEFAULT_BLOCK

    @property
    def n_blocks(self) -> int:
        return (self.seq_len + self.block - 1) // self.block

    @property
    def padded_len(self) -> int:
        return self.n_blocks * self.block

    @property
    def global_blocks(self) -> int:
        return max(1, (self.n_global + self.block - 1) // self.block) \
            if self.n_global > 0 else 0


def _bits() -> dict:
    if _FLEX:
        return _FLEX
    try:
        import torch
        from torch.nn.attention.flex_attention import BlockMask, flex_attention
    except Exception as exc:  # noqa: BLE001
        raise SparseUnavailable(f"flex_attention unimportable: {exc}") from exc
    settings_authority.raise_dynamo_cache_limits(64)
    _FLEX["BlockMask"] = BlockMask
    _FLEX["compiled"] = torch.compile(flex_attention, dynamic=False)
    return _FLEX


def probe() -> str:
    """"" if the block-sparse path is runnable here, else why not."""
    try:
        import torch  # noqa: F401 — the import IS the probe here
    except Exception as exc:  # noqa: BLE001
        return f"torch unimportable: {exc}"
    if not cuda_ready():
        return "no CUDA device"
    try:
        _bits()
    except SparseUnavailable as exc:
        return str(exc)
    return ""


def build_block_mask(scores: Any, k_blocks: int, geom: BlockGeometry,
                     heads: int) -> Any:
    """(X, NQ, NB) block scores -> ``BlockMask`` keeping the top ``k_blocks`` per row plus the forced local diagonal and the global prefix."""
    import torch

    bits = _bits()
    X, NQ, NB = scores.shape
    if NB != geom.n_blocks or NQ != geom.n_blocks:
        raise SparseUnavailable(
            f"score shape {tuple(scores.shape)} does not match geometry "
            f"(n_blocks={geom.n_blocks})")
    if heads % X:
        raise SparseUnavailable(f"heads {heads} not divisible by groups {X}")
    dev = scores.device
    B, g = geom.block, geom.global_blocks

    mark = torch.zeros((X, NQ, NB), dtype=torch.int32, device=dev)
    mark.scatter_(2, scores.topk(min(NB, max(1, int(k_blocks))), dim=-1).indices, 1)
    rows = torch.arange(NQ, device=dev)
    mark[:, rows, rows] = 1
    if g:
        mark[:, :, :g] = 1
        mark[:, rows < g, :] = 1

    last_partial = geom.padded_len != geom.seq_len
    part_num = torch.zeros((X, NQ), dtype=torch.int32, device=dev)
    part_idx = torch.zeros((X, NQ, NB), dtype=torch.int32, device=dev)
    if last_partial:
        part_num = mark[..., NB - 1].clone()
        part_idx[..., 0] = NB - 1
        mark[..., NB - 1] = 0

    pos = mark.cumsum(-1) - 1
    full_num = mark.sum(-1).to(torch.int32)
    ar = torch.arange(NB, device=dev, dtype=torch.int32).expand(X, NQ, NB)
    buf = torch.zeros((X, NQ, NB + 1), dtype=torch.int32, device=dev)
    buf.scatter_(2, torch.where(mark.bool(), pos, torch.full_like(pos, NB)), ar)
    full_idx = buf[..., :NB].contiguous()

    if X != heads:
        rep = heads // X
        full_num = full_num.repeat_interleave(rep, 0)
        full_idx = full_idx.repeat_interleave(rep, 0)
        part_num = part_num.repeat_interleave(rep, 0)
        part_idx = part_idx.repeat_interleave(rep, 0)

    seq_len = geom.seq_len

    def pad_mask_mod(b: Any, h: Any, qi: Any, ki: Any) -> Any:
        return ki < seq_len

    return bits["BlockMask"].from_kv_blocks(
        part_num[None], part_idx[None], full_num[None], full_idx[None],
        BLOCK_SIZE=(B, B), mask_mod=pad_mask_mod,
        seq_lengths=(geom.padded_len, geom.padded_len))


def measured_density(block_mask: Any, heads: int, n_blocks: int) -> float:
    """The kept fraction of key blocks this mask actually reads."""
    total = float(block_mask.full_kv_num_blocks.sum())
    if block_mask.kv_num_blocks is not None:
        total += float(block_mask.kv_num_blocks.sum())
    denom = float(heads * n_blocks * n_blocks) or 1.0
    return round(total / denom, 5)


def sparse_attend(query: Any, key: Any, value: Any, block_mask: Any,
                  kernel_options: Optional[dict] = None) -> Any:
    """(S, H, D) in and out — the layout an attention processor already holds."""
    bits = _bits()
    fn = bits["compiled"]
    if kernel_options:
        fn = _tuned(tuple(sorted(kernel_options.items())))
    out = fn(query.transpose(0, 1)[None], key.transpose(0, 1)[None],
             value.transpose(0, 1)[None], block_mask=block_mask)
    return out[0].transpose(0, 1)


def _tuned(opts_items: tuple) -> Any:
    key = "tuned:" + repr(opts_items)
    bits = _bits()
    if key not in bits:
        import torch
        from torch.nn.attention.flex_attention import flex_attention

        opts = dict(opts_items)
        flex: Any = flex_attention

        def _call(q: Any, k: Any, v: Any, block_mask: Any = None) -> Any:
            return flex(q, k, v, block_mask=block_mask, kernel_options=opts)

        bits[key] = torch.compile(_call, dynamic=False)
    return bits[key]


__all__ = [
    "DEFAULT_BLOCK",
    "BlockGeometry",
    "SparseUnavailable",
    "build_block_mask",
    "measured_density",
    "probe",
    "sparse_attend",
]
