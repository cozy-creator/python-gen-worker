"""Block-sparse attention MECHANISM (pgw#1043 §PRODUCTIZATION).

pgw#740 doctrine: the mechanism lives here, the per-model vocabulary lives in the
endpoint. This module knows how to turn *block scores* into a FlexAttention
``BlockMask`` and run the attention; it knows nothing about H3, about where the
scores came from, or about what a "global prefix" means beyond a row count.

What it is NOT: it is not a selector. A servable selector needs a trained index
branch per model (pgw#1043 §INDEXER), and that artifact is minted by the
conversion route and delivered through a binding slot — never bundled here.

Two facts this module exists to encode, both measured, both landmines:

1. **There is no eager mode.** Eager ``flex_attention`` silently IGNORES a
   manually built ``BlockMask``'s block structure and computes dense (§INDEXER
   rig red/green: eager == dense to 1e-7 under a 54%-kept mask). It also OOMs at
   video-DiT shapes. Every call here goes through the compiled callable.
2. **``block_mask`` is a KEYWORD argument.** ``flex_attention``'s fourth
   positional parameter is ``score_mod``; passing the mask positionally runs
   dense and announces itself only as a 299 GiB allocation failure. Measured on
   an H100, not theorised.

The mask builder is the ``topk``-direct path: ``topk`` already returns indices
and the forced blocks are known a priori, so the ascending kv index list is built
by mark -> cumsum -> scatter instead of by a full-width sort over the bool keep
tensor. It is bit-identical to §INDEXER's sort-based reference (asserted in
``tests/test_sparse_attention_pgw1043.py``) and materially cheaper.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from . import settings_authority

logger = logging.getLogger(__name__)

#: The block granularity the H3 probe settled on (block 64 buys 6% of budget for
#: 60% more kernel). Callers may override; nothing here assumes it.
DEFAULT_BLOCK = 128

_FLEX: dict = {}


class SparseUnavailable(RuntimeError):
    """The block-sparse path cannot run here. Callers DEGRADE to dense and say
    so — a sparse endpoint must never fail a request over an optional lane."""


@dataclass(frozen=True)
class BlockGeometry:
    """Everything the mask builder needs about one packed sequence.

    ``n_global`` is the count of leading rows that attend, and are attended,
    densely — H3's ``[text|audio|video]`` prefix. It is a ROW COUNT here and
    carries no modality semantics: the endpoint owns that vocabulary.
    """

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
    # The recompile ceiling is the SETTINGS AUTHORITY's to raise, never a second
    # writer's (pgw#1049). Sparse adds one compiled flex callable per distinct
    # kernel_options set, so it declares the shape count and asks.
    settings_authority.raise_dynamo_cache_limits(64)
    _FLEX["BlockMask"] = BlockMask
    _FLEX["compiled"] = torch.compile(flex_attention, dynamic=False)
    return _FLEX


def probe() -> str:
    """"" if the block-sparse path is runnable here, else why not. Called at
    setup so the refusal is a boot fact, not a first-request surprise."""
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return f"torch unimportable: {exc}"
    if not torch.cuda.is_available():
        return "no CUDA device"
    try:
        _bits()
    except SparseUnavailable as exc:
        return str(exc)
    return ""


def build_block_mask(scores: Any, k_blocks: int, geom: BlockGeometry,
                     heads: int) -> Any:
    """(X, NQ, NB) block scores -> ``BlockMask`` keeping the top ``k_blocks``
    per row plus the forced local diagonal and the global prefix.

    ``X`` is the score's own head dimension: ``heads`` for a per-head selector,
    a divisor of it for a grouped one (the group's rows are replicated, which is
    the honest cost of grouping — pgw#1043 measured the grouping CEILING at
    0.81/0.84 of per-head, so grouping is a reported control, not a default).
    """
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
    mark[:, rows, rows] = 1                       # forced local diagonal
    if g:
        mark[:, :, :g] = 1                        # global keys, every row
        mark[:, rows < g, :] = 1                  # global query rows are dense

    last_partial = geom.padded_len != geom.seq_len
    part_num = torch.zeros((X, NQ), dtype=torch.int32, device=dev)
    part_idx = torch.zeros((X, NQ, NB), dtype=torch.int32, device=dev)
    if last_partial:
        part_num = mark[..., NB - 1].clone()
        part_idx[..., 0] = NB - 1
        mark[..., NB - 1] = 0

    # Ascending scatter. Unmarked columns go to a DUMP slot that is sliced off;
    # routing them to slot 0 would clobber a real entry, which is exactly what
    # the reference paid a full-width sort to avoid.
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
        return ki < seq_len   # pad keys carry no mass

    return bits["BlockMask"].from_kv_blocks(
        part_num[None], part_idx[None], full_num[None], full_idx[None],
        BLOCK_SIZE=(B, B), mask_mod=pad_mask_mod,
        seq_lengths=(geom.padded_len, geom.padded_len))


def measured_density(block_mask: Any, heads: int, n_blocks: int) -> float:
    """The kept fraction of key blocks this mask actually reads. The number the
    wall is a function of — ``k`` is only the budget that was asked for."""
    total = float(block_mask.full_kv_num_blocks.sum())
    if block_mask.kv_num_blocks is not None:
        total += float(block_mask.kv_num_blocks.sum())
    denom = float(heads * n_blocks * n_blocks) or 1.0
    return round(total / denom, 5)


def sparse_attend(query: Any, key: Any, value: Any, block_mask: Any,
                  kernel_options: Optional[dict] = None) -> Any:
    """(S, H, D) in and out — the layout an attention processor already holds.

    ``block_mask`` goes in BY KEYWORD. See the module docstring: the alternative
    is a silent dense run."""
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
        # Through an Any-typed local: torch's overloads do not admit an untyped
        # kernel_options mapping, and the values are the measured per-mode tile.
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
