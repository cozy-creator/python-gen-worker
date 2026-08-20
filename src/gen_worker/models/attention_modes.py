"""Attention-execution vocabulary — the THIRD axis (pgw#1043 §PRODUCTIZATION).

``JobMetrics.lane`` says ``fp8-w8a8-dynamic+compiled`` and ``serving_mode`` says
``aot_graph``. Neither can say **which key blocks the attention actually read**,
and on a sparse-attention endpoint that is the single largest determinant of both
latency and the take the render is. The three axes are independent by
construction:

===============  ==================================  ==========================
axis             vocabulary                          who owns it
===============  ==================================  ==========================
execution lane   ``<weights>-<act>[-<scale>]+<exec>``  the CHECKPOINT's numerics
serving mode     ``eager | jit_graph | aot_graph``      the ARMED artifact
attention mode   ``dense | sparse-kNN``               the SELECTOR + its index
===============  ==================================  ==========================

Why this is not a lane token. The lane grammar is a NUMERICS descriptor and its
consumers — residency/VRAM planning, pricing, compiled graph identity,
``ResolvePinned`` — all ask it *"how big are the weights and what arithmetic"*.
Sparse attention changes neither operand format; it changes which blocks are
read. Growing the closed two-repo lane table by a cross product for an axis none
of those consumers use is exactly the mistake pgw#764/th#1293 declined to make
when ``+compiled`` could not distinguish an AOT replay from a JIT graph: the
answer there was a SEPARATE typed axis, and it is the answer here.

A mode token alone is also not enough, which is the second reason it cannot ride
the lane string: the honest report carries ``k`` and the **measured** kept
fraction, because two pods at ``sparse-k16`` on different sequence lengths do not
read the same density, and the density is what the wall is a function of.
"""

from __future__ import annotations

import re
from typing import Optional

import msgspec

#: Full attention. The permanent default and the only mode that needs no
#: artifact — an endpoint that reports nothing is dense by construction.
ATTENTION_DENSE = "dense"

#: ``sparse-k<N>``: block-sparse attention keeping the top ``N`` key blocks per
#: (query block, head), plus the protocol's forced local + global blocks. ``N``
#: is the BUDGET, not the achieved density — the density is measured and
#: reported beside it.
_SPARSE_RE = re.compile(r"^sparse-k(\d+)$")

#: The block granularity every sparse mode is quoted at today (pgw#1043's
#: probe: block 64 buys 6% of budget for 60% more kernel; 128 is the
#: granularity). Reported explicitly so a future re-block is visible.
DEFAULT_BLOCK_SIZE = 128


def valid_attention_mode(token: str) -> bool:
    tok = str(token or "").strip().lower()
    return tok == ATTENTION_DENSE or bool(_SPARSE_RE.match(tok))


def sparse_k_of(token: str) -> Optional[int]:
    """The block budget a ``sparse-kNN`` token names, or None for dense."""
    m = _SPARSE_RE.match(str(token or "").strip().lower())
    return int(m.group(1)) if m else None


def sparse_mode(k_blocks: int) -> str:
    if int(k_blocks) <= 0:
        raise ValueError(f"sparse_mode({k_blocks!r}): k must be positive")
    return f"sparse-k{int(k_blocks)}"


def known_attention_modes(max_k: int = 0) -> list[str]:
    """The enumerable vocabulary. ``sparse-kNN`` is a FAMILY, not a fixed list —
    any positive k is grammatical — so this returns dense plus the budgets a
    caller asks to enumerate (the ones §INDEXER measured by default)."""
    ks = (16, 32) if max_k <= 0 else tuple(range(1, max_k + 1))
    return [ATTENTION_DENSE] + [sparse_mode(k) for k in ks]


class AppliedAttention(msgspec.Struct, frozen=True, kw_only=True):
    """What the attention path ACTUALLY ran, reported by the code that installed
    it (pgw#1104's rule: only the code that did the thing can prove it did).

    ``density`` is the MEASURED kept fraction of key blocks at the shape the
    report was taken on — 0.0 when not measured. ``selector`` names where the
    block scores came from (``indexer`` / ``meanpool`` / ``oracle``) because a
    free mean-pool selector and a trained index branch are different quality
    claims at the same ``mode``. ``index_ref`` is the bound artifact the heads
    came from, so a render is traceable to the exact head that selected for it.
    """

    component: str
    #: The SPARSITY axis (``dense`` / ``sparse-k<N>``). Empty means this report
    #: says nothing about sparsity — which is what a backend-only report is, and
    #: is NOT the same as claiming dense.
    mode: str = ""
    #: th#1871 P1 (pgw#1225): the KERNEL axis — ``fa3``/``fa2``/``sdpa``/
    #: ``xformers``/``eager``, and what was asked for. A different question from
    #: ``mode``: `sparse-k8 on sdpa` and `sparse-k8 on fa3` are the same
    #: sparsity and a ~2x different number. Reporting them on one axis is how
    #: ie#707 stayed silent — the only reporter that existed validated against
    #: the sparsity grammar and RAISED on ``"sdpa"``, so 23 of 29 families
    #: reported nothing at all rather than the wrong thing.
    backend: str = ""
    backend_wanted: str = ""
    k_blocks: int = 0
    block_size: int = 0
    density: float = 0.0
    selector: str = ""
    index_ref: str = ""

    def detail(self) -> str:
        bits = [f"component={self.component}"]
        if self.mode:
            bits.append(f"attention={self.mode}")
        if self.backend:
            bits.append(f"backend={self.backend}")
        if self.backend_wanted and self.backend_wanted != self.backend:
            bits.append(f"backend_wanted={self.backend_wanted}")
        if self.k_blocks:
            bits.append(f"k={self.k_blocks}")
        if self.block_size:
            bits.append(f"block={self.block_size}")
        if self.density:
            bits.append(f"density={self.density:.4f}")
        if self.selector:
            bits.append(f"selector={self.selector}")
        if self.index_ref:
            bits.append(f"index={self.index_ref}")
        return " ".join(bits)


def most_sparse_mode(modes: list[str]) -> str:
    """The mode an INSTANCE reports when its components disagree.

    Sparse wins over dense and the smaller budget wins over the larger, for the
    same reason ``_most_quantized_lane`` picks the most-quantized: the report
    must never over-claim fidelity. A request that ran ANY component sparse did
    not run dense."""
    best = ATTENTION_DENSE
    best_k: Optional[int] = None
    for m in modes:
        tok = str(m or "").strip().lower()
        if not valid_attention_mode(tok) or tok == ATTENTION_DENSE:
            continue
        k = sparse_k_of(tok)
        if k is not None and (best_k is None or k < best_k):
            best, best_k = tok, k
    return best


__all__ = [
    "ATTENTION_DENSE",
    "DEFAULT_BLOCK_SIZE",
    "AppliedAttention",
    "known_attention_modes",
    "most_sparse_mode",
    "sparse_k_of",
    "sparse_mode",
    "valid_attention_mode",
]
