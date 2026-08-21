from __future__ import annotations

import re
from typing import Optional

import msgspec

ATTENTION_DENSE = "dense"

_SPARSE_RE = re.compile(r"^sparse-k(\d+)$")

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
    """The enumerable vocabulary."""
    ks = (16, 32) if max_k <= 0 else tuple(range(1, max_k + 1))
    return [ATTENTION_DENSE] + [sparse_mode(k) for k in ks]


class AppliedAttention(msgspec.Struct, frozen=True, kw_only=True):

    component: str
    mode: str = ""
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
    """The mode an INSTANCE reports when its components disagree."""
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
