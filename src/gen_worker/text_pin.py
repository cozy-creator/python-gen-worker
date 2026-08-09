"""Text-sequence pinning helper (SDK v2, pgw#647 — absorbs ie#544's
endpoint-side pins).

qwen-image / z-image / hidream-o1 feed a prompt-length-dependent SEQUENCE
dim into a statically compiled denoiser, so every distinct prompt length
minted a new graph — unbounded and un-warmable, every paid request missing
the shipped graph (ie#544). The v2 contract declares the axis
(``Compile(text_len=...)``); this helper is how the handler HONORS it.

ie#544's hard-won detail: dynamo guards on STRIDES, and ``.contiguous()``
is a no-op when dim 0 is size 1 — **a pin that fixes only the size is not a
pin**. This helper therefore always returns FRESH canonically-strided
allocations (``torch.zeros`` + ``copy_``), never a view or a
maybe-contiguous alias of the input.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple


class TextLengthExceededError(ValueError):
    """The encoded prompt is longer than the declared ``Compile(text_len=)``
    pin — a request outside the declared envelope. Typed refusal (same posture
    as over-rank LoRAs), never a silently minted graph."""


def pad_text_sequence(
    embeds: Any,
    length: int,
    *,
    mask: Optional[Any] = None,
    dim: int = 1,
) -> Tuple[Any, Optional[Any]]:
    """Pad ``embeds`` (and optionally ``mask``) along ``dim`` to exactly
    ``length`` tokens, returning fresh canonically-strided tensors.

    ``embeds`` is the encoder output ``[B, seq, hidden]`` (``dim=1``);
    ``mask`` is the matching attention mask ``[B, seq]`` (padded with 0).
    A sequence longer than ``length`` raises
    :class:`TextLengthExceededError` — the declared contract is the
    contract.
    """
    import torch

    n = int(length)
    if n <= 0:
        raise ValueError(f"pad_text_sequence: length must be positive, got {length}")
    seq = int(embeds.shape[dim])
    if seq > n:
        raise TextLengthExceededError(
            f"encoded text sequence ({seq} tokens) exceeds the declared "
            f"Compile(text_len={n}) pin — refuse or re-encode with "
            f"max_sequence_length={n}"
        )
    out_shape = list(embeds.shape)
    out_shape[dim] = n
    out = torch.zeros(
        out_shape, dtype=embeds.dtype, device=embeds.device
    )
    out.narrow(dim, 0, seq).copy_(embeds)
    out_mask: Optional[Any] = None
    if mask is not None:
        mask_dim = min(dim, mask.dim() - 1)
        m_shape = list(mask.shape)
        m_shape[mask_dim] = n
        out_mask = torch.zeros(m_shape, dtype=mask.dtype, device=mask.device)
        out_mask.narrow(mask_dim, 0, int(mask.shape[mask_dim])).copy_(mask)
    return out, out_mask


__all__ = ["TextLengthExceededError", "pad_text_sequence"]
