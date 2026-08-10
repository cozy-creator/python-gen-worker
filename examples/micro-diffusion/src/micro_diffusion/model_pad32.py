"""ie#637's shape, at micro scale: a sequence PADDED TO A MULTIPLE OF 32.

z-image's `patchify_and_embed` pads the flattened latent to a multiple of 32,
and that padded length is then an extent inside the transformer. Under
`torch.export` with both latent axes dynamic it becomes an algebraic function
of the declared symbols, and HOW it is spelled decides whether the exported
program is servable at all:

* **Upstream's spelling** — ``pad = (-L) % 32`` plus ``if pad > 0`` branches —
  DECIDES on the pad four times per sample, so export emits equality guards
  that PIN the declared symbols and ie#566's declared-range gate refuses the
  artifact (correctly: the graph took one branch at trace and would serve one
  latent size while advertising a range). That is :class:`MicroPad32Branchy`.
* **The fix's spelling** — the padded path taken UNCONDITIONALLY, an
  ``arange`` mask instead of a 1-D ``cat``, and the length written
  ``ceil(L/32)*32`` — carries no pinning guard, because
  ``Mod(32*FloorDiv(L+31,32), 32)`` folds to 0 statically while
  ``Mod(L + PythonMod(-L,32), 32)`` is a tautology sympy cannot prove. That is
  :class:`MicroPad32Denoiser`.

**What this family is FOR.** The endpoint half of that shipped as ie#637 and
was proven off-GPU on the guard algebra. What no run has ever reached is the
NEXT phase: whether AOTInductor CODEGENS ``32*FloorDiv(L+31,32)`` correctly
through export -> compile -> load -> serve. Two z-image pods died before it —
one on the gate, one on VRAM (ie#638) — so the question survives as ie#637's
open watch item and it gates the z-image confirmation buy. Here it costs
seconds and no pod.

Both classes share :class:`MicroDenoiser`'s weights and blocks; only the
padding spelling differs, so a difference in outcome is a difference in the
SPELLING and nothing else.
"""

from __future__ import annotations

from typing import List, Tuple

import torch

from .model import MicroDenoiser

#: z-image's own ``SEQ_MULTI_OF`` (`transformer_z_image.py:487`).
SEQ_MULTIPLE_OF = 32


def padded_length(length: int) -> int:
    """``ceil(L/32)*32`` — the spelling that FOLDS.

    Written with floor division rather than ``L + (-L) % 32`` deliberately:
    only this one lets ``sympy`` prove ``padded % 32 == 0`` statically, so the
    assertion below folds instead of landing as a real equality guard over the
    declared symbols. torch's ``PythonMod`` is not sympy's ``Mod`` and
    ``_is_tautology`` — correctly conservative — cannot discharge the other.
    """
    return ((length + SEQ_MULTIPLE_OF - 1) // SEQ_MULTIPLE_OF) * SEQ_MULTIPLE_OF


class _GridTokens:
    """Shared: ``(C, H, W)`` -> ``(H*W, C)`` tokens, and back."""

    @staticmethod
    def to_tokens(grid: torch.Tensor) -> Tuple[torch.Tensor, int]:
        channels, height, width = grid.shape
        seq = grid.reshape(channels, height * width).transpose(0, 1)
        return seq, seq.shape[0]

    @staticmethod
    def to_grid(out: torch.Tensor, like: torch.Tensor, arity: int
                ) -> torch.Tensor:
        channels, height, width = like.shape
        return out.transpose(1, 2).reshape(arity, channels, height, width)


class MicroPad32Denoiser(MicroDenoiser):
    """The FIXED spelling — ie#637's `z_image/pad32.py`, in miniature.

    The pad is computed in-graph, the padded path is taken unconditionally
    (with ``pad == 0`` the concatenated tail is empty and the value is
    identical), and the mask is an ``arange`` comparison rather than a 1-D
    ``cat`` whose empty-operand handling is itself a decision about the pad.
    """

    def forward(  # type: ignore[override]
        self,
        x: List[torch.Tensor],
        t: torch.Tensor,
        cond: List[torch.Tensor],
    ) -> torch.Tensor:
        tokens: List[torch.Tensor] = []
        length = 0
        for grid in x:
            seq, length = _GridTokens.to_tokens(grid)
            total = padded_length(length)
            # Upstream asserts exactly this, and it is the reason the spelling
            # matters: it folds to True statically here, and lands as an
            # equality guard over the declared symbols with the other spelling.
            assert total % SEQ_MULTIPLE_OF == 0
            tail = seq.new_zeros((total - length, seq.shape[1]))
            seq = torch.cat([seq, tail], dim=0)
            keep = (torch.arange(total, device=seq.device) < length)
            tokens.append(seq * keep[:, None].to(seq.dtype))
        out = super().forward(tokens, t, cond)
        return _GridTokens.to_grid(out[:, :length, :], x[0], len(x))


class MicroPad32Branchy(MicroDenoiser):
    """The UPSTREAM spelling — kept, and expected to be REFUSED.

    Three decisions on the pad, the same three z-image made: a ``PythonMod``
    length, a branch on ``pad > 0``, and a 1-D ``cat`` for the mask. Exported
    with both latent axes dynamic this pins the declared symbols and ie#566
    G3's declared-range gate refuses the artifact.

    It is here so the FIXED member's green means something: without a red twin
    on the same declaration, a pass proves only that this graph does not
    reach the gate.
    """

    def forward(  # type: ignore[override]
        self,
        x: List[torch.Tensor],
        t: torch.Tensor,
        cond: List[torch.Tensor],
    ) -> torch.Tensor:
        tokens: List[torch.Tensor] = []
        length = 0
        for grid in x:
            seq, length = _GridTokens.to_tokens(grid)
            pad = (-length) % SEQ_MULTIPLE_OF
            if pad > 0:                       # the guard, decision 1
                tail = seq.new_zeros((pad, seq.shape[1]))
                seq = torch.cat([seq, tail], dim=0)
                mask = torch.cat([                       # decision 2: 1-D cat
                    seq.new_ones((length,)), seq.new_zeros((pad,))], dim=0)
            else:
                mask = seq.new_ones((length,))           # decision 3
            tokens.append(seq * mask[:, None])
        out = super().forward(tokens, t, cond)
        return _GridTokens.to_grid(out[:, :length, :], x[0], len(x))


__all__ = ["MicroPad32Branchy", "MicroPad32Denoiser", "SEQ_MULTIPLE_OF",
           "padded_length"]
