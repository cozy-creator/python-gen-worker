"""The pgw#1080 RED CONTROL: upstream z-image's lazy CPU-pinned rope table.

Transcribed from `transformer_z_image.RopeEmbedder`, whose shape is the whole
reason ie#628 widened the meta-instantiation gate from ``__init__`` to CALL
time: the table is ``None`` until first use, and the build pins
``torch.device("cpu")``, which OVERRIDES any ambient context. Nothing happens
at construction — an ``__init__``-inspecting gate sees a clean instantiation —
and the violation lands mid-forward.

The GREEN twin is the base family: :class:`~micro_diffusion.model.MicroDenoiser`
registers its frequency table as a BUFFER at ``__init__`` with no device pin,
which is exactly ie#630's fix (`rope_buffers`), and it must mint green while
this one does not. Both halves are the control: a gate that fires on
correctly-authored code teaches authors to route around it.

This variant exists to FAIL. It is a gauntlet member whose declared
expectation is a refusal, so a green run here is news.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from .model import MicroConfig, MicroDenoiser


class PinnedRopeTable:
    """Not an ``nn.Module``, holds ``None``, builds under a device pin.

    All three properties are load-bearing and all three are upstream's:
    a plain object owns no buffers so nothing can register it; the lazy
    ``None`` moves the work out of ``__init__``; the pin makes the ambient
    device irrelevant.
    """

    def __init__(self, width: int) -> None:
        self.width = int(width)
        self.freqs_cis: Optional[torch.Tensor] = None

    def __call__(self, rows: int) -> torch.Tensor:
        if self.freqs_cis is None:
            with torch.device("cpu"):
                self.freqs_cis = torch.arange(
                    self.width, dtype=torch.float32) / float(self.width)
        return self.freqs_cis[None, :].expand(rows, self.width)


class MicroRopeDenoiser(MicroDenoiser):
    """The base denoiser with its table taken OFF the module and pinned."""

    def __init__(self, config: MicroConfig) -> None:
        super().__init__(config)
        self.rope = PinnedRopeTable(config.hidden)

    def forward(
        self, x: List[torch.Tensor], t: torch.Tensor, cond: List[torch.Tensor],
    ) -> torch.Tensor:
        temb = self.time_embed(t)
        outs: List[torch.Tensor] = []
        for index, tokens in enumerate(x):
            h = self.proj_in(tokens) + temb[index][None, :]
            # The ONE difference from the green twin: the table comes from a
            # pinned lazy build instead of a registered buffer.
            h = h + self.rope(h.shape[0]).to(h.dtype)
            for block in self.blocks:
                h = block(h, cond[index])
            outs.append(self.proj_out(self.norm_out(h)))
        return torch.stack(outs, dim=0)


__all__ = ["MicroRopeDenoiser", "PinnedRopeTable"]
