"""The pgw#1073 conv variant: the STATIC-ROWS graph class, at micro scale.

Every existing micro member is conv-free by construction, which buys the
3-entry ``dynamic-collapse`` declaration — and leaves the OTHER strategy,
``static-rows``, exercised by nothing smaller than sdxl (36 entries, ~95 min
per pod cycle). static-rows is the class pgw#1058 broke in: entry LABELS are
per-row static facts, and drift between the labels and the serve-side asks
admits nothing. This module is the smallest member that keeps that class
under test on every gauntlet run.

Deliberate structural choices, each an axis no other member carries:

* **Conv-bearing.** #730 ratified ``static-rows`` for conv-bearing graphs
  (symbolic latent H/W turns off inductor's channels-last layout opt on the
  convs). The denoiser is a real little conv UNet — down path, nested
  residual blocks, up path — so the declaration MUST be static-rows for the
  same measured reason sdxl's is.
* **An int64 timestep.** wan-2.2's shape, and the pgw#1058 defect class:
  dtype is a declared per-input fact, and a mixed int64/float32 signature
  keeps the declaration's dtype axis load-bearing on every cycle. The
  timestep drives an ``nn.Embedding`` lookup, so the integer input is
  structural (an index op), not a cast-away.
* **Deeper module nesting.** Blocks inside blocks inside stages
  (``_ConvStage`` -> ``_ResBlock`` -> ``_ConvGN``), against the flat two-deep
  micro DiT — exercises export's module-path naming on a genuinely nested
  tree.
* **A named persistent buffer** (``class_table``): the H3 pattern — a
  config-derived table that is part of the CHECKPOINT (persistent, in
  state_dict), not a lifted literal. pgw#857's seam from the other side:
  micro's rope table proves the non-persistent/literal half, this proves the
  named-component half.

Weights: derived deterministically from the SAME seed-997 checkpoint tree's
declared seed (the micro-4d/micro-escape precedent, one step further — those
reshape the base DiT's tensors, this family's tensors are a pure function of
``seed + 1`` because a conv module cannot load a DiT state dict). One catalog
checkpoint still serves every micro family.
"""

from __future__ import annotations

import torch
from torch import nn

from .model import MicroConfig

#: Timestep vocabulary for the embedding lookup — the int64 input indexes it.
NUM_TRAIN_TIMESTEPS = 16

#: The class-conditioning table rows (the named persistent buffer's extent).
NUM_CLASSES = 8


class _ConvGN(nn.Module):
    """Conv3x3 + GroupNorm + SiLU — the innermost nesting level."""

    def __init__(self, cin: int, cout: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1)
        self.norm = nn.GroupNorm(4, cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.silu(self.norm(self.conv(x)))


class _ResBlock(nn.Module):
    """Two nested _ConvGN with a residual — level two of the nesting."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block1 = _ConvGN(channels, channels)
        self.block2 = _ConvGN(channels, channels)

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x) + emb[:, :, None, None]
        return x + self.block2(h)


class _ConvStage(nn.Module):
    """A resolution stage holding a ModuleList of _ResBlocks — level three."""

    def __init__(self, channels: int, blocks: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_ResBlock(channels) for _ in range(blocks))

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, emb)
        return x


class MicroConvDenoiser(nn.Module):
    """The compile target ``unet``: ``forward(sample, timestep, cond)``.

    ``sample`` is a plain 4-D latent ``(B, C, H, W)`` — no containers,
    deliberately: the container seam belongs to ``micro``; this member's job
    is the conv/static-rows/mixed-dtype seam, and one variant per seam keeps
    a red diagnosable.

    ``timestep`` is ``(B,)`` **int64** and INDEXES ``time_embed`` — an
    embedding lookup, so the integer dtype is load-bearing in the graph.
    ``cond`` is ``(B, L, D)`` float32, mean-pooled and projected (a conv
    UNet's usual conditioning shortcut at toy width).
    """

    def __init__(self, config: MicroConfig) -> None:
        super().__init__()
        self.config = config
        width = max(16, config.hidden // 4)  # 32 at the default config
        self.width = width
        self.conv_in = nn.Conv2d(config.in_channels, width, 3, padding=1)
        self.time_embed = nn.Embedding(NUM_TRAIN_TIMESTEPS, width)
        self.cond_proj = nn.Linear(config.cond_dim, width)
        # The H3 pattern: a config-derived table that is CHECKPOINT STATE
        # (persistent=True -> in state_dict -> in the published tree), bound
        # like a weight, never a lifted literal (contrast micro's
        # non-persistent rope freqs, which prove the literal half of pgw#857).
        self.register_buffer(
            "class_table",
            torch.linspace(-1.0, 1.0, NUM_CLASSES * width).reshape(
                NUM_CLASSES, width),
            persistent=True)
        self.down = _ConvStage(width)
        self.pool = nn.Conv2d(width, width, 3, stride=2, padding=1)
        self.mid = _ConvStage(width)
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.post = _ConvStage(width, blocks=1)
        self.conv_out = nn.Conv2d(width, config.in_channels, 3, padding=1)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        emb = self.time_embed(timestep)              # int64 index -> (B, W)
        emb = emb + self.cond_proj(cond.mean(dim=1))  # float32 path joins
        emb = emb + self.class_table[0][None, :]      # the named buffer, read
        h = self.conv_in(sample)
        h = self.down(h, emb)
        h = self.mid(self.pool(h), emb)
        h = self.post(self.up(h), emb)
        return self.conv_out(h)


def build_conv_denoiser(config: MicroConfig) -> MicroConvDenoiser:
    """Construct with weights that are a pure function of ``config.seed``.

    ``seed + 1``, not ``seed``: the base DiT consumes the generator stream
    under ``seed`` in `weights.state_dict`, and two module families drawing
    the same stream would make their tensors coincide pairwise — harmless,
    but a needless aliasing between checkpoints that are supposed to be
    independent facts.
    """
    torch.manual_seed(int(config.seed) + 1)
    return MicroConvDenoiser(config)


__all__ = ["NUM_CLASSES", "NUM_TRAIN_TIMESTEPS", "MicroConvDenoiser",
           "build_conv_denoiser"]
