"""The micro family's two compile targets: a tiny DiT and a tiny decoder.

Shaped like the thing the fleet mints, sized so a whole mint is seconds.

Deliberate structural choices, each with a reason the mint can read:

* **No ``nn.Conv*`` anywhere.** #730 ratified ``dynamic-collapse`` for
  conv-free graphs and ``static-rows`` for conv-bearing ones — its 6.9% AOTI
  gap was root-caused as inductor's channels-last layout opt bailing when a
  convolution carries a free symbol. A conv here would force the declaration
  onto ``static-rows``, which is the strategy that makes sdxl 36 compiled graphs.
  Conv-free is what buys the 3-compiled graph declaration.
* **The denoiser takes a python ``list`` of latents**, z-image's real shape:
  CFG is batched by concatenating the list, so the pytree ARITY doubles and
  N=1/N=2 are two graphs. This is what puts a CONTAINER input, and a plain
  input immediately AFTER it, in every mint cycle — pgw#993 (declaration-side
  flattening) and pgw#994 (serve-side binding) both live on that seam.
* **The rope table is a registered buffer** (pgw#857): a plain attribute is
  lifted by ``torch.export`` as an anonymous ``_tensor_constant`` literal
  whose bytes then ship inside every compiled graph instead of staying a
  deduplicated weight.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch import nn


class MicroConfig:
    """The config object the export declaration reads widths off.

    A plain attribute holder rather than a dataclass: declaration files are
    exec'd by path without a ``sys.modules`` compiled graph, and ``@dataclass``
    resolves annotations through ``sys.modules[cls.__module__].__dict__``.
    """

    __slots__ = ("in_channels", "cond_dim", "cond_len", "hidden", "depth",
                 "vae_scale", "seed")

    def __init__(
        self,
        in_channels: int = 16,
        cond_dim: int = 32,
        cond_len: int = 16,
        hidden: int = 128,
        depth: int = 2,
        vae_scale: int = 8,
        seed: int = 997,
    ) -> None:
        self.in_channels = int(in_channels)
        self.cond_dim = int(cond_dim)
        self.cond_len = int(cond_len)
        self.hidden = int(hidden)
        self.depth = int(depth)
        self.vae_scale = int(vae_scale)
        self.seed = int(seed)

    def as_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__slots__}


class _TimestepEmbedding(nn.Module):
    """Sinusoidal features -> two Linears; the usual shape at toy width."""

    def __init__(self, hidden: int) -> None:
        super().__init__()
        self.hidden = hidden
        self.lin1 = nn.Linear(hidden, hidden)
        self.lin2 = nn.Linear(hidden, hidden)
        half = hidden // 2
        # pgw#857: a table, not a learned parameter, and therefore a BUFFER.
        # As a plain attribute torch.export would lift it as a literal and its
        # bytes would ship inside the compiled graph.
        self.register_buffer(
            "freqs",
            torch.exp(-torch.arange(half, dtype=torch.float32)
                      * (math.log(10000.0) / max(1, half - 1))),
            persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = t.float()[:, None] * self.freqs[None, :]
        feats = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.lin2(torch.nn.functional.silu(self.lin1(feats)))


class _CrossAttention(nn.Module):
    """Single-head cross attention from the token grid onto the condition."""

    def __init__(self, hidden: int, cond_dim: int) -> None:
        super().__init__()
        self.scale = hidden ** -0.5
        self.norm = nn.LayerNorm(hidden)
        self.to_q = nn.Linear(hidden, hidden, bias=False)
        self.to_k = nn.Linear(cond_dim, hidden, bias=False)
        self.to_v = nn.Linear(cond_dim, hidden, bias=False)
        self.to_out = nn.Linear(hidden, hidden)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        q = self.to_q(self.norm(x))
        k = self.to_k(cond)
        v = self.to_v(cond)
        attn = torch.softmax((q @ k.transpose(-1, -2)) * self.scale, dim=-1)
        return x + self.to_out(attn @ v)


class _Block(nn.Module):
    def __init__(self, hidden: int, cond_dim: int) -> None:
        super().__init__()
        self.attn = _CrossAttention(hidden, cond_dim)
        self.norm = nn.LayerNorm(hidden)
        self.mlp_in = nn.Linear(hidden, hidden * 2)
        self.mlp_out = nn.Linear(hidden * 2, hidden)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.attn(x, cond)
        h = self.mlp_out(torch.nn.functional.gelu(self.mlp_in(self.norm(x))))
        return x + h


class MicroDenoiser(nn.Module):
    """The compile target ``transformer``.

    ``forward(x, t, cond)`` where ``x`` and ``cond`` are python LISTS of
    length N and ``t`` is a plain ``(N,)`` tensor sitting between them.
    That ordering is the point: it is the exact arrangement whose flattened
    leaf positions diverge from its pre-flattening argument positions.

    ``x`` elements are TOKEN SEQUENCES ``(T, C)``, not latent grids
    ``(C, H, W)``. That is flux/qwen's real interface — patchify and
    unpatchify live in the pipeline — and it is also forced: a grid whose H
    and W are both dynamic makes every matmul's M extent a PRODUCT of two
    symbols, and the mint's ``torch.export.save``/``load`` hand-off to its
    compile child loses the facts inductor needs to lower a nonlinear size.
    Measured by the micro-mint rig on torch 2.13.0+cu130:
    ``LoweringException: RuntimeError: ('unexpected None!', 512*s18*s57)``,
    where the identical graph compiles fine WITHOUT the round trip. Filed as
    pgw#998; a linear token extent sidesteps it entirely.
    """

    def __init__(self, config: MicroConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden
        self.proj_in = nn.Linear(config.in_channels, hidden)
        self.time_embed = _TimestepEmbedding(hidden)
        self.blocks = nn.ModuleList(
            [_Block(hidden, config.cond_dim) for _ in range(config.depth)])
        self.norm_out = nn.LayerNorm(hidden)
        self.proj_out = nn.Linear(hidden, config.in_channels)

    # ------------------------------------------------------------------
    # The diffusers CONFIG surface, implemented deliberately (pgw#1014).
    #
    # `w8a8.load_w8a8_denoiser` builds its skeleton with
    # `cls.load_config(dir)` + `cls.from_config(cfg)` under
    # `accelerate.init_empty_weights()`. That is diffusers' `ConfigMixin`
    # API, so a denoiser that is a plain `nn.Module` — as most non-diffusers
    # runtimes are — CANNOT take the w8a8 load path at all until it exposes
    # it. Recorded as a real requirement on sdxl-class families rather than
    # worked around: this family implements the surface, it does not bypass
    # the loader.
    # ------------------------------------------------------------------

    @classmethod
    def load_config(cls, directory: str, **_kw: Any) -> Dict[str, Any]:
        return json.loads(
            (Path(directory) / "config.json").read_text("utf-8"))

    @classmethod
    def from_config(cls, config: Dict[str, Any], **_kw: Any) -> "MicroDenoiser":
        fields = set(MicroConfig().as_dict())
        return cls(MicroConfig(
            **{k: v for k, v in dict(config).items() if k in fields}))

    def forward(
        self,
        x: List[torch.Tensor],
        t: torch.Tensor,
        cond: List[torch.Tensor],
    ) -> torch.Tensor:
        temb = self.time_embed(t)
        outs: List[torch.Tensor] = []
        # Python-level iteration over the container: export UNROLLS it, the
        # way z-image's `for image, cap_feat in zip(...)` unrolls. Nothing
        # here reads a tensor VALUE, so there is no data-dependent branch.
        for i, tokens in enumerate(x):
            h = self.proj_in(tokens) + temb[i][None, :]
            for block in self.blocks:
                h = block(h, cond[i])
            outs.append(self.proj_out(self.norm_out(h)))
        return torch.stack(outs, dim=0)


class MicroDecoder(nn.Module):
    """The compile target ``decoder``: latent tokens -> pixel COMPILED GRAPHS.

    ``(N, T, C) -> (N, T, 3 * scale * scale)``: one ``scale x scale`` RGB compiled graph
    per latent token. The pixel-shuffle that turns those compiled graphs into an image
    is the pipeline's job, not this module's — the same division real
    pipelines make, where the transformer never sees the image layout.

    Conv-free on purpose (#730 ratified ``dynamic-collapse`` for conv-free
    graphs), and token-shaped for the same reason the denoiser is: every
    extent stays LINEAR in one symbol.
    """

    def __init__(self, config: MicroConfig) -> None:
        super().__init__()
        self.config = config
        scale = config.vae_scale
        self.proj_in = nn.Linear(config.in_channels, config.hidden)
        self.norm = nn.LayerNorm(config.hidden)
        self.proj_out = nn.Linear(config.hidden, 3 * scale * scale)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.silu(self.proj_in(latent))
        return torch.tanh(self.proj_out(self.norm(h)))

    # The same deliberate ConfigMixin surface the denoiser exposes, and for a
    # second reason (pgw#1080): a compile target the forge must be able to
    # build from CODE + CONFIG needs it, and `decoder` is one of this
    # family's two targets. A target without it is stranded on the
    # real-weight mint — honestly, by a typed refusal, but stranded.

    @classmethod
    def load_config(cls, directory: str, **_kw: Any) -> Dict[str, Any]:
        return json.loads(
            (Path(directory) / "config.json").read_text("utf-8"))

    @classmethod
    def from_config(cls, config: Dict[str, Any], **_kw: Any) -> "MicroDecoder":
        fields = set(MicroConfig().as_dict())
        return cls(MicroConfig(
            **{k: v for k, v in dict(config).items() if k in fields}))


__all__ = ["MicroConfig", "MicroDecoder", "MicroDenoiser",
           "MicroGridDenoiser"]


class MicroGridDenoiser(MicroDenoiser):
    """The pgw#998 shape: a 4-D latent GRID whose H and W are BOTH dynamic.

    Same weights and same blocks as :class:`MicroDenoiser` — only the
    interface differs. Each ``x`` element is ``(C, H, W)`` and is flattened to
    tokens INSIDE the forward, so every matmul's M extent becomes the PRODUCT
    ``H*W``: an extent that is NONLINEAR in the traced symbols.

    That is z-image's declared shape (``H_lat`` and ``W_lat`` on a 4-D latent
    under ``dynamic-collapse``), and it is what pgw#998 found unlowerable
    across the mint's own `torch.export.save`/`load` hand-off to its compile
    child. This class exists so the gauntlet keeps testing that seam on every
    run rather than trusting a changelog.
    """

    def forward(  # type: ignore[override]
        self,
        x: List[torch.Tensor],
        t: torch.Tensor,
        cond: List[torch.Tensor],
    ) -> torch.Tensor:
        tokens = []
        for grid in x:
            channels, height, width = grid.shape
            tokens.append(
                grid.reshape(channels, height * width).transpose(0, 1))
        out = super().forward(tokens, t, cond)
        channels, height, width = x[0].shape
        return out.transpose(1, 2).reshape(len(x), channels, height, width)
