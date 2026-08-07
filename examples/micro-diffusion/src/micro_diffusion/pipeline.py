"""The diffusers-SHAPED holder the slot resolves to.

``.denoiser`` and ``.decoder`` are the two compile targets; ``from_pretrained``
is the contract the SDK's slot loader calls and it reads only from the local
tree it is handed. If that tree does not exist yet it is GENERATED, never
fetched — see :mod:`micro_diffusion.weights`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import torch

from .model import MicroConfig, MicroDecoder, MicroDenoiser
from .weights import SEED, load_config, load_state, materialize


class MicroPipeline:
    def __init__(
        self, denoiser: MicroDenoiser, decoder: MicroDecoder, source: str = "",
    ) -> None:
        self.denoiser = denoiser
        self.decoder = decoder
        self.source = source
        self.config = denoiser.config

    @classmethod
    def from_pretrained(cls, path: str, **_kw: Any) -> "MicroPipeline":
        root = Path(path)
        if not (root / "config.json").is_file():
            # A pod whose binding resolved to an empty dir, or a boot before
            # the build step ran. Generating is CHEAPER than failing and is
            # the family's whole premise, so do it and say where.
            materialize(root, seed=SEED)
        config = load_config(root)
        state = load_state(root)
        denoiser = MicroDenoiser(config)
        decoder = MicroDecoder(config)
        denoiser.load_state_dict(
            {k[len("denoiser."):]: v for k, v in state.items()
             if k.startswith("denoiser.")}, strict=False)
        decoder.load_state_dict(
            {k[len("decoder."):]: v for k, v in state.items()
             if k.startswith("decoder.")}, strict=False)
        return cls(denoiser.eval(), decoder.eval(), source=str(root))

    def to(self, device: Any) -> "MicroPipeline":
        self.denoiser.to(device)
        self.decoder.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return next(self.denoiser.parameters()).device

    def unpatchify(self, cells: torch.Tensor, grid: int) -> torch.Tensor:
        """``(N, T, 3*s*s) -> (N, 3, grid*s, grid*s)``.

        In the PIPELINE, not in the decoder: the compiled targets take token
        sequences so every traced extent stays linear in one symbol, and the
        image layout is recovered outside the graph. That is the same
        division flux and qwen make.
        """
        scale = self.config.vae_scale
        batch = cells.shape[0]
        out = cells.reshape(batch, grid, grid, scale, scale, 3)
        out = out.permute(0, 5, 1, 3, 2, 4)
        return out.reshape(batch, 3, grid * scale, grid * scale)

    def __call__(
        self,
        latents: List[torch.Tensor],
        cond: List[torch.Tensor],
        *,
        grid: int,
        steps: int = 2,
        guidance: float = 4.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """A two-step Euler-ish loop with the fleet's real CFG shape.

        ``len(latents)`` IS the fork coordinate: 2 on the guided arm (the
        conditional and unconditional rows batched into ONE forward, the way
        z-image's ``latents.repeat(2,1,1,1)`` does), 1 on the turbo arm.
        """
        x = list(latents)
        for i in reversed(range(max(1, steps))):
            t = torch.full((len(x),), float(i * 100), device=x[0].device)
            with torch.no_grad():
                eps = self.denoiser(x, t, cond)
            if len(x) == 2:
                uncond, text = eps[0], eps[1]
                guided = uncond + guidance * (text - uncond)
                step = torch.stack([guided, guided], dim=0)
            else:
                step = eps
            x = [x[j] - step[j] / max(1, steps) for j in range(len(x))]
        with torch.no_grad():
            cells = self.decoder(x[-1][None, ...])
        return self.unpatchify(cells, grid)


__all__ = ["MicroConfig", "MicroPipeline"]
