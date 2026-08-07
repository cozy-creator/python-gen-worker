"""The diffusers-SHAPED holder the slot resolves to.

``.transformer`` and ``.decoder`` are the two compile targets; ``from_pretrained``
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


def _load_prefixed(module: Any, state: dict, prefix: str) -> None:
    """Load ``prefix.*`` out of ``state`` into ``module``, STRICTLY.

    Refuses by name when the prefix selects nothing — the failure mode that
    `strict=True` alone does NOT catch, because an empty dict against a
    module with parameters is just "missing keys", and the message would not
    say the interesting part: that the checkpoint uses a different layout.
    """
    sub = {k[len(prefix) + 1:]: v for k, v in state.items()
           if k.startswith(prefix + ".")}
    if not sub:
        raise ValueError(
            f"checkpoint carries no {prefix!r} tensors — its prefixes are "
            f"{sorted({k.split('.')[0] for k in state})!r}. This tree was "
            f"written by a different layout version; regenerate it "
            f"(`python -m micro_diffusion.weights --out <dir>`) rather than "
            f"loading a randomly-initialized module.")
    module.load_state_dict(sub, strict=True)


class MicroPipeline:
    def __init__(
        self, transformer: MicroDenoiser, decoder: MicroDecoder, source: str = "",
    ) -> None:
        self.transformer = transformer
        self.decoder = decoder
        self.source = source
        self.config = transformer.config

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
        transformer = MicroDenoiser(config)
        decoder = MicroDecoder(config)
        # STRICT, and it is load-bearing (pgw#999). With `strict=False` a
        # checkpoint written under a different prefix loads ZERO tensors and
        # leaves the module randomly initialized — silently. That is exactly
        # what renaming this family's denoiser `denoiser` -> `transformer` did:
        # the cached tree still said `denoiser.*`, every lookup missed, and two
        # processes then held two DIFFERENT random transformers. The rig read
        # that as a 0.151 numerics divergence and spent a session accusing the
        # numerics gate, which was right all along.
        #
        # A loader that cannot find its weights must fail, not shrug.
        _load_prefixed(transformer, state, "transformer")
        _load_prefixed(decoder, state, "decoder")
        return cls(transformer.eval(), decoder.eval(), source=str(root))

    def to(self, device: Any) -> "MicroPipeline":
        self.transformer.to(device)
        self.decoder.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return next(self.transformer.parameters()).device

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
                eps = self.transformer(x, t, cond)
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
