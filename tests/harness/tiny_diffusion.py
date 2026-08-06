"""pgw#978: a randomly-initialized latent-diffusion toy, built on this box.

The micro-mint rig needs a model that is *shaped* like the thing the fleet
mints — a cross-attention denoiser over a 4-channel latent, plus a decoder —
and is small enough that a whole mint fits inside a 4 GiB cap on a laptop card.
It must also need NO download: a rig whose first step is a network fetch is a
rig that stops working on a bad night, and the weights-locality rule
(Paul, amended 2026-08-06) bounds what may exist on this box by SIZE, so the
cheapest way to stay inside it is to never fetch anything.

So the checkpoint is *generated*: ``build_checkpoint`` seeds a generator, writes
the state dict and a small ``config.json``, and the whole tree is under 20 MB.
Two runs with the same seed produce byte-identical weights, which is what lets a
cell key mean anything across processes.

What is deliberately REAL here: the module is a ``torch.nn.Module`` with a
cross-attention block and a timestep embedding, so ``torch.export`` traces
something with branches, broadcasts and a matmul chain rather than a single
Linear. What is deliberately fake: there is no text encoder, no tokenizer and no
scheduler worth the name — the rig measures the MACHINERY, and a real sampler
would only add wall-clock.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch import nn

#: Latent channels, matching the SD-class convention the declaration mirrors.
LATENT_CHANNELS = 4
#: Latent grid the rig exports at. 32x32 is a 256x256 image at the usual /8.
LATENT_SIZE = 32
#: Cross-attention context width and its pinned token count.
CONTEXT_DIM = 64
TEXT_LEN = 77
#: Model width. Small enough that a full export + AOTI link is seconds, wide
#: enough that inductor has a real fusion decision to make.
WIDTH = 64

#: Fixed, so a checkpoint built twice is byte-identical and a cell key computed
#: on two boots agrees.
SEED = 978

WEIGHTS_NAME = "diffusion_pytorch_model.bin"
CONFIG_NAME = "config.json"


class _TimestepEmbedding(nn.Module):
    """Sinusoidal features -> two Linears. The usual shape, at toy width."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.lin1 = nn.Linear(width, width * 2)
        self.lin2 = nn.Linear(width * 2, width)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = self.width // 2
        freqs = torch.exp(
            -torch.arange(half, dtype=torch.float32, device=timestep.device)
            * (9.21034037 / max(1, half - 1)))
        angles = timestep.float()[:, None] * freqs[None, :]
        feats = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        return self.lin2(torch.nn.functional.silu(self.lin1(feats)))


class _CrossAttention(nn.Module):
    """Single-head cross attention from the latent grid onto the context.

    Single-head on purpose: the rig is measuring export/compile/publish, and a
    multi-head reshape adds trace nodes without adding a code path the mint
    treats differently.
    """

    def __init__(self, width: int, context_dim: int) -> None:
        super().__init__()
        self.scale = width ** -0.5
        self.to_q = nn.Linear(width, width, bias=False)
        self.to_k = nn.Linear(context_dim, width, bias=False)
        self.to_v = nn.Linear(context_dim, width, bias=False)
        self.to_out = nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        flat = self.norm(x.reshape(b, c, h * w).transpose(1, 2))
        q = self.to_q(flat)
        k = self.to_k(context)
        v = self.to_v(context)
        attn = torch.softmax((q @ k.transpose(1, 2)) * self.scale, dim=-1)
        out = self.to_out(attn @ v)
        return x + out.transpose(1, 2).reshape(b, c, h, w)


class TinyUNet(nn.Module):
    """The denoiser. Signature mirrors diffusers' ``UNet2DConditionModel``:
    ``(sample, timestep, encoder_hidden_states)`` positionally, which is what
    the export declaration's input rows bind to.
    """

    def __init__(
        self,
        width: int = WIDTH,
        context_dim: int = CONTEXT_DIM,
        latent_channels: int = LATENT_CHANNELS,
    ) -> None:
        super().__init__()
        self.width = width
        self.context_dim = context_dim
        self.latent_channels = latent_channels
        self.conv_in = nn.Conv2d(latent_channels, width, 3, padding=1)
        self.time_embed = _TimestepEmbedding(width)
        self.down = nn.Conv2d(width, width, 3, stride=2, padding=1)
        self.mid_norm = nn.GroupNorm(8, width)
        self.mid = nn.Conv2d(width, width, 3, padding=1)
        self.attn = _CrossAttention(width, context_dim)
        self.up = nn.ConvTranspose2d(width, width, 4, stride=2, padding=1)
        self.out_norm = nn.GroupNorm(8, width)
        self.conv_out = nn.Conv2d(width, latent_channels, 3, padding=1)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        h = self.conv_in(sample)
        temb = self.time_embed(timestep)[:, :, None, None]
        h = h + temb
        skip = h
        h = self.down(h)
        h = torch.nn.functional.silu(self.mid_norm(self.mid(h)))
        h = self.attn(h, encoder_hidden_states)
        h = self.up(h) + skip
        return self.conv_out(torch.nn.functional.silu(self.out_norm(h)))


class TinyDecoder(nn.Module):
    """Latent -> pixels. Present so the pipeline has a second component and the
    rig can prove the mint picks the DECLARED target rather than the first
    module it finds."""

    def __init__(self, width: int = WIDTH, latent_channels: int = LATENT_CHANNELS) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(latent_channels, width, 3, padding=1)
        self.up = nn.ConvTranspose2d(width, width, 4, stride=2, padding=1)
        self.conv_out = nn.Conv2d(width, 3, 3, padding=1)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        h = torch.nn.functional.silu(self.conv_in(latent))
        return torch.tanh(self.conv_out(self.up(h)))


class TinyDiffusionPipeline:
    """A diffusers-SHAPED holder: ``.unet`` is the compile target, ``.vae`` is
    the support object beside it.

    ``from_pretrained`` is the real contract the SDK's slot loader calls, and it
    reads only from the local tree it is handed — the rig never resolves a hub
    ref for weights.
    """

    def __init__(self, unet: TinyUNet, vae: TinyDecoder, source: str = "") -> None:
        self.unet = unet
        self.vae = vae
        self.source = source

    @classmethod
    def from_pretrained(cls, path: str, **_kw: Any) -> "TinyDiffusionPipeline":
        root = Path(path)
        config = json.loads((root / CONFIG_NAME).read_text())
        unet = TinyUNet(
            width=int(config["width"]),
            context_dim=int(config["context_dim"]),
            latent_channels=int(config["latent_channels"]))
        vae = TinyDecoder(
            width=int(config["width"]),
            latent_channels=int(config["latent_channels"]))
        state = torch.load(root / WEIGHTS_NAME, map_location="cpu", weights_only=True)
        unet.load_state_dict(state["unet"])
        vae.load_state_dict(state["vae"])
        return cls(unet.eval(), vae.eval(), source=str(root))

    def to(self, device: Any) -> "TinyDiffusionPipeline":
        self.unet.to(device)
        self.vae.to(device)
        return self

    def __call__(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        steps: int = 2,
    ) -> torch.Tensor:
        """A two-step Euler-ish loop. Enough to be a forward pass with a real
        sampler's shape; not enough to cost anything."""
        x = latents
        for i in reversed(range(steps)):
            t = torch.full((x.shape[0],), float(i * 100), device=x.device)
            x = x - self.unet(x, t, context) / max(1, steps)
        return self.vae(x)


def example_inputs(
    batch: int = 1,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """One legal call's tensors — the same shapes the declaration states."""
    gen = torch.Generator().manual_seed(SEED)
    return {
        "sample": torch.randn(
            batch, LATENT_CHANNELS, LATENT_SIZE, LATENT_SIZE, generator=gen,
        ).to(device=device, dtype=dtype),
        "timestep": torch.full((batch,), 100.0, device=device, dtype=dtype),
        "encoder_hidden_states": torch.randn(
            batch, TEXT_LEN, CONTEXT_DIM, generator=gen,
        ).to(device=device, dtype=dtype),
    }


#: pgw#978/pgw#981: the env that installs a SYNTHETIC runtime key so a cardless
#: box can complete the seal, publish and adopt legs.
SYNTHETIC_RUNTIME_ENV = "PGW978_SYNTHETIC_RUNTIME"

#: The values it installs. Every one is a real fact of a real runtime (an L4 at
#: sm_89, cuda 13.0) — nothing about how they are COMBINED into a cell key is
#: faked, which is the same seam `test_aot_mint_pgw723._gpu_runtime` has used
#: since pgw#723. sm_89 specifically because it is this box's own compute
#: capability; the box simply cannot execute cu130 today (pgw#981).
SYNTHETIC_RUNTIME = {
    "sku": "l4", "sm": "sm_89", "triton": "3.6.0", "cuda": "13.0",
    "cuda_driver": "13000", "image_digest": "sha256:" + "ab" * 32,
}


def install_synthetic_runtime_if_asked() -> bool:
    """Patch the runtime PROBES — and nothing else — when the env asks.

    A cell key requires an ``sm``, and a box with no usable CUDA build can state
    none: the mint refuses at seal with *"cannot state the compute capability
    (sm) of this runtime; an exported cell has no identity without it"*. That
    refusal is correct and must stay correct, so this does not weaken it — it
    supplies the one fact the box cannot measure, exactly the way pgw#723's
    tests have, and ONLY when explicitly asked.

    What this buys: the seal, publish and adopt legs run at all. Those are where
    the hub wire and the cross-pod filter live, and skipping them would remove
    most of what is left after the export. What it costs is stated in the rig's
    report and must never be implied away — the resulting cell's ``sm`` axis is
    a claim about a runtime this process is not, so a cell minted this way is a
    PLUMBING artifact and must never reach a shared namespace.

    Called at endpoint-module import time, which is the only hook the mint child
    (a fresh interpreter that shares nothing with the parent) offers.
    """
    import os

    if str(os.environ.get(SYNTHETIC_RUNTIME_ENV, "")).strip() not in ("1", "true", "yes"):
        return False
    import torch as _torch

    from gen_worker import aot_serve, compile_cache

    full = dict(SYNTHETIC_RUNTIME, torch=str(_torch.__version__))
    compile_cache.runtime_key = lambda: dict(full)  # type: ignore[assignment]
    aot_serve.runtime_key = lambda: {  # type: ignore[assignment]
        "sku": full["sku"], "sm": full["sm"], "torch": full["torch"],
        "cuda": full["cuda"]}
    return True


def build_checkpoint(root: Path, *, seed: int = SEED) -> Path:
    """Write a deterministic random checkpoint under ``root``; return the tree.

    Idempotent: an existing tree with the same config is left alone, so the rig
    re-runs without re-serializing.
    """
    root = Path(root)
    config = {
        "width": WIDTH,
        "context_dim": CONTEXT_DIM,
        "latent_channels": LATENT_CHANNELS,
        "latent_size": LATENT_SIZE,
        "text_len": TEXT_LEN,
        "seed": int(seed),
        "_class_name": "TinyDiffusionPipeline",
    }
    cfg_path = root / CONFIG_NAME
    if cfg_path.is_file() and (root / WEIGHTS_NAME).is_file():
        try:
            if json.loads(cfg_path.read_text()) == config:
                return root
        except ValueError:
            pass
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(seed))
    unet = TinyUNet(WIDTH, CONTEXT_DIM, LATENT_CHANNELS)
    vae = TinyDecoder(WIDTH, LATENT_CHANNELS)
    torch.save({"unet": unet.state_dict(), "vae": vae.state_dict()},
               root / WEIGHTS_NAME)
    cfg_path.write_text(json.dumps(config, indent=2, sort_keys=True))
    return root


def checkpoint_bytes(root: Path) -> int:
    """Total size of the built tree — the number the size carve-out bounds."""
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())


def parameter_count(module: Optional[nn.Module] = None) -> int:
    module = module if module is not None else TinyUNet()
    return sum(p.numel() for p in module.parameters())


__all__ = [
    "CONFIG_NAME",
    "CONTEXT_DIM",
    "LATENT_CHANNELS",
    "LATENT_SIZE",
    "SEED",
    "TEXT_LEN",
    "WEIGHTS_NAME",
    "WIDTH",
    "TinyDecoder",
    "TinyDiffusionPipeline",
    "TinyUNet",
    "build_checkpoint",
    "checkpoint_bytes",
    "example_inputs",
    "parameter_count",
]
