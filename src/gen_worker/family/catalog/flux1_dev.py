"""FLUX.1-dev, declared. The DECLARATION half of the catalog entry.

pgw#1326's catalog rule, literally: *"instead of importing diffusers, you
import the catalog."* An endpoint imports :class:`~gen_worker.family.catalog.
Flux1Dev` — the generated binding beside this file — and never touches
diffusers. This module is where diffusers is allowed, and only inside the
``build`` callables, which run at MINT time and on an eager-capable serving
pod. Importing this module imports no model code at all, which is what lets an
adopt-only serve role (pgw#1328) hold the bindings without acquiring diffusers.

**Checkpoint-free, and the architecture config is why.** The block below is
FLUX.1-dev's architecture, not any checkpoint's weights: every fine-tune that
shares it shares the graph classes, which is what lets one compiled cell serve
sixteen of them (DESIGN-RULINGS §4.27). A checkpoint whose config DIFFERS is a
different family, not another instance of this one — FLUX.1-schnell is the
live example (``guidance_embeds`` is False there, which changes the graph), and
it gets its own catalog entry rather than a fork arm here.

**Why the modules are wrapped.** Each ``build`` returns a thin wrapper whose
``forward`` is exactly the call the binding exposes. Two reasons, both
load-bearing: diffusers' own forwards take ``return_dict`` and other
non-tensor arguments that ``CallIngress`` v1 cannot record a pinned value for
(torchcg G4's known limit), so an unwrapped export would emit an unconstrained
parameter a caller has to remember to pass correctly; and the wrapper is where
the declaration states what the graph class RETURNS, instead of leaving it to
whatever container version of diffusers is installed.

Text encoders are deliberately absent: minting them is pgw#1331's lane, and a
declaration that named a runner nothing exports would fail its own build.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from ..spec import (
    Bucket,
    CallExample,
    GraphFamily,
    Loop,
    Parameter,
    Runner,
    Scheduler,
    Stage,
    TunedValues,
)

#: FLUX.1-dev's transformer architecture. Class-level truth: no weight, no
#: checkpoint ref, no tuned value appears here or can.
TRANSFORMER: Final[Mapping[str, Any]] = {
    "patch_size": 1,
    "in_channels": 64,
    "out_channels": 64,
    "num_layers": 19,
    "num_single_layers": 38,
    "attention_head_dim": 128,
    "num_attention_heads": 24,
    "joint_attention_dim": 4096,
    "pooled_projection_dim": 768,
    "guidance_embeds": True,
    "axes_dims_rope": (16, 56, 56),
}

#: FLUX.1-dev's VAE architecture (16 latent channels, /8 spatial).
VAE: Final[Mapping[str, Any]] = {
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 16,
    "block_out_channels": (128, 256, 512, 512),
    "layers_per_block": 2,
    "down_block_types": (
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
        "DownEncoderBlock2D",
    ),
    "up_block_types": (
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    ),
    "norm_num_groups": 32,
    "sample_size": 1024,
    "scaling_factor": 0.3611,
    "shift_factor": 0.1159,
}

#: Prompt tokens the T5 branch is padded to. A pinned length, not an axis: a
#: variable text dimension reaching a statically compiled denoiser mints a new
#: graph per prompt length — unbounded and un-warmable.
TEXT_TOKENS: Final = 512

#: The VAE's spatial stride, and the transformer's packing factor. Together
#: they turn a pixel edge into a token count, which is the ONLY arithmetic this
#: declaration does.
VAE_STRIDE: Final = 8
PATCH: Final = 2


def latent_edge(resolution: int) -> int:
    """Latent rows/cols for one pixel edge."""

    return resolution // VAE_STRIDE


def packed_tokens(resolution: int) -> int:
    """Packed transformer tokens for one pixel edge."""

    edge = latent_edge(resolution) // PATCH
    return edge * edge


class Flux1DevTuned(TunedValues, frozen=True):
    """FLUX.1-dev's tuned-value SCHEMA. The values are catalog, per release slot.

    Every field is a knob a checkpoint may legitimately have a different
    opinion about. Nothing here is a graph fact: a value that changed the graph
    would belong on a bucket axis, where the type system can keep the class set
    exhaustive.
    """

    steps: int = 28
    guidance: float = 3.5
    shift: float = 3.0
    #: A CLAMP, never a wire reshape: a request asking for more is served at
    #: this value with an adjustment recorded, not refused.
    max_guidance: float | None = None


class Flux1DevLoraTuned(TunedValues, frozen=True):
    """The LoRA-kind overlay for the same family: every field is "no opinion".

    ``None`` means the overlay declines to override the checkpoint's own tuned
    value. That is why every field is optional and none carries a real default:
    an overlay with an opinion it did not mean to have would silently retune
    every checkpoint it is applied to.
    """

    trigger_words: tuple[str, ...] = ()
    recommended_weight: float | None = None
    steps: int | None = None
    guidance: float | None = None
    shift: float | None = None


def _compute_dtype(layout: str) -> Any:
    """The compute dtype one layout token implies, for THIS family.

    A layout token is opaque to the SDK — it records the token and never
    interprets it (torchcg G15) — so the mapping to a torch dtype is the
    declaration's, stated once here rather than inferred anywhere.
    """

    import torch

    return torch.bfloat16 if layout == "bf16" else torch.float32


def _denoiser(layout: str) -> Any:
    """The transformer, wrapped so its traced call is the binding's call."""

    import torch
    from diffusers import FluxTransformer2DModel
    from torch import nn

    # `set_default_dtype` rather than `.to(dtype)`: a fake parameter cannot be
    # swapped in place, so the dtype has to be in force while the module is
    # BUILT. `fake_structure()` restores the process default afterwards.
    torch.set_default_dtype(_compute_dtype(layout))
    # Bound as a value, not called through the imported name: diffusers ships
    # no complete stubs, and this keeps the untyped boundary at ONE line per
    # build instead of a `type: ignore` on every attribute of the result.
    transformer: Any = FluxTransformer2DModel

    class _Denoiser(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.transformer = transformer(**dict(TRANSFORMER))

        def forward(
            self,
            hidden_states: Any,
            encoder_hidden_states: Any,
            pooled_projections: Any,
            timestep: Any,
            img_ids: Any,
            txt_ids: Any,
            guidance: Any,
        ) -> Any:
            return self.transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                guidance=guidance,
                return_dict=False,
            )[0]

    return _Denoiser().eval()


def _denoiser_example(bucket: Mapping[str, int], layout: str) -> CallExample:
    import torch

    dtype = _compute_dtype(layout)
    tokens = packed_tokens(int(bucket["resolution"]))
    return CallExample(
        params=(
            "hidden_states",
            "encoder_hidden_states",
            "pooled_projections",
            "timestep",
            "img_ids",
            "txt_ids",
            "guidance",
        ),
        kwargs={
            "hidden_states": torch.zeros(1, tokens, 64, dtype=dtype),
            "encoder_hidden_states": torch.zeros(1, TEXT_TOKENS, 4096, dtype=dtype),
            "pooled_projections": torch.zeros(1, 768, dtype=dtype),
            "timestep": torch.zeros(1, dtype=dtype),
            "img_ids": torch.zeros(tokens, 3, dtype=dtype),
            "txt_ids": torch.zeros(TEXT_TOKENS, 3, dtype=dtype),
            "guidance": torch.zeros(1, dtype=dtype),
        },
    )


def _decoder(layout: str) -> Any:
    """The VAE decoder half, wrapped: encode is not on the serve path."""

    import torch
    from diffusers import AutoencoderKL
    from torch import nn

    torch.set_default_dtype(_compute_dtype(layout))
    autoencoder: Any = AutoencoderKL

    class _Decoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vae = autoencoder(**dict(VAE))

        def forward(self, latents: Any) -> Any:
            return self.vae.decode(latents, return_dict=False)[0]

    return _Decoder().eval()


def _decoder_example(bucket: Mapping[str, int], layout: str) -> CallExample:
    import torch

    dtype = _compute_dtype(layout)
    edge = latent_edge(int(bucket["resolution"]))
    return CallExample(
        params=("latents",),
        kwargs={"latents": torch.zeros(1, 16, edge, edge, dtype=dtype)},
    )


#: FLUX.1-dev. Buckets are pixel edges; every runner has a class at each, so
#: the generated ``Literal`` is exhaustive and every selection resolves.
FLUX1_DEV: Final = GraphFamily(
    name="flux1_dev",
    tuned=Flux1DevTuned,
    lora_tuned=Flux1DevLoraTuned,
    buckets=(Bucket("resolution", (768, 1024)),),
    runners=(
        Runner("decoder", build=_decoder, example=_decoder_example, axes=("resolution",)),
        Runner("denoiser", build=_denoiser, example=_denoiser_example, axes=("resolution",)),
    ),
    loop=Loop(stages=(Stage("denoiser", repeat="steps"), Stage("decoder"))),
    parameters=(Parameter("steps", minimum=1, maximum=100),),
    scheduler=Scheduler("flow_match_euler_discrete", {"shift": 3.0}),
)

__all__ = [
    "FLUX1_DEV",
    "PATCH",
    "TEXT_TOKENS",
    "TRANSFORMER",
    "VAE",
    "VAE_STRIDE",
    "Flux1DevLoraTuned",
    "Flux1DevTuned",
    "latent_edge",
    "packed_tokens",
]
