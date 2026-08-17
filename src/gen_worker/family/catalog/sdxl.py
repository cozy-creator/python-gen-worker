"""Stable Diffusion XL, declared. The second catalog entry, and the contrast.

SDXL is here beside FLUX.1-dev on purpose: the two exercise opposite ends of
the declaration vocabulary from ONE authoring contract, which is the claim
pgw#1326's "One SDK shape" makes and this is the evidence for it.

* Flux is a DiT: packed tokens, a flow-match scheduler, a guidance embedding.
* SDXL is a U-Net: 4-channel latents, a conv-heavy graph, a micro-conditioning
  block (``text_embeds`` + ``time_ids``) that arrives as a MAPPING parameter —
  so the generated binding's ``added_cond_kwargs`` is typed
  ``Mapping[str, Tensor]`` from the leaves' own paths (torchcg G4), with
  nothing per-family in the generator to make that happen.

The conv-heavy shape is also why ``resolution`` is a bucket rather than a
symbolic axis: DESIGN-RULINGS §4.30 forbids trading a conv family's served
efficiency for compile speed, and a symbolic latent H/W turns off inductor's
channels-last layout optimisation (measured at +7.2% on sdxl). Buckets keep
each class statically specialised, and the closed ``Literal`` keeps the set
exhaustive.

Same rules as the Flux entry: diffusers is imported only inside ``build``, the
config below is architecture (checkpoint-free), and the text encoders belong to
pgw#1331.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final, Literal

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

#: SDXL's U-Net architecture. Every SDXL fine-tune shares it, which is exactly
#: why one compiled cell serves all of them.
UNET: Final[Mapping[str, Any]] = {
    "sample_size": 128,
    "in_channels": 4,
    "out_channels": 4,
    "center_input_sample": False,
    "flip_sin_to_cos": True,
    "freq_shift": 0,
    "down_block_types": ("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
    "up_block_types": ("CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"),
    "block_out_channels": (320, 640, 1280),
    "layers_per_block": 2,
    "cross_attention_dim": 2048,
    "transformer_layers_per_block": (1, 2, 10),
    "attention_head_dim": (5, 10, 20),
    "use_linear_projection": True,
    "addition_embed_type": "text_time",
    "addition_time_embed_dim": 256,
    "projection_class_embeddings_input_dim": 2816,
    "norm_num_groups": 32,
}

#: SDXL's VAE architecture (4 latent channels, /8 spatial).
VAE: Final[Mapping[str, Any]] = {
    "in_channels": 3,
    "out_channels": 3,
    "latent_channels": 4,
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
    "scaling_factor": 0.13025,
}

#: CLIP's pinned prompt length. Pinned, not an axis — same reason as Flux's.
TEXT_TOKENS: Final = 77

#: The VAE's spatial stride.
VAE_STRIDE: Final = 8

#: The two-batch classifier-free-guidance arity SDXL is traced at: cond and
#: uncond ride ONE call. A CFG arity change is a different graph, so it is a
#: fact of the declaration rather than a runtime flag.
CFG_BATCH: Final = 2

#: The micro-conditioning vector's width (original/crop/target sizes).
TIME_IDS: Final = 6


def latent_edge(resolution: int) -> int:
    """Latent rows/cols for one pixel edge."""

    return resolution // VAE_STRIDE


class SdxlTuned(TunedValues, frozen=True):
    """SDXL's tuned-value SCHEMA. Catalog stamps the values, per release slot."""

    scheduler: Literal["euler_a", "dpmpp_2m_karras", "dpmpp_2m_sde_karras"] = "euler_a"
    steps: int = 28
    guidance: float = 6.0
    negative: str = ""
    #: A CLAMP, never a wire reshape.
    max_guidance: float | None = None


class SdxlLoraTuned(TunedValues, frozen=True):
    """The LoRA-kind overlay: every field is "no opinion" unless stated."""

    trigger_words: tuple[str, ...] = ()
    recommended_weight: float | None = None
    steps: int | None = None
    guidance: float | None = None
    scheduler: Literal["euler_a", "dpmpp_2m_karras", "dpmpp_2m_sde_karras"] | None = None


def _compute_dtype(layout: str) -> Any:
    """The compute dtype one layout token implies, for THIS family.

    A layout token is opaque to the SDK — it records the token and never
    interprets it (torchcg G15) — so the mapping to a torch dtype is the
    declaration's, stated once here rather than inferred anywhere.
    """

    import torch

    return torch.bfloat16 if layout == "bf16" else torch.float32


def _denoiser(layout: str) -> Any:
    import torch
    from diffusers import UNet2DConditionModel
    from torch import nn

    torch.set_default_dtype(_compute_dtype(layout))
    # Bound as a value, not called through the imported name: diffusers ships
    # no complete stubs, and this keeps the untyped boundary at ONE line per
    # build instead of a `type: ignore` on every attribute of the result.
    unet: Any = UNet2DConditionModel

    class _Denoiser(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.unet = unet(**dict(UNET))

        def forward(
            self,
            sample: Any,
            timestep: Any,
            encoder_hidden_states: Any,
            added_cond_kwargs: Any,
        ) -> Any:
            return self.unet(
                sample=sample,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

    return _Denoiser().eval()


def _denoiser_example(bucket: Mapping[str, int], layout: str) -> CallExample:
    import torch

    dtype = _compute_dtype(layout)
    edge = latent_edge(int(bucket["resolution"]))
    return CallExample(
        params=("sample", "timestep", "encoder_hidden_states", "added_cond_kwargs"),
        kwargs={
            "sample": torch.zeros(CFG_BATCH, 4, edge, edge, dtype=dtype),
            # float32 deliberately: euler-family samplers present a float32
            # scalar timestep and ddim/dpmpp-family samplers present int64. The
            # integer -> float32 recast is `ingress_selection_v1`'s ONE
            # permitted dtype normalization, and it is value-preserving on a
            # rank-0 feed — so declaring float32 admits both without the
            # sampler becoming a compile axis.
            "timestep": torch.zeros((), dtype=torch.float32),
            "encoder_hidden_states": torch.zeros(
                CFG_BATCH, TEXT_TOKENS, 2048, dtype=dtype
            ),
            "added_cond_kwargs": {
                "text_embeds": torch.zeros(CFG_BATCH, 1280, dtype=dtype),
                "time_ids": torch.zeros(CFG_BATCH, TIME_IDS, dtype=dtype),
            },
        },
    )


def _decoder(layout: str) -> Any:
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
        kwargs={"latents": torch.zeros(1, 4, edge, edge, dtype=dtype)},
    )


#: Stable Diffusion XL.
SDXL: Final = GraphFamily(
    name="sdxl",
    tuned=SdxlTuned,
    lora_tuned=SdxlLoraTuned,
    buckets=(Bucket("resolution", (768, 1024)),),
    runners=(
        Runner("decoder", build=_decoder, example=_decoder_example, axes=("resolution",)),
        Runner("denoiser", build=_denoiser, example=_denoiser_example, axes=("resolution",)),
    ),
    loop=Loop(stages=(Stage("denoiser", repeat="steps"), Stage("decoder"))),
    parameters=(Parameter("steps", minimum=1, maximum=100),),
    scheduler=Scheduler("euler_discrete", {"timestep_spacing": "leading"}),
)

__all__ = [
    "CFG_BATCH",
    "SDXL",
    "TEXT_TOKENS",
    "TIME_IDS",
    "UNET",
    "VAE",
    "VAE_STRIDE",
    "SdxlLoraTuned",
    "SdxlTuned",
    "latent_edge",
]
