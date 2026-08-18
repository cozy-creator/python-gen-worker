"""FLUX.1-dev, declared. The DECLARATION half of the catalog entry.

pgw#1326's catalog rule, literally: *"instead of importing diffusers, you
import the catalog."* An endpoint imports :class:`~gen_worker.model.catalog.
Flux1Dev` — the generated binding beside this file — and never touches
diffusers. This module is where diffusers and transformers are allowed, and
only inside the ``build`` callables, which run at MINT time and on an
eager-capable serving pod.

**This module is MINT-SIDE and the serve role may not import it (pgw#1331).**
Everything the request path needs — the tuned schemas, the packing arithmetic,
the scheduler block's constants and the loop itself — lives in
:mod:`gen_worker.model.catalog.flux1_dev_serve`, which imports ``torch`` and
nothing above it. The import direction is one-way: this module reads from
there, never the reverse, so the family's shape arithmetic has exactly one
definition and an artifact cannot be minted at a shape the loop will not ask
for. ``scripts/lint_serve_role_closure.py`` enforces the split.

**Checkpoint-free, and the architecture config is why.** The blocks below are
FLUX.1-dev's architecture, not any checkpoint's weights: every fine-tune that
shares it shares the graph classes, which is what lets one compiled graph serve
sixteen of them (DESIGN-RULINGS §4.27). A checkpoint whose config DIFFERS is a
different family, not another instance of this one — FLUX.1-schnell is the
live example (``guidance_embeds`` is False there, which changes the graph), and
it gets its own catalog entry rather than a fork arm here.

**Why the modules are wrapped.** Each ``build`` returns a thin wrapper whose
``forward`` is exactly the call the binding exposes. Two reasons, both
load-bearing: the upstream forwards take ``return_dict`` and other non-tensor
arguments that ``CallIngress`` v1 cannot record a pinned value for (torchcg
G4's known limit), so an unwrapped export would emit an unconstrained parameter
a caller has to remember to pass correctly; and the wrapper is where the
declaration states what the graph class RETURNS, instead of leaving it to
whatever container version of a model library is installed.

**Four runners, which is the whole pipeline** (pgw#1331 closed the gap): the
two text encoders, the denoiser, and the VAE decoder. Nothing in FLUX.1-dev's
composition is left riding an eager model at serve time, and the scheduler that
threads them is bare typed math in :mod:`gen_worker.model.scheduler` rather
than an object.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

from ..spec import (
    Bucket,
    CallExample,
    GraphModelSpec,
    Loop,
    Parameter,
    Runner,
    Scheduler,
    Stage,
)
from .flux1_dev_serve import (
    CLIP_TOKENS,
    LATENT_CHANNELS,
    TEXT_TOKENS,
    Flux1DevLoraTuned,
    Flux1DevTuned,
    compute_dtype,
    latent_edge,
    packed_tokens,
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
    "latent_channels": LATENT_CHANNELS,
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

#: CLIP-L's text tower — FLUX.1-dev's ``text_encoder``. It contributes ONE
#: pooled vector, which conditions the transformer's modulation; its per-token
#: states are unused, which is why the wrapper returns only the pooled output.
CLIP_TEXT: Final[Mapping[str, Any]] = {
    "vocab_size": 49408,
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_hidden_layers": 12,
    "num_attention_heads": 12,
    "max_position_embeddings": CLIP_TOKENS,
    "hidden_act": "quick_gelu",
    "layer_norm_eps": 1e-5,
    "projection_dim": 768,
    "eos_token_id": 2,
}

#: T5-v1.1-XXL's encoder — FLUX.1-dev's ``text_encoder_2``, and the branch that
#: actually carries the prompt. 4.7B parameters, which is why leaving it eager
#: was the largest single hole in the compiled coverage.
T5_TEXT: Final[Mapping[str, Any]] = {
    "vocab_size": 32128,
    "d_model": 4096,
    "d_kv": 64,
    "d_ff": 10240,
    "num_layers": 24,
    "num_heads": 64,
    "relative_attention_num_buckets": 32,
    "relative_attention_max_distance": 128,
    "dropout_rate": 0.0,
    "layer_norm_epsilon": 1e-6,
    "feed_forward_proj": "gated-gelu",
    "is_encoder_decoder": False,
    "use_cache": False,
    "tie_word_embeddings": False,
}

#: FLUX.1-dev's scheduler block, as the checkpoint's own ``scheduler/
#: scheduler_config.json`` states it. It is DECLARED here — not hardcoded in
#: the math — so it rides the export digest: a re-declared schedule changes the
#: family's identity instead of silently changing every request.
SCHEDULER: Final[Mapping[str, bool | int | float | str]] = {
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": True,
    "base_shift": 0.5,
    "max_shift": 1.15,
    "base_image_seq_len": 256,
    "max_image_seq_len": 4096,
}


def _denoiser(layout: str) -> Any:
    """The transformer, wrapped so its traced call is the binding's call."""

    import torch
    from diffusers import FluxTransformer2DModel
    from torch import nn

    # `set_default_dtype` rather than `.to(dtype)`: a fake parameter cannot be
    # swapped in place, so the dtype has to be in force while the module is
    # BUILT. `fake_structure()` restores the process default afterwards.
    torch.set_default_dtype(compute_dtype(layout))
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

    dtype = compute_dtype(layout)
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

    torch.set_default_dtype(compute_dtype(layout))
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

    dtype = compute_dtype(layout)
    edge = latent_edge(int(bucket["resolution"]))
    return CallExample(
        params=("latents",),
        kwargs={"latents": torch.zeros(1, LATENT_CHANNELS, edge, edge, dtype=dtype)},
    )


def _clip(layout: str) -> Any:
    """CLIP-L's text tower, wrapped down to the ONE output Flux consumes.

    The wrapper returns ``pooler_output`` alone. Returning the per-token states
    beside it would put a 77x768 tensor in the graph class's egress that no
    caller reads, and an artifact's outputs are part of its ingress digest — so
    an unused output is a permanent contract, not dead weight that optimizes
    away.
    """

    import torch
    from torch import nn
    from transformers import CLIPTextConfig, CLIPTextModel

    torch.set_default_dtype(compute_dtype(layout))
    config: Any = CLIPTextConfig
    text_model: Any = CLIPTextModel

    class _Clip(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.text_encoder = text_model(config(**dict(CLIP_TEXT)))

        def forward(self, input_ids: Any) -> Any:
            return self.text_encoder(input_ids=input_ids).pooler_output

    return _Clip().eval()


def _clip_example(bucket: Mapping[str, int], layout: str) -> CallExample:
    import torch

    del bucket, layout  # token IDs are integers; the layout decides no dtype here
    return CallExample(
        params=("input_ids",),
        kwargs={"input_ids": torch.zeros(1, CLIP_TOKENS, dtype=torch.long)},
    )


def _t5(layout: str) -> Any:
    """T5-XXL's encoder, wrapped down to its last hidden state.

    This is the branch that carries the prompt: 512 tokens of 4096-dim states
    that the joint attention reads at every one of the denoiser's steps. It is
    also 4.7B parameters run once per request — the eager tail pgw#1331 exists
    to close.
    """

    import torch
    from torch import nn
    from transformers import T5Config, T5EncoderModel

    torch.set_default_dtype(compute_dtype(layout))
    config: Any = T5Config
    encoder: Any = T5EncoderModel

    class _T5(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.text_encoder = encoder(config(**dict(T5_TEXT)))

        def forward(self, input_ids: Any) -> Any:
            return self.text_encoder(input_ids=input_ids).last_hidden_state

    return _T5().eval()


def _t5_example(bucket: Mapping[str, int], layout: str) -> CallExample:
    import torch

    del bucket, layout
    return CallExample(
        params=("input_ids",),
        kwargs={"input_ids": torch.zeros(1, TEXT_TOKENS, dtype=torch.long)},
    )


#: FLUX.1-dev. Buckets are pixel edges; every runner that declares the axis has
#: a class at each, so the generated ``Literal`` is exhaustive and every
#: selection resolves. The text encoders declare NO axis: their token lengths
#: are pinned by the architecture (CLIP-L's learned positions) and by the
#: family (T5's 512), so bucketing them would generate classes nothing selects.
FLUX1_DEV: Final = GraphModelSpec(
    name="flux1_dev",
    tuned=Flux1DevTuned,
    lora_tuned=Flux1DevLoraTuned,
    buckets=(Bucket("resolution", (768, 1024)),),
    runners=(
        Runner("clip", build=_clip, example=_clip_example),
        Runner("decoder", build=_decoder, example=_decoder_example, axes=("resolution",)),
        Runner("denoiser", build=_denoiser, example=_denoiser_example, axes=("resolution",)),
        Runner("t5", build=_t5, example=_t5_example),
    ),
    loop=Loop(
        stages=(
            Stage("clip"),
            Stage("t5"),
            Stage("denoiser", repeat="steps"),
            Stage("decoder"),
        )
    ),
    parameters=(Parameter("steps", minimum=1, maximum=100),),
    # A set of ONE: FLUX.1-dev's tuned schema names no sampler because the
    # family serves exactly this schedule, so `inst.scheduler()` still takes no
    # argument and still returns the concrete class (pgw#1346 K10).
    schedulers={"flow_match_euler": Scheduler("flow_match_euler_discrete", SCHEDULER)},
)

__all__ = [
    "CLIP_TEXT",
    "FLUX1_DEV",
    "SCHEDULER",
    "T5_TEXT",
    "TRANSFORMER",
    "VAE",
]
