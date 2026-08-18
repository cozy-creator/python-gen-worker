"""FLUX.1-schnell, declared. The DECLARATION half of the catalog entry.

**This module is MINT-SIDE and the serve role may not import it (pgw#1331).**
Everything the request path needs lives in
:mod:`gen_worker.model.catalog.flux1_schnell_serve`.

**Schnell is FLUX.1-dev's architecture with ONE field flipped**, so this module
IMPORTS dev's blocks rather than copying them. That is the whole anti-
triplication argument of pgw#1346 B1, applied where it is actually true: the
two checkpoints publish nine architecture fields, eight identical, and
``guidance_embeds`` false here against true there. The VAE, CLIP-L and T5-XXL
blocks are shared outright — one autoencoder affine, one CLIP window, one T5
config, defined once in :mod:`~gen_worker.model.catalog.flux1_dev`.

**And that one field is why this is a MODEL and not an instance of dev.** It
removes an INPUT from the denoiser's traced call: ``pipeline_flux.py`` branches
on ``transformer.config.guidance_embeds`` and passes ``guidance=None``, so the
graph class differs. ``flux1_dev.py``'s own docstring names schnell as the live
example of exactly this, and B1's measurement confirmed it against the
published configs rather than against the model's reputation.

**Sourced, not guessed.** Every number below is checked by
``tests/test_flux1_schnell_pgw1346.py`` against
``tests/fixtures/flux1_schnell/``, which caches the serving release's own
published documents with their digests. The scheduler block matters most: it
rides the export digest, so a wrong ``shift`` would silently re-key the family
instead of failing.

**Three runners, not four.** CLIP-L, T5 and the denoiser are declared; the VAE
decode is not, because schnell's preset grid is RECTANGULAR — seven sizes
collapse onto four token coordinates, so a token bucket cannot tell a decoder
its output shape (1152x864 and 864x1152 are one bucket and two shapes). Dev
gets away with a decoder runner only because its own grid is square. This is
also what the endpoint does today: ``targets=("transformer",)``, narrowed from
the SDK default in ie#685.
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
from .flux1_dev import CLIP_TEXT, T5_TEXT, TRANSFORMER as DEV_TRANSFORMER, VAE
from .flux1_schnell_serve import (
    CLIP_TOKENS,
    TEXT_TOKENS,
    Flux1SchnellLoraTuned,
    Flux1SchnellTuned,
    compute_dtype,
)

#: FLUX.1-schnell's transformer architecture: dev's, with the ONE field the
#: published config actually differs on. Spelled as a derivation rather than a
#: copy so the eight shared numbers cannot drift apart, and so the difference is
#: the only thing a reader has to check.
TRANSFORMER: Final[Mapping[str, Any]] = {
    **DEV_TRANSFORMER,
    # The distillation's signature: no guidance embedding, therefore no
    # `guidance` input on the traced call.
    "guidance_embeds": False,
}

#: Schnell's scheduler block, as the serving release's own
#: ``scheduler/scheduler_config.json`` states it. STATIC shift, where dev's is
#: dynamic — the one scheduler difference between the two, and the reason the
#: batch plan could promise "zero scheduler math owed" for this batch:
#: ``flow_match_euler_discrete`` already implements both arms.
SCHEDULER: Final[Mapping[str, bool | int | float | str]] = {
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "use_dynamic_shifting": False,
    "base_shift": 0.5,
    "max_shift": 1.15,
    "base_image_seq_len": 256,
    "max_image_seq_len": 4096,
}

#: The four packed-token coordinates the endpoint's seven aspect rows produce.
#: Transposed presets collapse (ie#685), which is why seven rows are four
#: graph classes and not seven.
TOKEN_BUCKETS: Final = (3888, 4032, 4056, 4096)


def _denoiser(layout: str) -> Any:
    """The transformer, wrapped so its traced call is the binding's call."""

    import torch
    from diffusers import FluxTransformer2DModel
    from torch import nn

    torch.set_default_dtype(compute_dtype(layout))
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
        ) -> Any:
            # No `guidance=`: this checkpoint declares `guidance_embeds: false`,
            # and `FluxPipeline` passes the literal None on that arm. Threading
            # a None here would trace the same graph and document the wrong
            # contract; omitting the parameter is what states the contract.
            return self.transformer(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                timestep=timestep,
                img_ids=img_ids,
                txt_ids=txt_ids,
                return_dict=False,
            )[0]

    return _Denoiser().eval()


def _denoiser_example(bucket: Mapping[str, int], layout: str) -> CallExample:
    import torch

    dtype = compute_dtype(layout)
    tokens = int(bucket["tokens"])
    return CallExample(
        params=(
            "hidden_states",
            "encoder_hidden_states",
            "pooled_projections",
            "timestep",
            "img_ids",
            "txt_ids",
        ),
        kwargs={
            "hidden_states": torch.zeros(1, tokens, 64, dtype=dtype),
            "encoder_hidden_states": torch.zeros(1, TEXT_TOKENS, 4096, dtype=dtype),
            "pooled_projections": torch.zeros(1, 768, dtype=dtype),
            "timestep": torch.zeros(1, dtype=dtype),
            "img_ids": torch.zeros(tokens, 3, dtype=dtype),
            "txt_ids": torch.zeros(TEXT_TOKENS, 3, dtype=dtype),
        },
    )


def _clip(layout: str) -> Any:
    """CLIP-L's text tower, wrapped down to the pooled vector Flux consumes."""

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

    del bucket, layout
    return CallExample(
        params=("input_ids",),
        kwargs={"input_ids": torch.zeros(1, CLIP_TOKENS, dtype=torch.long)},
    )


def _t5(layout: str) -> Any:
    """T5-XXL's encoder, wrapped down to its last hidden state."""

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
    # 256, not dev's 512 — the pin is the family's, and it is what makes the
    # text dimension constant by construction (th#1126, ie#544).
    return CallExample(
        params=("input_ids",),
        kwargs={"input_ids": torch.zeros(1, TEXT_TOKENS, dtype=torch.long)},
    )


#: FLUX.1-schnell. The text encoders declare NO axis: their token lengths are
#: pinned by the architecture (CLIP-L's learned positions) and by the family
#: (T5's 256), so bucketing them would generate classes nothing selects.
#:
#: The ie#740 serving floor is preserved BY VALUE from the endpoint's retired
#: ``Slot`` (pgw#1346 K1): the bf16 lane's 36 GB. Schnell declares no fp8 lane —
#: its endpoint offers only ``plain.bf16@1`` — so there is no sm floor to carry,
#: and inventing one would decline cards that serve it today.
FLUX1_SCHNELL: Final = GraphModelSpec(
    name="flux1_schnell",
    tuned=Flux1SchnellTuned,
    lora_tuned=Flux1SchnellLoraTuned,
    buckets=(Bucket("tokens", TOKEN_BUCKETS),),
    layouts={"*": ("plain.bf16@1",)},
    layout_requirements={"plain.bf16@1": "vram36g"},
    runners=(
        Runner("clip", build=_clip, example=_clip_example),
        Runner("denoiser", build=_denoiser, example=_denoiser_example, axes=("tokens",)),
        Runner("t5", build=_t5, example=_t5_example),
    ),
    loop=Loop(
        stages=(
            Stage("clip"),
            Stage("t5"),
            Stage("denoiser", repeat="steps"),
        )
    ),
    # 1-4 steps is the distillation's published contract (ie#462).
    parameters=(Parameter("steps", minimum=1, maximum=4),),
    # A set of ONE (pgw#1346 K10): this family's tuned schema names no sampler
    # because it serves exactly this schedule, so `inst.scheduler()` still
    # takes no argument and still returns the concrete class.
    schedulers={"flow_match_euler": Scheduler("flow_match_euler_discrete", SCHEDULER)},
)

__all__ = [
    "FLUX1_SCHNELL",
    "SCHEDULER",
    "TOKEN_BUCKETS",
    "TRANSFORMER",
    "VAE",
]
