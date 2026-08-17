"""Stable Diffusion XL's SERVING half: tuned schemas and shape arithmetic.

The same two-module split ``flux1_dev_serve`` documents (pgw#1331), applied to
the second catalog entry so the convention is the catalog's and not one
family's exception: ``sdxl.py`` is the DECLARATION and imports diffusers inside
its ``build`` callables; this module is what the request path reads, and it
imports nothing above ``torch``.

SDXL is deliberately NOT taken end to end here. pgw#1331 covers ONE family, and
a hand-written SDXL loop with nothing measuring it is how a wrong schedule
ships. What this module carries is the half the split makes structural — the
tuned schemas a handler resolves against on every call, and the latent
arithmetic both halves must agree on — so the serve-role fence covers the whole
catalog rather than one entry of it, and the next family cannot regress the
property by being written the old way.

The consequence, stated rather than left to be discovered: ``Sdxl`` has no
``scheduler()`` method, because ``euler_discrete``'s math is not implemented in
:mod:`gen_worker.family.scheduler`. That is an ``AttributeError`` a type checker
reports on the author's machine, which is the intended shape of the gap.
"""

from __future__ import annotations

from typing import Any, Final, Literal

from ..spec import TunedValues

#: CLIP's pinned prompt length. Pinned, not an axis — a variable text dimension
#: reaching a statically compiled denoiser mints a new graph per prompt length.
TEXT_TOKENS: Final = 77

#: The VAE's spatial stride.
VAE_STRIDE: Final = 8

#: The two-batch classifier-free-guidance arity SDXL is traced at: cond and
#: uncond ride ONE call. A CFG arity change is a different graph, so it is a
#: fact of the declaration rather than a runtime flag.
CFG_BATCH: Final = 2

#: The micro-conditioning vector's width (original/crop/target sizes).
TIME_IDS: Final = 6

#: SDXL's latent channel count.
LATENT_CHANNELS: Final = 4


def latent_edge(resolution: int) -> int:
    """Latent rows/cols for one pixel edge."""

    return resolution // VAE_STRIDE


def compute_dtype(layout: str) -> Any:
    """The compute dtype one tensor-layout token implies, for THIS family.

    A layout token is opaque to the SDK — it records the token and never
    interprets it (torchcg G15) — so the mapping to a torch dtype is the
    family's, stated once here rather than inferred at each site that needs it.
    """

    import torch

    return torch.bfloat16 if layout == "bf16" else torch.float32


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


__all__ = [
    "CFG_BATCH",
    "LATENT_CHANNELS",
    "TEXT_TOKENS",
    "TIME_IDS",
    "VAE_STRIDE",
    "SdxlLoraTuned",
    "SdxlTuned",
    "compute_dtype",
    "latent_edge",
]
