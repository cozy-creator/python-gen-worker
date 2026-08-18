"""Thin model types for the ship-code-as-is author surface (pgw#1370 seam).

``main_v2``-shape endpoints say ``ctx.checkpoint_defaults(SDXL)`` and read a
typed ``SDXL.Defaults``: per-checkpoint serving values are MUTABLE PLATFORM
DEPLOY STATE (hub rows validated against the schema the release EXPORTS —
``gen-worker release derive`` stamps ``msgspec`` JSON Schema of this struct),
and the zero-arg construction IS the platform fallback.

This module is deliberately minimal: it is the seam pgw#1370's derive and the
Paul-reviewed sdxl ``main_v2.py`` build against. The full defaults design
program (pgw#1376) and the model-type package shape (pgw#1377) own the final
form; coordinate there before widening it.
"""

from __future__ import annotations

from typing import Any, Literal, Optional, Union

import msgspec

Number = Union[int, float]

#: The PLATFORM-WIDE sampler vocabulary (checkpoint metadata is written in
#: it). An endpoint types its REQUEST field with its own served subset; a
#: metadata value the endpoint does not serve warns and falls through to the
#: tree's shipped scheduler (main_v2 three-layer resolution).
SamplerName = Literal[
    "dpmpp_2m_karras", "dpmpp_2m", "dpmpp_sde_karras", "euler",
    "euler_trailing", "euler_a", "unipc", "ddim", "lcm", "heun",
]


class Knob(msgspec.Struct, frozen=True):
    """One resolvable serving knob: a default plus the checkpoint's range.

    ``resolve(None) -> default``; a caller value is CLAMPED into
    ``[lo, hi]`` caller-visibly via ``ctx.clamp`` — the API declares no
    bounds of its own, the checkpoint's knob owns the constraint.
    """

    default: Number
    lo: Optional[Number] = None
    hi: Optional[Number] = None
    name: str = ""

    def resolve(self, value: Optional[Number], ctx: Any) -> Number:
        if value is None:
            return self.default
        clamped: Any = ctx.clamp(
            self.name or "value",
            value,
            lo=self.lo,
            hi=self.hi,
            reason="outside the checkpoint's declared range",
        )
        # ctx.clamp answers float; keep an int knob's integer arithmetic.
        if isinstance(self.default, int) and float(clamped).is_integer():
            return int(clamped)
        return float(clamped)


#: INTERIM dtype resolution for lanes spelled as bare contract HANDLES.
#: A lane is a tensorfs layout contract; when the author imports the
#: contract OBJECT (``tensorfs.contracts.*``), dtype rides on it and this
#: table is not consulted. Bare handles resolve here until the canonical
#: per-model-type entries land in tensorfs ``spec/v1/contracts``
#: (coordinate: pgw#1376/pgw#1377 defaults+model-type design lane).
CONTRACT_DTYPES: dict[str, Any] = {}


def register_contract_dtype(handle: str, dtype: Any) -> None:
    known = CONTRACT_DTYPES.get(handle)
    if known is not None and known != dtype:
        raise ValueError(
            f"contract {handle!r} already resolves to {known!r}; refusing to "
            f"re-register it as {dtype!r}"
        )
    CONTRACT_DTYPES[handle] = dtype


def _seed_sdxl_contracts() -> None:
    try:
        import torch
    except ImportError:  # pragma: no cover - torch-less installs never derive
        return
    register_contract_dtype("sdxl.diffusers-bf16@1", torch.bfloat16)
    # The fp8-rowwise lane LOADS bf16 (the quantized artifact path is the fp8
    # pipeline's; the serve host's from_pretrained dtype stays bf16).
    register_contract_dtype("cozy.sdxl-fp8-rowwise@1", torch.bfloat16)


class _SdxlRecipe(msgspec.Struct, frozen=True):
    """One SERVING RECIPE: the typed axes that drive a request, whichever
    source it came from (the checkpoint's own defaults, or a riding
    distillation adapter's). Both Defaults types inherit it, so endpoint
    code holds ONE type, never a union (main_v2 pattern)."""

    # cfg=False -> guidance-off serving (batch-1; no unconditional branch).
    cfg: bool = True
    steps: Knob = msgspec.field(
        default_factory=lambda: Knob(
            default=28, lo=1, hi=80, name="num_inference_steps"
        )
    )
    guidance: Knob = msgspec.field(
        default_factory=lambda: Knob(
            default=6.0, lo=1.0, hi=15.0, name="guidance_scale"
        )
    )
    # Checkpoint metadata sampler preference, platform vocabulary; None =
    # the tree's shipped scheduler stands.
    sampler: Optional[SamplerName] = None
    # Pinned denoising ladder (belongs to the recipe's sampler).
    timesteps: tuple[int, ...] = ()


class _SdxlDefaults(_SdxlRecipe, frozen=True):
    """Per-checkpoint deploy row over platform fallbacks; zero-arg = the
    platform opinion (the schema the release exports)."""

    positive_preamble: str = ""
    negative_preamble: str = ""
    # A STEP-distilled checkpoint refuses a stacked step-distillation
    # adapter (cfg is a separate axis: guidance-distilled full-step
    # checkpoints have cfg off but step_distilled False).
    step_distilled: bool = False


class _SdxlLoraDefaults(_SdxlRecipe, frozen=True):
    """A distillation adapter's own recipe metadata (what ``turbo.defaults``
    reads as; the derive's fake-adapter enumeration instantiates it)."""

    cfg: bool = False
    steps: Knob = msgspec.field(
        default_factory=lambda: Knob(
            default=4, lo=1, hi=12, name="num_inference_steps"
        )
    )
    sampler: Optional[SamplerName] = "euler_trailing"


class SDXL:
    """The SDXL model type: today, its recipe/defaults schema."""

    #: The canonical layout contract (what omitting lanes= means).
    CANONICAL_CONTRACT = "sdxl.diffusers-bf16@1"

    Recipe = _SdxlRecipe
    Defaults = _SdxlDefaults

    class Lora:
        """SDXL adapter metadata (distillation LoRAs et al.)."""

        Defaults = _SdxlLoraDefaults




_seed_sdxl_contracts()

__all__ = [
    "CONTRACT_DTYPES",
    "Knob",
    "Number",
    "SDXL",
    "register_contract_dtype",
]
