"""Per-family export-input contracts for the AOT mint (#723).

``torch.export`` needs example inputs with the exact structure the target's
forward takes. That structure is FAMILY knowledge — SDXL's UNet wants
``added_cond_kwargs={"text_embeds", "time_ids"}``, LTX2 drives per-token
timesteps, z-image takes ragged lists of tensors (#729) — so it lives here,
registered per family, rather than as an ``if`` ladder inside the mint driver.

Composition itself is NOT family-specific and mirrors the dynamo mint
(``compile_cache.build``): ``load_from_pretrained`` -> ``place_pipeline``. That
is a parity requirement, not tidiness — the placement/low-VRAM flags are traced
INTO the graph, so a cell composed differently from the serving path is a cell
the serving path can never use (gw#391, ie#381 are both that bug).

The ONE deliberate divergence is the LoRA lane: the dynamo mint calls
``compile_cache.apply_lora_lane``, which installs zeroed branch BUFFERS —
module state that export would bake. The exported lane installs
``lora_lifted.install_lifted_lora_lanes`` instead, so the adapter arrives as
call arguments (#725 option 2). Same bucket, different installation, because
the two lanes disagree about what "dynamic" means.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, Dict, Mapping, Tuple

from .compile_cache import resolve_pipeline_class
from .models import lora_lifted
from .models.loading import load_from_pretrained
from .models.memory import place_pipeline

if TYPE_CHECKING:  # pragma: no cover — typing only, no runtime import cycle
    from .aot_mint import DynamicDim, ExportSpec

logger = logging.getLogger(__name__)

#: SDXL pooled-text-embedding width and micro-conditioning vector length. Both
#: are architectural constants of the SDXL conditioning contract, not config
#: fields — ``trt_engine._export_unet_onnx`` pins the same two.
SDXL_POOLED_DIM = 1280
SDXL_TIME_IDS = 6

InputBuilder = Callable[[Any, "ExportSpec"], Tuple[Tuple[Any, ...], Dict[str, Any]]]

_BUILDERS: Dict[str, InputBuilder] = {}


class InputContractError(RuntimeError):
    """No export-input contract is registered for a family, or it cannot be
    satisfied from the composed pipeline."""


def inputs_for(family: str) -> Callable[[InputBuilder], InputBuilder]:
    """Register the example-input builder for one family."""

    def register(fn: InputBuilder) -> InputBuilder:
        _BUILDERS[str(family)] = fn
        return fn

    return register


def builder_for(family: str) -> InputBuilder:
    fn = _BUILDERS.get(str(family))
    if fn is None:
        raise InputContractError(
            f"no AOT export-input contract registered for family "
            f"{family!r} (have: {sorted(_BUILDERS)!r}) — a family's forward "
            f"signature is family knowledge and must be declared before it "
            f"can be exported"
        )
    return fn


# ---------------------------------------------------------------------------
# SDXL
# ---------------------------------------------------------------------------


@inputs_for("sdxl")
def sdxl_unet_inputs(
    module: Any, spec: "ExportSpec",
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """Example inputs for the SDXL ``UNet2DConditionModel`` forward.

    Widths come from the module's own config (``in_channels``,
    ``cross_attention_dim``) rather than from constants, so a family member
    with a different conditioning width exports correctly instead of failing
    deep inside the trace. Shapes come from the FIRST declared shape row: the
    latent H/W axes are symbolic, so which row seeds the trace decides nothing
    about what the artifact admits — the declared range does.

    ``return_dict=False`` is passed because a dataclass output is not a valid
    export output; the consumer re-wraps.
    """
    import torch

    if not spec.shapes:
        raise InputContractError(
            "sdxl export needs at least one declared shape row (w, h) to seed "
            "the trace")
    width, height = int(spec.shapes[0][0]), int(spec.shapes[0][1])
    scale = 8
    latent_h, latent_w = height // scale, width // scale
    batch = 2 if _has_cfg(spec) else 1
    text_len = int(spec.text_lens[0]) if spec.text_lens else 77
    config = getattr(module, "config", None)
    latent_channels = int(getattr(config, "in_channels", 4) or 4)
    cross_dim = int(getattr(config, "cross_attention_dim", 2048) or 2048)
    dtype, device = _module_dtype_device(module)
    kw = {"dtype": dtype, "device": device}

    args = (
        torch.randn(batch, latent_channels, latent_h, latent_w, **kw),
        torch.tensor(1.0, device=device),
        torch.randn(batch, text_len, cross_dim, **kw),
    )
    kwargs: Dict[str, Any] = {
        "added_cond_kwargs": {
            "text_embeds": torch.randn(batch, SDXL_POOLED_DIM, **kw),
            "time_ids": torch.randn(batch, SDXL_TIME_IDS, **kw),
        },
        "return_dict": False,
    }
    kwargs.update(lifted_lora_kwargs(module, spec))
    return args, kwargs


def lifted_lora_kwargs(module: Any, spec: "ExportSpec") -> Dict[str, Any]:
    """The mandatory ``lora_a``/``lora_b`` call kwargs for a LoRA-bucket mint.

    #725 option 2: the adapter rides as a flat 1-D pair with static per-layer
    offsets, passed as CALL ARGUMENTS. The pair must be present at trace time —
    tracing without it traces the branchless graph and bakes the absence, which
    is the "missing FQN is the same bug in a different hat" case — and it must be
    NON-ZERO, because a zeroed B lets constant folding erase the branch and
    leaves the gate passing on nothing.

    Empty dict for a bucket-0 mint: the branchless pipeline is its own graph
    class and carries no adapter arguments at all.
    """
    if not int(spec.lora_bucket or 0):
        return {}
    import torch

    from .models import lora_lifted

    plan = lora_lifted.build_plan(module, int(spec.lora_bucket))
    lora_a, lora_b = plan.alloc(_module_dtype_device(module)[1])
    return {
        "lora_a": torch.randn_like(lora_a),
        "lora_b": torch.randn_like(lora_b),
    }


def _has_cfg(spec: "ExportSpec") -> bool:
    """Whether this cell's graph class is CFG-batched (ie#345).

    A CFG variant runs batch-2 graphs and a distilled variant batch-1; one
    class per artifact, never both. ``guidance_scales`` is the declaration:
    any scale above 1 means the batch is doubled.
    """
    return any(float(g) > 1.0 for g in spec.guidance_scales) \
        or not spec.guidance_scales


def _module_dtype_device(module: Any) -> Tuple[Any, Any]:
    import torch

    for source in (module.parameters, module.buffers):
        try:
            first = next(iter(source()))
        except (StopIteration, AttributeError, TypeError):
            continue
        return first.dtype, first.device
    return torch.bfloat16, torch.device("meta")


def latent_dims(
    spec: "ExportSpec",
    *,
    input_name: str = "sample",
    vae_scale: int = 8,
    downsamples: int = 8,
) -> Tuple["DynamicDim", ...]:
    """Declared symbolic latent H/W dims spanning ``spec.shapes``.

    Derived from the declared aspect set rather than hand-written bounds, so
    "these are the aspect ratios we serve" is stated once and the admissible
    range cannot drift from it. ``downsamples`` is the divisibility the UNet
    itself requires (SDXL downsamples 3x => multiples of 8 latent units); a
    range not expressible as that multiple is refused at
    ``aot_mint.dynamic_shapes_spec`` rather than silently 0/1-specialized.

    Bounds are ROUNDED OUT to the multiple, never in: rounding in would
    exclude a declared aspect ratio from the artifact's own contract, and B2
    measured that nothing at runtime would tell us.
    """
    from .aot_mint import DynamicDim

    if not spec.shapes:
        raise InputContractError(
            "cannot derive latent dims without declared shape rows")
    widths = [int(row[0]) // vae_scale for row in spec.shapes]
    heights = [int(row[1]) // vae_scale for row in spec.shapes]
    both = widths + heights
    low = (min(both) // downsamples) * downsamples
    high = -(-max(both) // downsamples) * downsamples
    if low < downsamples:
        low = downsamples
    return (
        DynamicDim(input_name, 2, low, high, multiple_of=downsamples),
        DynamicDim(input_name, 3, low, high, multiple_of=downsamples),
    )


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def compose(
    model: str, spec: "ExportSpec", request: Mapping[str, Any],
) -> Tuple[Any, Callable[[Any], Tuple[Tuple[Any, ...], Dict[str, Any]]]]:
    """Compose the pipeline for a mint; return it with its input builder.

    The sequence mirrors ``compile_cache.build`` exactly (see the module
    docstring for why that is a requirement rather than a convention). The
    returned builder takes the RESOLVED target module — not the pipeline — so
    widths and dtypes come off the thing actually being exported.
    """
    from diffusers import DiffusionPipeline

    builder = builder_for(spec.family)
    load_cls: Any = DiffusionPipeline
    pipeline_class = str(request.get("pipeline_class") or "").strip()
    if pipeline_class:
        # gw#586 call-path parity: trace through the SERVING pipeline class.
        load_cls = resolve_pipeline_class(pipeline_class)
    pipe = load_from_pretrained(
        load_cls, str(model),
        dtype=str(request.get("dtype") or "bf16"),
        storage_dtype=str(request.get("storage_dtype") or ""),
        declared_vram_gb=float(request.get("declared_vram_gb") or 0.0),
    )
    place_pipeline(pipe)
    if int(spec.lora_bucket or 0):
        # #725 option 2, NOT compile_cache.apply_lora_lane: the dynamo lane
        # installs zeroed branch BUFFERS (module state, which export would bake),
        # while the exported lane needs the adapter lifted to call arguments.
        # Same bucket, deliberately different installation.
        lora_lifted.install_lifted_lora_lanes(pipe, int(spec.lora_bucket))
    if callable(getattr(pipe, "set_progress_bar_config", None)):
        pipe.set_progress_bar_config(disable=True)
    return pipe, lambda module: builder(module, spec)


__all__ = [
    "InputContractError",
    "SDXL_POOLED_DIM",
    "SDXL_TIME_IDS",
    "builder_for",
    "compose",
    "inputs_for",
    "latent_dims",
    "sdxl_unet_inputs",
]
