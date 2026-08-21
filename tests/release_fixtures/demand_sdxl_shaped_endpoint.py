"""sdxl's DECLARATION shape exactly — the demand half (pgw#1600 acceptance (b)).

Every line that matters here is copied from the real endpoint
(`serverless-endpoints/sdxl/src/sdxl/main.py`), not invented:

* the payload dimensions itself through an ASPECT-RATIO ENUM over a bucket
  table, with no `width` / `height` field anywhere — which is precisely the
  case the platform's field-name vocabulary cannot read, and the reason
  `Shape(pixels=...)` exists;
* `_BUCKETS` is the endpoint's OWN table, passed to the annotation BY
  REFERENCE. There is no second spelling of the geometry to drift;
* the lane is a real ratified `(topology, quant)` pair;
* the demand formula carries tcg#80's measured basis on the term that one
  measurement can identify, and an explicitly UNCALIBRATED prior on the term
  it cannot.

The tree the derive traces against is the tiny SD15-class one, because this
repository ships no sdxl checkpoint and a mint on this box is banned. That is a
fixture detail and it is stated rather than hidden: what is under test is the
DECLARATION → DOCUMENT path, which does not read a weight.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any

import msgspec
import torch
from diffusers import StableDiffusionPipeline

from gen_worker import STATIC, LoadContext, Model, RequestContext, entrypoint, lane
from gen_worker.demand import Basis, MiB, Shape, const, per_mp_batch
from gen_worker.models import SDXL

SDXL_DIFFUSERS_BF16 = ("sdxl.diffusers@1", "plain.bf16@1")


class AspectRatio(StrEnum):
    RATIO_21_9 = "21:9"
    RATIO_16_9 = "16:9"
    RATIO_1_1 = "1:1"
    RATIO_9_16 = "9:16"
    RATIO_9_21 = "9:21"


#: SDXL's trained buckets (arXiv:2307.01952 Appendix I), ~1MP each.
_BUCKETS: dict[AspectRatio, tuple[int, int]] = {
    AspectRatio.RATIO_21_9: (1536, 640),
    AspectRatio.RATIO_16_9: (1344, 768),
    AspectRatio.RATIO_1_1: (1024, 1024),
    AspectRatio.RATIO_9_16: (768, 1344),
    AspectRatio.RATIO_9_21: (640, 1536),
}


class In(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str
    #: THE WHOLE POINT. Nothing about this field's NAME or TYPE says
    #: "1024x1024" — an aspect ratio is not a size. The annotation hands the
    #: platform the endpoint's own table, so the envelope is derived from the
    #: buckets the handler actually indexes.
    aspect_ratio: Annotated[AspectRatio, Shape(pixels=_BUCKETS)] = (
        AspectRatio.RATIO_1_1
    )


class Out(msgspec.Struct):
    model_used: str


class DemandSdxlShaped(
    Model[SDXL],
    lanes={SDXL_DIFFUSERS_BF16: lane(
        # ---- MEASURED, n=1, and it identifies exactly ONE term -------------
        # tcg#80's sm_89 acceptance run (2026-08-21), COLD DAEMON: the compiled
        # UNet allocated 4907 MiB inside the torch allocator at denoise (ONE
        # constant set, flat across 84 steps) and 1155 MiB OUTSIDE it — the
        # CUDA context plus cuDNN/cuBLAS workspaces — for a driver total of
        # 6649 MiB at denoise. The whole-request driver peak was 7792 MiB and
        # is owned by VAE DECODE, not by denoise.
        #
        # The out-of-allocator 1155 MiB is shape-INDEPENDENT by construction: a
        # CUDA context and a workspace pool are per-process, not per-pixel. So
        # this single point identifies a CONST term and nothing else, and it is
        # declared as measured with the run named.
        request=const(
            MiB(1155), basis=Basis.MEASURED,
            source="tcg#80 sm_89 acceptance run 2026-08-21, cold daemon, n=1: "
                   "out-of-allocator 1155 MiB (driver 6649 at denoise, "
                   "allocated 4907 MiB, whole-request peak 7792 MiB at VAE "
                   "decode). NEVER from a death trace (pgw#1601).",
        )
        # ---- UNCALIBRATED, and it says so ----------------------------------
        # ONE measurement cannot separate an intercept from a slope. 220
        # MiB/mp-batch is pgw#1577's bracket over the 519 MiB (model_offload)
        # and 1847 MiB (partial_resident) activation peaks at 1024x1024 with
        # the CFG pair — a DECLARED PRIOR, carried forward as one. pgw#1586
        # fits it from banked samples; pgw#1600's demand_miss falsifies it.
        + per_mp_batch(MiB(220)),
        resident=("vae",),
    )},
    shapes={"aspect": STATIC},
):
    pipe: Any

    def load(self, ctx: LoadContext[SDXL]) -> None:
        self.pipe = ctx.load(StableDiffusionPipeline)
        self.pipe.unet = ctx.compile(self.pipe.unet)


@entrypoint
def generate(ctx: RequestContext, payload: In, model: DemandSdxlShaped) -> Out:
    ctx.raise_if_cancelled()
    width, height = _BUCKETS[payload.aspect_ratio]
    with torch.inference_mode():
        model.pipe(
            prompt=payload.prompt,
            num_inference_steps=2,
            guidance_scale=7.5,
            width=min(64, width),
            height=min(64, height),
            callback_on_step_end=ctx.step_callback(2),
            output_type="latent",
        )
    return Out(model_used=ctx.checkpoint_ref)
