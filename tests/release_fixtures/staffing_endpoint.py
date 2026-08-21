from __future__ import annotations

from typing import Any

import msgspec

from gen_worker import Model, RequestContext, Resources, entrypoint, lane
from gen_worker.demand import GiB, MiB, const, per_frame, per_frame_squared
from gen_worker.models.tensor_layout_contract import LayoutRequirements


#: H3's packaged diffusers lane, as the v2 stamp PAIR (pgw#1621). The stand-in
#: `_Lane` object that used to sit here — a handle plus a hand-written
#: `dtype = "bfloat16"` — is inexpressible now and was always a second
#: producer of the load dtype: `plain.bf16@1` DECLARES bfloat16 and declares
#: the `capability_floor_sm` of 80 the tests below read, so the fixture types
#: neither. `minimax.h3-dit-diffusers@1` was this pair's v1 display name.
H3_LANE = ("minimax-h3.diffusers@1", "plain.bf16@1")


class MiniMaxH3:
    """The model type stand-in — only its name reaches the manifest."""

    name = "minimax-h3"


class H3Pipeline:
    pass


class GenerateInput(msgspec.Struct, forbid_unknown_fields=True):
    prompt: str


class GenerateOutput(msgspec.Struct):
    frames: int


class AnalyzeInput(msgspec.Struct, forbid_unknown_fields=True):
    samples: list[float]


class AnalyzeOutput(msgspec.Struct):
    bpm: float


class H3Model(
    Model[MiniMaxH3],
    # pgw#1599: the lane's value is its DEMAND FORMULA, not a VRAM string.
    # `"vram78g"` claimed one number for every request H3 would ever serve, and
    # H3 is the case that proves there is no such number — a longer video costs
    # linearly in frames and QUADRATICALLY in the attention term. Fixture-scale
    # coefficients; the sm floor is still derived — from the QUANT RULE now.
    lanes={H3_LANE: lane(
        request=const(GiB(1)) + per_frame(MiB(8)) + per_frame_squared(MiB(1)),
    )},
):
    def load(self, ctx: Any) -> None:
        self.pipe = ctx.load(H3Pipeline)


H3_STAFFING = Resources(
    vcpus=16,
    max_gpu_count=4,
    max_gpus_per_execution_group=4,
    parallel=("sequence",),
    requires=LayoutRequirements(recommended="ram96g"),
)


@entrypoint(resources=H3_STAFFING)
def generate(
    ctx: RequestContext, payload: GenerateInput, video: H3Model
) -> GenerateOutput:
    return GenerateOutput(frames=1)


@entrypoint(resources=Resources(vcpus=4))
def analyze(ctx: RequestContext, payload: AnalyzeInput) -> AnalyzeOutput:
    return AnalyzeOutput(bpm=float(len(payload.samples)))


@entrypoint
def control(
    ctx: RequestContext, payload: GenerateInput, video: H3Model
) -> GenerateOutput:
    return GenerateOutput(frames=1)
