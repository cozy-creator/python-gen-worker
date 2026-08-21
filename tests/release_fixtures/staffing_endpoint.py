"""minimax-h3's STAFFING ENVELOPE, v2-spelled (se#755 / pgw#1396).

The five facts H3's retired v1 ``Resources(...)`` carried and the v2 surface
had no syntax for: a vCPU floor, a host-RAM RECOMMENDATION, a GPU-count
ceiling, an execution-group width, and the sequence-parallel capability.

Three entrypoints, deliberately different:

* ``generate`` — the full envelope, model-bearing;
* ``analyze`` — WEIGHTLESS with ``vcpus=4``, the ``dj-utils`` /
  ``music-analysis`` shape that se#755 blocks;
* ``control`` — model-bearing with NO ``resources=``, which must emit exactly
  what it emitted before pgw#1396.
"""

from __future__ import annotations

from typing import Any

import msgspec

from gen_worker import Model, RequestContext, Resources, entrypoint, lane
from gen_worker.demand import GiB, MiB, const, per_frame, per_frame_squared
from gen_worker.models.tensor_layout_contract import LayoutRequirements


class _Lane:
    """A tensorfs layout contract stand-in: a stamp and a load dtype."""

    def __init__(self, stamp: str) -> None:
        self.stamp = stamp

    @property
    def dtype(self) -> str:
        return "bfloat16"


H3_LANE = _Lane("minimax.h3-dit-diffusers@1")


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
    # coefficients; the sm floor is still derived from the contract dtype.
    lanes={H3_LANE: lane(
        request=const(GiB(1)) + per_frame(MiB(8)) + per_frame_squared(MiB(1)),
    )},
    # No `shapes=`: `load` marks no compile target.
):
    def load(self, ctx: Any) -> None:
        self.pipe = ctx.load(H3Pipeline)


#: Hoisted so the three DiT entrypoints share one declaration rather than
#: three copies that can drift.
H3_STAFFING = Resources(
    # The CPU floor. H3's serve path is host-heavy by construction: the
    # residency sequencer parks and lands components between stages.
    vcpus=16,
    # The SHAPE the hub may staff — a ceiling, not a floor.
    max_gpu_count=4,
    max_gpus_per_execution_group=4,
    # The CAPABILITY: the vendored DiT declares `_cp_plan` and
    # `cp_attention.py` carries the exact-arm attention.
    parallel=("sequence",),
    # Host RAM is a RECOMMENDATION and can never be a minimum (Paul,
    # 2026-07-11: a RunPod GPU pod can neither select nor guarantee it).
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
