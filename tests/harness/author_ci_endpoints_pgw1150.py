"""pgw#1150 harness endpoints: compile-declaring families with a real denoise
stage, at CPU scale.

The author-CI harness measures ``stage_ms.<stage>`` across two arms of ONE
process, where the arms differ only in pgw#1142's serve posture. So the stage
here reads that posture exactly the way ``aot_serve``'s wrapper does — an
ordered-eager call runs the eager body — and the two sleep constants stand in
for the compiled and eager forwards. No torch, no GPU, no weights, no mint:
Paul's standing rule is that a compile runs on a pod, and what this file exists
to exercise is the ORCHESTRATION.

Three families, one property each:

* ``Fast`` — a healthy family: the compiled arm is ~3x, comfortably over the
  1.10 fleet bar. Its ``family`` is pgw#868's probe family so the real numerics
  rig can arm a real compiled graph under it.
* ``Regressed`` — the compiled arm is SLOWER. A speedup below the declared bar
  is a recorded FAILURE, never a proof, and this is the family that proves it.
* ``Blocked`` — two unresolved ``Compile.blockers``. A declared block is a
  LEGAL state (ie#664 §6): the run continues eager-only and the record stays
  ``never-run`` citing the blocker ids.
"""

from __future__ import annotations

import time

import msgspec

from gen_worker import (
    Compile, Dim, GraphClass, Input, MintBlocker, RequestContext, endpoint,
)
from gen_worker import serve_posture

#: The stage the records declare (`[speed] metric = "stage_ms.denoise"`).
STAGE = "denoise"

#: pgw#868's probe family and its declared band, so `test_numerics_gate_pgw868`
#: can arm a REAL compiled graph for this endpoint's declaration rather than a shape that
#: only looks like one.
PROBE_FAMILY = "pgw868-probe"
FLOOR, WARN = 0.995, 0.999

#: Sleeps, not measurements. Far enough apart that a 5-sample median cannot
#: cross the bar on scheduler noise, and small enough that the whole file runs
#: in about a second.
FAST_COMPILED_S = 0.010
FAST_EAGER_S = 0.030
REGRESSED_COMPILED_S = 0.030
REGRESSED_EAGER_S = 0.010

#: Every call records whether the eager-only order stood when it ran, so a test
#: can prove the compiled arm ran WITHOUT it and the eager arm WITH it.
POSTURE_SEEN: list[bool] = []


class GenIn(msgspec.Struct):
    prompt: str = ""


class GenOut(msgspec.Struct):
    ok: bool = True


def _denoise(ctx: RequestContext, compiled_s: float, eager_s: float) -> GenOut:
    """One request, bracketed on the stage the record names.

    Reading ``serve_posture.eager_only()`` here is not a test contrivance: it
    is the same decision ``aot_serve``'s wrapper makes at the call (pgw#1142's
    reversibility seam), which is the whole reason the eager arm can be the
    same process, the same weights and the same pipeline.
    """
    ordered_eager = serve_posture.eager_only()
    POSTURE_SEEN.append(ordered_eager)
    with ctx.stage(STAGE):
        time.sleep(eager_s if ordered_eager else compiled_s)
    return GenOut()


@endpoint(compile=Compile(
    family=PROBE_FAMILY, targets=("denoiser",), shapes=((8, 8), (16, 16)),
    text_len=0, numerics_floor=FLOOR, numerics_warn=WARN,
))
class Fast:
    def fast_generate(self, ctx: RequestContext, data: GenIn) -> GenOut:
        return _denoise(ctx, FAST_COMPILED_S, FAST_EAGER_S)


@endpoint(compile=Compile(
    # The same probe family as `Fast`, so its parity leg is HEALTHY and the
    # only thing this family fails is the SPEED bar. One variable per proof.
    family=PROBE_FAMILY, targets=("denoiser",), shapes=((8, 8), (16, 16)),
    text_len=0, numerics_floor=FLOOR, numerics_warn=WARN,
))
class Regressed:
    def regressed_generate(self, ctx: RequestContext, data: GenIn) -> GenOut:
        return _denoise(ctx, REGRESSED_COMPILED_S, REGRESSED_EAGER_S)


BLOCKER_IDS = ("OQ-1-pgw1150-unmeasured", "OQ-2-pgw1150-rank")

BLOCKED_COMPILE = Compile(
    family="pgw1150-blocked", targets=("denoiser",), shapes=((8, 8),), text_len=0,
    # `blockers` IS export-contract vocabulary, so this declaration registers
    # and the classes-and-family invariant applies to it.
    dims=(Dim("B", carried_by=(("sample", 0),)),),
    classes=(GraphClass(dims={"B": 1}),),
    inputs=(Input("sample", shape=("B", 8, 8), dtype="model"),),
    blockers=(
        MintBlocker(
            id=BLOCKER_IDS[0],
            what="Whole-graph export has never been measured on this lane.",
            evidence="harness fixture (pgw#1150).",
            resolves_when="ONE mint-lane measurement at the largest class.",
        ),
        MintBlocker(
            id=BLOCKER_IDS[1],
            what="The audio timestep is rank-1 on one route and rank-2 on "
                 "another, and the pytree input spec pins rank.",
            evidence="harness fixture (pgw#1150).",
            resolves_when="The endpoint normalizes rank at its boundary.",
        ),
    ),
)


@endpoint(compile=BLOCKED_COMPILE)
class Blocked:
    def blocked_generate(self, ctx: RequestContext, data: GenIn) -> GenOut:
        return _denoise(ctx, FAST_COMPILED_S, FAST_EAGER_S)
