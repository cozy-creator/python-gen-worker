"""A JOBS-ONLY package: `@job` functions and not one `@endpoint`.

This is the shape te#218 produced for every conversion package, so a worker
that cannot boot on it cannot run any migrated job. Kept in `harness/` rather
than inline in the test because `Worker` takes MODULE NAMES and walks them.
"""

from __future__ import annotations

import msgspec

from gen_worker import JobContext, Resources, job


class PlanIn(msgspec.Struct):
    rung: str = "w8a8"


class PlanOut(msgspec.Struct):
    rung: str
    vcpus: int


@job(name="plan-h3-svdq", resources=Resources(vcpus=4))
def plan_h3_svdq(ctx: JobContext, payload: PlanIn) -> PlanOut:
    ctx.progress(position=1, total=1, phase="plan")
    return PlanOut(rung=payload.rung, vcpus=4)


@job(name="bake-h3-modulation", resources=Resources(gpu=True), publishes=True)
def bake_h3_modulation(ctx: JobContext, payload: PlanIn) -> PlanOut:
    return PlanOut(rung=payload.rung, vcpus=0)
