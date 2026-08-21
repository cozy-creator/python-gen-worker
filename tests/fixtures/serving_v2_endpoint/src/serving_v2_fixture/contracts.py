"""The fixture's LANES — real tensor-layout **v2** stamp pairs.

pgw#1621: a lane is the ``(topology, quant)`` pair, both halves ratified
documents in the vendored ``spec/v2`` corpus, and `parse_lane_stamp` refuses
anything else at class-definition time. The v1 stand-ins this module used to
hold — ``LaneRef("sdxl.diffusers-bf16@1", dtype=torch.float32)``, a handle plus
a dtype and no layout — are exactly the shape that is now inexpressible: the
dtype is not the fixture's to choose, it is a field on the ratified quant rule.

That has a consequence this fixture cannot hide and should not: the pair below
declares ``bfloat16``/``float8_e4m3fn`` rather than the fixture's old
``float32``, and both rules state a real ``capability_floor_sm`` (80 and 89), so
this endpoint now derives a real placement row where it used to derive none.
"""

from __future__ import annotations

SDXL_DIFFUSERS_BF16 = ("sdxl.diffusers@1", "plain.bf16@1")
COZY_SDXL_FP8_ROWWISE = ("sdxl.diffusers@1", "cozy.fp8-rowwise@1")

#: The wire renderings, for tests that assert what the header produced.
SDXL_DIFFUSERS_BF16_ID = "sdxl.diffusers@1+plain.bf16@1"
COZY_SDXL_FP8_ROWWISE_ID = "sdxl.diffusers@1+cozy.fp8-rowwise@1"
