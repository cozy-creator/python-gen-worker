"""tensorfs#111 contract-object stand-ins (the fixture's ``tensorfs.contracts``).

The contract file imports ``from tensorfs import contracts``; until that
surface ships, a lane is anything satisfying the SDK's ``LaneContract``
Protocol (dtype + a string handle) — the vendored torchcg ``LaneRef`` is the
interim resolved form the adopt path already speaks.

dtype is float32 here because the fixture serves on CPU with fake weights.
"""

from __future__ import annotations

import torch

from gen_worker._vendor.torchcg import LaneRef

SDXL_DIFFUSERS_BF16 = LaneRef("sdxl.diffusers-bf16@1", dtype=torch.float32)
COZY_SDXL_FP8_ROWWISE = LaneRef("cozy.sdxl-fp8-rowwise@1", dtype=torch.float32)
