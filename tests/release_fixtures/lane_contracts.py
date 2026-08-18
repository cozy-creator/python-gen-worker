"""tensorfs#111 contract-object stand-ins for the release fixtures.

A lane IS a tensorfs layout contract — an imported OBJECT carrying the
handle and load dtype; never a bare string (Paul's contract-objects ruling;
the Model class header refuses dtype-less lanes at declaration). Until the
tensorfs surface ships, torchcg's resolved ``LaneRef`` is that shape.
"""

from __future__ import annotations

import torch
from torchcg import LaneRef

TINY_DIFFUSERS_FP32 = LaneRef("tiny.diffusers-fp32@1", dtype=torch.float32)
