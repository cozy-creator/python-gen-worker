"""from-scratch — example ``@endpoint(kind="conversion")`` that generates
random weights and publishes them as an orphan checkpoint (no parent lineage).

Tenant code never touches tensorhub's upload contract: it writes files
locally, yields ``ProducedFlavor``s, and ``cozy_convert.publish_flavors``
turns each into one Tensorhub commit.
"""

from __future__ import annotations

from typing import Iterator

import msgspec

from cozy_convert import ProducedFlavor
from gen_worker import ConversionContext, endpoint


class FromScratchInput(msgspec.Struct, forbid_unknown_fields=True):
    destination_repo: str
    seed: int = 42
    hidden_dim: int = 64


@endpoint(kind="conversion")
class FromScratch:
    """Generate random weights and emit them as an orphan checkpoint."""

    def generate(
        self,
        ctx: ConversionContext,
        payload: FromScratchInput,
    ) -> Iterator[ProducedFlavor]:
        import torch
        from safetensors.torch import save_file

        torch.manual_seed(payload.seed)
        weights = {
            "linear.weight": torch.randn(
                payload.hidden_dim, payload.hidden_dim, dtype=torch.float32,
            ),
            "linear.bias": torch.zeros(payload.hidden_dim, dtype=torch.float32),
        }
        weights_path = ctx.mktemp() / "weights.safetensors"
        save_file(weights, str(weights_path))
        yield ProducedFlavor(path=weights_path, flavor="fp32")


__all__ = ["FromScratchInput", "FromScratch"]
