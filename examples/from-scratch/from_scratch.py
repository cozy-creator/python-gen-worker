"""from-scratch — example ``@endpoint(kind="conversion")`` that generates
random weights and publishes them as an orphan checkpoint (no parent lineage).

The publish contract: a conversion handler writes files locally, calls
``cozy_convert.publish_flavors(ctx, flavors)`` — one Tensorhub commit per
``ProducedFlavor`` — and returns a result struct. Nothing publishes
implicitly; generator handlers are rejected for producer kinds.
"""

from __future__ import annotations

import msgspec

from cozy_convert import ProducedFlavor, publish_flavors
from gen_worker import ConversionContext, endpoint


class FromScratchInput(msgspec.Struct, forbid_unknown_fields=True):
    destination_repo: str
    seed: int = 42
    hidden_dim: int = 64


class FromScratchResult(msgspec.Struct):
    destination_repo: str
    revision_ids: list[str]
    published_files: int


@endpoint(kind="conversion")
class FromScratch:
    """Generate random weights and publish them as an orphan checkpoint."""

    def generate(
        self,
        ctx: ConversionContext,
        payload: FromScratchInput,
    ) -> FromScratchResult:
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

        commits = publish_flavors(
            ctx,
            [ProducedFlavor(path=weights_path, flavor="fp32")],
            destination_repo=payload.destination_repo,
        )
        return FromScratchResult(
            destination_repo=payload.destination_repo,
            revision_ids=[c.revision_id for c in commits],
            published_files=sum(c.uploaded + c.deduped for c in commits),
        )


__all__ = ["FromScratchInput", "FromScratchResult", "FromScratch"]
