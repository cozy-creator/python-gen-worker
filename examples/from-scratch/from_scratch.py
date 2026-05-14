"""from-scratch — example @conversion endpoint that generates random weights.

Class-shape #322/#332. Demonstrates the orphan-checkpoint pattern (issue
#20): tenant code emits a new checkpoint with no parent. The lineage
list is empty; the finalize handler accepts it; the checkpoint lands as
a root node in the DAG.

Tenant code never touches tensorhub's upload contract. It just:
  1. Generates weights
  2. Yields ProducedFlavors via the class-shape contract

The library handles the session lifecycle + upload + finalize via
``gen_worker.conversion.dispatch._finalize_produced_variants``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import msgspec

from gen_worker.api.decorators import conversion
from gen_worker.conversion import ConversionContext, ProducedFlavor


class FromScratchInput(msgspec.Struct, forbid_unknown_fields=True):
    destination_repo: str
    seed: int = 42
    hidden_dim: int = 64


@conversion(sub_kind="from-scratch")
class FromScratch:
    """Generate random weights and emit as an orphan checkpoint.

    No source repo, no parent lineage. The ``sub_kind='from-scratch'``
    declaration tells the library this produces checkpoints that
    genuinely have no upstream; the lineage array will be empty; the
    finalize handler accepts empty lineage as the root case.
    """

    def setup(self) -> None:
        pass

    @conversion.function(name="generate")
    def generate(
        self,
        ctx: ConversionContext,
        payload: FromScratchInput,
    ) -> Iterator[ProducedFlavor]:
        try:
            import torch  # noqa: WPS433
        except ImportError:
            raise RuntimeError("torch required for random-init example")

        torch.manual_seed(payload.seed)
        weights = {
            "linear.weight": torch.randn(
                payload.hidden_dim, payload.hidden_dim, dtype=torch.float32,
            ),
            "linear.bias": torch.zeros(payload.hidden_dim, dtype=torch.float32),
        }

        tmpdir = ctx.mktemp()
        weights_path = tmpdir / "weights.safetensors"
        try:
            from safetensors.torch import save_file  # noqa: WPS433
        except ImportError:
            raise RuntimeError("safetensors required for random-init example")
        save_file(weights, str(weights_path))

        # Attribute provenance (dtype/file_layout/file_type/kind) is no
        # longer stamped on produced checkpoints — server-side inference
        # reads the uploaded bytes for these.
        yield ProducedFlavor(path=weights_path, flavor="fp32")

    def shutdown(self) -> None:
        pass


__all__ = ["FromScratchInput", "FromScratch"]
