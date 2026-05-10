"""from-scratch — example @training_function that generates random weights.

Demonstrates the orphan-checkpoint pattern (issue #20): tenant code emits
a new checkpoint with no parent. The lineage list is empty; the finalize
handler accepts it; the checkpoint lands as a root node in the DAG.

Tenant code never touches tensorhub's upload contract. It just:
  1. Generates weights
  2. Returns them via the @training_function contract

The library handles the session lifecycle + upload + finalize via
dispatch._finalize_produced_variants.
"""

from __future__ import annotations

from pathlib import Path

import msgspec

from gen_worker.conversion import ConversionContext, ProducedFlavor, training_function


class FromScratchInput(msgspec.Struct, forbid_unknown_fields=True):
    destination_repo: str
    seed: int = 42
    hidden_dim: int = 64


@training_function(kind="from-scratch", concurrency="sequential")
def generate(
    ctx: ConversionContext,
    payload: FromScratchInput,
) -> list[ProducedFlavor]:
    """Generate random weights and emit as an orphan checkpoint.

    No source repo, no parent lineage. The `kind='from-scratch'`
    declaration tells the library this produces checkpoints that
    genuinely have no upstream; the lineage array will be empty; the
    finalize handler accepts empty lineage as the root case.
    """
    try:
        import torch  # noqa: WPS433
    except ImportError:
        raise RuntimeError("torch required for random-init example")

    torch.manual_seed(payload.seed)
    weights = {
        "linear.weight": torch.randn(payload.hidden_dim, payload.hidden_dim, dtype=torch.float32),
        "linear.bias": torch.zeros(payload.hidden_dim, dtype=torch.float32),
    }

    tmpdir = ctx.mktemp()
    weights_path = tmpdir / "weights.safetensors"
    try:
        from safetensors.torch import save_file  # noqa: WPS433
    except ImportError:
        raise RuntimeError("safetensors required for random-init example")
    save_file(weights, str(weights_path))

    return [
        ProducedFlavor(
            path=weights_path,
            flavor="fp32",
            attributes={
                "dtype": "fp32",
                "file_layout": "singlefile",
                "file_type": "safetensors",
                "kind": "model",
            },
        )
    ]
