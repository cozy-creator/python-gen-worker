from __future__ import annotations

from pathlib import Path

import pytest


def test_save_and_load_trainable_module_checkpoint_roundtrip(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    from gen_worker.trainer import load_trainable_module_checkpoint, save_trainable_module_checkpoint

    module = nn.Linear(4, 2, bias=True)
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-3)

    # Warm optimizer state so optimizer checkpointing path is exercised.
    x = torch.randn(3, 4)
    y = torch.randn(3, 2)
    loss = ((module(x) - y) ** 2).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    checkpoint_dir = tmp_path / "step-0001"
    meta = save_trainable_module_checkpoint(
        module=module,
        optimizer=optimizer,
        output_dir=str(checkpoint_dir),
        step=1,
        final=False,
        model_name_or_path="runwayml/stable-diffusion-v1-5",
    )

    assert "primary_path" in meta
    primary_path = Path(str(meta["primary_path"]))
    assert primary_path.exists()
    assert (checkpoint_dir / "checkpoint_meta.json").exists()

    saved_weight = module.weight.detach().clone()
    saved_bias = module.bias.detach().clone()

    # Corrupt params then verify load restores trainable tensors.
    with torch.no_grad():
        module.weight.add_(10.0)
        module.bias.sub_(10.0)

    load_trainable_module_checkpoint(
        module=module,
        optimizer=optimizer,
        checkpoint_dir=str(checkpoint_dir),
    )
    assert torch.allclose(module.weight.detach(), saved_weight)
    assert torch.allclose(module.bias.detach(), saved_bias)
