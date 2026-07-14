"""Block-window weight offload — degraded-mode rung 2 (ie#468).

The gw#460 windows in reverse: per-block weights rest in host RAM and stream
to the execution device only for that block's forward. CPU-safe: hook order,
rebind correctness, and fp8-window composition need no GPU
(``device="cpu"`` exercises the full rebind path).
"""

from __future__ import annotations

from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from gen_worker.models.loading import (  # noqa: E402
    apply_block_window_offload,
    apply_fp8_storage,
    block_offload_active,
)


def _tiny_t5() -> Any:
    from transformers import T5Config, T5EncoderModel

    cfg = T5Config(
        vocab_size=256, d_model=64, d_kv=16, d_ff=128,
        num_layers=2, num_heads=4,
    )
    return T5EncoderModel(cfg).to(torch.bfloat16).eval()


def _forward(model: Any, ids: Any) -> Any:
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids))
    return out.last_hidden_state.float()


def test_offload_preserves_outputs_and_rebinds():
    model = _tiny_t5()
    ids = torch.randint(0, 256, (1, 8))
    ref = _forward(model, ids)

    assert apply_block_window_offload(model, device="cpu") is True
    assert block_offload_active(model)

    parked = [
        p for m in model.modules() if isinstance(m, torch.nn.Linear)
        for p in m.parameters(recurse=False)
    ]
    host_ptrs = {p.data_ptr() for p in parked}

    out = _forward(model, ids)
    assert torch.allclose(ref, out, atol=1e-3)
    # post-hook rebound every window back to its host copy
    assert {p.data_ptr() for p in parked} == host_ptrs
    # second forward (hooks re-fire cleanly)
    out2 = _forward(model, ids)
    assert torch.allclose(out, out2)


def test_idempotent():
    model = _tiny_t5()
    assert apply_block_window_offload(model, device="cpu") is True
    parked = {p.data_ptr() for p in model.parameters()}
    assert apply_block_window_offload(model, device="cpu") is True  # no-op
    assert {p.data_ptr() for p in model.parameters()} == parked


def test_composes_with_fp8_windows():
    """fp8 storage windows (gw#460) + offload windows on the same blocks:
    fp8 bytes at rest in host RAM, upcast happens inside the forward, and the
    post-hooks land the params back on the host fp8 copies."""
    model = _tiny_t5()
    ids = torch.randint(0, 256, (1, 8))
    assert apply_fp8_storage(model) is True
    ref = _forward(model, ids)  # fp8-window baseline

    assert apply_block_window_offload(model, device="cpu") is True
    out = _forward(model, ids)
    assert torch.allclose(ref, out, atol=1e-3)
    fp8 = torch.float8_e4m3fn
    lin = [
        p for m in model.modules() if isinstance(m, torch.nn.Linear)
        for p in m.parameters(recurse=False)
    ]
    assert lin and all(p.dtype == fp8 for p in lin)
    assert all(p.device.type == "cpu" for p in lin)


def test_pipeline_component_targeting():
    class Pipe:
        def __init__(self) -> None:
            self.transformer = _tiny_t5()
            self.vae = torch.nn.Linear(4, 4)

    pipe = Pipe()
    assert apply_block_window_offload(pipe, device="cpu") is True
    assert block_offload_active(pipe)
    assert not getattr(pipe.vae, "_cozy_block_offload_applied", False)
