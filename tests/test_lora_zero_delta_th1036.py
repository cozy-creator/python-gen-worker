"""th#1036 zero-delta refusal lives in ``load_adapter_state_dict``.

One implementation of the attach-but-invisible rule: an adapter whose
low-rank product is provably zero is refused at PARSE time, typed, before it
can ride any attach path (executor overlays, BYO per-request loras, endpoint
code). The old endpoint-side copies checked ``any(tensor != 0)`` over the
WHOLE dict — vacuous whenever a (nonzero-by-construction) alpha key exists —
so these rows pin the pair-aware semantics, not just the happy path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pytest

torch = pytest.importorskip("torch")
st = pytest.importorskip("safetensors.torch")

from gen_worker.api.errors import RefCompatibilitySurprise  # noqa: E402
from gen_worker.utils.lora import load_adapter_state_dict  # noqa: E402


def _write(tmp_path: Path, tensors: Dict[str, "torch.Tensor"]) -> Path:
    p = tmp_path / "adapter.safetensors"
    st.save_file(tensors, str(p))
    return p


def test_healthy_kohya_adapter_loads(tmp_path: Path) -> None:
    path = _write(tmp_path, {
        "lora_unet_a.lora_down.weight": torch.randn(4, 8),
        "lora_unet_a.lora_up.weight": torch.randn(8, 4),
    })
    sd = load_adapter_state_dict(path, ref="t/healthy")
    # alpha injection still happens after the guard
    assert "lora_unet_a.alpha" in sd


def test_zero_up_half_is_refused_even_with_nonzero_down_and_alpha(tmp_path: Path) -> None:
    # delta = up @ down == 0 when up is all-zero; the stored alpha (nonzero
    # float) must not vouch for the delta — the exact hole in the deleted
    # endpoint-side any(tensor)!=0 copies.
    path = _write(tmp_path, {
        "lora_unet_a.lora_down.weight": torch.randn(4, 8),
        "lora_unet_a.lora_up.weight": torch.zeros(8, 4),
        "lora_unet_a.alpha": torch.tensor(4.0),
    })
    with pytest.raises(RefCompatibilitySurprise, match="NO visible delta"):
        load_adapter_state_dict(path, ref="t/zero-up")


def test_all_zero_adapter_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, {
        "lora_unet_a.lora_down.weight": torch.zeros(4, 8),
        "lora_unet_a.lora_up.weight": torch.zeros(8, 4),
    })
    with pytest.raises(RefCompatibilitySurprise, match="NO visible delta"):
        load_adapter_state_dict(path, ref="t/zero-both")


def test_one_live_pair_among_dead_ones_is_accepted(tmp_path: Path) -> None:
    path = _write(tmp_path, {
        "lora_unet_a.lora_down.weight": torch.zeros(4, 8),
        "lora_unet_a.lora_up.weight": torch.randn(8, 4),
        "lora_unet_b.lora_down.weight": torch.randn(4, 8),
        "lora_unet_b.lora_up.weight": torch.randn(8, 4),
    })
    load_adapter_state_dict(path, ref="t/one-live")


def test_peft_grammar_zero_B_is_refused(tmp_path: Path) -> None:
    path = _write(tmp_path, {
        "transformer.blocks.0.attn.to_q.lora_A.weight": torch.randn(4, 8),
        "transformer.blocks.0.attn.to_q.lora_B.weight": torch.zeros(8, 4),
    })
    with pytest.raises(RefCompatibilitySurprise, match="NO visible delta"):
        load_adapter_state_dict(path, ref="t/peft-zero-B")


def test_peft_grammar_healthy_loads(tmp_path: Path) -> None:
    path = _write(tmp_path, {
        "transformer.blocks.0.attn.to_q.lora_A.weight": torch.randn(4, 8),
        "transformer.blocks.0.attn.to_q.lora_B.weight": torch.randn(8, 4),
    })
    load_adapter_state_dict(path, ref="t/peft-ok")
