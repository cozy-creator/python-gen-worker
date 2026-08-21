"""A dual-expert w8a8 artifact must be served WHOLE."""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Any

import pytest


def _shard(path: Path, layers: tuple[str, ...]) -> None:
    header: dict[str, Any] = {"__metadata__": {"quant_scheme": "fp8-w8a8"}}
    offset = 0
    for layer in layers:
        header[f"{layer}.weight"] = {
            "dtype": "F8_E4M3", "shape": [16, 16],
            "data_offsets": [offset, offset + 256]}
        offset += 256
        header[f"{layer}.weight_scale"] = {
            "dtype": "F32", "shape": [16],
            "data_offsets": [offset, offset + 64]}
        offset += 64
    blob = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(b"\0" * offset)


@pytest.fixture()
def moe_tree(tmp_path: Path) -> Path:
    root = tmp_path / "wan-w8a8"
    root.mkdir()
    (root / "model_index.json").write_text(json.dumps({
        "_class_name": "WanPipeline",
        "transformer": ["diffusers", "WanTransformer3DModel"],
        "transformer_2": ["diffusers", "WanTransformer3DModel"],
    }))
    _shard(root / "transformer" / "model.safetensors",
           ("blocks.0.attn1.to_q", "blocks.0.ffn.net.0.proj"))
    _shard(root / "transformer_2" / "model.safetensors",
           ("blocks.0.attn1.to_q",))
    return root


def test_both_experts_are_detected(moe_tree: Path) -> None:
    from gen_worker.models.w8a8 import (detect_w8a8_artifact,
                                        detect_w8a8_artifacts)

    arts = detect_w8a8_artifacts(moe_tree)
    assert [a.component for a in arts] == ["transformer", "transformer_2"]
    assert len(arts[0].quantized) == 2 and len(arts[1].quantized) == 1
    assert detect_w8a8_artifact(moe_tree) == arts[0]


def test_the_pipeline_gets_every_expert_wired(
    moe_tree: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gen_worker.models import w8a8

    monkeypatch.setattr(w8a8, "w8a8_gemm_mode", lambda: "rowwise")
    monkeypatch.setattr(
        w8a8, "load_w8a8_denoiser",
        lambda root, art, **kw: f"<{art.component} on scaled_mm>")

    seen: dict[str, Any] = {}

    class _Cls:
        @staticmethod
        def from_pretrained(path: str, **kwargs: Any) -> Any:
            seen.update(kwargs)
            return type("P", (), {})()

    arts = w8a8.detect_w8a8_artifacts(moe_tree)
    w8a8.load_w8a8_pipeline(_Cls, moe_tree, arts[0])

    assert seen["transformer"] == "<transformer on scaled_mm>"
    assert seen["transformer_2"] == "<transformer_2 on scaled_mm>"
