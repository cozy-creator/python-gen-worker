from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

import pytest

_MODULE = Path(__file__).resolve().parents[1] / "benchmarks" / "pgw1548_pod_preflight.py"
_spec = importlib.util.spec_from_file_location("pgw1548_pod_preflight", _MODULE)
assert _spec and _spec.loader
preflight = importlib.util.module_from_spec(_spec)
sys.modules["pgw1548_pod_preflight"] = preflight
_spec.loader.exec_module(preflight)

EXIT_NONCONFORMING = preflight.EXIT_NONCONFORMING
Nonconforming = preflight.Nonconforming
assert_conforms = preflight.assert_conforms
main = preflight.main
survey = preflight.survey


def _tree(root: Path, components: dict[str, tuple[str, list[str]]]) -> Path:

    for component, (dtype, keys) in components.items():
        directory = root / component
        directory.mkdir(parents=True, exist_ok=True)
        header = {
            key: {"dtype": dtype, "shape": [2, 2], "data_offsets": [0, 8]}
            for key in keys
        }
        body = json.dumps(header).encode()
        with (directory / "model.safetensors").open("wb") as handle:
            handle.write(struct.pack("<Q", len(body)))
            handle.write(body)
            handle.write(b"\0" * 8)
    return root


SPLIT = [
    f"down_blocks.1.attentions.0.transformer_blocks.{i}.attn1.to_{axis}.weight"
    for i in range(2)
    for axis in ("q", "k", "v")
]
FUSED = [f"transformer_blocks.{i}.attn.qkv_proj.weight" for i in range(3)]


def test_a_conforming_tree_passes(tmp_path: Path) -> None:
    tree = _tree(tmp_path, {"unet": ("BF16", SPLIT), "vae": ("BF16", ["conv.weight"])})
    lines = assert_conforms(survey(tree), "sdxl.diffusers@1+plain.bf16@1", "bfloat16")
    assert any("split=6 fused=0" in line for line in lines)


def test_an_fp16_tree_under_a_bf16_lane_is_REFUSED(tmp_path: Path) -> None:

    tree = _tree(tmp_path, {"unet": ("F16", SPLIT)})
    with pytest.raises(Nonconforming, match="declares bfloat16"):
        assert_conforms(survey(tree), "sdxl.diffusers@1+plain.bf16@1", "bfloat16")


def test_integer_containers_do_not_count_as_a_dtype_violation(tmp_path: Path) -> None:
    """A real text encoder ships I64 position ids beside its BF16 weights."""

    tree = _tree(tmp_path, {"unet": ("BF16", SPLIT)})
    directory = tree / "text_encoder"
    directory.mkdir()
    header = {
        "embeddings.position_ids": {"dtype": "I64", "shape": [1, 77], "data_offsets": [0, 8]},
        "encoder.layer.0.attn.to_q.weight": {"dtype": "BF16", "shape": [2, 2], "data_offsets": [8, 16]},
    }
    body = json.dumps(header).encode()
    with (directory / "model.safetensors").open("wb") as handle:
        handle.write(struct.pack("<Q", len(body)))
        handle.write(body)
        handle.write(b"\0" * 16)
    assert_conforms(survey(tree), "sdxl.diffusers@1+plain.bf16@1", "bfloat16")


def test_a_FUSED_qkv_tree_is_REFUSED_even_at_the_right_dtype(tmp_path: Path) -> None:

    tree = _tree(tmp_path, {"unet": ("BF16", FUSED)})
    with pytest.raises(Nonconforming, match="FUSED attention key"):
        assert_conforms(survey(tree), "sdxl.diffusers@1+plain.bf16@1", "bfloat16")


def test_a_tree_carrying_NEITHER_convention_is_REFUSED_not_passed(tmp_path: Path) -> None:
    """The silence arm: nothing found is not the same as nothing wrong."""

    tree = _tree(tmp_path, {"unet": ("BF16", ["some.mlp.fc1.weight"])})
    with pytest.raises(Nonconforming, match="ZERO split attention keys"):
        assert_conforms(survey(tree), "sdxl.diffusers@1+plain.bf16@1", "bfloat16")


def test_a_projection_STUB_is_refused_by_name(tmp_path: Path) -> None:
    """A 128-byte TFSSTUB pointer reads its first 8 bytes as a huge length."""

    directory = tmp_path / "unet"
    directory.mkdir(parents=True)
    (directory / "model.safetensors").write_bytes(b"TFSSTUB1" + b"\0" * 120)
    with pytest.raises(Nonconforming, match="not credible"):
        survey(tmp_path)


def test_an_empty_tree_is_refused(tmp_path: Path) -> None:
    with pytest.raises(Nonconforming, match="no safetensors containers"):
        survey(tmp_path)


def test_an_unknown_lane_dtype_REFUSES_rather_than_guessing(tmp_path: Path) -> None:
    tree = _tree(tmp_path, {"unet": ("BF16", SPLIT)})
    with pytest.raises(Nonconforming, match="never guess"):
        assert_conforms(survey(tree), "x.y@1", "float4_exotic")


def test_the_cli_exits_91_on_a_nonconforming_tree(tmp_path: Path) -> None:
    """The exit code is the contract a rental script reads."""

    tree = _tree(tmp_path, {"unet": ("F16", SPLIT)})
    code = main([
        "--skip-fleet-line", "--tree", str(tree), "--lane", "sdxl.diffusers@1+plain.bf16@1",
    ])
    assert code == EXIT_NONCONFORMING


def test_the_cli_exits_0_on_a_conforming_tree(tmp_path: Path) -> None:
    tree = _tree(tmp_path, {"unet": ("BF16", SPLIT)})
    assert main([
        "--skip-fleet-line", "--tree", str(tree), "--lane", "sdxl.diffusers@1+plain.bf16@1",
    ]) == 0
