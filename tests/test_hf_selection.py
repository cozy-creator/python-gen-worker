from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from gen_worker.hf_selection import HFSelectionPolicy, finalize_diffusers_download, plan_diffusers_download


def _load_fixture(name: str) -> dict[str, Any]:
    p = Path(__file__).parent / "fixtures" / "hf_selection" / name
    return json.loads(p.read_text("utf-8"))


@pytest.mark.parametrize(
    "fixture_name",
    [
        "sd15_minimal_fp16.json",
        "sd15_sharded_unet_fp16.json",
        "zimage_minimal_fp16.json",
    ],
)
def test_hf_selection_fixtures(fixture_name: str) -> None:
    fx = _load_fixture(fixture_name)

    policy = HFSelectionPolicy(**fx["policy"])
    repo_files = list(fx["repo_files"])
    plan = plan_diffusers_download(
        model_index=fx["model_index"],
        repo_files=repo_files,
        policy=policy,
        weight_index_json_by_file=fx["weight_index_json_by_file"],
    )
    selected = finalize_diffusers_download(
        plan=plan,
        repo_files=repo_files,
        weight_index_json_by_file=fx["weight_index_json_by_file"],
    )

    assert selected == set(fx["expected_files"])


def test_hf_selection_prefers_smaller_candidate_when_dtype_equal() -> None:
    model_index = {
        "_class_name": "XPipeline",
        "transformer": ["diffusers", "XTransformer2DModel"],
    }
    repo_files = [
        "model_index.json",
        "transformer/config.json",
        "transformer/small.safetensors",
        "transformer/large.safetensors",
    ]
    policy = HFSelectionPolicy(weight_precisions=("fp16", "bf16"))
    sizes = {
        "transformer/small.safetensors": 10,
        "transformer/large.safetensors": 1000,
        "transformer/config.json": 1,
    }
    dtypes = {"transformer/small.safetensors": {"F16"}, "transformer/large.safetensors": {"F16"}}

    plan = plan_diffusers_download(
        model_index=model_index,
        repo_files=repo_files,
        policy=policy,
        repo_file_sizes=sizes,
        probe_safetensors_dtypes=lambda p: dtypes.get(p),
    )
    assert "transformer/small.safetensors" in plan.selected_files
    assert "transformer/large.safetensors" not in plan.selected_files


def test_hf_selection_prefers_smaller_sharded_candidate_when_dtype_equal() -> None:
    model_index = {
        "_class_name": "XPipeline",
        "transformer": ["diffusers", "XTransformer2DModel"],
    }
    repo_files = [
        "model_index.json",
        "transformer/config.json",
        "transformer/a.safetensors.index.json",
        "transformer/a-00001-of-00002.safetensors",
        "transformer/a-00002-of-00002.safetensors",
        "transformer/b.safetensors.index.json",
        "transformer/b-00001-of-00002.safetensors",
        "transformer/b-00002-of-00002.safetensors",
    ]
    policy = HFSelectionPolicy(weight_precisions=("fp16", "bf16"))
    sizes = {
        "transformer/a.safetensors.index.json": 10,
        "transformer/a-00001-of-00002.safetensors": 100,
        "transformer/a-00002-of-00002.safetensors": 100,
        "transformer/b.safetensors.index.json": 10,
        "transformer/b-00001-of-00002.safetensors": 200,
        "transformer/b-00002-of-00002.safetensors": 200,
    }
    idx_json = {
        "transformer/a.safetensors.index.json": {
            "weight_map": {"x": "a-00001-of-00002.safetensors", "y": "a-00002-of-00002.safetensors"}
        },
        "transformer/b.safetensors.index.json": {
            "weight_map": {"x": "b-00001-of-00002.safetensors", "y": "b-00002-of-00002.safetensors"}
        },
    }
    dtypes = {
        "transformer/a-00001-of-00002.safetensors": {"F16"},
        "transformer/b-00001-of-00002.safetensors": {"F16"},
    }

    plan = plan_diffusers_download(
        model_index=model_index,
        repo_files=repo_files,
        policy=policy,
        weight_index_json_by_file=idx_json,
        repo_file_sizes=sizes,
        probe_safetensors_dtypes=lambda p: dtypes.get(p),
    )
    assert "transformer/a.safetensors.index.json" in plan.selected_files
    assert "transformer/a-00001-of-00002.safetensors" in plan.selected_files
    assert "transformer/a-00002-of-00002.safetensors" in plan.selected_files
    assert "transformer/b.safetensors.index.json" not in plan.selected_files


def test_hf_selection_ignores_large_export_formats_by_default() -> None:
    model_index = {
        "_class_name": "XPipeline",
        "unet": ["diffusers", "UNet2DConditionModel"],
    }
    repo_files = [
        "model_index.json",
        "unet/config.json",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "unet/model.onnx_data",
        "unet/diffusion_flax_model.msgpack",
        "unet/openvino_model.xml",
    ]
    policy = HFSelectionPolicy(weight_precisions=("fp16", "bf16"))

    plan = plan_diffusers_download(
        model_index=model_index,
        repo_files=repo_files,
        policy=policy,
        repo_file_sizes={
            "unet/diffusion_pytorch_model.fp16.safetensors": 123,
            "unet/model.onnx_data": 999999999,
            "unet/diffusion_flax_model.msgpack": 999999999,
        },
        probe_safetensors_dtypes=lambda p: {"F16"} if p.endswith(".safetensors") else None,
    )
    assert "unet/diffusion_pytorch_model.fp16.safetensors" in plan.selected_files
    assert "unet/config.json" in plan.selected_files
    assert "unet/model.onnx_data" not in plan.selected_files
    assert "unet/diffusion_flax_model.msgpack" not in plan.selected_files
    assert "unet/openvino_model.xml" not in plan.selected_files
