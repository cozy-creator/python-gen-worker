"""Integration tests: real tiny HF models, one per conversion direction.

Network required (huggingface.co). Each test stays under ~a minute — the
models are hf-internal-testing fixtures of a few MB.

Directions covered:
  - HF ingest (classify + selective snapshot_download), transformers class
  - HF ingest, diffusers class
  - dtype cast (fp32 -> fp16) via the streaming writer
  - full flavor tree build on a diffusers pipe (per-component cast + passthrough)
  - repackage diffusers -> singlefile (SDXL key mapping)
  - quant nf4 (skipped unless bitsandbytes installed)
  - gguf (skipped unless the llama.cpp toolchain is on PATH)
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import pytest
torch = pytest.importorskip("torch")
from safetensors.torch import load_file

from gen_worker.convert.clone import OutputSpec, build_flavor_tree
from gen_worker.convert.ingest import ingest_huggingface
from gen_worker.convert.writer import streaming_dtype_cast

_TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"
_TINY_SDXL = "hf-internal-testing/tiny-sdxl-pipe"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def tiny_llama(tmp_path_factory) -> object:
    return ingest_huggingface(_TINY_LLAMA, tmp_path_factory.mktemp("llama"))


@pytest.fixture(scope="module")
def tiny_sdxl(tmp_path_factory) -> object:
    return ingest_huggingface(_TINY_SDXL, tmp_path_factory.mktemp("sdxl"))


def test_plan_huggingface_pre_download_identity(tmp_path: Path) -> None:
    """th#592 download-skip: the plan derives a complete per-file content
    identity (lfs sha256 / git blob oid) from metadata alone, its bank key is
    stable across calls, and feeding the plan into ingest_huggingface yields
    the same selection the un-planned ingest makes."""
    from gen_worker.convert.bank import flavor_bank_key
    from gen_worker.convert.ingest import ingest_huggingface as ingest, plan_huggingface

    plan = plan_huggingface(_TINY_SDXL)
    files = plan.bank_files()
    assert files, "every selected file must carry a provider content id"
    assert all(cid.startswith(("sha256:", "git:")) for _, _, cid in files)
    assert {p for p, _, _ in files} == set(plan.classification.allow_patterns)

    key1 = flavor_bank_key(plan, "bf16-diffusers-safetensors", layout_hint="diffusers")
    key2 = flavor_bank_key(plan_huggingface(_TINY_SDXL), "bf16-diffusers-safetensors",
                           layout_hint="diffusers")
    assert key1 and key1 == key2, "bank key must be reproducible from metadata"

    src = ingest(_TINY_SDXL, tmp_path / "planned", plan=plan)
    assert src.source_revision == plan.revision
    assert (src.dir / "model_index.json").is_file()


def test_ingest_transformers_selects_safetensors_not_onnx(tiny_llama) -> None:
    src = tiny_llama
    assert src.classification.strategy == "transformers"
    assert src.layout == "singlefile"
    assert (src.dir / "model.safetensors").is_file()
    assert (src.dir / "config.json").is_file()
    assert (src.dir / "tokenizer.json").is_file()
    assert not (src.dir / "onnx").exists(), "onnx must be excluded by the classifier"
    assert src.repo_spec["kind"] == "model"
    assert src.repo_spec["library_name"] == "transformers"
    assert src.source_revision, "resolved commit sha expected"


def test_cast_direction_fp16(tiny_llama, tmp_path: Path) -> None:
    result = streaming_dtype_cast(
        tiny_llama.dir / "model.safetensors", tmp_path / "fp16",
        target_dtype=torch.float16)
    assert result["converted_count"] > 0
    loaded = load_file(str(result["output_paths"][0]))
    assert all(t.dtype == torch.float16 for t in loaded.values() if t.is_floating_point())


def test_ingest_diffusers_pipe(tiny_sdxl) -> None:
    src = tiny_sdxl
    assert src.classification.strategy == "diffusers"
    assert src.layout == "diffusers"
    assert (src.dir / "model_index.json").is_file()
    assert (src.dir / "unet" / "diffusion_pytorch_model.safetensors").is_file()
    assert (src.dir / "tokenizer" / "vocab.json").is_file()
    assert src.model_family == "sdxl"


def test_flavor_tree_cast_diffusers(tiny_sdxl, tmp_path: Path) -> None:
    """End-to-end flavor build: per-component fp16 cast + config passthrough."""
    tree, attrs = build_flavor_tree(
        tiny_sdxl,
        OutputSpec(dtype="fp16", file_layout="diffusers", file_type="safetensors"),
        tmp_path / "flavor",
    )
    assert attrs["dtype"] == "fp16"
    assert (tree / "model_index.json").is_file()
    assert (tree / "scheduler" / "scheduler_config.json").is_file()
    unet = load_file(str(tree / "unet" / "diffusion_pytorch_model.safetensors"))
    assert all(t.dtype == torch.float16 for t in unet.values() if t.is_floating_point())
    te = load_file(str(tree / "text_encoder" / "model.safetensors"))
    assert all(t.dtype == torch.float16 for t in te.values() if t.is_floating_point())


def test_repackage_direction_diffusers_to_singlefile(tiny_sdxl, tmp_path: Path) -> None:
    from gen_worker.convert.repackage import diffusers_to_singlefile

    out = tmp_path / "model.safetensors"
    diffusers_to_singlefile(tiny_sdxl.dir, out, model_family="sdxl")
    assert out.is_file() and out.stat().st_size > 0
    sd = load_file(str(out))
    assert any(k.startswith("model.diffusion_model.") for k in sd), "unet keys expected"
    assert any(k.startswith("first_stage_model.") for k in sd), "vae keys expected"
    assert any(k.startswith("conditioner.") for k in sd), "text encoder keys expected"


@pytest.mark.skipif(
    any(importlib.util.find_spec(m) is None
        for m in ("bitsandbytes", "transformers", "accelerate")),
    reason="bitsandbytes/transformers/accelerate not installed",
)
def test_quant_direction_nf4(tiny_llama, tmp_path: Path) -> None:
    from gen_worker.convert.convert import run_inline_conversion

    result = run_inline_conversion(
        source_path=tiny_llama.dir / "model.safetensors",
        out_dir=tmp_path / "nf4",
        target_dtype="nf4",
        target_file_type="safetensors",
        source_repo_dir=tiny_llama.dir,
    )
    assert result.output_paths
    assert result.attributes.get("quant_library") in {"bitsandbytes", "bnb"}


@pytest.mark.skipif(
    shutil.which("convert_hf_to_gguf.py") is None,
    reason="llama.cpp toolchain (convert_hf_to_gguf.py) not on PATH",
)
def test_gguf_direction(tiny_llama, tmp_path: Path) -> None:
    from gen_worker.convert.convert import run_inline_conversion

    result = run_inline_conversion(
        source_path=tiny_llama.dir / "model.safetensors",
        out_dir=tmp_path / "gguf",
        target_dtype="f16",
        target_file_type="gguf",
        source_repo_dir=tiny_llama.dir,
    )
    assert result.output_paths and result.output_paths[0].suffix == ".gguf"
    assert not (tmp_path / "gguf" / "_gguf_work").exists()


def test_calibrated_dtype_refused(tiny_llama, tmp_path: Path) -> None:
    from gen_worker.convert.convert import InlineConversionNotPossible, run_inline_conversion

    with pytest.raises(InlineConversionNotPossible) as exc:
        run_inline_conversion(
            source_path=tiny_llama.dir / "model.safetensors",
            out_dir=tmp_path / "nvfp4",
            target_dtype="nvfp4",
            source_repo_dir=tiny_llama.dir,
        )
    assert exc.value.deferred_requirement is not None
