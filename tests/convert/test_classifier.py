"""Unit tests for the small HF repo classifier (gen_worker.convert.classifier)."""

from __future__ import annotations

import pytest

from gen_worker.convert.classifier import RepoRefusal, classify_repo

_DIFFUSERS = [
    "model_index.json",
    "README.md",
    "scheduler/scheduler_config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
    "text_encoder/config.json",
    "text_encoder/model.safetensors",
    "text_encoder/model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.bf16.safetensors",
    "transformer/diffusion_pytorch_model.fp16.safetensors",
    "transformer/diffusion_pytorch_model.safetensors",
    "unet/demo.png",
]


def test_diffusers_one_weight_set_per_component() -> None:
    c = classify_repo(_DIFFUSERS)
    assert c.strategy == "diffusers"
    assert c.runtime_library == "diffusers"
    allow = set(c.allow_patterns)
    # bf16 preferred where present; next preferred tag (fp16) beats untagged;
    # untagged is the fallback when no preferred tag exists.
    assert "transformer/diffusion_pytorch_model.bf16.safetensors" in allow
    assert "transformer/diffusion_pytorch_model.fp16.safetensors" not in allow
    assert "transformer/diffusion_pytorch_model.safetensors" not in allow
    assert "text_encoder/model.fp16.safetensors" in allow
    assert "text_encoder/model.safetensors" not in allow
    assert "vae/diffusion_pytorch_model.safetensors" in allow
    # configs + model_index always ride along; demo media never.
    assert "model_index.json" in allow
    assert "scheduler/scheduler_config.json" in allow
    assert "unet/demo.png" not in allow


def test_diffusers_dtype_preference_fp16() -> None:
    c = classify_repo(_DIFFUSERS, dtype_pref=("fp16",))
    allow = set(c.allow_patterns)
    assert "transformer/diffusion_pytorch_model.fp16.safetensors" in allow
    assert "text_encoder/model.fp16.safetensors" in allow
    assert "transformer/diffusion_pytorch_model.bf16.safetensors" not in allow


def test_diffusers_skips_root_allinone_checkpoints() -> None:
    # SD1.5 shape: the repo ships all-in-one root checkpoints (12GB) on top
    # of the component tree — a diffusers-layout ingest must never pull them
    # (found live in e2e J7: 14.7GB selected instead of 2.75GB, ENOSPC on
    # the ingest pod).
    files = _DIFFUSERS + [
        "v1-5-pruned.safetensors",
        "v1-5-pruned-emaonly.safetensors",
    ]
    c = classify_repo(files, dtype_pref=("fp16",))
    assert c.strategy == "diffusers"
    allow = set(c.allow_patterns)
    assert "v1-5-pruned.safetensors" not in allow
    assert "v1-5-pruned-emaonly.safetensors" not in allow
    assert "transformer/diffusion_pytorch_model.fp16.safetensors" in allow


def test_transformers_with_sharded_index_and_onnx_excluded() -> None:
    files = [
        "config.json", "generation_config.json", "tokenizer.json",
        "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
        "onnx/model.onnx",
        "pytorch_model.bin",
    ]
    c = classify_repo(files, config_json={"architectures": ["LlamaForCausalLM"]})
    assert c.strategy == "transformers"
    allow = set(c.allow_patterns)
    assert "model-00001-of-00002.safetensors" in allow
    assert "model.safetensors.index.json" in allow
    assert "onnx/model.onnx" not in allow
    assert "pytorch_model.bin" not in allow
    assert c.attrs.get("architecture") == "LlamaForCausalLM"


def test_peft_adapter() -> None:
    c = classify_repo(["adapter_config.json", "adapter_model.safetensors", "README.md"])
    assert c.strategy == "peft"
    assert c.runtime_library == "peft"
    assert "adapter_model.safetensors" in c.allow_patterns


def test_sentence_transformers() -> None:
    c = classify_repo([
        "modules.json", "config.json", "model.safetensors",
        "1_Pooling/config.json", "tokenizer.json",
    ])
    assert c.strategy == "sentence_transformers"


def test_gguf_quant_preference_and_explicit_pick() -> None:
    files = ["model.Q4_K_M.gguf", "model.Q8_0.gguf", "README.md"]
    c = classify_repo(files)
    assert c.strategy == "gguf"
    assert [p for p in c.allow_patterns if p.endswith(".gguf")] == ["model.Q8_0.gguf"]

    c2 = classify_repo(files, gguf_quant="q4_k_m")
    assert [p for p in c2.allow_patterns if p.endswith(".gguf")] == ["model.Q4_K_M.gguf"]

    with pytest.raises(RepoRefusal) as exc:
        classify_repo(files, gguf_quant="q2_k")
    assert exc.value.reason == "gguf_quant_not_found"


def test_native_lora_via_kohya_metadata() -> None:
    c = classify_repo(
        ["my_lora.safetensors", "README.md"],
        sizes={"my_lora.safetensors": 40 * 1024 * 1024},
        safetensors_metadata={"ss_network_module": "networks.lora"},
    )
    assert c.strategy == "native_lora"
    assert c.runtime_library == "diffusers-lora"


def test_native_lora_via_readme_tags() -> None:
    c = classify_repo(
        ["style.safetensors"],
        sizes={"style.safetensors": 2 * 1024 * 1024 * 1024},
        readme_tags=["lora", "text-to-image"],
    )
    assert c.strategy == "native_lora"


def test_aio_singlefile() -> None:
    c = classify_repo(
        ["juggernaut-xl.safetensors"],
        sizes={"juggernaut-xl.safetensors": 6 * 1024 * 1024 * 1024},
    )
    assert c.strategy == "aio_singlefile"
    assert c.runtime_library == "diffusers-single-file"


@pytest.mark.parametrize("files,reason", [
    (["pytorch_model.bin"], "pickle_only"),
    (["model.onnx"], "onnx_only"),
    (["tf_model.h5"], "tf_only"),
    (["flax_model.msgpack"], "flax_only"),
    (["weights.mlpackage"], "coreml_only"),
    (["model.engine"], "tensorrt_only"),
    (["README.md"], "unknown_shape"),
])
def test_refusals(files: list[str], reason: str) -> None:
    with pytest.raises(RepoRefusal) as exc:
        classify_repo(files)
    assert exc.value.reason == reason


def test_too_large_refused() -> None:
    with pytest.raises(RepoRefusal) as exc:
        classify_repo(
            ["model_index.json", "transformer/x.safetensors"],
            sizes={"transformer/x.safetensors": 200 * 1024 * 1024 * 1024},
        )
    assert exc.value.reason == "too_large"
