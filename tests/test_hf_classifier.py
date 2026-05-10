"""Regression tests for gen_worker.conversion.hf_classifier.

Covers every classifier strategy and refusal
path. Tests use synthetic file listings (no HF network calls) so they're fast
and deterministic.
"""

from __future__ import annotations

import pytest

from gen_worker.conversion.hf_classifier import (
    ClassificationInputs,
    RepoFlaxOnly,
    RepoMissingSafetensors,
    RepoOnnxOnly,
    RepoPickleOnly,
    RepoTfOnly,
    RepoUnknownShape,
    classify_huggingface_repo,
    select_for_classification,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ci(
    paths: list[str],
    *,
    sizes: dict[str, int] | None = None,
    model_index: dict | None = None,
    config: dict | None = None,
    adapter: dict | None = None,
    modules: list | dict | None = None,
    config_st: dict | None = None,
    frontmatter: dict | None = None,
    safetensors_md: dict | None = None,
    safetensors_path: str | None = None,
) -> ClassificationInputs:
    return ClassificationInputs(
        file_paths=paths,
        file_sizes=sizes or {},
        model_index_json=model_index,
        config_json=config,
        adapter_config_json=adapter,
        modules_json=modules,
        config_sentence_transformers_json=config_st,
        readme_frontmatter=frontmatter or {},
        root_safetensors_metadata=safetensors_md,
        root_safetensors_path=safetensors_path,
    )


# ---------------------------------------------------------------------------
# (a) SDXL diffusers — drop ONNX/OpenVINO/Flax/PNGs/vae_decoder/vae_encoder
# ---------------------------------------------------------------------------

SDXL_PATHS = [
    "model_index.json",
    "README.md",
    "LICENSE.md",
    # unet
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "unet/diffusion_pytorch_model.safetensors",  # unsuffixed (fp32)
    "unet/diffusion_pytorch_model.bin",  # pickle — must not be selected
    "unet/config.json",
    "unet/model.onnx",  # junk
    "unet/model.onnx_data",  # junk
    "unet/openvino_model.xml",  # junk
    "unet/openvino_model.bin",  # junk
    # vae
    "vae/diffusion_pytorch_model.fp16.safetensors",
    "vae/diffusion_pytorch_model.safetensors",
    "vae/config.json",
    "vae/diffusion_flax_model.msgpack",  # junk
    # text_encoder + text_encoder_2
    "text_encoder/model.fp16.safetensors",
    "text_encoder/model.safetensors",
    "text_encoder/config.json",
    "text_encoder/flax_model.msgpack",
    "text_encoder_2/model.fp16.safetensors",
    "text_encoder_2/model.safetensors",
    "text_encoder_2/config.json",
    # tokenizers
    "tokenizer/vocab.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer_2/vocab.json",
    "tokenizer_2/merges.txt",
    "tokenizer_2/special_tokens_map.json",
    "tokenizer_2/tokenizer_config.json",
    # scheduler
    "scheduler/scheduler_config.json",
    # JUNK to be skipped
    "vae_decoder/config.json",
    "vae_decoder/model.onnx",
    "vae_decoder/openvino_model.xml",
    "vae_encoder/config.json",
    "vae_encoder/model.onnx",
    "vae_1_0/config.json",
    "01.png",
    "comparison.png",
    "pipeline.png",
    "sd_xl_offset_example-lora_1.0.safetensors",
]


def test_sdxl_diffusers_classification():
    inputs = _ci(
        SDXL_PATHS,
        model_index={
            "_class_name": "StableDiffusionXLPipeline",
            "unet": ["diffusers", "UNet2DConditionModel"],
            "vae": ["diffusers", "AutoencoderKL"],
            "text_encoder": ["transformers", "CLIPTextModel"],
            "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
            "tokenizer": ["transformers", "CLIPTokenizer"],
            "tokenizer_2": ["transformers", "CLIPTokenizer"],
            "scheduler": ["diffusers", "EulerDiscreteScheduler"],
        },
    )
    cls = classify_huggingface_repo(inputs)
    assert cls.strategy == "diffusers"
    assert cls.runtime_library == "diffusers"
    assert cls.refusal is None


def test_sdxl_diffusers_selection():
    inputs = _ci(
        SDXL_PATHS,
        model_index={
            "_class_name": "StableDiffusionXLPipeline",
            "unet": ["diffusers", "UNet2DConditionModel"],
            "vae": ["diffusers", "AutoencoderKL"],
            "text_encoder": ["transformers", "CLIPTextModel"],
            "text_encoder_2": ["transformers", "CLIPTextModelWithProjection"],
            "tokenizer": ["transformers", "CLIPTokenizer"],
            "tokenizer_2": ["transformers", "CLIPTokenizer"],
            "scheduler": ["diffusers", "EulerDiscreteScheduler"],
        },
    )
    cls = classify_huggingface_repo(inputs)
    sel = select_for_classification(cls, inputs, dtype_pref=("fp16", "bf16"))

    selected = set(sel.selected_paths)
    # Required canonical SDXL files
    must_have = {
        "model_index.json",
        "unet/diffusion_pytorch_model.fp16.safetensors",
        "unet/config.json",
        "vae/diffusion_pytorch_model.fp16.safetensors",
        "vae/config.json",
        "text_encoder/model.fp16.safetensors",
        "text_encoder/config.json",
        "text_encoder_2/model.fp16.safetensors",
        "text_encoder_2/config.json",
        "tokenizer/vocab.json",
        "tokenizer/merges.txt",
        "tokenizer/special_tokens_map.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer_2/vocab.json",
        "tokenizer_2/merges.txt",
        "tokenizer_2/special_tokens_map.json",
        "tokenizer_2/tokenizer_config.json",
        "scheduler/scheduler_config.json",
        "README.md",
        "LICENSE.md",
    }
    assert must_have.issubset(selected), f"missing: {must_have - selected}"

    # Must NOT include any pickle, ONNX, OpenVINO, Flax, PNG, or sibling LoRA
    must_not_have = {
        "unet/diffusion_pytorch_model.bin",
        "unet/model.onnx",
        "unet/model.onnx_data",
        "unet/openvino_model.xml",
        "unet/openvino_model.bin",
        "vae/diffusion_flax_model.msgpack",
        "text_encoder/flax_model.msgpack",
        "01.png",
        "comparison.png",
        "pipeline.png",
        "sd_xl_offset_example-lora_1.0.safetensors",  # at root, not in component
        # Pruned subdirs
        "vae_decoder/config.json",
        "vae_decoder/model.onnx",
        "vae_encoder/config.json",
        "vae_encoder/model.onnx",
        "vae_1_0/config.json",
        # Unsuffixed (fp32) safetensors when we have fp16 picked
        "unet/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
        "text_encoder/model.safetensors",
        "text_encoder_2/model.safetensors",
    }
    assert not (must_not_have & selected), f"must not include: {must_not_have & selected}"

    # Attributes
    assert sel.attrs["runtime_library"] == "diffusers"
    assert sel.attrs["pipeline_class"] == "StableDiffusionXLPipeline"


# ---------------------------------------------------------------------------
# (b) Llama-3 transformers fp16 sharded
# ---------------------------------------------------------------------------

def test_llama3_transformers_sharded():
    paths = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "README.md",
        "LICENSE",
        "USE_POLICY.md",
        # Pickle siblings to confirm rejection
        "original/consolidated.00.pth",
        "pytorch_model.bin",
    ]
    inputs = _ci(
        paths,
        config={
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
        },
    )
    cls = classify_huggingface_repo(inputs)
    assert cls.strategy == "transformers"
    assert cls.runtime_library == "transformers"
    assert cls.subtype == "decoder"

    sel = select_for_classification(cls, inputs, dtype_pref=("bf16", "fp16"))
    selected = set(sel.selected_paths)
    # Required transformers files
    must_have = {
        "config.json", "generation_config.json", "tokenizer.json",
        "tokenizer_config.json", "special_tokens_map.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "README.md", "LICENSE", "USE_POLICY.md",
    }
    assert must_have.issubset(selected), f"missing: {must_have - selected}"
    # No pickle
    assert "pytorch_model.bin" not in selected
    assert "original/consolidated.00.pth" not in selected
    assert sel.attrs["runtime_library"] == "transformers"
    assert sel.attrs["architecture"] == "LlamaForCausalLM"
    assert sel.attrs["subtype"] == "decoder"


# ---------------------------------------------------------------------------
# (c) Qwen-VL — transformers VLM (preprocessor sidecar)
# ---------------------------------------------------------------------------

def test_qwenvl_transformers_vlm_subtype():
    paths = [
        "config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "model.safetensors.index.json",
        "model-00001-of-00005.safetensors",
        "model-00002-of-00005.safetensors",
        "model-00003-of-00005.safetensors",
        "model-00004-of-00005.safetensors",
        "model-00005-of-00005.safetensors",
        "README.md",
    ]
    inputs = _ci(
        paths,
        config={
            "architectures": ["Qwen2VLForConditionalGeneration"],
            "model_type": "qwen2_vl",
        },
    )
    cls = classify_huggingface_repo(inputs)
    assert cls.strategy == "transformers"
    assert cls.subtype == "vlm"

    sel = select_for_classification(cls, inputs)
    selected = set(sel.selected_paths)
    assert "preprocessor_config.json" in selected
    assert "processor_config.json" in selected
    assert "chat_template.jinja" in selected
    assert "model.safetensors.index.json" in selected
    assert "model-00003-of-00005.safetensors" in selected
    assert sel.attrs["subtype"] == "vlm"
    assert sel.attrs["architecture"] == "Qwen2VLForConditionalGeneration"


# ---------------------------------------------------------------------------
# (d) Pre-quantized FP8 — config.quantization_config marks subtype
# ---------------------------------------------------------------------------

def test_qwen_fp8_quantized_subtype():
    paths = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors",
        "README.md",
    ]
    inputs = _ci(
        paths,
        config={
            "architectures": ["Qwen2ForCausalLM"],
            "model_type": "qwen2",
            "quantization_config": {
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
            },
        },
    )
    cls = classify_huggingface_repo(inputs)
    assert cls.strategy == "transformers"
    assert cls.subtype == "quantized"

    sel = select_for_classification(cls, inputs)
    assert sel.attrs["quant_scheme"] == "fp8"
    assert sel.attrs["subtype"] == "quantized"


# ---------------------------------------------------------------------------
# (e) GGUF — picks correct quant level
# ---------------------------------------------------------------------------

def test_gguf_default_quant_pick():
    paths = [
        "README.md",
        "Meta-Llama-3.1-8B-Instruct-Q2_K.gguf",
        "Meta-Llama-3.1-8B-Instruct-Q3_K_M.gguf",
        "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf",
        "Meta-Llama-3.1-8B-Instruct-Q6_K.gguf",
        "Meta-Llama-3.1-8B-Instruct-Q8_0.gguf",
        "Meta-Llama-3.1-8B-Instruct-f16.gguf",
        "tokenizer.model",
    ]
    inputs = _ci(paths)
    cls = classify_huggingface_repo(inputs)
    assert cls.strategy == "gguf"
    assert cls.runtime_library == "llama-cpp"

    sel = select_for_classification(cls, inputs)
    # Default preference is now PRECISION-FIRST (#72): F16 wins over Q4_K_M.
    assert "Meta-Llama-3.1-8B-Instruct-f16.gguf" in sel.selected_paths
    assert "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" not in sel.selected_paths
    assert "Meta-Llama-3.1-8B-Instruct-Q2_K.gguf" not in sel.selected_paths
    assert "tokenizer.model" in sel.selected_paths
    assert "README.md" in sel.selected_paths
    assert sel.attrs["dtype"] == "f16"
    assert sel.attrs["quant_scheme"] == "F16"


def test_gguf_explicit_quant_request():
    paths = [
        "README.md",
        "model-Q4_K_M.gguf",
        "model-Q8_0.gguf",
    ]
    inputs = _ci(paths)
    cls = classify_huggingface_repo(inputs)
    sel = select_for_classification(cls, inputs, gguf_quant="Q8_0")
    assert "model-Q8_0.gguf" in sel.selected_paths
    assert "model-Q4_K_M.gguf" not in sel.selected_paths
    assert sel.attrs["quant_scheme"] == "Q8_0"


def test_gguf_unknown_quant_raises():
    paths = ["model-Q4_K_M.gguf"]
    inputs = _ci(paths)
    cls = classify_huggingface_repo(inputs)
    with pytest.raises(ValueError, match="not present"):
        select_for_classification(cls, inputs, gguf_quant="Q1_K")


# ---------------------------------------------------------------------------
# (f) Sentence-Transformers — recurse into module subdirs
# ---------------------------------------------------------------------------

def test_sentence_transformers_root_path_module():
    # Real all-MiniLM-L6-v2 ships modules.json with the Transformer module at
    # path: "" (i.e. at the repo root). Files like model.safetensors,
    # config.json, tokenizer.json sit alongside modules.json — they are NOT
    # in a 0_Transformer/ subdir.
    paths = [
        "modules.json",
        "config_sentence_transformers.json",
        "config.json",
        "model.safetensors",         # 91 MB - the actual weights
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "1_Pooling/config.json",
        "README.md",
        # junk
        "tf_model.h5",
        "pytorch_model.bin",
        "rust_model.ot",
        "onnx/model.onnx",
        "openvino/openvino_model.bin",
    ]
    inputs = _ci(
        paths,
        sizes={
            "model.safetensors": 90868376,
            "tf_model.h5": 91005696,
            "pytorch_model.bin": 90888945,
        },
        modules=[
            {"idx": 0, "name": "0_Transformer", "path": "", "type": "sentence_transformers.models.Transformer"},
            {"idx": 1, "name": "1_Pooling", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
        ],
        config_st={"__version__": {"sentence_transformers": "2.0.0"}},
    )
    cls = classify_huggingface_repo(inputs)
    assert cls.strategy == "sentence_transformers"

    sel = select_for_classification(cls, inputs)
    selected = set(sel.selected_paths)
    # Critical assertion: the actual weights MUST be selected
    assert "model.safetensors" in selected, \
        f"sentence-transformers selector missed model.safetensors at root: {selected}"
    assert "config.json" in selected
    assert "tokenizer.json" in selected
    assert "1_Pooling/config.json" in selected
    # Junk dropped
    assert "tf_model.h5" not in selected
    assert "pytorch_model.bin" not in selected
    assert "rust_model.ot" not in selected
    assert "onnx/model.onnx" not in selected
    assert "openvino/openvino_model.bin" not in selected


def test_sentence_transformers_modules():
    paths = [
        "modules.json",
        "config_sentence_transformers.json",
        "config.json",
        "sentence_bert_config.json",
        "README.md",
        # 0_Transformer module — has its own transformers shape
        "0_Transformer/config.json",
        "0_Transformer/tokenizer.json",
        "0_Transformer/tokenizer_config.json",
        "0_Transformer/model.safetensors",
        # 1_Pooling module
        "1_Pooling/config.json",
        # 2_Normalize
        "2_Normalize/config.json",
    ]
    inputs = _ci(
        paths,
        modules=[
            {"idx": 0, "name": "0_Transformer", "path": "0_Transformer", "type": "sentence_transformers.models.Transformer"},
            {"idx": 1, "name": "1_Pooling", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
            {"idx": 2, "name": "2_Normalize", "path": "2_Normalize", "type": "sentence_transformers.models.Normalize"},
        ],
    )
    cls = classify_huggingface_repo(inputs)
    assert cls.strategy == "sentence_transformers"
    assert cls.runtime_library == "sentence-transformers"

    sel = select_for_classification(cls, inputs)
    selected = set(sel.selected_paths)
    must_have = {
        "modules.json", "config_sentence_transformers.json", "config.json",
        "sentence_bert_config.json", "README.md",
        "0_Transformer/config.json", "0_Transformer/tokenizer.json",
        "0_Transformer/tokenizer_config.json", "0_Transformer/model.safetensors",
        "1_Pooling/config.json",
        "2_Normalize/config.json",
    }
    assert must_have.issubset(selected), f"missing: {must_have - selected}"
    assert sel.attrs["runtime_library"] == "sentence-transformers"


# ---------------------------------------------------------------------------
# (g) PEFT canonical — adapter_config.json with base_model_name_or_path
# ---------------------------------------------------------------------------

def test_peft_canonical_lineage_extracted():
    paths = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "README.md",
    ]
    inputs = _ci(
        paths,
        adapter={
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": "meta-llama/Llama-3.2-3B",
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q_proj", "v_proj"],
        },
    )
    cls = classify_huggingface_repo(inputs)
    assert cls.strategy == "peft_canonical"
    assert cls.runtime_library == "peft"

    sel = select_for_classification(cls, inputs)
    assert sel.attrs["peft_type"] == "LORA"
    assert sel.attrs["task_type"] == "CAUSAL_LM"
    assert sel.attrs["base_model_lineage"] == "meta-llama/Llama-3.2-3B"
    assert sel.attrs["lineage_source"] == "adapter_config_json"
    assert sel.attrs["r"] == "16"
    assert sel.attrs["lora_alpha"] == "32"
    assert "adapter_config.json" in sel.selected_paths
    assert "adapter_model.safetensors" in sel.selected_paths


def test_peft_pickle_only_refused():
    paths = [
        "adapter_config.json",
        "adapter_model.bin",  # pickle, no safetensors sibling
    ]
    inputs = _ci(paths, adapter={"peft_type": "LORA"})
    cls = classify_huggingface_repo(inputs)
    with pytest.raises(RepoMissingSafetensors):
        select_for_classification(cls, inputs)


# ---------------------------------------------------------------------------
# (h) Native LoRA, kohya safetensors metadata
# ---------------------------------------------------------------------------

def test_native_lora_kohya_metadata_lineage():
    paths = [
        "README.md",
        "SDXL-VintageMagStyle-Lora.safetensors",
        "Examples/sample-1.png",
        "Examples/sample-2.png",
    ]
    inputs = _ci(
        paths,
        sizes={"SDXL-VintageMagStyle-Lora.safetensors": 60_000_000},  # 60MB
        safetensors_md={
            "ss_network_module": "networks.lora",
            "ss_network_dim": "32",
            "ss_network_alpha": "32",
            "ss_base_model_version": "stable-diffusion-xl-1.0",
            "ss_sd_model_name": "stable-diffusion-xl-base-1.0",
        },
        safetensors_path="SDXL-VintageMagStyle-Lora.safetensors",
    )
    cls = classify_huggingface_repo(inputs)
    assert cls.strategy == "native_lora"
    assert cls.runtime_library == "diffusers-lora"

    sel = select_for_classification(cls, inputs)
    assert "SDXL-VintageMagStyle-Lora.safetensors" in sel.selected_paths
    assert "README.md" in sel.selected_paths
    # Demo PNGs dropped
    assert "Examples/sample-1.png" not in sel.selected_paths
    # Lineage from kohya metadata
    assert sel.attrs["base_model_lineage"] == "stable-diffusion-xl-1.0"
    assert sel.attrs["lineage_source"] == "kohya_safetensors_metadata"
    assert sel.attrs["network_module"] == "networks.lora"
    assert sel.attrs["network_dim"] == "32"


# ---------------------------------------------------------------------------
# (i) Native LoRA, README YAML frontmatter only (no kohya metadata)
# ---------------------------------------------------------------------------

def test_native_lora_yaml_frontmatter_lineage():
    paths = [
        "README.md",
        "MyStyle-v1.safetensors",
    ]
    inputs = _ci(
        paths,
        sizes={"MyStyle-v1.safetensors": 50_000_000},
        frontmatter={
            "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
            "tags": ["stable-diffusion-xl", "lora"],
        },
    )
    cls = classify_huggingface_repo(inputs)
    assert cls.strategy == "native_lora"
    sel = select_for_classification(cls, inputs)
    assert sel.attrs["base_model_lineage"] == "stabilityai/stable-diffusion-xl-base-1.0"
    assert sel.attrs["lineage_source"] == "yaml_frontmatter"


# ---------------------------------------------------------------------------
# (j) Native LoRA with no lineage signals → unknown
# ---------------------------------------------------------------------------

def test_native_lora_no_lineage_unknown():
    paths = [
        "README.md",
        "MysteryLora.safetensors",
    ]
    inputs = _ci(
        paths,
        sizes={"MysteryLora.safetensors": 80_000_000},
        # No frontmatter, no kohya metadata
    )
    cls = classify_huggingface_repo(inputs)
    assert cls.strategy == "native_lora"
    sel = select_for_classification(cls, inputs)
    assert sel.attrs["base_model_lineage"] == "unknown"
    assert sel.attrs["lineage_source"] == "none"
    # Still ingested — anonymous community uploads are a real case
    assert "MysteryLora.safetensors" in sel.selected_paths


# ---------------------------------------------------------------------------
# (k) Pickle-only LLM — refused
# ---------------------------------------------------------------------------

def test_pickle_only_llm_refused():
    paths = [
        "config.json",
        "tokenizer.json",
        "pytorch_model.bin",
        "README.md",
    ]
    # config.json is present but no safetensors, so transformers strategy
    # doesn't trigger. Falls through to refusal logic.
    inputs = _ci(paths, config={"architectures": ["LlamaForCausalLM"]})
    cls = classify_huggingface_repo(inputs)
    assert cls.refusal is not None
    assert isinstance(cls.refusal, RepoPickleOnly)


# ---------------------------------------------------------------------------
# (l) ONNX-only — refused
# ---------------------------------------------------------------------------

def test_onnx_only_refused():
    paths = [
        "model.onnx",
        "model.onnx_data",
        "README.md",
    ]
    inputs = _ci(paths)
    cls = classify_huggingface_repo(inputs)
    assert cls.refusal is not None
    assert isinstance(cls.refusal, RepoOnnxOnly)


# ---------------------------------------------------------------------------
# (m) FLUX (already-clean diffusers) regression guard
# ---------------------------------------------------------------------------

def test_flux_diffusers_regression_guard():
    paths = [
        "model_index.json",
        "README.md",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
        "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "text_encoder/config.json",
        "text_encoder/model-00001-of-00002.safetensors",
        "text_encoder/model-00002-of-00002.safetensors",
        "text_encoder/model.safetensors.index.json",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/special_tokens_map.json",
        "scheduler/scheduler_config.json",
    ]
    inputs = _ci(
        paths,
        model_index={
            "_class_name": "FluxPipeline",
            "transformer": ["diffusers", "FluxTransformer2DModel"],
            "vae": ["diffusers", "AutoencoderKL"],
            "text_encoder": ["transformers", "CLIPTextModel"],
            "tokenizer": ["transformers", "CLIPTokenizer"],
            "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
        },
    )
    cls = classify_huggingface_repo(inputs)
    assert cls.strategy == "diffusers"
    assert cls.refusal is None

    sel = select_for_classification(cls, inputs, dtype_pref=("bf16", "fp16"))
    selected = set(sel.selected_paths)
    # Already-clean repo; everything safetensors gets selected
    for p in paths:
        if not p.endswith((".png", ".jpg", ".bin")):
            assert p in selected, f"missing: {p}"
    assert sel.attrs["pipeline_class"] == "FluxPipeline"


# ---------------------------------------------------------------------------
# Refusal: TF-only
# ---------------------------------------------------------------------------

def test_tf_only_refused():
    paths = ["tf_model.h5", "config.json", "README.md"]
    # config.json present but no safetensors — falls to refusal path
    inputs = _ci(paths, config={"architectures": ["BertForSequenceClassification"]})
    cls = classify_huggingface_repo(inputs)
    assert cls.refusal is not None
    assert isinstance(cls.refusal, RepoTfOnly)


# ---------------------------------------------------------------------------
# Refusal: Flax-only
# ---------------------------------------------------------------------------

def test_flax_only_refused():
    paths = ["flax_model.msgpack", "config.json", "README.md"]
    inputs = _ci(paths, config={"architectures": ["FlaxGPT2Model"]})
    cls = classify_huggingface_repo(inputs)
    assert cls.refusal is not None
    assert isinstance(cls.refusal, RepoFlaxOnly)


# ---------------------------------------------------------------------------
# Refusal: unknown shape (empty repo)
# ---------------------------------------------------------------------------

def test_unknown_shape_refused():
    paths = ["README.md", "junk.txt"]
    inputs = _ci(paths)
    cls = classify_huggingface_repo(inputs)
    assert cls.refusal is not None
    assert isinstance(cls.refusal, RepoUnknownShape)


# ---------------------------------------------------------------------------
# AIO/Singlefile: full SD checkpoint at root
# ---------------------------------------------------------------------------

def test_aio_singlefile_full_checkpoint():
    paths = [
        "README.md",
        "v1-5-pruned-emaonly.safetensors",
        "v1-inference.yaml",
    ]
    inputs = _ci(
        paths,
        sizes={"v1-5-pruned-emaonly.safetensors": 4_265_000_000},  # 4.3 GB — above LoRA threshold
        # No kohya metadata, no `lora` tag → AIO not native_lora
    )
    cls = classify_huggingface_repo(inputs)
    assert cls.strategy == "aio_singlefile"
    assert cls.runtime_library == "diffusers-single-file"

    sel = select_for_classification(cls, inputs)
    assert "v1-5-pruned-emaonly.safetensors" in sel.selected_paths
    assert "v1-inference.yaml" in sel.selected_paths
    assert "README.md" in sel.selected_paths


# ---------------------------------------------------------------------------
# Defense-in-depth: pickle slipped into a per-strategy selector → blocked
# ---------------------------------------------------------------------------

def test_diffusers_optional_components_skipped_on_pickle_only():
    # SDXL family / SD-1.5 family ship safety_checker + feature_extractor as
    # pickle-only. Those are optional; we should skip them, not refuse the
    # whole clone.
    paths = [
        "model_index.json",
        "unet/diffusion_pytorch_model.safetensors",
        "unet/config.json",
        "vae/diffusion_pytorch_model.safetensors",
        "vae/config.json",
        "text_encoder/model.safetensors",
        "text_encoder/config.json",
        "tokenizer/vocab.json",
        "tokenizer/merges.txt",
        "scheduler/scheduler_config.json",
        # Optional pickle-only components — must not cause refusal
        "safety_checker/pytorch_model.bin",
        "safety_checker/config.json",
        "feature_extractor/preprocessor_config.json",
    ]
    inputs = _ci(
        paths,
        model_index={
            "_class_name": "StableDiffusionPipeline",
            "unet": ["diffusers", "UNet2DConditionModel"],
            "vae": ["diffusers", "AutoencoderKL"],
            "text_encoder": ["transformers", "CLIPTextModel"],
            "tokenizer": ["transformers", "CLIPTokenizer"],
            "scheduler": ["diffusers", "PNDMScheduler"],
            "safety_checker": ["diffusers_legacy.safety_checker", "StableDiffusionSafetyChecker"],
            "feature_extractor": ["transformers", "CLIPFeatureExtractor"],
        },
    )
    cls = classify_huggingface_repo(inputs)
    assert cls.strategy == "diffusers"
    sel = select_for_classification(cls, inputs)
    selected = set(sel.selected_paths)
    # Required components are there
    assert "unet/diffusion_pytorch_model.safetensors" in selected
    assert "vae/diffusion_pytorch_model.safetensors" in selected
    # Optional pickle-only is skipped (no refusal)
    assert "safety_checker/pytorch_model.bin" not in selected
    # safety_checker config.json may be picked up (it's a config, not a weight),
    # but the pickle weight is definitely not.
    # feature_extractor has no weight at all, just a preprocessor config — fine.


def test_pickle_in_diffusers_dropped_via_blocklist():
    # If a strategy somehow slips a .bin past, _finalize blocklists it.
    paths = [
        "model_index.json",
        "unet/diffusion_pytorch_model.safetensors",
        "unet/config.json",
        "unet/diffusion_pytorch_model.bin",  # pickle — must not survive
    ]
    inputs = _ci(
        paths,
        model_index={
            "_class_name": "StableDiffusionPipeline",
            "unet": ["diffusers", "UNet2DConditionModel"],
        },
    )
    cls = classify_huggingface_repo(inputs)
    sel = select_for_classification(cls, inputs)
    assert "unet/diffusion_pytorch_model.bin" not in sel.selected_paths


# ---------------------------------------------------------------------------
# Always-include allowlist: README/LICENSE always present at root → always picked
# ---------------------------------------------------------------------------

def test_always_include_license_when_present():
    paths = [
        "model_index.json",
        "unet/diffusion_pytorch_model.safetensors",
        "unet/config.json",
        "LICENSE",
        "NOTICE",
    ]
    inputs = _ci(
        paths,
        model_index={"_class_name": "StableDiffusionPipeline",
                     "unet": ["diffusers", "UNet2DConditionModel"]},
    )
    cls = classify_huggingface_repo(inputs)
    sel = select_for_classification(cls, inputs)
    assert "LICENSE" in sel.selected_paths
    assert "NOTICE" in sel.selected_paths


def test_gguf_default_now_picks_f16_not_q4():
    # Default prefers F16 (highest fidelity) over Q4_K_M (lossy).
    # when both are available.
    paths = [
        "README.md",
        "model-Q4_K_M.gguf",
        "model-Q8_0.gguf",
        "model-f16.gguf",
        "model-bf16.gguf",
    ]
    inputs = _ci(paths)
    cls = classify_huggingface_repo(inputs)
    sel = select_for_classification(cls, inputs)  # no quant request
    selected = set(sel.selected_paths)
    assert "model-f16.gguf" in selected, f"expected f16 default, got {selected}"
    assert "model-Q4_K_M.gguf" not in selected
    assert "model-Q8_0.gguf" not in selected
    assert sel.attrs["dtype"] == "f16"
    assert sel.attrs["quant_scheme"] == "F16"


def test_gguf_fuzzy_4bit_resolves_to_q4_k_m():
    # Fuzzy `4bit` token resolves to Q4_K_M when available.
    paths = ["README.md", "model-Q4_K_M.gguf", "model-Q4_K_S.gguf", "model-f16.gguf"]
    inputs = _ci(paths)
    cls = classify_huggingface_repo(inputs)
    sel = select_for_classification(cls, inputs, gguf_quant="4bit")
    assert "model-Q4_K_M.gguf" in sel.selected_paths
    assert "model-Q4_K_S.gguf" not in sel.selected_paths
    assert sel.attrs["dtype"] == "q4_k_m"


def test_gguf_fuzzy_4bit_falls_back_to_q4_k_s_when_q4_k_m_missing():
    # Fuzzy table cascades through alternatives.
    paths = ["README.md", "model-Q4_K_S.gguf", "model-f16.gguf"]
    inputs = _ci(paths)
    cls = classify_huggingface_repo(inputs)
    sel = select_for_classification(cls, inputs, gguf_quant="4bit")
    assert "model-Q4_K_S.gguf" in sel.selected_paths
    assert sel.attrs["dtype"] == "q4_k_s"


def test_gguf_fuzzy_16bit_prefers_bf16_over_f16():
    paths = ["README.md", "model-bf16.gguf", "model-f16.gguf"]
    inputs = _ci(paths)
    cls = classify_huggingface_repo(inputs)
    sel = select_for_classification(cls, inputs, gguf_quant="16bit")
    assert "model-bf16.gguf" in sel.selected_paths
    assert "model-f16.gguf" not in sel.selected_paths
    assert sel.attrs["dtype"] == "bf16"


def test_gguf_fuzzy_unresolvable_raises_clear_error():
    # 5bit fuzzy token, repo only has 4bit + 16bit — fuzzy resolution fails
    paths = ["README.md", "model-Q4_K_M.gguf", "model-f16.gguf"]
    inputs = _ci(paths)
    cls = classify_huggingface_repo(inputs)
    import pytest
    with pytest.raises(ValueError, match="could not resolve"):
        select_for_classification(cls, inputs, gguf_quant="5bit")


# ---------------------------------------------------------------------------
# Issue #71: structured base-model lineage on LoRA / PEFT clones
# ---------------------------------------------------------------------------

def test_peft_lineage_extracts_repo_and_family():
    paths = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "README.md",
    ]
    inputs = _ci(
        paths,
        adapter={
            "peft_type": "LORA",
            "task_type": "CAUSAL_LM",
            "base_model_name_or_path": "meta-llama/Llama-3.2-3B",
            "r": 16,
            "lora_alpha": 32,
        },
    )
    cls = classify_huggingface_repo(inputs)
    sel = select_for_classification(cls, inputs)
    assert sel.attrs["base_model_repo"] == "meta-llama/Llama-3.2-3B"
    assert sel.attrs["base_model_family"] == "llama-3-2-3b"
    assert sel.attrs["lineage_source"] == "adapter_config_json"


def test_native_lora_lineage_yaml_frontmatter():
    paths = ["README.md", "MyStyle.safetensors"]
    inputs = _ci(
        paths,
        sizes={"MyStyle.safetensors": 50_000_000},
        frontmatter={
            "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
            "tags": ["stable-diffusion-xl", "lora"],
        },
    )
    cls = classify_huggingface_repo(inputs)
    sel = select_for_classification(cls, inputs)
    assert sel.attrs["base_model_repo"] == "stabilityai/stable-diffusion-xl-base-1.0"
    assert sel.attrs["base_model_family"] == "sdxl"
    assert sel.attrs["lineage_source"] == "yaml_frontmatter"


def test_native_lora_lineage_kohya_with_specific_hint():
    paths = ["README.md", "VintageStyle.safetensors"]
    inputs = _ci(
        paths,
        sizes={"VintageStyle.safetensors": 60_000_000},
        safetensors_md={
            "ss_network_module": "networks.lora",
            "ss_network_dim": "32",
            "ss_network_alpha": "32",
            "ss_base_model_version": "stable-diffusion-xl-1.0",
            "ss_sd_model_name": "juggernautXL_v8Rundiffusion.safetensors",
        },
        safetensors_path="VintageStyle.safetensors",
    )
    cls = classify_huggingface_repo(inputs)
    sel = select_for_classification(cls, inputs)
    # Family resolved from kohya version
    assert sel.attrs["base_model_family"] == "sdxl"
    # Specific hint preserved separately
    assert sel.attrs["base_model_specific_hint"] == "juggernautXL_v8Rundiffusion.safetensors"
    assert sel.attrs["lineage_source"] == "kohya_safetensors_metadata"
    # Network metadata
    assert sel.attrs["network_module"] == "networks.lora"
    assert sel.attrs["network_dim"] == "32"


def test_native_lora_lineage_unknown():
    # No YAML frontmatter, no kohya metadata, no tags → unknown
    paths = ["README.md", "MysteryLora.safetensors"]
    inputs = _ci(paths, sizes={"MysteryLora.safetensors": 80_000_000})
    cls = classify_huggingface_repo(inputs)
    sel = select_for_classification(cls, inputs)
    assert sel.attrs["base_model_family"] == "unknown"
    assert sel.attrs["lineage_source"] == "none"
    # Still ingested
    assert "MysteryLora.safetensors" in sel.selected_paths


def test_base_model_families_module():
    from gen_worker.conversion.base_model_families import (
        kohya_to_family, civitai_to_family, repo_to_family, tags_to_family,
    )
    # Kohya
    assert kohya_to_family("stable-diffusion-xl-1.0") == "sdxl"
    assert kohya_to_family("flux1-dev") == "flux1-dev"
    assert kohya_to_family("sd_v1.5") == "sd15"
    assert kohya_to_family("") is None
    # Civitai
    assert civitai_to_family("SDXL 1.0") == "sdxl"
    assert civitai_to_family("Flux.1 D") == "flux1-dev"
    assert civitai_to_family("Pony") == "sdxl-pony"
    assert civitai_to_family("Illustrious") == "sdxl-illustrious"
    # Repo
    assert repo_to_family("stabilityai/stable-diffusion-xl-base-1.0") == "sdxl"
    assert repo_to_family("black-forest-labs/FLUX.2-klein-4B") == "flux2-klein-4b"
    assert repo_to_family("meta-llama/Llama-3.2-3B") == "llama-3-2-3b"
    # Tags
    assert tags_to_family(["stable-diffusion-xl", "lora"]) == "sdxl"
    assert tags_to_family(["flux", "text-to-image"]) == "flux1-dev"
    assert tags_to_family([]) is None
    assert tags_to_family(None) is None
