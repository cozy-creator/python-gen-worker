"""The small HF variant selector (#366, replaces the 522-LOC planner).

``select_hf_files`` decides which repo files ``snapshot_download`` pulls:
one weight set per component directory (flavor > bf16 > fp16 > untagged,
safetensors preferred), all configs/tokenizers, root-weights repos supported,
unknown layouts fall back to the whole repo (returns None).
"""

from __future__ import annotations

from gen_worker.models.download import select_hf_files

_DIFFUSERS_REPO = [
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
    "vae/diffusion_pytorch_model.fp16.safetensors",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model.bf16.safetensors",
    "transformer/diffusion_pytorch_model.fp16.safetensors",
    "transformer/diffusion_pytorch_model.safetensors",
]


def test_diffusers_repo_picks_one_weight_set_per_component() -> None:
    sel = select_hf_files(_DIFFUSERS_REPO)
    assert sel is not None
    # All non-weight files included.
    assert "model_index.json" in sel
    assert "scheduler/scheduler_config.json" in sel
    assert "tokenizer/tokenizer.json" in sel
    # transformer: bf16 preferred over fp16/untagged.
    assert "transformer/diffusion_pytorch_model.bf16.safetensors" in sel
    assert "transformer/diffusion_pytorch_model.fp16.safetensors" not in sel
    assert "transformer/diffusion_pytorch_model.safetensors" not in sel
    # vae/text_encoder have no bf16 -> fp16 wins over untagged.
    assert "vae/diffusion_pytorch_model.fp16.safetensors" in sel
    assert "vae/diffusion_pytorch_model.safetensors" not in sel


def test_flavor_overrides_the_default_preference() -> None:
    sel = select_hf_files(_DIFFUSERS_REPO, flavor="fp16")
    assert sel is not None
    assert "transformer/diffusion_pytorch_model.fp16.safetensors" in sel
    assert "transformer/diffusion_pytorch_model.bf16.safetensors" not in sel


def test_safetensors_preferred_over_bin() -> None:
    files = [
        "model_index.json",
        "unet/config.json",
        "unet/diffusion_pytorch_model.bin",
        "unet/diffusion_pytorch_model.safetensors",
    ]
    sel = select_hf_files(files)
    assert sel is not None
    assert "unet/diffusion_pytorch_model.safetensors" in sel
    assert "unet/diffusion_pytorch_model.bin" not in sel


def test_root_weights_repo_selects_weights_and_sidecars() -> None:
    files = [
        "config.json",
        "tokenizer.json",
        "model.bf16.safetensors",
        "model.fp16.safetensors",
        "README.md",
    ]
    sel = select_hf_files(files)
    assert sel is not None
    assert "model.bf16.safetensors" in sel
    assert "model.fp16.safetensors" not in sel
    assert "config.json" in sel and "tokenizer.json" in sel


def test_unrecognized_layout_downloads_whole_repo() -> None:
    # Weights nested under non-diffusers dirs (ComfyUI split checkpoints).
    files = [
        "split_files/diffusion_models/model.safetensors",
        "split_files/vae/ae.safetensors",
        "notes.txt",
    ]
    assert select_hf_files(files) is None


def test_no_weights_at_all_downloads_whole_repo() -> None:
    assert select_hf_files(["README.md", "config.json"]) is None
    assert select_hf_files([]) is None
