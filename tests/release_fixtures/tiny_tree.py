"""A tiny SD15-class CONFIG-ONLY checkpoint tree for the derive tests."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

_WEIGHT_PATTERNS = ("*.safetensors", "*.bin", "*.pt", "*.ckpt")


def build_pipe() -> Any:
    import torch
    from diffusers import (
        AutoencoderKL,
        DDIMScheduler,
        StableDiffusionPipeline,
        UNet2DConditionModel,
    )
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

    torch.manual_seed(0)
    unet: Any = UNet2DConditionModel(
        sample_size=8,
        in_channels=4,
        out_channels=4,
        layers_per_block=1,
        block_out_channels=(8, 16),
        down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
        up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"),
        cross_attention_dim=16,
        attention_head_dim=2,
        norm_num_groups=4,
    )
    vae: Any = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(8,),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=4,
        sample_size=32,
    )
    text_encoder = CLIPTextModel(
        CLIPTextConfig(
            hidden_size=16,
            intermediate_size=16,
            num_attention_heads=2,
            num_hidden_layers=2,
            vocab_size=1000,
            max_position_embeddings=77,
            projection_dim=16,
        )
    )
    vocab_dir = Path(tempfile.mkdtemp(prefix="tiny-vocab-"))
    letters = [chr(code) for code in range(ord("a"), ord("z") + 1)]
    vocab = {
        token: index
        for index, token in enumerate(
            ["<|startoftext|>", "<|endoftext|>"]
            + [f"{letter}</w>" for letter in letters]
            + letters
        )
    }
    (vocab_dir / "vocab.json").write_text(json.dumps(vocab))
    (vocab_dir / "merges.txt").write_text("#version: 0.2\n")
    tokenizer = CLIPTokenizer(str(vocab_dir / "vocab.json"), str(vocab_dir / "merges.txt"))
    tokenizer.model_max_length = 77
    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    unet.eval()
    vae.eval()
    text_encoder.eval()
    return StableDiffusionPipeline(
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )


def save_config_only(tree: Path) -> Path:
    pipe = build_pipe()
    pipe.save_pretrained(tree)
    removed = 0
    for pattern in _WEIGHT_PATTERNS:
        for weight_file in Path(tree).rglob(pattern):
            weight_file.unlink()
            removed += 1
    assert removed > 0, "the saved tree carried no weight files to delete"
    return Path(tree)
