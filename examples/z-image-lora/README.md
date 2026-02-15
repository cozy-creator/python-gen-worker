# z-image-lora

Dynamic LoRA loading example demonstrating the z-image pattern.

## Overview

This example shows how to load custom LoRAs at runtime, similar to fal.ai's z-image/turbo/lora endpoint. LoRAs are passed as `Asset` references in the request payload, downloaded and cached by the worker, then applied to the base Z-Image Turbo pipeline.

**Key features:**
- Base model (Z-Image Turbo) is loaded once and kept in VRAM via `ModelRef(Src.FIXED, "z-image-turbo")`
- LoRAs are passed as `Asset` in the request (external URLs or Cozy Hub file refs)
- LoRAs are loaded dynamically per-request
- LoRAs are unloaded after each request to avoid VRAM accumulation
- Multiple LoRAs can be combined with individual weights

## Pattern: LoRAs as Assets, Not Model Config

LoRAs are **not** declared in `cozy.toml [models]` because they're dynamic per-request. Instead, they're passed as `Asset` in the payload:

```python
class LoraSpec(msgspec.Struct):
    file: Asset        # LoRA weights (safetensors)
    weight: float      # LoRA strength (0.0-1.0+)
    adapter_name: str  # Optional name
```

The worker materializes `Asset` references before calling your function, so `lora.file.local_path` points to the downloaded file.

## Example Requests

### Basic generation (no LoRAs)

```json
{
    "prompt": "a beautiful landscape at sunset",
    "num_inference_steps": 9,
    "guidance_scale": 0.0
}
```

### With a single LoRA

```json
{
    "prompt": "a portrait in anime style",
    "loras": [
        {
            "file": {"ref": "https://example.com/anime-style.safetensors"},
            "weight": 0.8
        }
    ],
    "num_inference_steps": 9,
    "guidance_scale": 0.0
}
```

### With multiple LoRAs

```json
{
    "prompt": "a cyberpunk city with neon lights",
    "loras": [
        {
            "file": {"ref": "https://example.com/cyberpunk-style.safetensors"},
            "weight": 0.7,
            "adapter_name": "cyberpunk"
        },
        {
            "file": {"ref": "https://example.com/neon-lights.safetensors"},
            "weight": 0.5,
            "adapter_name": "neon"
        }
    ],
    "negative_prompt": "blurry, low quality",
    "num_inference_steps": 9,
    "guidance_scale": 0.0
}
```

### Using Cozy Hub file refs

```json
{
    "prompt": "product photography, white background",
    "loras": [
        {
            "file": {"ref": "loras/product-photography-v1"},
            "weight": 1.0
        }
    ]
}
```

## VRAM Considerations

- **Base model**: depends on the base model; Turbo-class image models still typically require >10GB for 1024x1024 at reasonable speed
- **LoRAs**: ~10-100MB each when loaded
- **Working memory**: ~3-4GB during inference

Total recommended VRAM: 16GB+ for comfortable operation with LoRAs.

LoRAs are unloaded after each request using `pipeline.unload_lora_weights()`, so VRAM usage returns to baseline between requests.

## Dependencies

- `gen-worker[torch]` - Worker SDK with torch support
- `diffusers` - Pipeline framework
- `transformers` - Model loading
- `accelerate` - Device placement
- `peft` - LoRA support in diffusers
- `pillow` - Image handling
