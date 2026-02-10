# multi-sdxl-checkpoints

Example showing payload-based model selection with multiple SDXL variants.

## Overview

This example demonstrates how to efficiently support multiple model fine-tunes (checkpoints) in a single endpoint package. The request payload specifies which model to use, and the orchestrator routes requests to workers that have the requested model available.

## How It Works

### 1. Declare Models in Config

```toml
[models]
sdxl-base = "hf:stabilityai/stable-diffusion-xl-base-1.0"
sdxl-turbo = "hf:stabilityai/sdxl-turbo"
dreamshaper = "hf:Lykon/dreamshaper-xl-v2-turbo"
juggernaut = "hf:RunDiffusion/Juggernaut-XL-v9"
realvisxl = "hf:SG161222/RealVisXL_V5.0"
animagine = "hf:cagliostrolab/animagine-xl-3.1"
playground-v25 = "hf:playgroundai/playground-v2.5-1024px-aesthetic"
ssd-1b = "hf:segmind/SSD-1B"
pony-v6 = "hf:Runware/pony-xl-v6-diffusers"
illustrious-xl = "hf:drawhisper/illustrious-xl"
```

Keys (left side) are endpoint-local identifiers used in requests.
Values (right side) are model refs. `hf:` loads from Hugging Face; `cozy:` loads from Cozy Hub.

### 2. Use Payload-Based Model Selection

```python
@worker_function(ResourceRequirements())
def generate(
    ctx: ActionContext,
    pipeline: Annotated[
        DiffusionPipeline,
        ModelRef(Src.PAYLOAD, "model_key")  # Reads from payload.model_key
    ],
    payload: GenerateInput,
) -> GenerateOutput:
    ...
```

`ModelRef(Src.PAYLOAD, "model_key")` tells the worker to look up `payload.model_key` and use that as the model key.

### 3. Scheduler Routes Intelligently

Workers report model availability in heartbeats:
- `vram_models`: Models loaded in GPU memory (hot)
- `disk_models`: Models cached on disk (warm)

The scheduler prioritizes:
1. **Hot workers**: Model already in VRAM → instant inference
2. **Warm workers**: Model on disk → fast load (seconds)
3. **Cold workers**: Model not present → download + load (minutes)

## Example Requests

### Using the base SDXL model

```json
{
    "prompt": "a beautiful mountain landscape at sunrise",
    "model_key": "sdxl-base",
    "num_inference_steps": 28,
    "guidance_scale": 7.5
}
```

### Using DreamShaper

```json
{
    "prompt": "ethereal fantasy portrait, soft lighting",
    "model_key": "dreamshaper",
    "negative_prompt": "blurry, low quality, distorted"
}
```

### Using Juggernaut

```json
{
    "prompt": "photorealistic portrait of a woman, studio lighting",
    "model_key": "juggernaut",
    "num_inference_steps": 30,
    "guidance_scale": 8.0
}
```

## VRAM Considerations

### LRU Eviction

When a worker needs to load a new model but VRAM is full, the least-recently-used model is evicted. This happens automatically via the ModelCache.

### Routing Efficiency

With multiple workers, the scheduler distributes requests to maximize cache hits:
- Worker A has `sdxl-base` hot → gets base requests
- Worker B has `sdxl-turbo` hot → gets turbo requests
- Worker C has `dreamshaper` hot → gets dreamshaper requests

### Cold Starts

If no worker has the requested model:
1. Request goes to any capable worker
2. Worker downloads model from Hugging Face or Cozy Hub (cached to disk)
3. Worker loads model into VRAM
4. Inference runs
5. Model stays in VRAM for future requests

Cold starts are slower but only happen once per model per worker.

## Model Specification Rules

1. **Keys are endpoint-local**: `"sdxl-base"` only has meaning within this endpoint package
2. **Values are model refs**: `hf:` refs load from Hugging Face; `cozy:` refs load from Cozy Hub
3. **Payload uses keys**: Requests specify `model_key: "sdxl-base"`, not the full model ref
4. **Scheduler uses keys**: Routing decisions use the endpoint-local keys

This separation allows:
- Changing the underlying model without changing client code
- Different endpoint packages to use different versions of the "same" key
- Clear ownership of model configuration in `cozy.toml`
