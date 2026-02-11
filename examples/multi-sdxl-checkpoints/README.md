# multi-sdxl-checkpoints

Payload-based model selection for multiple SDXL checkpoints using a single endpoint (`generate`).

## Overview

- Requests select `model_key` from `cozy.toml [models]`
- Worker resolves that key via `ModelRef(Src.PAYLOAD, "model_key")`
- Scheduler routes by warm/hot model cache when possible

## Model Map

Defined in `cozy.toml`:

- `sdxl-base`
- `dreamshaper`
- `juggernaut`
- `realvisxl`
- `animagine`
- `playground-v25`
- `ssd-1b`
- `pony-v6`
- `illustrious-xl`

## Example Request

```json
{
  "prompt": "cinematic mountain valley at sunrise",
  "model_key": "sdxl-base",
  "num_inference_steps": 25,
  "guidance_scale": 7.0,
  "width": 1024,
  "height": 1024,
  "seed": 42
}
```

## Notes

- SDXL defaults are quality-oriented (`25+` steps, CFG 7).
- Input payload uses a single prompt pair (`prompt`, `negative_prompt`).
  The worker mirrors those into SDXL's secondary text inputs (`prompt_2`, `negative_prompt_2`).
- `preset=auto` keeps SDXL quality defaults consistent across model keys.
- On <=10GB GPUs, attention/VAE slicing + VAE tiling are enabled, and `preset=auto` downscales oversized requests to avoid OOM.
