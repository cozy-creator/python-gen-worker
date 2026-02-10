# sd15

Stable Diffusion 1.5 example using Cozy’s injection pattern.

- The worker function only defines input/output + runs inference.
- Model selection + downloading is handled by the worker runtime via `cozy.toml`.
- The worker clamps `num_inference_steps` to a minimum of 25 for quality.

Config:

```toml
[models]
sd15 = "hf:stable-diffusion-v1-5/stable-diffusion-v1-5"
```

Code uses:

```py
pipeline: Annotated[StableDiffusionPipeline, ModelRef(Src.FIXED, "sd15")]
```
