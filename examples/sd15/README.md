# sd15

Stable Diffusion 1.5 example using Cozy’s injection pattern.

- The worker function only defines input/output + runs inference.
- Model selection + downloading is handled by the worker runtime via `[tool.cozy.models]`.

Config:

```toml
[tool.cozy.models]
sd15 = "stable-diffusion-v1-5/stable-diffusion-v1-5"
```

Code uses:

```py
pipeline: Annotated[StableDiffusionPipeline, ModelRef(Src.DEPLOYMENT, "sd15")]
```
