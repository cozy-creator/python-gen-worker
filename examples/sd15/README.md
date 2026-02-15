# sd15

Stable Diffusion 1.5 example using Cozy’s injection pattern.

- The worker function only defines input/output + runs inference.
- Model selection + downloading is handled by the worker runtime via `cozy.toml`.
- The worker clamps `num_inference_steps` to a minimum of 25 for quality.

Config:

```toml
[models]
sd15 = "cozy:stable-diffusion-v1-5/stable-diffusion-v1-5:latest"
sd15_fp8 = { ref = "cozy:stable-diffusion-v1-5/stable-diffusion-v1-5:latest", dtypes = ["fp8"] }
sd15_int8 = { ref = "cozy:stable-diffusion-v1-5/stable-diffusion-v1-5:latest", dtypes = ["int8"] }
sd15_int4 = { ref = "cozy:stable-diffusion-v1-5/stable-diffusion-v1-5:latest", dtypes = ["int4"] }
```

Code uses:

```py
pipeline: Annotated[StableDiffusionPipeline, ModelRef(Src.FIXED, "sd15")]
```

There are two functions:

- `generate`: fp16/bf16 (default)
- `generate_fp8`: fp8-only
- `generate_int8`: int8-only
- `generate_int4`: int4-only

Notes on FP8:

- FP8 support here is **weight-only** quantization via `torchao` (Diffusers TorchAoConfig).
- GPUs vary: FP8 acceleration typically requires newer NVIDIA GPUs (e.g. Ada/Hopper class).

Notes on INT8/INT4:

- INT8/INT4 support here is **weight-only** quantization via `torchao` (Diffusers TorchAoConfig).
- INT4 is experimental for diffusion; expect quality regressions or incompatibilities.
