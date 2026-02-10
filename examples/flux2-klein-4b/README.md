# flux2-klein-4b

FLUX.2-klein-4B example using Cozy’s injection pattern.

- The worker function only defines input/output + runs inference.
- Model selection + downloading is handled by the platform via `cozy.toml [models]`.
- This model is treated as a turbo model: the worker forces `num_inference_steps=8`.

Steps:

- `num_inference_steps` is accepted in the payload, but it is clamped to `[4, 8]` (rounded) for predictable cost/latency.

Config:

```toml
[models]
flux2-klein-4b = "hf:black-forest-labs/FLUX.2-klein-4B"
flux2-klein-4b_fp8 = { ref = "hf:black-forest-labs/FLUX.2-klein-4B", dtypes = ["fp8"] }
flux2-klein-4b_int8 = { ref = "hf:black-forest-labs/FLUX.2-klein-4B", dtypes = ["int8"] }
flux2-klein-4b_int4 = { ref = "hf:black-forest-labs/FLUX.2-klein-4B", dtypes = ["int4"] }
```

Code uses:

```py
pipeline: Annotated[Flux2KleinPipeline, ModelRef(Src.DEPLOYMENT, "flux2-klein-4b")]
```

There are two endpoints:

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
