# flux2-klein-4b

FLUX.2-klein turbo example using Cozy’s injection pattern (4B + 9B variants).

- The worker function only defines input/output + runs inference.
- Model selection + downloading is handled by the platform via `cozy.toml [models]`.
- This model is treated as a turbo model: the worker forces `num_inference_steps=8`.

Steps:

- `num_inference_steps` is accepted in the payload, but it is clamped to `[4, 8]` (rounded) for predictable cost/latency.

Code uses:

```py
pipeline: Annotated[
  Flux2KleinPipeline,
  ModelRef(Src.FIXED, "flux2-klein-4b", ref="black-forest-labs/FLUX.2-klein-4B", dtypes=("bf16",)),
]
```

Functions:

- `generate`: 4B bf16 (regular turbo baseline)
- `generate_fp8`: 4B fp8
- `generate_9b`: 9B bf16 (regular turbo baseline)
- `generate_9b_fp8`: 9B fp8
- `generate_int8`: int8-only
- `generate_int4`: int4-only

Notes on FP8:

- FP8 support here is **weight-only** quantization via `torchao` (Diffusers TorchAoConfig).
- GPUs vary: FP8 acceleration typically requires newer NVIDIA GPUs (e.g. Ada/Hopper class).

Notes on INT8/INT4:

- INT8/INT4 support here is **weight-only** quantization via `torchao` (Diffusers TorchAoConfig).
- INT4 is experimental for diffusion; expect quality regressions or incompatibilities.
