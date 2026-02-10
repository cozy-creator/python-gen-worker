# flux2-klein-4b

FLUX.2-klein-4B example using Cozy’s injection pattern.

- The worker function only defines input/output + runs inference.
- Model selection + downloading is handled by the platform via `cozy.toml [models]`.
- This model is treated as a turbo model: the worker forces `num_inference_steps=8`.

Config:

```toml
[models]
flux2-klein-4b = "hf:black-forest-labs/FLUX.2-klein-4B"
```

Code uses:

```py
pipeline: Annotated[Flux2KleinPipeline, ModelRef(Src.FIXED, "flux2-klein-4b")]
```
