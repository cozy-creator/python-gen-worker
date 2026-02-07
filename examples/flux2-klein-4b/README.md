# flux2-klein-4b

FLUX.2-klein-4B example using Cozy’s injection pattern.

- The worker function only defines input/output + runs inference.
- Model selection + downloading is handled by the worker runtime via `[tool.cozy.models]`.

Config:

```toml
[tool.cozy.models]
flux2-klein-4b = "hf:black-forest-labs/FLUX.2-klein-4B"
```

Code uses:

```py
pipeline: Annotated[Flux2KleinPipeline, ModelRef(Src.DEPLOYMENT, "flux2-klein-4b")]
```
