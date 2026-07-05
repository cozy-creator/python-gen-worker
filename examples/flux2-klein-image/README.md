# flux2-klein-image

Real GPU inference: FLUX.2-klein-4B (turbo, 4-8 step distilled) text-to-image.
Send `{"prompt": "a lighthouse at dusk"}`, get back a PNG.

## What it demonstrates

- `@endpoint(model=HF(...), resources=Resources(vram_gb=20))` — an HF binding
  narrowed with `files=` to skip a redundant root-level single-file
  checkpoint the source repo ships alongside its real diffusers-layout
  weights (transformer/text_encoder/vae subfolders), roughly halving the
  download.
- A real ~7.75GB (bf16) diffusion transformer + Qwen3-based text encoder,
  loaded via `Flux2KleinPipeline.from_pretrained` and injected into
  `setup()` fully constructed — dtype and device placement are worker
  policy, not endpoint code.
- Image output through `gen_worker.io.write_image(..., as_type=ImageAsset)` —
  PNGs exceed the inline threshold and ride the stored blob_ref path.

Ported from `inference-endpoints/flux.2-klein-4b`, which is written against
the retired pre-#368 authoring surface (`@inference`/`Case`/`HFRepo`/
`parametrize`); this is a minimal single-function (`generate_turbo`) port
onto the current `@endpoint`/`HF`/`Resources` v2 surface.

Driven end to end by the cozy e2e J6 GPU cloud journey (`task e2e-gpu-cloud`
in the e2e repo): real RunPod RTX 4090 pod -> HF download -> IN_VRAM
residency -> 4 real generations -> billing captured per request -> scale to
zero.
