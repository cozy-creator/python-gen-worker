# sd15-image

Real GPU inference: Stable Diffusion 1.5 text-to-image. Send
`{"prompt": "a lighthouse at dusk"}`, get back a WebP.

## What it demonstrates

- `@endpoint(model=HF(...), resources=Resources(vram_gb=6))` — a
  HuggingFace model binding (fp16 variant only, via `files=` allow
  patterns) downloaded through `ensure_local` and injected into
  `setup()` as a constructed `StableDiffusionPipeline` (dtype and
  device placement are worker policy, not endpoint code).
- Image outputs through `gen_worker.io.write_image` — WebP is the
  framework default; rendered images exceed the inline threshold, so they
  ride the stored blob_ref path.
- GPU jobs serialize on the worker's GPU semaphore; no CUDA OOM under
  concurrent submits.

Driven end to end by the cozy e2e J4 GPU journey (`task e2e-gpu` in the
e2e repo): cold HF download -> IN_VRAM residency -> 8 real generations ->
billing captured per request.
