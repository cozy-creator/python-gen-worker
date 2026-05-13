# medasr-transcribe

Speech-to-text using Google's MedASR (Wav2Vec2-CTC). Takes an audio `Asset`, returns the transcribed text.

## What it demonstrates
- **Model bindings** — `model` and `processor` are loaded from the module-level `Repo(...)` binding declared in `@inference_function(models={...})` and passed in as function arguments; the worker handles loading and caching.
- **`Asset` inputs** — the caller uploads audio via the file API; the worker materializes it and your function gets `payload.audio.local_path`.
- **GPU-bound inference** — the `endpoint.toml` build profile declares `accelerator = "cuda"` + `cuda = "12.8"`, and the `@inference_function`'s per-function `Resources` declares the VRAM floor.
- Mono-down + resample preprocessing in tenant code (the SDK doesn't dictate format).

## When to copy it
- Any audio/transcription endpoint with a HuggingFace `transformers` model.
- More generally: any inference endpoint that takes a binary blob (audio, image, video) as input via `Asset`.

## Files
- `src/medasr_transcribe/main.py` — handler + CTC decoding.
- `endpoint.toml` — declares the `medasr` model ref + hardware requirements.
