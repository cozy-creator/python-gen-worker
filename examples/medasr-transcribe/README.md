# medasr-transcribe

Speech-to-text using Google's MedASR (Wav2Vec2-CTC). Takes an audio `Asset`, returns the transcribed text.

## What it demonstrates
- **`ModelRef` injection** — `model` and `processor` are loaded from the declared `[models].medasr` ref and passed in as function arguments; the worker handles loading and caching.
- **`Asset` inputs** — the caller uploads audio via the file API; the worker materializes it and your function gets `payload.audio.local_path`.
- **GPU-bound inference** with `[host.requirements] cuda_min = "12.8"` and a VRAM budget.
- Mono-down + resample preprocessing in tenant code (the SDK doesn't dictate format).

## When to copy it
- Any audio/transcription endpoint with a HuggingFace `transformers` model.
- More generally: any inference endpoint that takes a binary blob (audio, image, video) as input via `Asset`.

## Files
- `src/medasr_transcribe/main.py` — handler + CTC decoding.
- `endpoint.toml` — declares the `medasr` model ref + hardware requirements.
