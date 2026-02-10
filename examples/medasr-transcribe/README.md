MedASR transcription worker example.

Contents:

- `src/medasr_transcribe` with a single `@worker_function` (`medasr_transcribe`).
- `pyproject.toml` with deps.
- `cozy.toml` with Cozy build-time metadata (name/main/gen_worker, optional models/resources).
- `Dockerfile` that installs deps and bakes `/app/.cozy/manifest.json` via `python -m gen_worker.discover`.

Notes:

- This model is gated on Hugging Face (`google/medasr`). You will need to:
  - accept the license/terms on the model page, and
  - set `HUGGINGFACE_HUB_TOKEN` (or `HF_TOKEN`) to a token with access.
