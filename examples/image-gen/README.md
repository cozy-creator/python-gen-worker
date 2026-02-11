Image generation worker example (diffusers + torch).

What this contains:

- Python module `image_gen` with @worker_function for SDXL-style inference.
- `pyproject.toml` + `uv.lock` with tenant deps.
- `cozy.toml` with Cozy build-time metadata (name/main/gen_worker, optional models/resources).
- `Dockerfile` that installs deps and bakes `/app/.cozy/manifest.json` via `python -m gen_worker.discover`.

Notes:

- Install deps with `uv sync` and generate `uv.lock` for reproducible builds.
- Model choice can be dynamic at runtime via request payload (model_ref).
- Deploy by publishing this folder to Cozy Hub (Dockerfile-first build inputs).

Defaults:

- `num_inference_steps` defaults to `25` and is clamped to `[20, 50]` (rounded).
- `guidance_scale` (CFG) defaults to `7.0` and is configurable via the payload.
