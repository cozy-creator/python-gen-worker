Image generation worker example (diffusers + torch).

What this contains:

- Python module `image_gen` with @worker_function for SDXL-style inference.
- `pyproject.toml` with tenant deps and `[tool.cozy]` deployment config.

Notes:

- Install deps with `uv sync` and generate `uv.lock` for reproducible builds.
- gen-builder reads `[tool.cozy].functions.modules` from `pyproject.toml`.
- Model choice can be dynamic at runtime via request payload (model_ref).
- Deploy with gen-builder by pointing the build source at this folder.
