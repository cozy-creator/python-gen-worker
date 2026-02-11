Qwen Image 2512 worker example (diffusers + torch).

What this contains:

- Python module `qwen_image_2512` with `@worker_function` for Qwen Image inference.
- `pyproject.toml` + `uv.lock` with tenant deps.
- `cozy.toml` with Cozy build-time metadata (name/main/gen_worker, optional models/resources).
- `Dockerfile` that installs deps and bakes `/app/.cozy/manifest.json` via `python -m gen_worker.discover`.

Notes:

- Install deps with `uv sync` and generate `uv.lock` for reproducible builds.
- This endpoint uses the fixed model mapping key `qwen_image` from `cozy.toml`.
- Deploy by publishing this folder through Cozy Hub `publish-dir`.

Defaults:

- Uses Qwen Image guidance style (`true_cfg_scale` + `negative_prompt`).
- `preset=quality` defaults to `50` steps and `true_cfg_scale=4.0`.
- `preset=balanced` defaults to `40` steps and `true_cfg_scale=3.5`.
- `preset=fast` defaults to `28` steps and `true_cfg_scale=3.0`.
