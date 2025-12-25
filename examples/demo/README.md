Demo worker functions (minimal).

What this contains:

- Python module `demo` with several @worker_function examples.
- `pyproject.toml` with deps and `[tool.cozy]` deployment config.

Notes:

- Install deps with `uv sync` and generate `uv.lock` for reproducible builds.
- gen-builder reads `[tool.cozy].functions.modules` from `pyproject.toml`.
- Deploy with gen-builder by pointing the build source at this folder.
