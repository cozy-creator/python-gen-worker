Orchestrator examples migrated to gen-worker functions.

Contents:

- `src/examples` with @worker_function-decorated examples.
- `pyproject.toml` with deps.
- `[tool.cozy]` deployment config in `pyproject.toml`.

Notes:

- Install deps with `uv sync` (or `pip install -e .`).
- gen-builder reads `[tool.cozy]` from `pyproject.toml` to discover modules and base image.
