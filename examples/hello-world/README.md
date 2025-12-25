Hello-world worker example.

Contents:

- `src/hello_world` with a single @worker_function.
- `pyproject.toml` with deps.
- `[tool.cozy]` deployment config in `pyproject.toml`.

Notes:

- Install deps with `uv sync` (or `pip install -e .`).
- gen-builder reads `[tool.cozy]` from `pyproject.toml` to discover modules and base image.
