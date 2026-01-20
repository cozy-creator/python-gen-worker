# Hello-world worker example

Minimal example of a Cozy worker function.

## Contents

- `src/hello_world` - module with a single `@worker_function`
- `pyproject.toml` - dependencies and `[tool.cozy]` deployment config

## Configuration

The `[tool.cozy]` section in `pyproject.toml` configures how gen-builder builds your worker:

```toml
[project]
requires-python = ">=3.12"  # Python version constraint (currently only 3.12 supported)

[tool.cozy.build]
# Option 1: Simple GPU selection (recommended)
gpu = true           # Build GPU variant
torch = ">=2.9"      # Optional torch version constraint

# Option 2: GPU with specific CUDA version
# gpu = true
# cuda = ">=12.6"    # Optional: filter CUDA versions

# Option 3: Specify exact base image (overrides all constraints)
# base_image = "cozycreator/python-worker:cuda12.8-torch2.9"
```

### Available build options

| Option | Description | Example |
|--------|-------------|---------|
| `gpu` | Enable GPU support | `true` or `false` |
| `cuda` | CUDA version constraint (advanced) | `">=12.6"`, `">=12.8"` |
| `torch` | PyTorch version constraint | `">=2.9"` |
| `base_image` | Exact base image (overrides all) | `"cozycreator/python-worker:cuda13-torch2.9"` |

### Available base images

- `cozycreator/python-worker:cuda12.6-torch2.9`
- `cozycreator/python-worker:cuda12.8-torch2.9`
- `cozycreator/python-worker:cuda13-torch2.9`
- `cozycreator/python-worker:cpu-torch2.9`

## Local development

```bash
# Install dependencies
uv sync

# Run locally (requires gen-worker to be running)
python -m hello_world
```
