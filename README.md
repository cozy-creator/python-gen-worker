This is a python package, called gen_worker, which provides the worker runtime SDK:

- Orchestrator gRPC client + job loop
- Function discovery via @worker_function
- ActionContext + errors + progress events
- Model downloading from the Cozy hub (async + retries + progress)
- Output saving via the Cozy hub file API (ctx.save_bytes/ctx.save_file -> Asset refs)

Torch-based model memory management is optional and installed via extras.

---

Files in `python-worker/src/gen_worker/pb` are generated from the `.proto` definitions in `gen-orchestrator/proto`.

Assuming `gen-orchestrator` is checked out as a sibling repo, regenerate stubs with:

`task -d python-worker proto`

This runs `uv sync --extra dev` and then `grpc_tools.protoc` against `../gen-orchestrator/proto`.

Install modes:

- Core only: `gen-worker`
- Torch runtime add-on: `gen-worker[torch]` (torch + torchvision + torchaudio + safetensors + flashpack + numpy)

Example tenant projects live in `./examples`. They use:

- `pyproject.toml` + `uv.lock` for dependencies (no requirements.txt)
- `[tool.cozy]` in `pyproject.toml` for deployment config (functions.modules, runtime.base_image, etc.)

Dependency policy:

- Require `pyproject.toml` and/or `uv.lock`
- Do not use `requirements.txt`
- Put Cozy deployment config in `pyproject.toml` under `[tool.cozy]`

Example:

```toml
[tool.cozy]
functions.modules = ["functions"]

[tool.cozy.runtime]
base_image = "ghcr.io/cozy/python-worker:cuda12.1-torch2.6"
```

Function signature:

```python
from typing import Annotated, Iterator

import msgspec

from gen_worker import ActionContext, ResourceRequirements, worker_function
from gen_worker.injection import ModelArtifacts, ModelRef, ModelRefSource as Src

class Input(msgspec.Struct):
    prompt: str
    model_key: str = "default"

class Output(msgspec.Struct):
    text: str

@worker_function(ResourceRequirements())
def run(
    ctx: ActionContext,
    # The worker injects cached handles based on the ModelRef.
    # ModelRef(Src.DEPLOYMENT, ...) is fixed by deployment configuration (or a literal model id).
    artifacts: Annotated[ModelArtifacts, ModelRef(Src.DEPLOYMENT, "google/functiongemma-270m-it")],
    payload: Input,
) -> Output:
    return Output(text=f"prompt={payload.prompt} model_root={artifacts.root_dir}")

class Delta(msgspec.Struct):
    delta: str

@worker_function(ResourceRequirements())
def run_incremental(ctx: ActionContext, payload: Input) -> Iterator[Delta]:
    for ch in payload.prompt:
        if ctx.is_canceled():
            raise InterruptedError("canceled")
        yield Delta(delta=ch)
```

Dynamic checkpoints:

- Prefer deployment-defined allowlists. Requests pick a **key/label** from the payload
  (e.g. `payload.model_key`), and the worker resolves it via a deployment-provided mapping.
- Use `ModelRef(Src.PAYLOAD, "model_key")` for this pattern (the payload value is a key, not a raw HF id).

Build contract (gen-builder):

- Tenant code + `pyproject.toml`/`uv.lock` are packaged together
- gen-builder layers tenant code + deps on top of a python-worker base image
- gen-orchestrator deploys the resulting worker image

---

## Manual builds (without gen-builder)

You can build worker images directly using Docker, without gen-builder.

### 1. Project structure

```
my-worker/
├── pyproject.toml      # dependencies + [tool.cozy] config
├── uv.lock             # lockfile (recommended)
└── src/
    └── my_module/
        └── __init__.py # contains @worker_function decorated functions
```

### 2. Copy the Dockerfile template

Copy `Dockerfile.template` from this repo to your project as `Dockerfile`:

```bash
cp /path/to/python-worker/Dockerfile.template ./Dockerfile
```

Or write your own:

```dockerfile
ARG BASE_IMAGE=cozycreator/python-worker:cuda12.8-torch2.9
FROM ${BASE_IMAGE}

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir uv
RUN if [ -f /app/uv.lock ]; then uv sync --frozen --no-dev; else uv sync --no-dev; fi

# Generate function manifest at build time
RUN mkdir -p /app/.cozy && python -m gen_worker.discover > /app/.cozy/manifest.json

ENTRYPOINT ["python", "-m", "gen_worker.entrypoint"]
```

### 3. Build

```bash
# CPU only
docker build -t my-worker --build-arg BASE_IMAGE=cozycreator/python-worker:cpu-torch2.9 .

# CUDA 12.8 (default)
docker build -t my-worker .

# CUDA 13.0
docker build -t my-worker --build-arg BASE_IMAGE=cozycreator/python-worker:cuda13-torch2.9 .
```

### 4. Run

```bash
docker run -e ORCHESTRATOR_URL=http://orchestrator:8080 my-worker
```

The worker will:
1. Read the manifest from `/app/.cozy/manifest.json`
2. Self-register with the orchestrator
3. Start listening for tasks

### Available base images

| Image | GPU | CUDA | PyTorch |
|-------|-----|------|---------|
| `cozycreator/python-worker:cpu-torch2.9` | No | - | 2.9.1 |
| `cozycreator/python-worker:cuda12.6-torch2.9` | Yes | 12.6 | 2.9.1 |
| `cozycreator/python-worker:cuda12.8-torch2.9` | Yes | 12.8 | 2.9.1 |
| `cozycreator/python-worker:cuda13-torch2.9` | Yes | 13.0 | 2.9.1 |

### What happens automatically

- **Function discovery**: `gen_worker.discover` scans for `@worker_function` decorators
- **Manifest generation**: Input/output schemas extracted from msgspec types
- **Self-registration**: Worker registers its functions with orchestrator on startup

No gen-builder required for local development or custom CI pipelines.

---

Env hints:

- `SCHEDULER_ADDR` sets the primary scheduler address.
- `SCHEDULER_ADDRS` (comma-separated) provides seed addresses for leader discovery.
- `WORKER_JWT` is accepted as the auth token if `AUTH_TOKEN` is not set.
- `SCHEDULER_JWKS_URL` enables verification of `WORKER_JWT` before connecting.
- JWT verification uses RSA and requires PyJWT crypto support (installed by default via `PyJWT[crypto]`).
- `WORKER_MAX_INPUT_BYTES`, `WORKER_MAX_OUTPUT_BYTES`, `WORKER_MAX_UPLOAD_BYTES` cap payload sizes.
- `WORKER_MAX_CONCURRENCY` limits concurrent runs; `ResourceRequirements(max_concurrency=...)` limits per-function.
- `COZY_HUB_URL` base URL for Cozy hub downloads (used by core downloader).
- `COZY_HUB_TOKEN` optional bearer token for Cozy hub downloads.
- `MODEL_MANAGER_CLASS` optional ModelManager plugin (module:Class) loaded at startup.

Error hints:

- Use `gen_worker.errors.RetryableError` in worker functions to flag retryable failures.

API note:

- `output_format` is an orchestrator HTTP response preference (queue vs long-poll bytes/url) and does not change worker behavior; workers persist outputs as `Asset` refs via the Cozy hub file API.
