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
deployment = "my-worker"  # Default deployment ID

[tool.cozy.build]
gpu = true
torch = ">=2.9"
```

### Development

| Command | Purpose |
|---------|---------|
| `uv build` | Verify package builds correctly |
| `uv run mypy src/gen_worker` | Type checking |
| `uv run pytest` | Run tests |
| `rm -rf dist/ && uv build && uv publish` | Publish to PyPI |

### Deployment ID

The deployment ID identifies your worker in the orchestrator. It can be specified in two ways:

1. **In pyproject.toml** (recommended): Set `[tool.cozy].deployment` for a self-describing project
2. **In build request**: Pass `deployment` field when calling the gen-builder API

**Precedence**: Build request > pyproject.toml

**Validation rules**:
- 3-63 characters
- Lowercase alphanumeric and hyphens only
- Must start with a letter
- No consecutive hyphens or trailing hyphen

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

---

## Model Availability and Cache-Aware Routing

Workers report model availability to the orchestrator for intelligent job routing. The orchestrator prefers workers that already have the required model ready.

### Model States

| State | Location | Description |
|-------|----------|-------------|
| **Hot** | VRAM | Model loaded in GPU memory - instant inference |
| **Warm** | Disk | Model cached on local disk - fast load (seconds), no download |
| **Cold** | None | Model not present - requires download + load (minutes) |

### Heartbeat Reporting

Workers report two model lists in each heartbeat:
- `vram_models` - Models currently loaded in VRAM (hot)
- `disk_models` - Models cached on disk but not in VRAM (warm)

The orchestrator uses this to route jobs:
1. First preference: Workers with model in VRAM (instant)
2. Second preference: Workers with model on disk (fast load)
3. Last resort: Any capable worker (will need download)

### ModelCache

The `ModelCache` class tracks model states and provides availability checks:

```python
from gen_worker.model_cache import ModelCache, ModelLocation

cache = ModelCache(max_vram_gb=20.0)

# Register models
cache.mark_loaded_to_vram("model-a", pipeline, size_gb=8.0)
cache.mark_cached_to_disk("model-b", Path("/cache/model-b"), size_gb=10.0)

# Check availability
cache.is_in_vram("model-a")      # True
cache.is_on_disk("model-b")      # True
cache.are_models_available(["model-a", "model-b"])  # True (both ready)

# Get model lists for heartbeat
cache.get_vram_models()   # ["model-a"]
cache.get_disk_models()   # ["model-b"]
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKER_MAX_VRAM_GB` | Auto-detect | Maximum VRAM to use for models |
| `WORKER_VRAM_SAFETY_MARGIN_GB` | 3.5 | Reserved VRAM for working memory |
| `WORKER_MODEL_CACHE_DIR` | `/tmp/model_cache` | Directory for disk-cached models |
| `WORKER_MAX_CONCURRENT_DOWNLOADS` | 2 | Max parallel model downloads |

### Progressive Availability

Workers can accept jobs as soon as required models are ready. If a function needs model A and model B:
- Jobs requiring only model A can run while model B is still downloading
- The `are_models_available(model_ids)` method checks if all required models are ready

### Concurrent Inference (Thread Safety)

Diffusers schedulers maintain internal state that gets corrupted with concurrent access, causing `IndexError: index N is out of bounds`. The worker handles this automatically by creating a fresh scheduler for each request.

**How it works:**
- Heavy components (UNet, VAE, text encoders) are shared in VRAM (~10+ GB)
- Only the scheduler (~few KB) is recreated per-request
- Uses `Pipeline.from_pipe()` with a fresh scheduler from `scheduler.from_config()`

**For custom model managers:**
```python
from gen_worker.model_interface import ModelManagementInterface

class MyModelManager(ModelManagementInterface):
    def get_for_inference(self, model_id: str) -> Optional[Any]:
        """Return thread-safe pipeline with fresh scheduler."""
        base = self._pipelines.get(model_id)
        if not base or not hasattr(base, 'scheduler'):
            return base
        fresh_scheduler = base.scheduler.from_config(base.scheduler.config)
        return type(base).from_pipe(base, scheduler=fresh_scheduler)
```

References:
- [HuggingFace Server Guide](https://huggingface.co/docs/diffusers/using-diffusers/create_a_server)
- [GitHub Issue #3672](https://github.com/huggingface/diffusers/issues/3672)

---

API note:

- `output_format` is an orchestrator HTTP response preference (queue vs long-poll bytes/url) and does not change worker behavior; workers persist outputs as `Asset` refs via the Cozy hub file API.
