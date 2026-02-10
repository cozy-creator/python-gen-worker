# gen-worker

A Python SDK for building serverless api-endpoints for AI inference. Just write your custom function, create a manifest specifying what model-weights you need from Cozy-Hub, and then deploy it! We take care of the rest!

## Tenant Worker Build Contract (Dockerfile-First)

When publishing a tenant worker, Cozy expects a **Dockerfile-first** project layout.

Build inputs MUST include:

- `cozy.toml` (Cozy manifest; used at build/publish time)
- `Dockerfile` (builds the worker image)
- tenant code (`pyproject.toml`, `uv.lock`, `src/`, etc.)

The built image MUST:

1. Install `gen-worker` (so discovery + runtime can run).
2. Bake endpoint discovery output at build time:

```dockerfile
RUN mkdir -p /app/.cozy && python -m gen_worker.discover > /app/.cozy/manifest.json
```

3. Use the Cozy worker runtime as the ENTRYPOINT:

```dockerfile
ENTRYPOINT ["python", "-m", "gen_worker.entrypoint"]
```

Notes:

- `cozy.toml` is **not required** to be present in the final image; it is a build-time input.
- The platform reads `/app/.cozy/manifest.json` from the built image and stores it in Cozy Hub DB for routing/invocation.

## Installation

Start a python project, and then run:

```bash
uv add gen-worker
```

With PyTorch support:

```bash
uv add gen-worker[torch]
```

## Quick Start

```python
import msgspec
from gen_worker import ActionContext, worker_function

class Input(msgspec.Struct):
    prompt: str

class Output(msgspec.Struct):
    text: str

@worker_function()
def generate(ctx: ActionContext, payload: Input) -> Output:
    return Output(text=f"Hello, {payload.prompt}!")
```

## Features

- **Function discovery** - Automatic detection of `@worker_function` decorated functions
- **Schema generation** - Input/output schemas extracted from msgspec types
- **Model injection** - Dependency injection for ML models with caching
- **Streaming output** - Support for incremental/streaming responses
- **Progress reporting** - Built-in progress events via `ActionContext`
- **Perf metrics** - Best-effort per-run metrics emitted to gen-orchestrator (`metrics.*` worker events)
- **File handling** - Upload/download assets via Cozy hub file API
- **Model caching** - LRU cache with VRAM/disk management and cache-aware routing

## Usage

### Basic Function

```python
import msgspec
from gen_worker import ActionContext, worker_function

class Input(msgspec.Struct):
    prompt: str

class Output(msgspec.Struct):
    result: str

@worker_function()
def my_function(ctx: ActionContext, payload: Input) -> Output:
    return Output(result=f"Processed: {payload.prompt}")
```

### Streaming Output

```python
from typing import Iterator

class Delta(msgspec.Struct):
    chunk: str

@worker_function()
def stream(ctx: ActionContext, payload: Input) -> Iterator[Delta]:
    for word in payload.prompt.split():
        if ctx.is_canceled():
            raise InterruptedError("canceled")
        yield Delta(chunk=word)
```

### Model Injection

```python
from typing import Annotated
from diffusers import DiffusionPipeline
from gen_worker.injection import ModelRef, ModelRefSource as Src

@worker_function()
def generate(
    ctx: ActionContext,
    pipe: Annotated[DiffusionPipeline, ModelRef(Src.FIXED, "my-model")],
    payload: Input,
) -> Output:
    # Use the injected pipeline (loaded/cached by the worker's model manager).
    return Output(result="done")
```

### Payload-Selected Model (Short Key)

If you want the client payload to choose which repo to run, declare a short-key
mapping in `cozy.toml` and use `ModelRef(PAYLOAD, ...)`:

```toml
[models]
sd15 = "hf:stable-diffusion-v1-5/stable-diffusion-v1-5"
flux = "hf:black-forest-labs/FLUX.2-klein-4B"
```

```python
from typing import Annotated
import msgspec
from diffusers import DiffusionPipeline
from gen_worker import ActionContext, worker_function
from gen_worker.injection import ModelRef, ModelRefSource as Src

class Input(msgspec.Struct):
    prompt: str
    model: str  # must be one of: "sd15" | "flux"

@worker_function()
def generate(
    ctx: ActionContext,
    pipe: Annotated[DiffusionPipeline, ModelRef(Src.PAYLOAD, "model")],
    payload: Input,
):
    ...
```

Note: by default the worker requires payload model selection to use a known
short-key from `[models]` (cozy.toml) (it will not accept arbitrary repo refs in
the payload).

### Saving Files

```python
@worker_function()
def process(ctx: ActionContext, payload: Input) -> Output:
    # Save bytes and get asset reference
    asset = ctx.save_bytes(f"runs/{ctx.run_id}/outputs/output.png", image_bytes)
    return Output(result=asset.ref)
```

## Dev HTTP Runner (Local Inference Without gen-orchestrator)

For local testing of a built worker image (without standing up gen-orchestrator),
run the dev HTTP runner and write outputs to a mounted local directory.

Container example:

```bash
docker run --rm --gpus all -p 8081:8081 \
  -v "$(pwd)/out:/outputs" \
  <your-worker-image> \
  python -m gen_worker.testing.http_runner --listen 0.0.0.0:8081 --outputs /outputs
```

Invoke an endpoint:

```bash
curl -sS -X POST 'http://localhost:8081/v1/run/generate' \
  -H 'content-type: application/json' \
  -d '{"payload": {"prompt": "a tiny robot watering a bonsai, macro photo"}}'
```

Outputs are written under `/outputs/runs/<run_id>/outputs/...` (matching Cozy ref semantics).

## Configuration

### cozy.toml

```toml
schema_version = 1
name = "my-worker"
main = "my_pkg.main"
gen_worker = ">=0.2.0,<0.3.0"

[models]
sdxl = { ref = "hf:stabilityai/stable-diffusion-xl-base-1.0", dtypes = ["fp16","bf16"] }
```

`[models]` entries support two forms:

- String form (defaults to `dtypes=["fp16","bf16"]`):
  - `sd15 = "hf:stable-diffusion-v1-5/stable-diffusion-v1-5"`
- Table form:
  - `flux_fp8 = { ref = "hf:black-forest-labs/FLUX.2-klein-4B", dtypes = ["fp8"] }`

### Environment Variables

Orchestrator-injected (production contract):

| Variable | Default | Description |
|----------|---------|-------------|
| `SCHEDULER_ADDR` | - | Scheduler address workers should dial |
| `SCHEDULER_ADDRS` | - | Optional comma-separated seed addresses for leader discovery |
| `WORKER_JWT` | - | Worker-connect JWT (required; claims are authoritative) |

Local dev / advanced (not injected by orchestrator):

| Variable | Default | Description |
|----------|---------|-------------|
| `SCHEDULER_JWKS_URL` | - | Optional: verify WORKER_JWT locally against scheduler JWKS |
| `SCHEDULER_JWT_ISSUER` | - | Optional: expected `iss` when verifying WORKER_JWT locally |
| `SCHEDULER_JWT_AUDIENCE` | - | Optional: expected `aud` when verifying WORKER_JWT locally |
| `USE_TLS` | `false` | Local-dev knob for plaintext vs TLS gRPC; production typically terminates TLS upstream |
| `WORKER_MAX_CONCURRENCY` | - | Max concurrent task executions |
| `WORKER_MAX_INPUT_BYTES` | - | Max input payload size |
| `WORKER_MAX_OUTPUT_BYTES` | - | Max output payload size |
| `WORKER_MAX_UPLOAD_BYTES` | - | Max file upload size |
| `WORKER_MAX_VRAM_GB` | Auto | Maximum VRAM for models |
| `WORKER_VRAM_SAFETY_MARGIN_GB` | 3.5 | Reserved VRAM for working memory |
| `WORKER_MODEL_CACHE_DIR` | `/tmp/model_cache` | Disk cache directory |
| `WORKER_MAX_CONCURRENT_DOWNLOADS` | 2 | Max parallel model downloads |
| `COZY_HUB_URL` | - | Local dev only: Cozy Hub base URL (used only if you enable Cozy Hub API resolve) |
| `WORKER_ALLOW_COZY_HUB_API_RESOLVE` | `false` | Local dev only: allow the worker to call Cozy Hub resolve APIs |
| `COZY_HUB_TOKEN` | - | Local dev only: Cozy Hub bearer token (only used when `WORKER_ALLOW_COZY_HUB_API_RESOLVE=1`) |
| `HF_TOKEN` | - | Hugging Face token (for private `hf:` refs) |

## Metrics

The worker can emit best-effort performance/debug metrics to gen-orchestrator via `WorkerEvent` messages.

See `docs/metrics.md`.

### Hugging Face (`hf:`) download behavior

By default, `hf:` model refs **do not download the full repo**. The worker uses `huggingface_hub.snapshot_download(allow_patterns=...)` to avoid pulling huge legacy weights.

Defaults:
- Download only what a diffusers pipeline needs (derived from `model_index.json`).
- Skip `safety_checker` and `feature_extractor` by default.
- Download only reduced-precision **safetensors** weights (`fp16`/`bf16`); never download `.ckpt` or `.bin` by default.
- For sharded safetensors, also download the `*.safetensors.index.json` and the referenced shard files.

Overrides:
- `COZY_HF_COMPONENTS="unet,vae,text_encoder,tokenizer,scheduler"`: hard override component list.
- `COZY_HF_INCLUDE_OPTIONAL_COMPONENTS=1`: include components like `safety_checker` / `feature_extractor` if present.
- `COZY_HF_WEIGHT_PRECISIONS="fp16,bf16"`: change which weight suffixes are accepted (add `fp32` only if you really need it).
- `COZY_HF_ALLOW_ROOT_JSON=1`: allow additional small root `*.json` files (some repos need extra root config).
- `COZY_HF_FULL_REPO_DOWNLOAD=1`: disable filtering and download the entire repo (not recommended; can be 10s of GB).

### Cozy Hub (`cozy:`) download behavior

Cozy snapshot/object file downloads are written to `*.part` and then atomically renamed on success. If a `*.part` file exists from a previous interrupted download, the worker attempts to resume it using HTTP `Range` requests (if supported by the presigned object-store URL), and falls back to a full re-download if Range is not supported.

## Docker Deployment

### Project Structure

```
my-worker/
├── pyproject.toml
├── uv.lock
└── src/
    └── my_module/
        └── main.py
```

### Local Dev Build (Using Root `Dockerfile`)

For production, use the `cozyctl` CLI to build and deploy worker-images to our network. But for local testing, you can build images using our provided `Dockerfile`:

```bash
# Build an example using the same root Dockerfile
docker build -t sd15-worker -f Dockerfile examples/sd15

# Run
docker run \
  -e SCHEDULER_ADDR=orchestrator:8080 \
  -e WORKER_JWT='<worker-connect-jwt>' \
  sd15-worker
```

Canonical local dev build args (GPU, CUDA 12.6, torch 2.10.x, Python 3.12):

```bash
cd ~/cozy/python-gen-worker

docker build \
  --build-arg PYTHON_VERSION=3.12 \
  --build-arg UV_TORCH_BACKEND=cu126 \
  --build-arg TORCH_SPEC='~=2.10.0' \
  -f Dockerfile \
  -t my-worker:dev \
  examples/sd15
```

Optional build args:

```bash
docker build \
  --build-arg PYTHON_VERSION=3.12 \
  --build-arg UV_TORCH_BACKEND=cu128 \
  --build-arg TORCH_SPEC=">=2.9,<3" \
  -t my-worker -f Dockerfile examples/sd15
```

### Build Base

Worker images build directly from a Python+uv base image:

- `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`

PyTorch/CUDA dependencies are installed as part of your worker's dependency set during image build.

## Publish/Promote Lifecycle

Control-plane behavior (cozy-hub + orchestrator):

- Every publish creates a new immutable internal `release_id`.
- End users invoke endpoints by `tenant/project/endpoint` (default `prod`) or `tenant/project/endpoint@tag`.
- `project` is derived from `pyproject.toml` `[project].name` and normalized to a URL-safe slug.
- Endpoint names are derived from worker function names and slugified to URL-safe form (for example, `medasr_transcribe` -> `medasr-transcribe`).
- Publishing does not move traffic by default.
- Promoting an endpoint/tag moves traffic to that release.
- Rollback is just retargeting the tag to an older release.

## Model Cache

Workers report model availability for intelligent job routing:

| State | Location | Latency |
|-------|----------|---------|
| Hot | VRAM | Instant |
| Warm | Disk | Seconds |
| Cold | None | Minutes (download required) |

## Dev Testing (Mock Orchestrator)

For local end-to-end tests without standing up `gen-orchestrator`, use the one-off mock orchestrator invoke command (curl-like workflow). It starts a temporary scheduler, waits for a worker to connect, sends one `TaskExecutionRequest`, prints the result, and exits.

Start your worker container first:

```bash
docker run --rm \
  --add-host=host.docker.internal:host-gateway \
  -e SCHEDULER_ADDR=host.docker.internal:8080 \
  -e WORKER_JWT='dev-worker-jwt' \
  <your-worker-image>
```

In another terminal, send one request:

```bash
python -m gen_worker.testing.mock_orchestrator \
  --listen 0.0.0.0:8080 \
  --run hello \
  --payload-json '{"name":"world"}'
```

Run the command again with a different payload whenever you want to send another request.

```python
from gen_worker.model_cache import ModelCache

cache = ModelCache(max_vram_gb=20.0)
cache.mark_loaded_to_vram("model-a", pipeline, size_gb=8.0)
cache.is_in_vram("model-a")  # True
cache.get_vram_models()      # ["model-a"]
```

## Error Handling

```python
from gen_worker.errors import RetryableError, ValidationError, FatalError

@worker_function()
def process(ctx: ActionContext, payload: Input) -> Output:
    if not payload.prompt:
        raise ValidationError("prompt is required")  # 400, no retry

    try:
        result = call_external_api()
    except TimeoutError:
        raise RetryableError("API timeout")  # Will be retried

    return Output(result=result)
```

## Development

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests
uv run pytest

# Type checking
uv run mypy src/gen_worker

# Build
uv build
```

### Regenerating Protobuf Stubs

Requires `gen-orchestrator` as a sibling repo:

```bash
uv sync --extra dev
python -m grpc_tools.protoc -I../gen-orchestrator/proto --python_out=src/gen_worker/pb --grpc_python_out=src/gen_worker/pb ../gen-orchestrator/proto/*.proto
```

## License

MIT
