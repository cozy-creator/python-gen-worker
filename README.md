# gen-worker

A Python SDK for building serverless functions for AI inference. Write your function, declare required model refs, publish an endpoint release, and invoke it via Cozy's control plane.

## Tenant Worker Build Contract (Dockerfile-First)

When publishing a tenant worker, Cozy expects a **Dockerfile-first** project layout.

Build inputs MUST include:

- `endpoint.toml` (Cozy manifest; used at build/publish time)
- `Dockerfile` (builds the worker image)
- tenant code (`pyproject.toml`, `uv.lock`, `src/`, etc.)

The built image MUST:

1. Install `gen-worker` (so discovery + runtime can run).
2. Bake function discovery output (manifest) at build time:

```dockerfile
RUN mkdir -p /app/.tensorhub && python -m gen_worker.discover > /app/.tensorhub/endpoint.lock
```

3. Use the Cozy worker runtime as the ENTRYPOINT:

```dockerfile
ENTRYPOINT ["python", "-m", "gen_worker.entrypoint"]
```

Notes:

- `endpoint.toml` is **not required** to be present in the final image; it is a build-time input.
- The platform reads `/app/.tensorhub/endpoint.lock` from the built image and stores it in Cozy Hub DB for routing/invocation.

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
from gen_worker import RequestContext, worker_function

class Input(msgspec.Struct):
    prompt: str

class Output(msgspec.Struct):
    text: str

@worker_function()
def generate(ctx: RequestContext, payload: Input) -> Output:
    return Output(text=f"Hello, {payload.prompt}!")
```

## Features

- **Function discovery** - Automatic detection of `@worker_function` decorated functions
- **Schema generation** - Input/output schemas extracted from msgspec types
- **Model injection** - Dependency injection for ML models with caching
- **Streaming output** - Support for incremental/streaming responses
- **Progress reporting** - Built-in progress events via `RequestContext`
- **Perf metrics** - Best-effort per-run metrics emitted to gen-orchestrator (`metrics.*` worker events)
- **Trainer runtime mode** - SDK-native trainer loop via `WORKER_MODE=trainer`
- **File handling** - Upload/download assets via Cozy hub file API
- **Model caching** - LRU cache with VRAM/disk management and cache-aware routing

## Authoring Endpoints

Three endpoint types are supported — **inference**, **conversion**, and
**training**. See `docs/endpoint-authoring.md` for the full manual covering
`RequestContext`, model injection (fixed and payload-selected), streaming
output, file persistence, conversion reserved-name payloads
(`source`/`destination`/`outputs`), and the trainer class contract
(`setup`/`configure`/`prepare_batch`/`train_step`/`state_dict`/`load_state_dict`).

Training runs use trainer mode:

```bash
WORKER_MODE=trainer \
TRAINER_JOB_SPEC_PATH=/app/.cozy/trainer_job.json \
python -m gen_worker.entrypoint
```

## Dev HTTP Runner (Local Inference Without gen-orchestrator)

For local testing of a built worker image (without standing up gen-orchestrator),
run the dev HTTP runner and write outputs to a mounted local directory.

Container example:

```bash
docker run --rm --gpus all -p 8081:8081 \
  -v "$(pwd)/out:/outputs" \
  -e TENSORHUB_URL='http://host.docker.internal:7777' \
  <your-worker-image> \
  python -m gen_worker.testing.http_runner --listen 0.0.0.0:8081 --outputs /outputs
```

Prefetch a public model (example: SD1.5):

```bash
curl -sS -X POST 'http://localhost:8081/v1/models/prefetch' \
  -H 'content-type: application/json' \
  -d '{"models":[{"ref":"runwayml/stable-diffusion-v1-5","dtypes":["bf16","fp16"]}]}'
```

Invoke a function:

```bash
curl -sS -X POST 'http://localhost:8081/v1/request/generate' \
  -H 'content-type: application/json' \
  -d '{"payload": {"prompt": "a tiny robot watering a bonsai, macro photo"}}'
```

Outputs are written under `/outputs/jobs/<request_id>/outputs/...` (matching Cozy ref semantics).

## Configuration

### endpoint.toml

```toml
schema_version = 1
name = "my-worker"
main = "my_pkg.main"

[functions.generate]
batch_dimension = "items"  # optional

[models]
sdxl = { ref = "stabilityai/stable-diffusion-xl-base-1.0", attributes = { dtype = ["fp16", "bf16"] } }

[models.generate]
dreamshaper = { ref = "lykon/dreamshaper-xl-v2-turbo", attributes = { dtype = ["fp16", "bf16"] } }

[resources]
max_inflight_requests = 1
```

### Environment Variables

Orchestrator-injected (production contract):

| Variable | Default | Description |
|----------|---------|-------------|
| `WORKER_MODE` | `inference` | Runtime mode selector (`inference` or `trainer`) |
| `PUBLIC_ORCHESTRATOR_GRPC_ADDR` | - | Scheduler address workers should dial |
| `SCHEDULER_ADDRS` | - | Optional comma-separated LB seed addresses |
| `WORKER_JWT` | - | Worker-connect JWT (required; claims are authoritative) |

Local dev / advanced (not injected by orchestrator):

| Variable | Default | Description |
|----------|---------|-------------|
| `SCHEDULER_JWKS_URL` | - | Optional: verify WORKER_JWT locally against scheduler JWKS |
| `SCHEDULER_JWT_ISSUER` | - | Optional: expected `iss` when verifying WORKER_JWT locally |
| `SCHEDULER_JWT_AUDIENCE` | - | Optional: expected `aud` when verifying WORKER_JWT locally |
| `USE_TLS` | `false` | Local-dev knob for plaintext vs TLS gRPC; production typically terminates TLS upstream |
| `LB_ONLY_RETRIES` | `true` | Retry via configured LB endpoint(s) only; ignore direct owner redirect hints |
| `RECONNECT_DELAY` | `0.1` | Base reconnect backoff in seconds (exponential) |
| `RECONNECT_MAX_DELAY` | `1.0` | Reconnect backoff cap in seconds |
| `RECONNECT_JITTER_SECONDS` | `0.1` | Added jitter upper bound in seconds, capped by `RECONNECT_MAX_DELAY` |
| `MAX_RECONNECT_ATTEMPTS` | `0` | Max reconnect attempts (`0` = infinite retries) |
| `WORKER_MAX_CONCURRENCY` | - | Max concurrent request executions |
| `WORKER_MAX_INPUT_BYTES` | - | Max input payload size |
| `WORKER_MAX_OUTPUT_BYTES` | - | Max output payload size |
| `WORKER_MAX_UPLOAD_BYTES` | - | Max file upload size |
| `WORKER_MAX_VRAM_GB` | Auto | Maximum VRAM for models |
| `WORKER_VRAM_SAFETY_MARGIN_GB` | 3.5 | Reserved VRAM for working memory |
| `COZY_INFERENCE_MEMORY_MODE` | `auto` | Force a low-VRAM ladder step: `auto`, `off`, `vae_only`, `model_offload`, `group_offload`, `sequential` |
| `COZY_INFERENCE_VRAM_SAFETY_MARGIN_GB` | `2.0` | VRAM headroom (GB) the worker reserves for activations in the low-VRAM preflight |
| `COZY_INFERENCE_VAE_SLICE_VRAM_GB` | `10.0` | Total-VRAM threshold below which `auto` enables VAE slicing/tiling + attention slicing |
| `COZY_INFERENCE_MODEL_OFFLOAD_VRAM_GB` | `8.0` | Total-VRAM threshold below which `auto` enables `enable_model_cpu_offload()` |
| `COZY_INFERENCE_GROUP_OFFLOAD_VRAM_GB` | `6.0` | Total-VRAM threshold below which `auto` enables leaf-level group offload |
| `COZY_INFERENCE_AUTO_DISK_OFFLOAD` | `1` | Auto-enable disk offload when CPU RAM is tight; set to `0` to disable |
| `COZY_INFERENCE_DISK_OFFLOAD_RAM_GB` | `16.0` | Available-RAM threshold below which disk offload activates |
| `COZY_OFFLOAD_DIR` | `/tmp/cozy-offload` | Directory used by group offload when CPU RAM is insufficient |
| `TENSORHUB_CACHE_DIR` | `~/.cache/tensorhub` | TensorHub cache root; worker CAS defaults derive from this (`${TENSORHUB_CACHE_DIR}/cas/...`) |
| `WORKER_LOCAL_MODEL_CACHE_DIR` | `/tmp/tensorhub/local-model-cache` | Optional local (non-NFS) cache for snapshot localization |
| `WORKER_REGISTER_TIMEOUT_S` | `90` | Startup watchdog: fail fast if worker never registers with scheduler |
| `WORKER_WARN_MODEL_RESOLVE_S` | `30` | Emit `request.model_resolve.stuck` warning after this duration |
| `WORKER_WARN_MODEL_LOAD_S` | `60` | Emit `request.model_load.stuck` warning after this duration |
| `WORKER_WARN_INFERENCE_S` | `60` | Emit `request.inference.stuck` warning after this duration |
| `WORKER_MAX_CONCURRENT_DOWNLOADS` | 2 | Max parallel model downloads |
| `TENSORHUB_URL` | - | Cozy Hub base URL (used for public model requests and, if enabled, Cozy Hub API resolve) |
| `WORKER_ALLOW_TENSORHUB_API_RESOLVE` | `false` | Local dev only: allow the worker to call Cozy Hub resolve APIs |
| `TENSORHUB_TOKEN` | - | Cozy Hub bearer token (optional; enables ingest-if-missing for public models, if Cozy Hub requires auth) |
| `TRAINER_JOB_SPEC_PATH` | `/app/.cozy/trainer_job.json` | Trainer-mode JSON job manifest path |
| `TRAINER_PLUGIN` | - | Trainer plugin import (`module:symbol`); optional if provided in job JSON |
| `TRAINER_CHECKPOINTS_DIR` | `/tmp/training/checkpoints` | Local checkpoint output directory in trainer mode |
| `TRAINER_SAMPLES_DIR` | `/tmp/training/samples` | Local sample output directory in trainer mode |
| `TRAINER_EVENTS_PATH` | - | Optional line-delimited JSON lifecycle event log for trainer mode |

## Robust low-VRAM inference

When a pipeline is larger than the available VRAM on the host, the worker
does not crash with `torch.cuda.OutOfMemoryError`. It applies a progressive
offload ladder:

    off          no optimizations
    vae_only     VAE slicing + VAE tiling (+ attention slicing where available)
    model_offload  enable_model_cpu_offload()         (~10% slower)
    group_offload  leaf-level group offload w/ CUDA streams (~25% slower)
    sequential   enable_sequential_cpu_offload()     (~50%+ slower)

### Worker baseline (always on)

After a diffusers pipeline is injected via a `ModelRef` annotation, the
worker in `_inject_pipeline()` runs:

1. A VRAM preflight — if the estimated model size does not fit in free VRAM
   (minus a safety margin), it skips `.to("cuda")` and installs
   `enable_model_cpu_offload()` (or leaf-level group offload on very-small
   GPUs) directly on the CPU-resident pipeline.
2. `.to(device)` wrapped in up to three attempts. On
   `torch.cuda.OutOfMemoryError` it flushes memory, escalates the pipeline
   one ladder step (model → group → sequential), and retries.
3. A baseline `apply_low_vram_config(pipeline, mode="auto")` pass that turns
   on VAE tiling/slicing + attention slicing. Safe no-ops on pipelines that
   don't expose those methods.

Around the tenant's inference call, the worker additionally catches
`torch.cuda.OutOfMemoryError` (for single-output functions), escalates each
injected pipeline one step up the ladder, and retries the call up to twice.
Each transition emits a `low_vram_mode_applied` or `inference.oom_retry`
worker event.

### Endpoint-authoring helper

Endpoints that want explicit control over the mode can call
`gen_worker.apply_low_vram_config(pipeline, mode=...)`. The default
`mode="auto"` uses `COZY_INFERENCE_MEMORY_MODE` when set and otherwise picks
the least-aggressive ladder step that fits the total VRAM of the host:

```python
from gen_worker import apply_low_vram_config, with_oom_retry

with _lock_for_pipeline(pipeline):
    apply_low_vram_config(pipeline, mode="sequential", logger=logger)
    result = with_oom_retry(pipeline, prompt="...", num_inference_steps=8, pipelines=[pipeline])
```

### Disk offload (tight CPU RAM)

When `COZY_INFERENCE_AUTO_DISK_OFFLOAD=1` (default) and available RAM is
below `COZY_INFERENCE_DISK_OFFLOAD_RAM_GB` (default 16 GB), group offload
stores offloaded weights on disk at `COZY_OFFLOAD_DIR`
(default `/tmp/cozy-offload`) instead of CPU RAM. This is the only path
that handles FLUX-class models on 8 GB-VRAM + 16 GB-RAM hosts, at the cost
of much higher inference latency.

### Operator observability

Worker events emitted by the ladder:

- `low_vram_mode_applied` — payload includes `model_id`, `stage`
  (`preflight` | `baseline` | `oom_escalation`), `requested_mode`, and the
  booleans for each enabler that was applied.
- `inference.oom_retry` — payload includes `function_name` and `attempt`.

Operators diagnosing "why is my endpoint slow" on undersized hardware should
grep for these two event types.

## Metrics

The worker can emit best-effort performance/debug metrics to gen-orchestrator via `WorkerEvent` messages.

See the **Observability** section in `docs/endpoint-authoring.md` for the event catalog (request lifecycle, startup phases, per-run `metrics.*`, and cache inventory).

### Model Download Behavior

Model refs are plain lower-case strings:
- `owner/repo`
- `owner/repo:tag`
- `owner/repo:tag#flavor`
- `owner/repo@blake3:<digest>`
- `owner/repo@blake3:<digest>#flavor`

Tags are mutable pointers that resolve to published checkpoints. Flavors select a concrete artifact within that checkpoint, such as `bf16`, `fp8`, or `int4`.

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
docker build -t medasr-worker -f Dockerfile examples/medasr-transcribe

# Run
docker run \
  -e PUBLIC_ORCHESTRATOR_GRPC_ADDR=orchestrator:8080 \
  -e WORKER_JWT='<worker-connect-jwt>' \
  medasr-worker
```

Canonical local dev build args (GPU, CUDA 12.6, torch 2.11.x, Python 3.12):

```bash
cd ~/cozy/python-gen-worker

docker build \
  --build-arg PYTHON_VERSION=3.12 \
  --build-arg UV_TORCH_BACKEND=cu126 \
  --build-arg TORCH_SPEC='~=2.11.0' \
  -f Dockerfile \
  -t my-worker:dev \
  examples/medasr-transcribe
```

Optional build args:

```bash
docker build \
  --build-arg PYTHON_VERSION=3.12 \
  --build-arg UV_TORCH_BACKEND=cu128 \
  --build-arg TORCH_SPEC=">=2.9,<3" \
  -t my-worker -f Dockerfile examples/medasr-transcribe
```

### Build Base

Worker images build directly from a Python+uv base image:

- `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`

PyTorch/CUDA dependencies are installed as part of your worker's dependency set during image build.

## Publish/Promote Lifecycle

Control-plane behavior (tensorhub + orchestrator):

- Every publish creates a new immutable internal `release_id`.
- End users invoke functions by `owner/endpoint/function` (default `prod`) or `owner/endpoint/function:tag`.
- `endpoint` is derived from `endpoint.toml` `name` and normalized to a URL-safe slug.
- `function` names are derived from Python `@worker_function` names and normalized to URL-safe slugs (for example, `medasr_transcribe` -> `medasr-transcribe`).
- Publishing does not move traffic by default.
- Promoting a function tag moves traffic to that release.
- Rollback is just retargeting the tag to an older release.

## Model Cache

Workers report model availability for intelligent job routing:

| State | Location | Latency |
|-------|----------|---------|
| Hot | VRAM | Instant |
| Warm | Disk | Seconds |
| Cold | None | Minutes (download required) |

## Dev Testing (Mock Orchestrator)

For local end-to-end tests without standing up `gen-orchestrator`, use the one-off mock orchestrator invoke command (curl-like workflow). It starts a temporary scheduler, waits for a worker to connect, sends one `JobExecutionRequest`, prints the result, and exits.

Start your worker container first:

```bash
docker run --rm \
  --add-host=host.docker.internal:host-gateway \
  -e PUBLIC_ORCHESTRATOR_GRPC_ADDR=host.docker.internal:8080 \
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
def process(ctx: RequestContext, payload: Input) -> Output:
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

### Worker Wire Protocol

The worker advertises a protocol `MAJOR.MINOR` in `WorkerRegistration` (`protocol_major`, `protocol_minor`).

- Current runtime constants live in `src/gen_worker/wire_protocol.py`.
- Orchestrator compatibility policy/ranges are documented in `../gen-orchestrator/docs/worker_wire_protocol.md`.

## License

MIT
