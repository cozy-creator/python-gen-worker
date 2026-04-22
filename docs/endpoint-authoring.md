# Endpoint Author Manual

This manual covers how to write Python code using the `gen-worker` library to
publish endpoints to Cozy. Three endpoint types are supported:

- **Inference** — request/response functions, optionally streaming.
- **Conversion** — functions that produce weight artifacts on a destination repo.
- **Training** — long-running jobs implemented as a stateful trainer class.

The library handles discovery, scheduling, model loading, cancellation, artifact
uploads, and terminal reporting. Your code owns the function body only.

---

## 1. Terminology

- `owner` — publishing namespace (an org slug in URLs; canonical ID is a UUID).
- `endpoint` — published unit, built from your source. Your `endpoint.toml`
  `name` becomes the endpoint slug.
- `function` — an invokable unit inside an endpoint release. Names are derived
  from `@worker_function` names (normalized to a URL-safe slug, e.g.
  `medasr_transcribe` → `medasr-transcribe`).
- `release_id` — immutable identifier for a published endpoint release.
- Invoke reference — `owner/endpoint/function[:tag]` (default tag `prod`).
- `invoker` / `invoker_id` — the identity performing an invocation.
- `request` / `job` — one execution of a function (legacy: "task"/"action").

---

## 2. Project Layout

A tenant endpoint is a Dockerfile-first project:

```
my-endpoint/
├── endpoint.toml
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── src/
    └── my_pkg/
        └── main.py
```

The built image must:

1. Install `gen-worker`.
2. Bake the discovery manifest at build time:
   ```dockerfile
   RUN mkdir -p /app/.tensorhub \
     && python -m gen_worker.discover > /app/.tensorhub/endpoint.lock
   ```
3. Use `gen_worker.entrypoint` as the container entrypoint:
   ```dockerfile
   ENTRYPOINT ["python", "-m", "gen_worker.entrypoint"]
   ```

`endpoint.toml` is a build-time input; it does not need to be shipped inside the
image. The control plane reads `/app/.tensorhub/endpoint.lock` from the image
and stores it for routing.

---

## 3. `endpoint.toml` Reference

```toml
schema_version = 1
name = "my-endpoint"
main = "my_pkg.main"         # Python import path that discovery scans

[host.requirements]
cuda = "12.8"                # presence indicates GPU requirement
compute_capabilities = ["8.0", "8.6"]   # optional

[resources]
vram_gb = 12
ram_gb = 32
cpu_cores = 8
disk_gb = 80
max_inflight_requests = 1

# Fixed model keyspace (available to any function)
[models]
sdxl = { ref = "stabilityai/stable-diffusion-xl-base-1.0", attributes = { dtype = ["fp16", "bf16"] } }

# Per-function keyspace (for payload-driven model selection)
[models.generate]
dreamshaper = "lykon/dreamshaper-xl-v2-turbo"
juggernaut  = "rundiffusion/juggernaut-xl-v9"
```

The shorthand `key = "owner/repo"` omits attributes — the tensorhub resolver matches any variant and picks the tag/latest.

---

## 4. Inference Endpoints

### Minimal handler

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

Inputs and outputs must be `msgspec.Struct` types. The discovery manifest
records their schemas.

### `@worker_function` resources

Optional `ResourceRequirements` hints for the scheduler:

```python
from gen_worker import worker_function
from gen_worker.api.decorators import ResourceRequirements

@worker_function(resources=ResourceRequirements(
    batch_size_target=4,
    prefetch_depth=2,
    requires_gpu=True,
    min_vram_gb=8.0,
))
def generate(ctx, payload): ...
```

Common fields: `batch_size_{min,target,max}`, `prefetch_depth`, `max_wait_ms`,
`memory_hint_mb`, `kind`, `requires_gpu`, `min_vram_gb`, `vram_multiplier`,
`compute_capability_min`, `supported_precisions`, `supported_conversion_profiles`.

### `RequestContext`

What your function receives as its first argument.

Identity and environment:

- `request_id`, `job_id`, `parent_request_id`, `child_request_id`
- `owner`, `invoker_id`, `workspace_scope_id`
- `device` (torch device), `timeout_ms`, `deadline`, `time_remaining_s()`
- `resolved_cozy_models_by_id`, `required_models`

Lifecycle:

- `is_canceled() -> bool` — cooperative cancellation check.
- `done` (a `threading.Event`) and `cancel()`.
- `progress(progress: float, stage: str | None)` — emit progress events.
- `log(message, level)` — structured log.
- `emit(event_type, payload)` — custom event.

Output persistence:

- `save_bytes(ref, data) -> Asset`
- `save_file(ref, local_path) -> Asset`
- `save_checkpoint(ref, local_path, format=...) -> Tensors`
- `save_checkpoint_bytes(ref, data, format=...) -> Tensors`
- `open_output_stream()` — incremental byte streams.
- `open_checkpoint_stream(ref, format=...)` — incremental weight artifacts;
  `finalize()` returns a `Tensors`.

Batching hints: `preferred_batch_size()`, `prefetch_depth()`.

### Model Injection

Declare a model dependency with `Annotated[..., ModelRef(...)]`. The worker
loads and caches the model; your function receives the live object.

**Fixed** — the model key is baked into the signature:

```python
from typing import Annotated
from diffusers import DiffusionPipeline
from gen_worker.api.injection import ModelRef, ModelRefSource as Src

@worker_function()
def generate(
    ctx: RequestContext,
    pipe: Annotated[DiffusionPipeline, ModelRef(Src.FIXED, "sdxl")],
    payload: Input,
) -> Output: ...
```

**Payload-selected** — the caller chooses from the function's keyspace:

```python
class Input(msgspec.Struct):
    prompt: str
    model: str   # must match a key in [models.generate]

@worker_function()
def generate(
    ctx: RequestContext,
    pipe: Annotated[DiffusionPipeline, ModelRef(Src.PAYLOAD, "model")],
    payload: Input,
) -> Output: ...
```

Payload selection rejects arbitrary repo refs; only short keys declared in
`[models.<function>]` are accepted by default.

### Streaming Output

Return `Iterator[T]`, `Generator[T, None, None]`, or `AsyncIterator[T]`. Each
yielded struct is flushed to the caller as a delta.

```python
from typing import Iterator

class Delta(msgspec.Struct):
    chunk: str

@worker_function()
def stream(ctx: RequestContext, payload: Input) -> Iterator[Delta]:
    for word in payload.prompt.split():
        if ctx.is_canceled():
            raise InterruptedError("canceled")
        yield Delta(chunk=word)
```

For Hugging Face `TextIteratorStreamer` integration, see
`gen_worker.api.streaming.iter_transformers_text_deltas` (forwards a cancel
check while iterating).

### Saving Output Files

```python
@worker_function()
def render(ctx: RequestContext, payload: Input) -> Output:
    asset = ctx.save_bytes(
        f"jobs/{ctx.request_id}/outputs/out.png",
        image_bytes,
    )
    return Output(image=asset)
```

For large outputs, stream instead of buffering:

```python
with ctx.open_output_stream(
    f"jobs/{ctx.request_id}/outputs/out.bin"
) as out:
    for chunk in produce():
        out.write(chunk)
    asset = out.finalize()
```

### Error Types

Raise these from `gen_worker.errors` to shape retry behavior:

- `ValidationError` — bad input; no retry, returned to caller as 4xx.
- `RetryableError` — transient; scheduler may retry.
- `ResourceError` — resource exhausted.
- `CanceledError` — run canceled cooperatively.
- `AuthError` — permission denied.
- `FatalError` — unrecoverable.
- `OutputTooLargeError(size_bytes, max_bytes)` — returned output exceeds limit.

Any uncaught exception is treated as a fatal run failure.

---

## 5. Conversion Endpoints

Conversion endpoints are regular `@worker_function` handlers that declare
**reserved-name** payload fields. The orchestrator inspects them by name and
routes/validates the job accordingly; there is no separate decorator.

```python
import msgspec
from gen_worker import RequestContext, Tensors, worker_function
from gen_worker.api.types import SourceRepo, DestinationRepo, OutputSpec

class ConvertInput(msgspec.Struct):
    source: SourceRepo            # reserved name: what to convert
    destination: DestinationRepo  # reserved name: where to publish
    outputs: list[OutputSpec]     # one entry per emitted variant

class ConvertOutput(msgspec.Struct):
    weights: Tensors

@worker_function()
def convert(ctx: RequestContext, payload: ConvertInput) -> ConvertOutput:
    # ... do the conversion, write local file ...
    tensors = ctx.save_checkpoint(
        f"jobs/{ctx.request_id}/outputs/weights.safetensors",
        "/tmp/converted.safetensors",
        format="safetensors",
    )
    return ConvertOutput(weights=tensors)
```

### Reserved types

- **`SourceRepo`**: `ref` (`owner/repo[:tag][@<digest>]`), optional `variant_id`
  (highest-priority explicit variant selector), optional `attributes` dict
  (subset-containment match; well-known keys include `dtype`, `file_layout`,
  `file_type`, `quant_library`, plus family-specific `quant_*` keys — see
  tensorhub `docs/variant_attributes.md`).
- **`DestinationRepo`**: `ref` and `tags` list. After your function returns
  success, the library applies each tag to the new checkpoint atomically.
- **`OutputSpec`**: `attributes` dict. Every entry in `payload.outputs`
  produces one variant on the destination's new checkpoint. You may augment
  `attributes` at upload time with runtime-discovered provenance
  (e.g. `quant_library_version`); the stored attribute bag is the union. The
  attributes must satisfy tensorhub's per-family validation at commit time.

### Streaming large artifacts

```python
with ctx.open_checkpoint_stream(
    f"jobs/{ctx.request_id}/outputs/weights.safetensors",
    format="safetensors",
) as out:
    for chunk in produce_chunks():
        out.write(chunk)
    tensors = out.finalize()
```

`Tensors` fields surfaced to callers: `ref`, `owner`, `local_path`, `format`,
`size_bytes`, `sha256`, `blake3`, `blob_digest`, `snapshot_digest`. It mirrors
`Asset` but is the first-class type for weight payloads.

---

## 6. Training Endpoints

Training endpoints are published the same way as inference endpoints, but are
invoked by the worker running with `WORKER_MODE=trainer`. The trainer body is
a **class** (or class instance) with canonical hooks. The runtime owns the
outer loop, cadence, checkpoint/sample writes, artifact uploads, cancellation,
and terminal reporting.

### Trainer class contract

Required methods:

```python
from gen_worker import StepContext, StepResult

class MyTrainer:
    def setup(self, ctx: StepContext) -> None: ...
    def configure(self, ctx: StepContext) -> dict[str, object]: ...
    def prepare_batch(self, raw_batch, state, ctx: StepContext): ...
    def train_step(self, prepared_batch, state, ctx: StepContext) -> StepResult: ...
    def state_dict(self, state) -> dict[str, object]: ...
    def load_state_dict(self, state, payload, ctx: StepContext) -> None: ...
```

Optional checkpoint hooks (recommended when you produce real weight files):

```python
def save_checkpoint(self, *, state, step, output_dir, final, ctx) -> dict | None: ...
def load_checkpoint(self, *, state, checkpoint_dir, payload, ctx) -> None: ...
```

Point to your class from the job spec (`"trainer": "my_pkg.train:MyTrainer"`)
or via `TRAINER_PLUGIN=my_pkg.train:MyTrainer`.

Incompatibility policy is buyer-beware: if the resolved model layout is
incompatible with your trainer, fail fast inside `configure()` with a clear
error.

### `StepContext`

Attributes the runtime populates for every hook call:

- `job` — `TrainingJobSpec` (`request_id`, `max_steps`, `trainer_api_version`,
  `metric_every`, `checkpoint_every`, `sample_every`, `owner`, `release_ref`,
  `hyperparams`).
- `model_handles` — dict of resolved model components.
- `dataset`, `optimizer`, `scheduler`.
- `device` (e.g. `"cuda:0"`), `dtype` (e.g. `"bf16"`).
- `is_canceled()` — cancellation check (check it at step boundaries).

### `StepResult`

```python
@dataclass
class StepResult:
    metrics: Mapping[str, float]
    debug:   Mapping[str, Any] | None = None
    control: StepControlHints | None = None  # skip_cadence_emit, backoff_seconds
```

The runtime normalizes metric names: `loss`/`train_loss` → `train/loss`;
`lr`/`learning_rate`/`train_lr` → `train/lr`. Missing `train/lr` is extracted
from `debug`, state values, or optimizer param groups if possible.

### Ownership split

The runtime owns:

- Lifecycle, cancellation, timeout, retry, terminal state.
- Ref resolution, input downloads, parquet→Arrow batch feeding.
- Cadence-driven metric/checkpoint/sample emission.
- Local artifact writes and uploads.

Your trainer owns:

- Dataset shaping and batch preparation.
- Forward/backward/update math.
- Prompt/mask/curriculum logic.
- Trainer state serialization (`state_dict` / `load_state_dict`).

### Optional helpers (`gen_worker.trainer.helpers`)

Use only if they fit your endpoint:

- `seed_everything(seed)` — seeds `random` and, if available, `torch`.
- `to_float_scalar(value)` — normalize tensor/scalar loss values.
- `build_default_adamw_bundle(model_or_params, hyperparams=...)` — AdamW with
  cosine/warmup scheduler.
- `save_trainable_module_checkpoint(...)` / `load_trainable_module_checkpoint(...)` —
  LoRA-style module + optimizer checkpoint serialization.

### Running locally

```bash
WORKER_MODE=trainer \
TRAINER_JOB_SPEC_PATH=/path/to/trainer_job.json \
python -m gen_worker.entrypoint
```

### Job spec (v1)

```json
{
  "trainer_api_version": "v1",
  "request_id": "run_123",
  "trainer": "my_pkg.train:MyTrainer",
  "max_steps": 1000,
  "metric_every": 10,
  "checkpoint_every": 200,
  "sample_every": 200,
  "owner": "org_id",
  "release_ref": "org/repo:latest",
  "hyperparams": {"learning_rate": 1e-4},
  "dataset": {
    "parquet_paths": ["/data/train.parquet"],
    "batch_size": 32,
    "readahead": 2,
    "columns": ["image_ref", "caption"]
  }
}
```

`mock_batches` is supported for local smoke runs. `inputs` accepts
`base_model_ref`/`base_model_url`, `dataset_parquet_refs`/`..._urls`, and
`resume_checkpoint_ref`/`..._url` — the runtime materializes these before
`setup()` is called.

### Artifact layout

Everything resolves from `TRAINER_ARTIFACTS_DIR` (default `/tmp/training`):

- `${TRAINER_ARTIFACTS_DIR}/checkpoints/step-%08d.json` — periodic checkpoints.
- `${TRAINER_ARTIFACTS_DIR}/checkpoints/final.json` — final marker at completion.
- `${TRAINER_ARTIFACTS_DIR}/samples/step-%08d-%02d.json` — sample artifacts.
- `${TRAINER_ARTIFACTS_DIR}/metrics/events.jsonl` — JSONL event stream.

Checkpoint JSON payload shape: `{"step": ..., "request_id": ..., "state": {...}}`.
JSON writes are atomic (`tempfile + fsync + rename`) where possible. Overrides:
`TRAINER_CHECKPOINTS_DIR`, `TRAINER_SAMPLES_DIR`, `TRAINER_METRICS_DIR`,
`TRAINER_EVENTS_PATH`.

### Resume

Set `resume_from_latest: true`. The runtime scans `step-*.json` (ignoring
corrupt files), picks the highest valid step, and applies the serialized state
via `load_state_dict(state, payload, ctx)`. Step counters continue from the
resumed step. If `final.json` already exists, the runtime short-circuits as
completed to avoid duplicate terminal emission. An explicit
`resume_checkpoint_path` overrides the scan when provided.

### Sample prompts

The job spec `sample_prompts` list accepts:

- plain strings (`t2i` by default), or
- objects with `mode`, `prompt`, `instruction`, `source_image`, `seed`.

An optional `sample_seed` provides a fixed fallback seed.

### Orchestrated mode

Set `TRAINER_ORCHESTRATED=1` to enable strict startup checks; a capability
token is then required (via `TRAINER_CAPABILITY_TOKEN` or job spec
`capability_token`). Cancellation is checked via `TRAINER_CANCELLED` env flag
or `TRAINER_CANCEL_FILE`. Max runtime via `TRAINER_MAX_RUNTIME_SECONDS`
(surfaced as cancel reason `timeout`).

Upload endpoints (bearer-auth using the capability token):

- `TRAINER_UPLOAD_METRICS_URL`, `TRAINER_UPLOAD_CHECKPOINT_URL`,
  `TRAINER_UPLOAD_SAMPLE_URL`, `TRAINER_UPLOAD_TERMINAL_URL`.

Deterministic failure categories surfaced in events: `startup`, `input`,
`auth`, `model-load`, `train-step`, `upload`.

### Event stream (v1)

`events.jsonl` carries one JSON object per line with stable fields:

- `schema_version: "trainer_event.v1"`
- `event`: `started | metric | checkpoint | sample | completed | failed`
- `request_id`, `seq`, `timestamp_ms`
- event-specific payload (`name`, `value`, `step`, `path`, `error`, ...)

---

## 7. Observability (worker-emitted)

You do not need to emit these yourself; they are listed so you know what
signals exist when debugging a run.

### Inference request lifecycle

Per-request events emitted by the worker:

- `request.received`, `request.started`
- `request.model_resolve.{started,completed,failed,stuck}`
- `request.model_load.{started,completed,failed,stuck}`
- `request.inference.{started,completed,failed,stuck}`
- `request.completed`, `request.failed`

Stuck warnings are controlled by env:

- `WORKER_WARN_MODEL_RESOLVE_S` (default 30)
- `WORKER_WARN_MODEL_LOAD_S` (default 60)
- `WORKER_WARN_INFERENCE_S` (default 60)

### Worker startup

Emitted as `worker.startup.phase` events:

`boot`, `cache_preflight_started`, `cache_preflight_ok|failed`,
`cache_preflight_fallback_attempt|enabled`, `scheduler_connecting`,
`registered`, `ready`, `startup_timeout_unregistered`.

Registration timeout: `WORKER_REGISTER_TIMEOUT_S` (default 90s). On top-level
crash, a `worker.fatal` event is emitted with `phase`, `exception_class`,
`exception_message`, `traceback`, `exit_code`.

### Per-run performance metrics

Best-effort events, low-cardinality payloads (numbers and small strings only):

- `metrics.compute.started` — `{ "at": "<rfc3339>" }`
- `metrics.compute.completed` — `{ "at": "<rfc3339>" }`
- `metrics.fetch` — `{ "ms": <int> }` (0 for warm disk hits)
- `metrics.gpu_load` — `{ "ms": <int> }`
- `metrics.inference` — `{ "ms": <int> }`
- `metrics.tokens` — `{ "output_tokens": <int> }` (when applicable)
- `metrics.job` — one end-of-run extended payload (schema version 1), including
  `function_name`, `cache_state`, per-model details, `pipeline_init_ms`,
  `inference_ms`, optional post/resources keys.

Separately, the worker emits `model.cached` and `models.disk_inventory` with
`disk_backend`, `disk_fstype`, `disk_volume_key` so the scheduler knows which
shared volumes hold which models. None of these are required for correctness;
they must never fail a run.

---

## 8. Local Testing

### Dev HTTP runner (inference)

```bash
docker run --rm --gpus all -p 8081:8081 \
  -v "$(pwd)/out:/outputs" \
  -e TENSORHUB_URL='http://host.docker.internal:7777' \
  <your-image> \
  python -m gen_worker.testing.http_runner --listen 0.0.0.0:8081 --outputs /outputs
```

Invoke:

```bash
curl -sS -X POST 'http://localhost:8081/v1/request/generate' \
  -H 'content-type: application/json' \
  -d '{"payload":{"prompt":"hello"}}'
```

Outputs land under `/outputs/jobs/<request_id>/outputs/...`, matching Cozy ref
semantics.

### Mock orchestrator (one-shot request)

Start the worker container pointing at `host.docker.internal:8080`, then:

```bash
python -m gen_worker.testing.mock_orchestrator \
  --listen 0.0.0.0:8080 \
  --run generate \
  --payload-json '{"prompt":"hello"}'
```

### Trainer smoke run

```bash
WORKER_MODE=trainer \
TRAINER_JOB_SPEC_PATH=./trainer_job.example.json \
python -m gen_worker.entrypoint
```

Confirm: metrics appear in `events.jsonl`, checkpoints include a serialized
`state` payload, and the resume path restores through `load_state_dict`.
