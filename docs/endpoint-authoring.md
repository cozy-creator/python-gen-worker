# Endpoint Authoring Guide

A guide for writing Python endpoints with `gen-worker`. Three endpoint kinds:

| Kind | Decorator | Lives in | What it does |
|---|---|---|---|
| Inference | `@inference_function` | `gen_worker` | Request/response (optionally streaming) |
| Conversion | `@training_function(kind="format-conversion")` | `gen_worker.conversion` | Produces new weight artifacts on a destination repo |
| Training | trainer class | `gen_worker.trainer` | Long-running stateful job, periodic checkpoints |

The SDK handles discovery, scheduling, model loading, cancellation, file I/O, and terminal reporting. You write the function body.

---

## 1. Project Layout

A tenant endpoint is Dockerfile-first:

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

The Dockerfile must:

1. Install `gen-worker` and your dependencies.
2. Bake the discovery manifest at build time:
   ```dockerfile
   RUN mkdir -p /app/.tensorhub && \
       uv run python -m gen_worker.discovery > /app/.tensorhub/endpoint.lock
   ```
3. Use `gen_worker.entrypoint` as the entrypoint:
   ```dockerfile
   ENTRYPOINT ["uv", "run", "python", "-m", "gen_worker.entrypoint"]
   ```

`endpoint.toml` is a build-time input; it does not need to be in the final image. The control plane reads `/app/.tensorhub/endpoint.lock` and stores it for routing.

---

## 2. `endpoint.toml` Reference

```toml
schema_version = 1
name = "my-endpoint"
main = "my_pkg.main"   # Python import path that discovery scans

[host.requirements]
cuda = "12.8"                            # presence indicates GPU requirement
compute_capabilities = ["8.0", "8.6"]    # optional, supported SM list

[resources]
vram_gb = 12
ram_gb = 32
cpu_cores = 8
disk_gb = 80

# Static model refs: declared up front, pre-resolved at deploy time, available
# to any function. Selectors live in the ref string itself:
#   "owner/repo"               → latest tag, default flavor
#   "owner/repo:prod"          → prod tag, default flavor
#   "owner/repo:prod#int4"     → prod tag, int4 flavor
#   "owner/repo@blake3:<hex>"  → pinned checkpoint
[models]
sdxl = "stabilityai/stable-diffusion-xl-base-1.0:prod"

# Per-function model keyspace — payload chooses one key at invoke time.
[models.generate]
dreamshaper = "lykon/dreamshaper-xl-v2-turbo"
juggernaut  = "rundiffusion/juggernaut-xl-v9"
```

The publishing identity must have read access to every declared ref; deployment fails fast otherwise.

---

## 3. Inference Endpoints

### Minimal handler

```python
import msgspec
from gen_worker import RequestContext, inference_function

class Input(msgspec.Struct):
    prompt: str

class Output(msgspec.Struct):
    text: str

@inference_function
def generate(ctx: RequestContext, payload: Input) -> Output:
    return Output(text=f"Hello, {payload.prompt}!")
```

Inputs and outputs must be `msgspec.Struct`. The discovery step records their JSON schemas.

### Resource hints

```python
from gen_worker import inference_function, ResourceRequirements

@inference_function(resources=ResourceRequirements(
    accelerator="cuda",
    requires_gpu=True,
    min_vram_gb=8.0,
    cuda_compute_min=8.0,
    required_libraries=["torch", "diffusers"],
))
def generate(ctx, payload): ...
```

Fields: `kind`, `accelerator` (`"none"` | `"cuda"`), `requires_gpu`, `min_vram_gb`, `cuda_compute_min` (SM floor as float, e.g. `8.0`), `required_libraries`.

### Scaling hints

Tell the orchestrator how your endpoint's resource use depends on the payload. The runtime uses these to pre-empt OOMs and to size pods more efficiently. You declare *which* dimensions matter; the orchestrator learns the coefficients from observed runs.

`ScalingHints` is a frozen `msgspec.Struct` — pass it via `scaling_hints=`:

**VRAM scales with payload (image generation):**

```python
from gen_worker import inference_function, ScalingHints

@inference_function(scaling_hints=ScalingHints(
    vram_must_fit="full_model",                          # anchor: whole bf16 must fit
    vram_size_multiplier=1.0,                            # × source.size_facts["full_model"]
    vram_base=2 * 1024**3,                               # + 2 GB working memory
    vram_scales_with=("payload.width", "payload.height"),
))
def generate_image(ctx, payload): ...
```

**Runtime scales with payload (LLM):**

```python
@inference_function(scaling_hints=ScalingHints(
    vram_must_fit="largest_component",                   # streaming-load: only need largest single shard
    vram_size_multiplier=1.0,
    runtime_scales_with=("payload.max_tokens",),
))
def chat(ctx, payload): ...
```

Fields:
- `vram_must_fit`: anchor — `"full_model"` (whole model must fit) or `"largest_component"` (only largest single component must fit, for streaming-load paths). Tensorhub records both numbers on each checkpoint at ingest; this picks which one the orchestrator multiplies.
- `vram_base` (int bytes): constant overhead (CUDA kernels, working memory).
- `vram_size_multiplier` (float): multiplier on `source.size_facts[vram_must_fit]`.
- `vram_scales_with` / `runtime_scales_with` (tuples of payload field paths): the orchestrator learns coefficients from observation.

All optional — declare only the dimensions your endpoint actually depends on.

### Model injection

Declare a model dependency with `Annotated[..., ModelRef(...)]`. The worker loads and caches the object; your function receives the live instance.

**Fixed** — key baked into the signature:

```python
from typing import Annotated
from diffusers import DiffusionPipeline
from gen_worker import ModelRef, ModelRefSource as Src

@inference_function
def generate(
    ctx: RequestContext,
    pipe: Annotated[DiffusionPipeline, ModelRef(Src.FIXED, "sdxl")],
    payload: Input,
) -> Output: ...
```

**Payload-selected** — caller chooses from the function's `[models.<fn>]` keyspace:

```python
class Input(msgspec.Struct):
    prompt: str
    model: str   # must match a key in [models.generate]

@inference_function
def generate(
    ctx: RequestContext,
    pipe: Annotated[DiffusionPipeline, ModelRef(Src.PAYLOAD, "model")],
    payload: Input,
) -> Output: ...
```

Only short keys declared in `[models.<function>]` are accepted; arbitrary repo refs from the payload are rejected.

### Streaming output

Return `Iterator[T]`, `Generator[T, None, None]`, or `AsyncIterator[T]`. Each yielded struct is flushed as a delta.

```python
from typing import Iterator

class Delta(msgspec.Struct):
    chunk: str

@inference_function
def stream(ctx: RequestContext, payload: Input) -> Iterator[Delta]:
    for word in payload.prompt.split():
        ctx.raise_if_canceled()
        yield Delta(chunk=word)
```

For Hugging Face `TextIteratorStreamer`, `gen_worker.iter_transformers_text_deltas` forwards a cancel check while iterating.

### Loading inputs and saving outputs

`Asset` is a small typed pointer to a file — codecs are free functions in
`gen_worker.io`, not methods on the struct. This keeps `Asset` cheap to
import (no PIL/numpy pull) and gives every endpoint the same one-liner.

```python
from gen_worker import io as gw_io

# Decode an Asset as a PIL image (requires gen-worker[images]):
img = gw_io.read_image(payload.image)                       # default mode="RGB"

# Decode an Asset as (numpy float32, sample_rate) (requires gen-worker[audio]):
speech, sr = gw_io.read_audio(payload.audio, target_sample_rate=16000)

# Raw bytes / file handles / existence probe (no extras needed):
data = gw_io.read_bytes(payload.file)
with gw_io.open(payload.file, "rb") as f:
    head = f.read(64)
if gw_io.exists(payload.optional_file):
    ...

# Encode and save an image output (replaces the older ctx.save_image()):
out = gw_io.write_image(ctx, "out", img, format="webp", quality=90)
return Output(image=out)
```

If `Asset.local_path` is not set (the platform didn't materialize the file),
these functions raise `ValidationError` with the asset ref in the message —
no need for per-endpoint `local_path is None` checks.

Install the codec extras you need:

```bash
pip install gen-worker[images,audio]
```

### Saving output files

```python
@inference_function
def render(ctx: RequestContext, payload: Input) -> Output:
    asset = ctx.save_bytes(
        f"jobs/{ctx.request_id}/outputs/out.png",
        image_bytes,
    )
    return Output(image=asset)
```

For large outputs, stream:

```python
with ctx.open_output_stream(
    f"jobs/{ctx.request_id}/outputs/out.bin"
) as out:
    for chunk in produce():
        out.write(chunk)
    asset = out.finalize()
```

---

## 4. `RequestContext`

What every endpoint function receives as its first argument.

**Identity / environment**
- `ctx.request_id`, `ctx.job_id`, `ctx.parent_request_id`, `ctx.child_request_id`
- `ctx.owner`
- `ctx.device` (torch device), `ctx.timeout_ms`, `ctx.deadline`, `ctx.time_remaining_s()`
- `ctx.compute` — resolved hardware spec (accelerator, vram_gb, gpu_tier, etc.)

**Lifecycle**
- `ctx.is_canceled() -> bool` — cooperative cancellation check
- `ctx.raise_if_canceled(message: str = "request canceled") -> None` — raise `CanceledError(message)` if canceled; the canonical one-liner inside long-running loops
- `ctx.progress(progress: float, stage: str | None = None)` — emit a progress event
- `ctx.log(message, level)` — structured log
- `ctx.emit(event_type, payload)` — custom event

**Output persistence**
- `ctx.save_bytes(ref, data) -> Asset` — small inline payloads
- `ctx.save_file(ref, local_path) -> Asset` — non-tensor files
- `ctx.save_checkpoint(ref, local_path, format=..., flavor=...) -> Tensors` — tensor weights (requires repo-job scope; set by conversion/training jobs)
- `ctx.save_checkpoint_bytes(ref, data, format=...) -> Tensors`
- `ctx.open_output_stream(ref, ...)` / `ctx.open_checkpoint_stream(ref, ...)` — chunked uploads

For ref strings: `jobs/{ctx.request_id}/outputs/<path>` is the canonical layout for ephemeral per-job output. Other prefixes need explicit scope from the orchestrator.

### Kind-specific context subclasses

The SDK ships three `RequestContext` subclasses for non-inference endpoint kinds. The worker constructs the matching subclass before dispatch based on your function's declared kind; you type the matching subclass as your `ctx:` parameter and get autocomplete for the methods you can call.

- **Inference handlers** — receive `RequestContext`. Just the base surface above.
- **Conversion handlers** (`@training_function(kind="format-conversion")`, `kind="quantization"`, `kind="fine-tuning"`, etc.) — receive `ConversionContext`. Adds `publish_repo_revision`, `read_repo_metadata`, `write_repo_metadata`, `materialize_blob`, plus the conversion-helper API (`mktemp`, `checkpoint_dir`, `open_output_writer`, `copy_unconverted_components`, `cancelled`).
- **Dataset handlers** (`@training_function(kind="dataset-generation")`) — receive `DatasetContext`. Adds `publish_dataset_revision`, `resolve_dataset`, `materialize_blob`.
- **Trainer-class endpoints** — receive `TrainingContext`. Adds `read_repo_metadata`, `write_repo_metadata`. `save_checkpoint` is on the base, so trainer code can use it from any context.

```python
from gen_worker import (
    ConversionContext, DatasetContext, RequestContext, TrainingContext,
    inference_function,
)
from gen_worker.conversion import training_function

@inference_function
def infer(ctx: RequestContext, payload: Input) -> Output: ...

@training_function(kind="format-conversion")
def convert(ctx: ConversionContext, source: Source, specs: list[Spec]) -> list[ProducedFlavor]: ...

@training_function(kind="dataset-generation")
def gen_corpus(ctx: DatasetContext, payload: GenerateInput) -> list[ProducedFlavor]: ...
```

Admin-plane visibility toggles (`publish_endpoint`, `publish_media`, `publish_checkpoint`, etc.) are no longer on `RequestContext`. Endpoint authors should never need them — they belong in `cozyctl` or the tensorhub UI.

### How model downloads work

You never call resolve from inside an endpoint. The orchestrator pre-resolves every model ref the job needs (static refs from `endpoint.toml [models]`, plus any runtime refs from the request payload) and ships `{snapshot_digest, files: [{path, blake3, presigned_url}]}` to the worker over gRPC. The `ModelRef` injection paths shown above just consume that pre-resolved manifest — the worker downloads from the presigned URLs and hands you the loaded object.

If the invoker doesn't have read access to a runtime-specified repo, the orchestrator fails the invoke before dispatching — your function never starts.

---

## 5. Error Types

Raise these from `gen_worker` (or let them propagate) to shape retry/terminal behavior:

| Exception | Outcome |
|---|---|
| `ValidationError` | 4xx to caller, no retry |
| `RetryableError` | Transient; scheduler may retry |
| `ResourceError` | Resource exhausted (e.g. OOM) |
| `CanceledError` | Cooperative cancellation — what `ctx.raise_if_canceled()` raises; `except CanceledError` in cleanup code |
| `AuthError` | Permission denied |
| `FatalError` | Unrecoverable |
| `OutputTooLargeError(size_bytes, max_bytes)` | Output exceeds limit |
| `InputTooLargeError(size_bytes, max_bytes, source)` | Input exceeds limit |

Any uncaught exception becomes a fatal run failure.

---

## 6. Conversion Endpoints

Conversion endpoints take a source model + an output spec, produce new weight artifacts on a destination repo. Use `gen_worker.conversion.training_function(kind="format-conversion")`.

```python
import msgspec
import torch
from gen_worker.conversion import (
    ConversionContext,
    ProducedFlavor,
    Source,
    training_function,
)

class CastSpec(msgspec.Struct):
    dtype: str

@training_function(kind="format-conversion")
def cast_dtype(
    ctx: ConversionContext,
    source: Source,
    specs: list[CastSpec],
) -> list[ProducedFlavor]:
    writer = ctx.open_output_writer()
    for component, name, tensor in source.iter_tensors():
        if ctx.cancelled:
            break
        writer.write(component, name, tensor.to(torch.bfloat16))

    return [ProducedFlavor(path=writer.finalize(), flavor=specs[0].dtype)]
```

Reserved signature names: `ctx`, `source`, `destination`, `specs`, `datasets`. Everything else is decoded from the request payload by name.

- `source` is materialized from the request's `source.ref` (and optional `checkpoint_id`); orchestrator has already verified the invoker can read it.
- `destination` carries the destination repo + tags. After your function returns, the SDK uploads each `ProducedFlavor` and applies the tags atomically.
- Each `ProducedFlavor` becomes one checkpoint flavor on the destination repo. `flavor` is the user-facing name (e.g. `int4-awq`, `bf16-singlefile`).

For calibrated quantization, dataset shaping helpers, etc., see `gen_worker.conversion`'s submodules.

---

## 7. Training Endpoints

Training is published like any other endpoint but invoked with `WORKER_MODE=trainer`. Your trainer is a **class** with canonical hooks; the runtime owns the outer loop, cadence, checkpointing, uploads, cancellation.

```python
from gen_worker.trainer import StepContext, StepResult

class MyTrainer:
    def setup(self, ctx: StepContext) -> None: ...
    def configure(self, ctx: StepContext) -> dict[str, object]: ...
    def prepare_batch(self, raw_batch, state, ctx: StepContext): ...
    def train_step(self, prepared_batch, state, ctx: StepContext) -> StepResult: ...
    def state_dict(self, state) -> dict[str, object]: ...
    def load_state_dict(self, state, payload, ctx: StepContext) -> None: ...
    # Optional, recommended:
    def save_checkpoint(self, *, state, step, output_dir, final, ctx) -> dict | None: ...
    def load_checkpoint(self, *, state, checkpoint_dir, payload, ctx) -> None: ...
```

Point to the class via the job spec's `"trainer": "my_pkg.train:MyTrainer"` field, or `TRAINER_PLUGIN=my_pkg.train:MyTrainer`.

**Ownership split**:

The runtime handles: lifecycle, cancellation, timeouts, ref resolution, dataset materialization, cadence-driven metric/checkpoint/sample emission, artifact uploads, terminal reporting.

Your trainer handles: dataset shaping, batch preparation, forward/backward/update math, prompt/mask/curriculum logic, state serialization.

**StepResult**:

```python
@dataclass
class StepResult:
    metrics: Mapping[str, float]      # {"loss": ..., "lr": ...}
    debug:   Mapping[str, Any] | None = None
    control: StepControlHints | None = None  # skip_cadence_emit, backoff_seconds
```

The runtime normalizes metric names (`loss` → `train/loss`, `lr` → `train/lr`).

**Optional helpers** (`gen_worker.trainer.helpers`):
- `seed_everything(seed)`
- `to_float_scalar(value)` — normalize tensor/scalar loss
- `build_default_adamw_bundle(model_or_params, hyperparams=...)`
- `save_trainable_module_checkpoint(...)` / `load_trainable_module_checkpoint(...)`

---

## 8. Local Testing

### Inference endpoint

```bash
uv run python -m gen_worker.discovery       # writes endpoint.lock to stdout
WORKER_MODE=invoke uv run python -m gen_worker.entrypoint
```

Hit it with a local invoke (`curl` against the dev server, or the in-repo dev runner).

### Trainer smoke run

```bash
WORKER_MODE=trainer \
TRAINER_JOB_SPEC_PATH=./trainer_job.example.json \
uv run python -m gen_worker.entrypoint
```

Confirm: metrics appear in `events.jsonl`, checkpoints serialize a `state` payload, and the resume path restores through `load_state_dict`.

---

## 9. Examples

Working endpoints to copy from in `python-gen-worker/examples/`:

- `marco-polo/` — minimal inference endpoint
- `medasr-transcribe/` — audio transcription with a HF model dependency
- `openai-codex/` — text generation
- `training-smoke/` — minimal trainer
- `from-scratch/` — boilerplate template
