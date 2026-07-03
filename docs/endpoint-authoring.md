# Endpoint Authoring Guide

Write a Python function, ship it as a serverless endpoint. The SDK handles
discovery, scheduling, model loading, cancellation, streaming, file I/O, and
terminal reporting. You write the endpoint class.

> **Note (API migration):** the runnable current-API endpoint is the class
> shape shown in the quick-start below and in
> [`examples/marco-polo/`](../examples/marco-polo/). The *advanced* examples
> further down this page still use the pre-#322 `@inference_function` /
> `@training_function` **function** shape and no longer import — they're kept
> for the concepts (Resources, bindings, dispatch, streaming) and are being
> rewritten in #368. For copy-paste code, prefer the
> [README](../README.md) and the examples.

Three endpoint kinds:

| Kind       | Decorator                                          | Module                  | What it does                              |
|------------|----------------------------------------------------|-------------------------|-------------------------------------------|
| Inference  | `@inference` class + `@invocable` methods          | `gen_worker`            | Request/response, optionally streaming    |
| Conversion | `@conversion(sub_kind="...")` class                | `gen_worker.conversion` | Produces new weight artifacts on a repo  |
| Training   | trainer class                                      | `gen_worker.trainer`    | Long-running, periodic checkpoints        |

---

## Quick start — the minimum viable endpoint

Two files when deploying through Tensorhub's generated-Dockerfile path.
Tensorhub generates the Dockerfile when `endpoint.toml` has build hints,
installs your dependencies, runs discovery, and wires the runtime entrypoint.

**`endpoint.toml`**

```toml
schema_version = 1
main = "myendpoint.main"

[[build.profiles]]
name = "default"
accelerator = "none"
python = "3.12"
dependencies = ["gen-worker>=0.7.5", "msgspec"]
```

**`main.py`**

```python
import msgspec
from gen_worker import RequestContext, inference, invocable

class Input(msgspec.Struct):
    prompt: str

class Output(msgspec.Struct):
    text: str

@inference()
class Echo:
    def setup(self) -> None:
        pass

    @invocable(name="run")
    def run(self, ctx: RequestContext, payload: Input) -> Output:
        return Output(text=f"got: {payload.prompt}")
```

That's everything. Discovery still runs at image-build time and bakes
`/app/.tensorhub/endpoint.lock`; in generated-Dockerfile builds Tensorhub writes the Dockerfile
that performs those contract steps.

---

## The three layers

Each layer is independent and can be minimal or maximal:

| File                 | Declares                                                                  |
|----------------------|---------------------------------------------------------------------------|
| `endpoint.toml`      | The entry module + where to run the image, plus optional build hints      |
| `Dockerfile`         | Optional when generated Dockerfile builds are enough; required for custom build steps |
| `main.py` (decorators) | What the functions do + each function's runtime envelope and model bindings |

- `endpoint.toml` reference: [endpoint-toml.md](endpoint-toml.md)
- Dockerfile contract: [dockerfile.md](dockerfile.md)
- The rest of this doc covers the Python side.

---

## `Resources` — per-function envelope + cost shape

`Resources` is declared **per function** so the worker can self-advertise an
accurate availability map at boot (skip functions whose hardware envelope this
host can't satisfy) and the orchestrator can route on accurate per-function
needs.

```python
from gen_worker import Resources

_flux_dispatch = Resources(
    # Static placement envelope (hard gates)
    requires_gpu=True,
    min_vram_gb=14.0,
    min_compute_capability=8.0,
    required_libraries=("torch", "diffusers"),

    # Dynamic cost shape (admission + scheduling)
    vram_must_fit="full_model",
    vram_base=500 * 1024 * 1024,
    vram_size_multiplier=1.10,
    vram_scales_with=("width", "height", "num_images_per_prompt"),
    runtime_scales_with=("num_inference_steps", "num_images_per_prompt"),
)
```

| Field                    | Purpose                                                                                                         |
|--------------------------|-----------------------------------------------------------------------------------------------------------------|
| `accelerator`            | `"cuda"`, `"none"`, or unset. `"gpu"`/`"cpu"` are rejected at decoration time (use `"cuda"`/`"none"`).           |
| `requires_gpu`           | Implies `accelerator="cuda"` for placement.                                                                     |
| `min_vram_gb`            | Hard VRAM floor in GiB. Function unavailable on hosts below this.                                              |
| `min_compute_capability`       | Minimum SM compute capability (e.g. `8.0`). Function unavailable on hosts below this.                          |
| `required_libraries`     | Python package names the function needs (`"flash_attn"`, `"bitsandbytes"`, …). Worker checks import at boot.   |
| `vram_must_fit`          | `"full_model"` or `"largest_component"`. Picks which `size_facts` entry the orchestrator uses for admission.   |
| `vram_base`              | Constant VRAM overhead in bytes.                                                                                |
| `vram_size_multiplier`   | Multiplier on `size_facts[vram_must_fit]` when computing admission VRAM.                                       |
| `vram_scales_with`       | Payload field names that grow VRAM. Coefficients learned per `(function, gpu_class, field)`.                   |
| `runtime_scales_with`    | Payload field names that grow runtime. Coefficients learned per `(function, gpu_class, field)`.                |

`vram_scales_with` and `runtime_scales_with` must reference real fields on the
payload struct. A name that doesn't match a payload field is a silent bug — the
cost-shape coefficient never fires — so keep them in sync with the struct.

See [scaling-hints.md](scaling-hints.md) for the cost-shape fields in depth.

---

## `Repo` + chainable methods

A binding is constructed module-level. `Repo` is both the repo handle and a
usable binding with defaults (`tag="prod"`, no flavor, no override).

```python
from gen_worker import Repo

flux = Repo("black-forest-labs/flux.2-klein-4b-base")

flux                                          # bare repo, defaults: tag="prod", no flavor
flux.flavor("nf4")                            # pin a flavor
flux.tag("canary")                            # pin a non-prod tag
flux.tag("canary").flavor("nf4")              # both (order doesn't matter)
flux.flavor("nf4").allow_override(Flux2KleinPipeline)
```

All modifier methods return new immutable instances; chain order is commutative.

| Method                       | Returns | Effect                                                                                                                       |
|------------------------------|---------|------------------------------------------------------------------------------------------------------------------------------|
| `Repo.flavor(name)`          | `Repo`  | New Repo with the flavor set                                                                                                |
| `Repo.tag(name)`             | `Repo`  | New Repo with the tag set                                                                                                   |
| `Repo.allow_override(*cls)`  | `Repo`  | New Repo allowing caller substitution within the supplied class allowlist. Zero-arg call raises `ValueError` at decoration. |

---

## Fixed pick

Function pins one specific `(repo, flavor?, tag?)`:

```python
from diffusers import Flux2KleinPipeline
from gen_worker import Repo, Resources, inference_function

flux = Repo("black-forest-labs/flux.2-klein-4b-base")
_flux_bf16 = Resources(requires_gpu=True, min_vram_gb=22.0)

@inference_function(
    resources=_flux_bf16,
    models={"pipeline": flux.flavor("bf16")},
)
def generate_bf16(ctx, pipeline: Flux2KleinPipeline, payload: GenerateInput) -> GenerateOutput:
    return _generate(ctx, pipeline, payload)
```

The pick resolves to a concrete checkpoint at deploy time; the worker downloads
and caches it before serving traffic.

---

## Dispatch pick — payload-driven selection

Function pins a set of picks keyed by a `Literal[...]`-typed discriminator
field on the payload. At invoke time the discriminator selects which pick to
use.

```python
from typing import Literal
import msgspec
from gen_worker import Repo, Resources, dispatch, inference_function

class BnbInput(msgspec.Struct):
    variant: Literal["nf4", "int8"]
    prompt: str
    num_inference_steps: int = 4
    width: int = 1024
    height: int = 1024
    num_images_per_prompt: int = 1

flux = Repo("black-forest-labs/flux.2-klein-4b-base")

_flux_dispatch = Resources(
    requires_gpu=True,
    min_vram_gb=14.0,                # largest pick the table can resolve to
    vram_must_fit="full_model",
    vram_base=500 * 1024 * 1024,
    vram_size_multiplier=1.10,
    vram_scales_with=("width", "height", "num_images_per_prompt"),
    runtime_scales_with=("num_inference_steps", "num_images_per_prompt"),
)

@inference_function(
    resources=_flux_dispatch,
    models={
        "pipeline": dispatch(
            field="variant",
            table={
                "nf4":  flux.flavor("nf4"),
                "int8": flux.flavor("int8"),
            },
        ),
    },
)
def generate_bnb(ctx, pipeline: Flux2KleinPipeline, payload: BnbInput) -> GenerateOutput: ...
```

Decoration-time validation:

- The `field` name must exist on the payload struct.
- The field must be `Literal[...]`-typed (or `Optional[Literal[...]]`).
- Every `table` key must be a member of the Literal.

---

## `allow_override(*classes)` — caller substitution

`.allow_override(...)` lets the invoker substitute the binding default with an
arbitrary ref of their choice. The tenant supplies an explicit pipeline-class
allowlist; the orchestrator rejects overrides whose `pipeline_class` is not in
that list.

```python
from diffusers import StableDiffusionXLPipeline

@inference_function(
    resources=Resources(requires_gpu=True, min_vram_gb=12.0),
    models={
        "pipe": sdxl.flavor("bf16").allow_override(StableDiffusionXLPipeline),
    },
)
def generate(ctx, pipe: StableDiffusionXLPipeline, payload: Input) -> Output: ...
```

Multiple acceptable classes:

```python
.allow_override(Flux2KleinPipeline, Flux2KleinKontextPipeline)
```

Classes may be passed as class objects (preferred — autocomplete + import-time
check) or string FQNs (escape hatch). **Bare zero-arg `.allow_override()` is a
decoration-time error** — the framework does not auto-derive the constraint
from the function's parameter annotation.

### The six override error codes

If the invoker supplies `_models.<param>`, the orchestrator validates the
override before dispatch and rejects with one of:

| Error code                       | Meaning                                                                |
|----------------------------------|------------------------------------------------------------------------|
| `unknown_override_param`         | `_models[<x>]` names a param that doesn't exist on the function       |
| `model_override_not_allowed`     | The binding has no `.allow_override(...)` declared                     |
| `override_ref_not_found`         | Tensorhub returns 404 for the supplied ref                             |
| `override_tag_not_found`         | The ref exists but the requested tag isn't published                   |
| `override_flavor_not_found`      | The ref+tag exist but the flavor isn't in the checkpoint group         |
| `incompatible_pipeline_class`    | Supplied `pipeline_class` is not in the binding's allowlist            |

All return HTTP 400 with a typed error code in the body.

---

## Multi-param injection — each binding is independent

A function can declare multiple injected params in `models={...}`. Each entry
is its own binding with its own `(repo, flavor, tag)` pick and its own
optional `.allow_override(...)` modifier. The invoker's `_models` dict is
keyed by the same param names, so each is overridden independently.

```python
flux       = Repo("black-forest-labs/flux.2-klein-4b-base")
flux_lora  = Repo("black-forest-labs/flux-lora-collection")

@inference_function(
    resources=_flux,
    models={
        "pipeline":   flux.flavor("nf4"),                                 # fixed, NOT overridable
        "adapter":    flux_lora.flavor("realism").allow_override(LoRA),   # fixed default + overridable
        "controlnet": dispatch(
            field="controlnet_kind",
            table={
                "depth": Repo("...flux-controlnet-depth").flavor("bf16"),
                "canny": Repo("...flux-controlnet-canny").flavor("bf16"),
            },
        ).allow_override(FluxControlNetModel),                            # dispatch + overridable
    },
)
def generate_with_adapter(
    ctx,
    pipeline: Flux2KleinPipeline,
    adapter: LoRA,
    controlnet: FluxControlNetModel,
    payload: AdapterInput,
) -> GenerateOutput: ...
```

Three bindings, three independent override policies:

- `pipeline` — fixed, no override. Invoker cannot substitute it.
- `adapter` — fixed default + overridable within the `LoRA` allowlist.
- `controlnet` — dispatch default + overridable within the
  `FluxControlNetModel` allowlist. Invoker may pick a table key via the
  discriminator OR send `_models.controlnet` to bypass the table entirely.

An invocation overriding two of three:

```json
{
  "prompt": "A red bicycle",
  "controlnet_kind": "depth",
  "_models": {
    "adapter":    "acme/my-custom-lora:prod#realism",
    "controlnet": {"ref": "acme/my-controlnet", "flavor": "bf16"}
  }
}
```

### Multi-model two-stage example — SDXL base + refiner

Two pipelines injected into one function; both optionally overridable.

```python
sdxl_base    = Repo("stabilityai/stable-diffusion-xl-base-1.0")
sdxl_refiner = Repo("stabilityai/stable-diffusion-xl-refiner-1.0")

_sdxl_two_stage = Resources(
    requires_gpu=True,
    min_vram_gb=24.0,                       # BOTH models resident
    vram_must_fit="full_model",
    vram_scales_with=("width", "height", "num_images_per_prompt"),
    runtime_scales_with=("base_steps", "refiner_steps", "num_images_per_prompt"),
)

class SDXLTwoStageInput(msgspec.Struct):
    prompt: str
    negative_prompt: str = ""
    base_steps: int = 30
    refiner_steps: int = 10
    high_noise_frac: float = 0.8
    width: int = 1024
    height: int = 1024
    num_images_per_prompt: int = 1
    seed: int | None = None

@inference_function(
    resources=_sdxl_two_stage,
    models={
        "base":    sdxl_base.flavor("bf16").allow_override(StableDiffusionXLPipeline),
        "refiner": sdxl_refiner.flavor("bf16").allow_override(StableDiffusionXLImg2ImgPipeline),
    },
)
def generate_with_refiner(
    ctx: RequestContext,
    base: StableDiffusionXLPipeline,
    refiner: StableDiffusionXLImg2ImgPipeline,
    payload: SDXLTwoStageInput,
) -> GenerateOutput:
    latent = base(
        prompt=payload.prompt, negative_prompt=payload.negative_prompt,
        num_inference_steps=payload.base_steps,
        width=payload.width, height=payload.height,
        num_images_per_prompt=payload.num_images_per_prompt,
        denoising_end=payload.high_noise_frac,
        output_type="latent",
    ).images
    images = refiner(
        prompt=payload.prompt, negative_prompt=payload.negative_prompt,
        num_inference_steps=payload.refiner_steps,
        denoising_start=payload.high_noise_frac,
        image=latent,
    ).images
    return GenerateOutput(image=gw_io.write_image(ctx, "out", images[0]))
```

Four invocation scenarios:

```jsonc
// (a) defaults — both checkpoints from the bindings
{ "prompt": "A red bicycle", "base_steps": 30, "refiner_steps": 10 }

// (b) override just `base` — refiner stays default
{
  "prompt": "A red bicycle",
  "_models": { "base": "acme/sdxl-architecture-finetune:prod#bf16" }
}

// (c) override both
{
  "prompt": "A red bicycle",
  "_models": {
    "base":    "acme/sdxl-architecture-finetune:prod#bf16",
    "refiner": { "ref": "acme/sdxl-refiner-tuned", "flavor": "bf16" }
  }
}

// (d) class mismatch on `base` — rejected with incompatible_pipeline_class
{
  "prompt": "A red bicycle",
  "_models": { "base": "runwayml/stable-diffusion-v1-5" }
}
```

---

## The reserved `_models` invocation field

The orchestrator strips `_models` from the payload before dispatch and uses it
to compute the resolved binding for each param. Two accepted shapes:

**Structured** (preferred for typed clients):

```json
{
  "prompt": "A red bicycle",
  "_models": {
    "pipeline": {"ref": "acme/my-flux-finetune", "tag": "prod", "flavor": "bf16"}
  }
}
```

**String shorthand**:

```json
{
  "prompt": "A red bicycle",
  "_models": {"pipeline": "acme/my-flux-finetune:prod#bf16"}
}
```

Both forms normalize to `(ref, tag, flavor)`. Grammar:
`<owner>/<repo>[:<tag>][#<flavor>]`. Tag defaults to `"prod"`; flavor is
optional.

Payload structs cannot use the reserved field name `_models` — decoration
fails at discovery time if you try.

### Atomic substitution

If the invoker supplies overrides for multiple params and **any one** fails
validation, the whole request is rejected before dispatch. No partial
substitution. The error response names which `_models[<param>]` failed.

Atomic is the safe default: tenants' function bodies assume the whole binding
set is valid, and half-substituting could deliver a mismatched pair the tenant
didn't authorize.

### Cross-param compatibility is tenant-side

The framework only enforces per-param class allowlist and ref-resolves-cleanly
checks. It does NOT verify that an override for `base` is compatible with the
override or default for `refiner` (shared VAE, latent space, training-data
alignment, etc.).

Tenants who need strict cross-param coupling have three options:

1. **Don't enable `.allow_override(...)`** — lock both params to known-good defaults.
2. **Narrow the allowlist to a single class** — overrides must at least share architecture.
3. **Add a runtime cross-check** inside the function body and raise a typed
   error if the pair is incompatible at invoke time.

The framework gives the building blocks; tenants compose the policy.

---

## Streaming output

Return an `Iterator[T]`, `Generator[T, None, None]`, or `AsyncIterator[T]`.
Each yielded struct is flushed as a delta.

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

For Hugging Face `TextIteratorStreamer`,
`gen_worker.iter_transformers_text_deltas` forwards a cancel check while
iterating.

---

## Loading inputs and saving outputs

`Asset` is a typed pointer to a file. Codecs live as free functions in
`gen_worker.io` — keeps `Asset` cheap to import (no PIL/numpy pull) and gives
every endpoint the same one-liner.

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

# Encode and save an image output:
out = gw_io.write_image(ctx, "out", img, format="webp", quality=90)
return Output(image=out)
```

If `Asset.local_path` is not set (the platform didn't materialize the file),
these functions raise `ValidationError` with the asset ref in the message — no
need for per-endpoint `local_path is None` checks.

For arbitrary outputs:

```python
@inference_function
def render(ctx: RequestContext, payload: Input) -> Output:
    asset = ctx.save_bytes(
        f"jobs/{ctx.request_id}/outputs/out.png",
        image_bytes,
    )
    return Output(image=asset)
```

Streaming large outputs:

```python
with ctx.open_output_stream(f"jobs/{ctx.request_id}/outputs/out.bin") as out:
    for chunk in produce():
        out.write(chunk)
    asset = out.finalize()
```

---

## `RequestContext`

Every endpoint function receives the matching context subclass as its first
argument.

**Identity / environment**

- `ctx.request_id`, `ctx.job_id`, `ctx.parent_request_id`, `ctx.child_request_id`
- `ctx.owner`
- `ctx.device` (torch device), `ctx.timeout_ms`, `ctx.deadline`, `ctx.time_remaining_s()`
- `ctx.compute` — resolved hardware spec (`accelerator`, `vram_gb`, `gpu_tier`, …)

**Lifecycle**

- `ctx.is_canceled() -> bool` — cooperative cancellation check
- `ctx.raise_if_canceled(message="request canceled")` — the canonical one-liner inside long-running loops; raises `CanceledError`
- `ctx.progress(progress: float, stage: str | None = None)` — emit a progress event
- `ctx.log(message, level)` — structured log
- `ctx.emit(event_type, payload)` — custom event

**Output persistence**

- `ctx.save_bytes(ref, data) -> Asset` — small inline payloads
- `ctx.save_file(ref, local_path) -> Asset` — non-tensor files
- `ctx.save_checkpoint(ref, local_path, format=..., flavor=...) -> Tensors` — tensor weights
- `ctx.save_checkpoint_bytes(ref, data, format=...) -> Tensors`
- `ctx.open_output_stream(ref, ...)` / `ctx.open_checkpoint_stream(ref, ...)` — chunked uploads

The canonical layout for ephemeral per-job output is
`jobs/{ctx.request_id}/outputs/<path>`. Other prefixes need explicit scope from
the orchestrator.

### Kind-specific context subclasses

The SDK ships three `RequestContext` subclasses for non-inference kinds. The
worker constructs the matching subclass before dispatch; you type the
matching subclass on your handler and get autocomplete for the methods you
can call.

- **Inference handlers** — `RequestContext`. Just the base surface above.
- **Conversion handlers** — `ConversionContext`. Adds `publish_repo_revision`,
  `read_repo_metadata`, `write_repo_metadata`, `materialize_blob`, plus the
  conversion helpers (`mktemp`, `checkpoint_dir`, `open_output_writer`,
  `copy_unconverted_components`, `cancelled`).
- **Dataset handlers** (`@training_function(kind="dataset-generation")`) —
  `DatasetContext`. Adds `publish_dataset_revision`, `resolve_dataset`,
  `materialize_blob`.
- **Trainer-class endpoints** — `TrainingContext`. Adds `read_repo_metadata`,
  `write_repo_metadata`. `save_checkpoint` is on the base.

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

### How model downloads work

You never call resolve from inside an endpoint. The orchestrator pre-resolves
every model ref the job needs (default picks from `models={...}` plus any
caller-supplied overrides from the request's `_models` field) and ships
`{snapshot_digest, files: [{path, blake3, presigned_url}]}` to the worker
over gRPC. The binding-injection paths shown above just consume that
pre-resolved manifest — the worker downloads from the presigned URLs and
hands you the loaded object.

If the invoker doesn't have read access to a runtime-specified repo, the
orchestrator fails the invoke before dispatching — your function never starts.

---

## Error types

Raise these from `gen_worker` (or let them propagate) to shape retry/terminal
behavior:

| Exception                              | Outcome                                                                       |
|----------------------------------------|-------------------------------------------------------------------------------|
| `ValidationError`                      | 4xx to caller, no retry                                                       |
| `RetryableError`                       | Transient; scheduler may retry                                                |
| `ResourceError`                        | Resource exhausted (e.g. OOM)                                                 |
| `CanceledError`                        | Cooperative cancellation — what `ctx.raise_if_canceled()` raises              |
| `AuthError`                            | Permission denied                                                             |
| `FatalError`                           | Unrecoverable                                                                 |
| `OutputTooLargeError(size, max)`       | Output exceeds limit                                                          |
| `InputTooLargeError(size, max, source)`| Input exceeds limit                                                           |

Any uncaught exception becomes a fatal run failure.

---

## Conversion endpoints

Conversion endpoints take a source model + an output spec and produce new
weight artifacts on a destination repo.

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

Reserved signature names: `ctx`, `source`, `destination`, `specs`, `datasets`.
Everything else is decoded from the request payload by name.

- `source` is materialized from the request's `source.ref` (and optional
  `checkpoint_id`); the orchestrator has already verified the invoker can
  read it.
- `destination` carries the destination repo + tags. After your function
  returns, the SDK uploads each `ProducedFlavor` and applies the tags
  atomically.
- Each `ProducedFlavor` becomes one checkpoint flavor on the destination
  repo. `flavor` is the user-facing name (e.g. `int4-awq`, `bf16-singlefile`).

For calibrated quantization, dataset-shaping helpers, etc., see
`gen_worker.conversion`'s submodules.

---

## Training endpoints

Training is published like any other endpoint but invoked with
`WORKER_MODE=trainer`. Your trainer is a **class** with canonical hooks; the
runtime owns the outer loop, cadence, checkpointing, uploads, cancellation.

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

Point to the class via the job spec's `"trainer": "my_pkg.train:MyTrainer"`
field.

**Ownership split:**

The runtime handles: lifecycle, cancellation, timeouts, ref resolution,
dataset materialization, cadence-driven metric/checkpoint/sample emission,
artifact uploads, terminal reporting.

Your trainer handles: dataset shaping, batch preparation, forward/backward/
update math, prompt/mask/curriculum logic, state serialization.

Use this hook contract when your endpoint owns the native PyTorch loop. That
should feel like normal PyTorch or Hugging Face code: tokenize/collate a batch,
call `model(**batch)`, backpropagate `loss`, then step the optimizer and
scheduler.

For external engines such as Hugging Face `Trainer`, Accelerate scripts, or
Ostris AI Toolkit, keep the endpoint as a thin adapter around that engine:
materialize TensorHub inputs, write the engine config or dataset layout, run the
engine, and expose the produced checkpoints/samples to the runtime. Do not
duplicate the engine's inner loop inside `train_step` unless TensorHub needs
per-step control.

**StepResult:**

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

## Local testing

### Inference endpoint

The canonical local-test recipe is `gen-worker run`:

```bash
# Single-class, single-method endpoint — both inferred from your code.
gen-worker run --payload '{"prompt": "hello"}'

# Pick a specific class + method.
gen-worker run --class MyEndpoint --method generate --payload '{"prompt":"hello"}'

# Read payload from a file.
gen-worker run --method generate --payload-file ./fixtures/sample.json

# Override the model via payload._models (same shape as production).
gen-worker run --payload '{"prompt":"x","_models":{"pipe":"other/repo:tag#flavor"}}'

# Air-gapped — fail rather than fetch from the registry on cache miss.
gen-worker run --offline --payload '{"prompt":"x"}'
```

Result goes to stdout (one JSON object per line, msgspec-encoded); events
from `ctx.emit / progress / log` go to stderr as JSON lines so the result
on stdout stays pipeable (`gen-worker run ... | jq .value`). Exit codes:
`0` success, `1` user exception, `2` usage / validation error, `3` model
resolution failure, `130` SIGINT. See [local-dev.md](local-dev.md) for
the full design, the SIGINT story, and the `--offline` story.

Fallback (when you want to wire your own HTTP harness):

```bash
uv run python -m gen_worker.discovery       # writes endpoint.lock to stdout
WORKER_MODE=invoke uv run python -m gen_worker.entrypoint
```

### Trainer smoke run

```bash
WORKER_MODE=trainer \
TRAINER_JOB_SPEC_PATH=./trainer_job.example.json \
uv run python -m gen_worker.entrypoint
```

Confirm: metrics appear in `events.jsonl`, checkpoints serialize a `state`
payload, the resume path restores through `load_state_dict`.

---

## Per-modality cookbooks

For shipping production endpoints by modality, see the cookbooks. Each
is a one-page recipe covering the class shape, acceleration stack,
per-model recommended config, and a complete working example:

- [cookbook-image-diffusion.md](cookbook-image-diffusion.md) — Flux, SDXL,
  SD3, Qwen-Image, Sana-Sprint. torch.compile + FBCache/DeepCache + NVFP4.
- [cookbook-video-diffusion.md](cookbook-video-diffusion.md) — LTX-Video,
  HunyuanVideo, Wan2.2, Mochi. TeaCache + FP8/NVFP4 + xDiT sequence
  parallelism.
- [cookbook-audio.md](cookbook-audio.md) — Stable Audio style SerialWorkers
  and Chatterbox-style BatchedWorkers via vLLM.
- [cookbook-stages.md](cookbook-stages.md) — `@inference.stage`
  annotations for multi-stage pipelines (3D, large DiTs with batch-
  friendly encoders).

## Examples

Working endpoints to copy from in `examples/`:

- `marco-polo/` — minimal inference endpoint
- `training-smoke/` — minimal trainer
- `from-scratch/` — boilerplate template
