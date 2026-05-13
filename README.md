# gen-worker

Python SDK for writing **endpoints** that run on Cozy's worker pool. You write a
decorated function, the SDK handles discovery, scheduling, model loading,
cancellation, file I/O, streaming, and reporting back to the control plane.

Three endpoint kinds:

- **Inference** — request/response, optionally streaming.
- **Training** — long-running, stateful, periodic checkpoints.
- **Conversion** — produces weight artifacts on a destination repo.

## Install

```bash
pip install gen-worker[torch]   # for inference with PyTorch
pip install gen-worker          # plain Python (e.g. API-proxy endpoints)
```

Optional extras: `[images]` for `gw.io.read_image / write_image`,
`[audio]` for `gw.io.read_audio`, `[trainer]` for trainer-class endpoints.

## Minimum viable endpoint

Three files. Any base image. No model injection, no fancy decorators.

**`endpoint.toml`** (5 lines for CUDA, 4 for CPU):

```toml
schema_version = 1
main = "myendpoint.main"

[[build.profiles]]
name = "default"
accelerator = "none"
```

**`Dockerfile`** (your choice of base, your build steps):

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . /app
RUN pip install -e .
RUN mkdir -p /app/.tensorhub \
    && python -m gen_worker.discovery > /app/.tensorhub/endpoint.lock
ENTRYPOINT ["python", "-m", "gen_worker.entrypoint"]
```

**`main.py`**:

```python
import msgspec
from gen_worker import RequestContext, inference_function

class Input(msgspec.Struct):
    prompt: str

class Output(msgspec.Struct):
    text: str

@inference_function
def run(ctx: RequestContext, payload: Input) -> Output:
    return Output(text=f"got: {payload.prompt}")
```

That's it. `cozyctl endpoint deploy` (or the platform UI) takes it from here.

## Adding a model

Declare model dependencies on the decorator's `models={...}` kwarg. The worker
loads and caches each binding; your function receives the live instance.

```python
from diffusers import StableDiffusionXLPipeline
from gen_worker import Repo, Resources, inference_function

sdxl = Repo("stabilityai/stable-diffusion-xl-base-1.0")

@inference_function(
    resources=Resources(requires_gpu=True, min_vram_gb=12.0),
    models={"pipe": sdxl.flavor("bf16")},
)
def generate(ctx, pipe: StableDiffusionXLPipeline, payload: Input) -> Output:
    images = pipe(payload.prompt).images
    return Output(image=gw_io.write_image(ctx, "out", images[0]))
```

`Resources` is the per-function hardware envelope plus dynamic cost shape (used
by the orchestrator for placement and admission). `Repo(ref).flavor(name)` is
the binding — see [docs/endpoint-authoring.md](docs/endpoint-authoring.md) for
the full grammar.

## Three binding shapes

**Fixed pick** — function pins one specific `(repo, flavor?, tag?)`:

```python
models={"pipe": Repo("acme/flux").flavor("bf16")}
```

**Dispatch pick** — payload-driven, keyed by a `Literal[...]`-typed field:

```python
from typing import Literal

class Input(msgspec.Struct):
    variant: Literal["nf4", "int8"]
    prompt: str

@inference_function(
    resources=Resources(requires_gpu=True, min_vram_gb=14.0),
    models={"pipe": dispatch(
        field="variant",
        table={
            "nf4":  flux.flavor("nf4"),
            "int8": flux.flavor("int8"),
        },
    )},
)
def generate(ctx, pipe, payload: Input) -> Output: ...
```

**Override-allowed** — caller may substitute the default, subject to a
pipeline-class allowlist the tenant declares:

```python
models={"pipe": flux.flavor("bf16").allow_override(StableDiffusionXLPipeline)}
```

The caller then sends `{"prompt": "...", "_models": {"pipe": "acme/my-finetune:prod#bf16"}}`
to substitute. Class mismatch → request rejected before dispatch.

## Public surface

Top-level `gen_worker` exports only what endpoint authors need:

- Decorators + bindings: `inference_function`, `Resources`, `Repo`, `Dispatch`, `dispatch`
- Context types: `RequestContext`, `ConversionContext`, `DatasetContext`, `TrainingContext`
- Value types: `Asset`, `Tensors`, `Compute`, `LoraSpec`
- Errors: `ValidationError`, `RetryableError`, `FatalError`, `ResourceError`,
  `AuthError`, `CanceledError`, `OutputTooLargeError`, `InputTooLargeError`,
  `WorkerError`
- Helpers: `Clamp`, `iter_transformers_text_deltas`, `load_loras`,
  `apply_low_vram_config`, `with_oom_retry`
- I/O codecs: `gen_worker.io` (`read_image`, `read_audio`, `write_image`,
  `read_bytes`, `open`, `exists`)

Training and conversion live in their own submodules: `gen_worker.trainer`,
`gen_worker.conversion`, `gen_worker.clone`.

## Documentation

- [docs/endpoint-authoring.md](docs/endpoint-authoring.md) — full reference: the
  three layers, `Resources`, bindings, `dispatch`, `allow_override`,
  multi-param injection, the `_models` envelope, atomic substitution.
- [docs/endpoint-toml.md](docs/endpoint-toml.md) — `endpoint.toml` reference:
  build modes, placement fields, build hints, `BASE_IMAGE` injection.
- [docs/dockerfile.md](docs/dockerfile.md) — the three Dockerfile contract
  points, when `ARG BASE_IMAGE` matters, multi-profile builds.
- [docs/scaling-hints.md](docs/scaling-hints.md) — `Resources` cost-shape
  fields used by the orchestrator for admission and scheduling.
- [docs/endpoint-envs.md](docs/endpoint-envs.md) — tenant-defined envs/secrets
  attached to a deployed endpoint at runtime.

## Examples

Working endpoints to copy from in `examples/`:

- `marco-polo/` — minimal inference endpoint
- `medasr-transcribe/` — audio transcription with a Hugging Face model
- `openai-codex/` — text generation
- `training-smoke/` — minimal trainer
- `from-scratch/` — boilerplate template
