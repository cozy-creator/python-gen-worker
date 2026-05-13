# gen-worker

Python SDK for writing **endpoints** that run on Cozy's worker pool. You write
a decorated Python function; the SDK handles discovery, scheduling, model
loading, cancellation, file I/O, and reporting to the control plane.

Three endpoint kinds:

- **Inference** — request/response (optionally streaming).
- **Training** — long-running, stateful, can publish checkpoints back to a repo.
- **Conversion** — produces weight artifacts on a destination repo.

## Install

```bash
uv add gen-worker          # core
uv add gen-worker[torch]   # with PyTorch
```

## Quick start

```python
import msgspec
from gen_worker import RequestContext, inference_function

class Input(msgspec.Struct):
    prompt: str

class Output(msgspec.Struct):
    text: str

@inference_function
def hello(ctx: RequestContext, payload: Input) -> Output:
    return Output(text=f"Hello, {payload.prompt}!")
```

Pair it with an `endpoint.toml`:

```toml
schema_version = 1
name = "hello"
main = "my_pkg.main"   # import path that contains your @inference_function

[resources]
ram_gb = 2
cpu_cores = 1
```

…and a `Dockerfile`:

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY . /app
RUN pip install uv && uv sync --frozen
RUN mkdir -p /app/.tensorhub && \
    uv run python -m gen_worker.discovery > /app/.tensorhub/endpoint.lock
ENTRYPOINT ["uv", "run", "python", "-m", "gen_worker.entrypoint"]
```

Publish with `cozyctl endpoint deploy` (or via the platform UI). The control
plane reads `/app/.tensorhub/endpoint.lock` from the image and routes invocations.

## Reference

See [`docs/endpoint-authoring.md`](docs/endpoint-authoring.md) for the full
authoring guide: model injection, streaming, file uploads, training/conversion
contracts, error types, and local testing.

## Public surface

The top-level `gen_worker` module exports only what endpoint authors need:

- Decorators: `inference_function`, `Resources`
- Bindings: `Repo`, `Dispatch`, `dispatch`
- Context: `RequestContext` (inference; the base), `ConversionContext` (transform / conversion endpoints), `DatasetContext` (dataset-generation), `TrainingContext` (trainer-class)
- Types: `Asset`, `Tensors`, `Compute`, `LoraSpec`
- Errors: `ValidationError`, `RetryableError`, `FatalError`, `ResourceError`,
  `AuthError`, `CanceledError`, `OutputTooLargeError`, `WorkerError`
- Helpers: `Clamp`, `iter_transformers_text_deltas`, `load_loras`,
  `apply_low_vram_config`, `with_oom_retry`

Training and conversion live in their own submodules: `gen_worker.trainer`,
`gen_worker.conversion`, `gen_worker.clone`.

## Migrating 0.6.x → 0.7.0

The 0.7.0 cut replaces the `Annotated[T, ModelRef(...)]` injection pattern and
the `endpoint.toml [models]` table with a single `models={...}` kwarg on
`@inference_function`. `ResourceRequirements` and `ScalingHints` merged into
one `Resources` struct (declared **per function**). The `require_vram` /
`require_compute_capability` / `require_cuda_library` runtime helpers are
gone — the worker now boot-checks each function's `Resources` envelope
against host hardware and self-advertises only runnable functions.

```python
# 0.6.x:
from gen_worker import ModelRef, ResourceRequirements, ScalingHints, inference_function
from gen_worker.capability import require_vram

@inference_function(
    resources=ResourceRequirements(min_vram_gb=4.0),
    scaling_hints=ScalingHints(vram_scales_with=("width", "height")),
)
def generate(
    ctx,
    pipe: Annotated[FluxPipeline, ModelRef(Src.FIXED, ref="acme/flux", flavor="bf16")],
    payload: Input,
) -> Output:
    require_vram(22 * 1024**3)
    ...

# 0.7.0:
from gen_worker import Repo, Resources, inference_function

flux = Repo("acme/flux")

@inference_function(
    resources=Resources(
        requires_gpu=True,
        min_vram_gb=22.0,
        vram_scales_with=("width", "height"),
    ),
    models={"pipe": flux.flavor("bf16")},
)
def generate(ctx, pipe: FluxPipeline, payload: Input) -> Output:
    ...
```

Bare imports of the removed symbols (`ModelRef`, `ModelRefSource`, `Src`,
`ResourceRequirements`, `ScalingHints`, `require_vram`, etc.) raise
`ImportError` with a one-line migration pointer.
