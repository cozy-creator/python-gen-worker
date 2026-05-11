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

- Decorators: `inference_function`, `realtime_function`, `ResourceRequirements`
- Injection: `ModelRef`, `ModelRefSource`
- Context: `RequestContext`, `RealtimeSocket`
- Types: `Asset`, `Tensors`, `Compute`, `LoraSpec`
- Errors: `ValidationError`, `RetryableError`, `FatalError`, `ResourceError`,
  `AuthError`, `CanceledError`, `OutputTooLargeError`, `WorkerError`
- Helpers: `Clamp`, `iter_transformers_text_deltas`, `load_loras`,
  `apply_low_vram_config`, `with_oom_retry`

Training and conversion live in their own submodules: `gen_worker.trainer`,
`gen_worker.conversion`, `gen_worker.clone`.
