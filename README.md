# gen-worker

Python SDK for writing **endpoints** that run on Cozy's worker pool. You write
one decorated function or class; the SDK handles discovery, scheduling, model
download + placement, cancellation, file I/O, streaming, and reporting back to
the control plane.

## Install

```bash
pip install gen-worker[torch]   # for PyTorch inference/training
pip install gen-worker          # plain Python (e.g. API-proxy endpoints)
```

Optional extras: `[images]` / `[audio]` / `[video]` for media I/O,
`[vision]` for torchvision.

## Hello world

**`pyproject.toml`** — the one config value:

```toml
[tool.gen_worker]
main = "myendpoint.main"
```

**`main.py`**:

```python
import msgspec
from gen_worker import RequestContext, endpoint

class Input(msgspec.Struct):
    prompt: str

class Output(msgspec.Struct):
    text: str

@endpoint
def echo(ctx: RequestContext, payload: Input) -> Output:
    return Output(text=f"got: {payload.prompt}")
```

Run it locally, no orchestrator:

```bash
gen-worker run --payload '{"prompt": "hello"}'
```

`cozyctl build` / `cozyctl deploy` take it from here — the full path to a
deployed, billed endpoint is [tensorhub docs/writing-endpoints.md](https://github.com/cozy-creator/tensorhub/blob/master/docs/writing-endpoints.md).

Full API reference: [docs/endpoint-authoring.md](docs/endpoint-authoring.md).
The public surface is whatever `gen_worker/__init__.py` exports.

## Local development

```bash
gen-worker run --payload '{"prompt": "hello"}'  # one-shot in-process
gen-worker run --list                            # describe functions (JSON)
gen-worker serve                                 # warm local server
gen-worker invoke <fn> prompt=hello              # client for serve
gen-worker prefetch                              # weights only, no GPU
```

stdout for results, stderr for events; exit 0 / 1 / 2 / 3 / 130 for success /
user-exception / usage / model-resolution / SIGINT. Details:
[docs/local-dev.md](docs/local-dev.md); host contract:
[docs/host-integration.md](docs/host-integration.md).

### Running tests

```bash
uv run --extra dev pytest
```

Plain `uv run pytest` would fall through to a global launcher — always pass
`--extra dev`. **Never `pip install` gen-worker globally:** a stale
`~/.local` install silently shadows the working tree (`tests/conftest.py`
hard-fails if `gen_worker` resolves outside `src/`).

## Documentation

- [docs/endpoint-authoring.md](docs/endpoint-authoring.md) — the `@endpoint`
  reference: bindings, variants, Resources, contexts, streaming, runtimes.
- [docs/local-dev.md](docs/local-dev.md) — the CLI: `run`/`serve`/`invoke`/
  `prefetch`, `field=value` grammar, `--offline`, exit codes.
- [docs/dockerfile.md](docs/dockerfile.md) — bring-your-own-Dockerfile contract.
- [docs/host-integration.md](docs/host-integration.md) — the versioned contract
  a host orchestrator (cozy-local) drives gen-worker over.
- [docs/multi-gpu.md](docs/multi-gpu.md), [docs/compile-cache.md](docs/compile-cache.md),
  [docs/convert.md](docs/convert.md) — measured results and the rulings from them.

Tenant endpoint envs/secrets are tensorhub's contract:
[docs/endpoint-envs-api.md](https://github.com/cozy-creator/tensorhub/blob/master/docs/endpoint-envs-api.md).

## Examples

`examples/` — `marco-polo` (minimal CPU endpoint: sync, async, streaming),
`sd15-image`, `sd15-hub-image`, `flux2-klein-image` (real GPU inference).
