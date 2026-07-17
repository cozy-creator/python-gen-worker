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

## Adding a model

Hold state in a class: `setup()` runs once, every public method is one
routable function. The worker downloads the binding, constructs the pipeline
from the `setup()` annotation, and owns device placement + low-VRAM offload —
endpoint code never touches `.to("cuda")` or offload config.

```python
from diffusers import StableDiffusionXLPipeline
from gen_worker import HF, RequestContext, Resources, endpoint

@endpoint(
    model=HF("stabilityai/stable-diffusion-xl-base-1.0", dtype="bf16"),
    resources=Resources(vram_gb=12),
)
class Generate:
    def setup(self, pipe: StableDiffusionXLPipeline) -> None:
        self.pipe = pipe

    def generate(self, ctx: RequestContext, payload: Input) -> Output:
        image = self.pipe(payload.prompt, generator=ctx.generator(42)).images[0]
        return Output(text=ctx.save_image(image).ref)
```

`Resources(vram_gb=N)` recommends a card size — the total VRAM of the smallest
card the function targets, not free bytes. It's an optional placement hint: the
platform reserves ~1 GB for driver/framebuffer/CUDA-context overhead, so
`vram_gb=24` serves on any 24 GB card.

Bindings: `HF(id, revision=, dtype=, subfolder=, files=, storage_dtype=)`,
`Hub(ref, tag=, flavor=, storage_dtype=)`, `Civitai(id, version=)`, `ModelScope(id, ...)`.
The slot name comes from the `models={}` key or the `setup()` parameter —
never a constructor argument. `storage_dtype="fp8"` keeps denoiser weights in
fp8-E4M3 storage with per-layer upcast to the compute `dtype` (half the VRAM
on any card); fp8-stored `#fp8` flavors get the same treatment automatically.

Curated checkpoint selection is a runtime payload argument: a handler declares
`model: SomeModelChoice` (a `ModelChoice` enum of `Model` rows, each carrying a
`ModelRef` binding + typed per-model defaults) and reads `payload.model.defaults`
typed — one `generate(model=)` replaces N near-identical functions. `model:
SomeModelChoice | ModelRef` opens BYOM. Streaming = an async-generator handler.
Engine-hosted endpoints declare `runtime="vllm"` and get a booted,
health-checked server subprocess injected into `setup()`.

`Slot(pipeline_cls, selected_by=, default_checkpoint=, default_config=)` is
the hub-resolved alternative to `ModelChoice`: the model SET lives in
platform config, not code. `ctx.slots["<name>"]` returns a typed
`ResolvedSlot` — repo-metadata inference defaults (a
`gen_worker.families.FamilyDefaults` vocabulary, tensorhub-validated)
merged over the endpoint's code `default_config=` preset (which LOSES to
repo metadata — a recipe of last resort).

Full reference: [docs/endpoint-authoring.md](docs/endpoint-authoring.md).

## Public surface

- The decorator + bindings: `endpoint`, `Resources`, `Compile`, `HF`, `Hub`,
  `Civitai`, `ModelScope`, `ModelRef`
- Model selection: `Model`, `ModelChoice`, `ModelDefaults`, `Slot`,
  `ResolvedSlot`, `gen_worker.families.FamilyDefaults`
- Contexts: `RequestContext` (≤15 members), `ConversionContext`,
  `DatasetContext`, `TrainingContext`
- Errors: `ValidationError`, `RetryableError`, `CanceledError`, `FatalError`
- Streaming: `BatchItemDelta`, `IncrementalTokenDelta`, `Done`, `Error`
- Value types: `Asset`, `ImageAsset`, `AudioAsset`, `VideoAsset`
- I/O codecs: `gen_worker.io`

The conversion ETL (hub ingest, dtype cast / quant, clone, Tensorhub
publish) is `gen_worker.convert` (see [docs/convert.md](docs/convert.md)).

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
- [docs/endpoint-envs.md](docs/endpoint-envs.md) — tenant envs/secrets.

## Examples

- `examples/marco-polo/` — minimal inference endpoint (sync, async, streaming)
