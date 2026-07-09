# Endpoint authoring

API reference for the `@endpoint` surface. For the platform-side narrative —
quickstart, deploy, pricing, model-binding practice, the DON'Ts — read
[tensorhub docs/writing-endpoints.md](https://github.com/cozy-creator/tensorhub/blob/master/docs/writing-endpoints.md)
first.

One decorator: `@endpoint`. A plain function for stateless endpoints; a class
with an optional `setup()` when you hold state (model weights, an engine).

## Stateless: a function

```python
import msgspec
from gen_worker import RequestContext, endpoint

class In(msgspec.Struct):
    text: str

class Out(msgspec.Struct):
    reply: str

@endpoint
def echo(ctx: RequestContext, data: In) -> Out:
    return Out(reply=data.text)
```

Payload and return types are `msgspec.Struct`s — they validate the wire
payload and compile into the endpoint's public schema. Put input bounds on
the struct (`Annotated[int, msgspec.Meta(ge=1, le=50)]`), not in handler code.

## Stateful: a class

Every public method taking `(self, ctx, payload)` is a routable function;
prefix helpers with `_`. `setup()` runs once before the first request;
`shutdown()` (optional) runs at process end.

```python
from diffusers import FluxPipeline
from gen_worker import HF, RequestContext, Resources, endpoint

@endpoint(
    model=HF("black-forest-labs/FLUX.1-dev", dtype="bf16"),
    resources=Resources(vram_gb=24),
)
class Generate:
    def setup(self, model: FluxPipeline) -> None:
        self.model = model

    def generate(self, ctx: RequestContext, p: In) -> Out:
        img = self.model(prompt=p.text, generator=ctx.generator(42)).images[0]
        return Out(reply=ctx.save_image(img).ref)
```

The worker downloads the model, constructs the pipeline from the `setup()`
annotation (`FluxPipeline` → `from_pretrained`; `str`/`Path` → the local
snapshot dir), and owns device placement and low-VRAM offload. Endpoint code
never calls `.to("cuda")`, `enable_model_cpu_offload()`, or `empty_cache()`.

## Bindings

The slot name is the `models={}` key (or, with the single-binding `model=`
shorthand, the `setup()` parameter name). It is never a constructor argument.

```python
HF("owner/repo", revision=..., dtype=..., subfolder=..., files=(...), storage_dtype=...)
Hub("owner/repo", tag="prod", flavor="", storage_dtype="")   # tensorhub
Civitai("123456", version="789")             # civitai model id
ModelScope("owner/repo", revision=..., files=(...))
```

`files` are `snapshot_download` allow-patterns for split-checkpoint repos.

`storage_dtype="fp8"` keeps denoiser weights in fp8-E4M3 STORAGE with
per-layer upcast to the compute `dtype` (diffusers layerwise casting) — half
the denoiser VRAM on any card, no fp8 silicon required. Snapshots whose
weights are already fp8-stored (an `#fp8` flavor) get the same treatment
automatically; endpoint code stays precision-agnostic and
`ModelEvent.vram_bytes` reports the measured resident size. Quantized
formats are platform-produced stored flavors (`#fp8`, `#nvfp4` on Blackwell)
— there is no runtime "quantize my model" kwarg. The one exception is the
EMERGENCY rung (automatic on CUDA hosts): when
even the downloaded flavor cannot fit free VRAM, the loading layer
runtime-quantizes the denoiser to 4-bit nf4 with a loud warning (quality
below platform standards) rather than falling straight to CPU offload.
Fit ladder: bf16 → `#fp8` → `#nvfp4` (Blackwell) → emergency-nf4 → offload.

## Variants

One handler body, N separately-placeable routable functions:

```python
@endpoint(
    model=HF("org/base", dtype="bf16"),
    resources=Resources(vram_gb=24),
    variants={
        "generate-fp8": (HF("org/base-fp8"), Resources(vram_gb=14)),
    },
)
class Generate:
    def setup(self, model: FluxPipeline): self.model = model
    def generate(self, ctx, p: In) -> Out: ...
```

Each variant key is a routable function name with its own binding and
`Resources` (falls back to the class values). The base method-named function
is stamped only when the class declares `model=`/`models=`. If your payload
echoes the variant in a `variant` field, type it `Literal[...]` — members are
validated against the declared variants at import time.

## Resources

```python
Resources(gpu=True, vram_gb=24, compute_capability=8.0, libraries=("nunchaku",))
```

`vram_gb=`/`compute_capability=` imply `gpu=True`.

## Kinds

`@endpoint(kind="conversion" | "training" | "dataset")` selects the context
subclass the handler receives: `ConversionContext` adds `save_checkpoint` /
`mktemp` / `source` / `destination`; `DatasetContext` adds
`publish_dataset_revision` / `resolve_dataset`; `TrainingContext` adds the
repo-metadata RPCs.

Producer endpoints publish **explicitly**: write files locally, call
`gen_worker.convert.publish_flavors(ctx, flavors)` — one Tensorhub commit per
`ProducedFlavor` (path = file or directory) — and return a result struct:

```python
@endpoint(kind="conversion")
class Convert:
    def run(self, ctx: ConversionContext, p: In) -> Out:
        out_dir = ctx.mktemp()
        ...  # write model files under out_dir
        commits = publish_flavors(
            ctx, [ProducedFlavor(path=out_dir, flavor="bf16")],
            destination_repo=p.destination_repo,
        )
        return Out(revision_ids=[c.revision_id for c in commits])
```

Generator handlers are rejected for producer kinds — yielding streams
chunks, it never publishes.

## Streaming

An async-generator handler streams (inference kinds only); each yielded
struct is one chunk:

```python
async def stream(self, ctx, p: In) -> AsyncIterator[Out]:
    async for tok in self.engine.generate(p.text):
        ctx.raise_if_cancelled()
        yield Out(reply=tok)
```

For multi-item binary streams yield `gen_worker.BatchItemDelta(index=,
total=, item_id=, finished=, error=, chunk=, content_type=)` — no ad-hoc
field names.

## Engine-hosted runtimes

`@endpoint(runtime="vllm")` (or `"llama-server"`) makes the worker boot the
engine server around `setup()`: download the bound model, start the
subprocess, wait for `/health`, and inject a `ServerHandle` (base_url +
process control) into any setup parameter annotated with it:

```python
from gen_worker.runtimes.server import ServerHandle

@endpoint(model=HF("org/llm"), resources=Resources(vram_gb=40), runtime="vllm")
class Chat:
    def setup(self, model: str, server: ServerHandle) -> None:
        self.base_url = server.base_url
```

The worker aborts the boot on failure and stops the server at teardown.

## RequestContext

At most 15 members:

| member | |
|---|---|
| `request_id` | unique id for this request |
| `models` | resolved model refs by slot |
| `device` | the torch device to run on |
| `generator(seed)` | seeded `torch.Generator` on `device` |
| `deadline`, `time_remaining()` | absolute deadline / seconds left |
| `cancelled`, `raise_if_cancelled()` | THE cancellation spelling |
| `progress(fraction, stage=)`, `log(msg, level=)` | events to the caller |
| `save_bytes/file/image/audio/video` | persist outputs → typed `Asset` |

## Project config

`pyproject.toml` carries the one config value (there is no endpoint.toml):

```toml
[tool.gen_worker]
main = "my_endpoint.main"
```

## Errors

Raise `ValidationError` (bad input, don't retry), `RetryableError`,
`CanceledError`, or `FatalError`. Anything else is reported as an internal
error.

## Local dev

```bash
gen-worker run --payload '{"text":"marco"}'   # one-shot; picks the function
gen-worker run --list                          # machine-readable description
gen-worker serve                               # warm server on a unix socket
gen-worker invoke <fn> text=marco              # client for serve
gen-worker run --attach ...                    # route run through warm serve
gen-worker prefetch                            # download weights, no GPU
```

See [local-dev.md](local-dev.md) for the `field=value` payload grammar,
`--offline`, exit codes, and SIGINT semantics.
