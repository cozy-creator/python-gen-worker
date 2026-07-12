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
Hub("owner/repo", tag="latest", flavor="", storage_dtype="")  # tensorhub
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

## Model selection: `model=` is a payload argument (pgw#509)

Checkpoint selection is a runtime PAYLOAD FIELD, not a build-time fan-out. A
handler whose payload declares a field typed with a `ModelChoice` subclass
picks, per request, which curated checkpoint runs against the resident base.
16 near-identical fine-tunes = ONE `generate(model=)`, not 16 functions.

Declare the curated set as DATA — one row per checkpoint carrying its
`ModelRef` binding + typed per-model defaults:

```python
class SdxlDefaults(ModelDefaults, frozen=True):
    scheduler: Literal["euler_a", "dpmpp_2m_karras"]
    steps: int
    guidance: float

class SdxlModel(ModelChoice[SdxlDefaults], enum.Enum):
    WAI  = Model("wai-illustrious",     Hub("tensorhub/wai-illustrious"), SdxlDefaults("euler_a", 28, 6.0), hot=True)
    PONY = Model("cyberrealistic-pony", Hub("tensorhub/cyberrealistic-pony"), SdxlDefaults("dpmpp_2m_karras", 30, 5.0))

class TextToImage(msgspec.Struct):
    prompt: str
    model: SdxlModel = SdxlModel.WAI          # curated-only

@endpoint(models={"pipeline": Hub("tensorhub/wai-illustrious"), "vae": Hub("tensorhub/sdxl-vae")})
class SDXLFamily:
    def setup(self, pipeline, vae): ...
    def generate(self, ctx, p: TextToImage) -> ImageOutput:
        d = p.model.defaults                  # typed SdxlDefaults — no ctx.models sniffing
        ...
```

On the wire a pick is its id string (`"wai-illustrious"`); the JSON schema is
a closed `enum` (the curated allowlist). Defaults are manifest-exported so the
catalog/UI can render `steps: 28` before submit. Discovery emits the whole set
(each choice's binding + defaults + `hot`/`price` hints) for the scheduler to
warm-pool per checkpoint.

**BYOM is the field TYPE.** `model: SdxlModel` is curated-only; `model:
SdxlModel | ModelRef` additionally accepts an arbitrary client-supplied
`ModelRef` (bring-your-own-model). No `@byom` decorator, no `sources=` —
architecture compatibility is derived from the pipeline the endpoint's
`models=` loads. Per-method policy falls out of method=contract: a `generate`
method can be BYOM-open while a `generate_turbo` method (fixed distillation
LoRA) stays curated.

Divergent WIRE contracts are separate METHODS, not `Optional` fields; only
weight-sharing forces one class. A distilled turbo that shares the base is a
`generate_turbo` method on the same class (shares the resident base); a
standalone distilled checkpoint is a separate class/endpoint.

## Lanes: multi-model classes with shared components (gw#479)

A class binding 2+ pipeline slots whose snapshots share byte-identical
components (content-keyed by the files' blake3 digests) loads the shared set
ONCE; each slot's exclusive weights (its transformer) are an independent
residency entry the worker LRU-swaps under VRAM pressure:

```python
@endpoint(models={"t2i": Hub("org/base"), "edit": Hub("org/edit")})
class Generate:
    def setup(self, t2i: QwenImagePipeline, edit: QwenImageEditPlusPipeline): ...
    def generate(self, ctx, p: In) -> Out: ...  # picks self.t2i / self.edit
```

This COMPENSATES for split-vendor base+edit releases (Qwen, HiDream, Wan
t2v/i2v); unified models (one transformer doing t2i + edit) bind one model.

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

Live deltas are droppable; the completed request keeps a terminal record:
the worker folds every yielded delta into a `StreamResult` (`text`,
per-`item_id` `texts`, batch `items`, `usage`) and serializes it as the
request's output, so a client that never attached to the stream still
retrieves the result. Token endpoints should yield one
`gen_worker.TokenUsage(prompt_tokens=, completion_tokens=,
tokens_per_second=)` at the end of the stream — billing reads it from the
terminal `StreamResult.usage`.

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

### llama.cpp / GGUF

For `runtime="llama-server"` the bound snapshot may be the `.gguf` file or
a dir holding exactly one GGUF model (split shards count as one; several
quants fail closed — pin the flavor). Unless `-ngl`/`-c` are pinned in
`extra_args`, the worker reads the GGUF header and sizes `-ngl` + context
to the free-VRAM budget, degrading through fewer GPU layers (down to
CPU-only) instead of failing the boot. The serve image provides the
`llama-server` binary (native-build image class); gen-worker adds no
Python binding dependency.

`gen_worker.runtimes.llama` has the streaming client half —
`chat_deltas(server, messages, ...)` / `completion_deltas(server, prompt,
...)` are sync generators yielding `IncrementalTokenDelta` then one
`TokenUsage`, so a handler is one `yield from`:

```python
from gen_worker.runtimes.llama import chat_deltas
from gen_worker.runtimes.server import ServerHandle

@endpoint(model=Hub("org/llm-gguf"), resources=Resources(vram_gb=24),
          runtime="llama-server")
class Chat:
    def setup(self, model: str, server: ServerHandle) -> None:
        self.server = server

    def chat(self, ctx, p: ChatIn) -> Iterator[IncrementalTokenDelta]:
        yield from chat_deltas(self.server, p.messages,
                               max_tokens=p.max_tokens,
                               cancelled=lambda: ctx.cancelled)
```

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
