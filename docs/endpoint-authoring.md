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
    resources=Resources(gpu=True),
)
class Generate:
    def setup(self, model: FluxPipeline) -> None:
        self.pipeline = model

    def generate(self, ctx: RequestContext, p: In) -> Out:
        view = ctx.for_request(self.pipeline, seed=42)
        img = view(prompt=p.text, generator=view.generator).images[0]
        return Out(reply=ctx.save_image(img).ref)
```

Handlers take exactly `(self, ctx, payload)` and use the instance state
`setup()` stored — one live instance == one binding set, so "which
checkpoint am I?" never needs asking; the runtime routed the request here
because the bindings match. Anything request-scoped (sampler/scheduler
state, seed/generator, latents) lives in a per-request VIEW
(`ctx.for_request`), never assigned onto `self.pipeline` — diffusers
schedulers are stateful and shared, so per-request instance mutation
corrupts concurrent requests. RETURN the output object; the SDK owns
encode + upload (hand-uploading inside the handler opts out of the
encode/upload tail overlap).

The worker downloads the model, constructs the pipeline from the `setup()`
annotation (`FluxPipeline` → `from_pretrained`; `str`/`Path` → the local
snapshot dir), and owns device placement and low-VRAM offload. Endpoint code
never calls `.to("cuda")`, `enable_model_cpu_offload()`, or `empty_cache()`.

## Tensor state: `register_buffer`, never a plain attribute (pgw#857)

If your model class holds a tensor that is **not** a learned parameter — a rope
frequency table, a precomputed mask, any "just a cache" — **register it as a
buffer**:

```python
# WRONG — a plain attribute
self.pos_freqs = torch.cat([...])          # not in state_dict

# RIGHT
self.register_buffer("pos_freqs", torch.cat([...]), persistent=False)
```

**The discriminator is ASSIGNMENT STYLE, not class ancestry.** It is tempting to
assume "it is an `nn.Module`, so its tensors are tracked" — that is false.
Measured: diffusers' `QwenEmbedRope` **is** an `nn.Module` and its
`state_dict()` is **empty**, because its two 2 MB rope tables are plain
attributes. z-image's `RopeEmbedder` has the same problem for a different
reason (it is not an `nn.Module` at all), which is why "is it a module" is the
wrong test to carry away.

**Why it matters beyond tidiness.** A tensor outside `state_dict` is not a
weight the CAS delivers and rebinds at load. `torch.export` lifts it as an
anonymous `_tensor_constant{N}` **literal**, and a literal's bytes ship *inside*
the compiled cell. You have moved MBs out of the dedup'd weight store into
every compiled artifact, and coupled cell identity to a number you probably
meant to be a cache. Nothing breaks today (pgw#857 folds literal VALUES into
the cell key), but it is pure cost.

A registered buffer costs nothing and stays a weight.

## Imports go at module top — including torch

Write `import torch` (and every other heavy dep) at module top like any
normal Python. Build-time discovery imports your module to read `@endpoint`
metadata; when a heavy dep isn't installed in the discovery environment, the
SDK stubs it (allowlist: torch, torchvision, torchaudio, triton, xformers,
flash_attn, bitsandbytes — extend via
`[tool.gen_worker] discovery_heavy_deps = ["my_heavy_lib"]`), so the import
costs nothing. The old convention of deferring `import torch` into handler
bodies is retired — don't do it.

The one boundary: don't EXECUTE heavy-dep code at module scope
(`DTYPE = torch.bfloat16`, `torch.cuda.is_available()` at import time).
Under a stub that fails discovery loudly with a message naming the fix —
move the code into `setup()` or the handler.

Discovery also hard-fails when any module in your package fails to import
for any OTHER reason (missing non-heavy dep, SyntaxError): a broken
submodule fails the build with the real traceback instead of silently
dropping its functions from the manifest.

## Bindings

The slot name is the `models={}` key (or, with the single-binding `model=`
shorthand, the `setup()` parameter name). It is never a constructor argument.

```python
HF("owner/repo", revision=..., dtype=..., subfolder=..., files=(...), components=(...), storage_dtype=...)
Hub("owner/repo", tag="latest", flavor="", components=(...), storage_dtype="")  # tensorhub
Civitai("123456", version="789")             # civitai model id
ModelScope("owner/repo", revision=..., files=(...))
```

`files` are `snapshot_download` allow-patterns for split-checkpoint repos.
Measured worth checking for: FLUX.2-klein-4B's DiT is **~7.75 GB bf16**, and
the source repo ships a redundant root-level single-file checkpoint alongside
its real diffusers-layout weights (`transformer/`, `text_encoder/`, `vae/`).
Narrowing with `files=` **roughly halves the download**. Vendor repos
carrying both layouts are common — check before assuming the repo size is
the model size.

`components` (tensorhub/huggingface only) fetches only the named pipeline
component subfolders instead of the whole repo — root config files
(`model_index.json` and other root `*.json`) are always kept alongside. The
win case: a slot binds a full pipeline repo but only needs ONE component out
of it, e.g. `Hub("owner/sdxl-repo", components=("vae",))` for a VAE swap —
`unet`/`text_encoder`/etc. never download. Civitai/modelscope reject it
(civitai artifacts aren't component-structured; modelscope has `files=`).

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

## Model selection — the design decisions

The API surface is `Slot`, `ModelChoice`, `Model`, `ModelDefaults` and
`gen_worker.families` (read their signatures). What the code does not tell
you is why they are shaped this way:

- **Checkpoint selection is a runtime PAYLOAD FIELD, not a build-time
  fan-out** (pgw#509). 16 near-identical fine-tunes = ONE `generate(model=)`,
  not 16 functions.
- **BYOM is the field TYPE**, not a decorator. `model: SdxlModel` is
  curated-only; `model: SdxlModel | ModelRef` is BYOM-open. There is no
  `@byom` and no `sources=`; architecture compatibility derives from the
  pipeline the endpoint loads. Per-method policy falls out of
  method=contract.
- **Divergent WIRE contracts are separate METHODS, not `Optional` fields.**
  Only weight-sharing forces one class.
- **The model SET is CATALOG, not code** (th#767) — adding a checkpoint must
  not require a software release. That is what `Slot` is for; `ModelChoice`
  bakes the set into the image and is the first-party-recipe case.
- **The component tree is DERIVED from the pipeline class**, never declared.
  Declaring a part as a sibling slot (`"vae": Slot(AutoencoderKL)`) or a
  `str` modifier slot beside a pipeline slot is a **decoration-time error**:
  component overrides are catalog data (th#1116) and adapters ride the model
  binding. Explicit multi-slot declaration survives only for runtimes the SDK
  cannot introspect (llama/gguf, custom engines).
- **Code owns the config SCHEMA; the catalog owns the VALUES.** The schema
  derives from the context annotation (`ctx: RequestContext[SdxlDefaults]`) —
  `default_config=` was deleted rather than kept as a second answer.
- **Component sharing is AUTOMATIC by content address**, refcounted across
  picks — `share_components=` was deleted for the same reason.
- **The boot warm plan is DERIVED, not written.** There is no `warmup=`
  payload dict, precisely so it cannot drift from the schema.
- **Per-request state lives in a VIEW** (`ctx.for_request`), never assigned
  onto the instance: diffusers schedulers are stateful and shared, so
  per-request instance mutation corrupts concurrent requests. The SDK owns
  the sampler-name table (`gen_worker.view.SAMPLERS`) — endpoints never
  define private sampler maps.

## Resources

Declare ONLY what the endpoint cannot run without:

```python
Resources(gpu=True, libraries=("nunchaku",))
```

Fields: `gpu`, `gpu_count`, `libraries`, `strict_vram` (bindings that
cannot tolerate CPU-resident weights), `vcpus`, `compute_capability`, and
the two hints `vram_gb_hint` / `ram_gb_hint`.

**Hints vs. gates — the distinction is the whole contract.**

`vram_gb_hint` is an optional FIRST-BUILD placement hint used only before
th#683 profiling measurements exist; `ram_gb_hint` (pgw#670) is its
host-side twin and does not imply `gpu=True`. Both are allocation-time
asks the platform may miss: never a gate, ceiling, or reservation.

`compute_capability` (pgw#660) is the opposite — a HARD GPU-architecture
floor the scheduler filters offers on and refuses to rent below. Declare
the dotted capability the way NVIDIA writes it (`8.9`, `"8.9"`, or
`"sm_89"`, never the bare SM code `89`); it implies `gpu=True`.

```python
# scaled_mm is sm_89+ or nothing — an incapability, not a slow rung.
Resources(gpu=True, compute_capability=8.9, libraries=("modelopt",))
```

Declare it ONLY for genuine incapability. A function that merely runs
*better* on newer silicon declares nothing and lets the fit ladder choose;
over-declaring shrinks the rentable pool for no reason. Omitting it is
always safe — no key, no gate, today's behaviour.

`vram_gb` and `ram_gb` remain deleted from SDK v2 (measured requirements
belong to the profiler; host RAM is an opportunistic tier), as does v1's
`min_compute_capability` spelling — the hub rejects that key outright.

## Kinds

`@endpoint(kind="conversion" | "training" | "dataset")` selects the context
subclass the handler receives: `ConversionContext` adds `save_checkpoint` /
`mktemp` / `source` / `destination`; `DatasetContext` adds
`publish_dataset_revision` / `resolve_dataset`; `TrainingContext` adds the
typed training-metric emitter.

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

@endpoint(model=HF("org/llm"), resources=Resources(gpu=True), runtime="vllm")
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

@endpoint(model=Hub("org/llm-gguf"), resources=Resources(gpu=True),
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
| `defaults` | the catalog-resolved recipe, typed as `RequestContext[D]` |
| `for_request(pipeline, sampler=, seed=)` | per-request view (own scheduler over shared weights) |
| `device` | the torch device to run on |
| `generator(seed)` | seeded `torch.Generator` on `device` |
| `deadline` | absolute deadline |
| `cancelled`, `raise_if_cancelled()` | THE cancellation spelling |
| `progress(fraction, stage=)` | USER-facing status event (the job card) |
| `log(msg, level=, **fields)` | PLATFORM/OPERATOR diagnostic, never user-facing (pgw#508) |
| `save_bytes/file/image/audio/video` | persist outputs → typed `Asset` |

Logging rule of thumb: module-level `logging.getLogger(__name__)` for
boot-time/cross-request logging; `ctx.log` for anything scoped to THIS
request you'd want when debugging it; `ctx.progress` for what the human
watching the job should see.

## Project config

`pyproject.toml` carries the one config value gen-worker itself reads:

```toml
[tool.gen_worker]
main = "my_endpoint.main"
```

gen-worker needs no `endpoint.toml`. **tensorhub's builder requires one** —
a build tarball without it is rejected with `ErrNoEndpointToml`
(`internal/builder/validation.go`). It carries build profiles and platform
metadata, not SDK config.

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
