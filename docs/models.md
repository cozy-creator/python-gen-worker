# Models — the one authoring contract

A **model** is described, declared, bound and called exactly one way. There is
no second: the runtime never accepts a bare pipeline or module, so a catalog
model and an author-defined inline one are identical at the call site and
promoting one into the catalog is a copy rather than a rewrite.

The declaration is a `ModelSpec` (eager) or a `GraphModelSpec` (with graph
classes); codegen turns it into a `Model` subclass, and a handler parameter
annotated with that subclass receives an instance of it. Class and instance are
Python's own, not two SDK types (pgw#1346).

```python
from gen_worker import RequestContext, endpoint
from gen_worker.model.catalog import Sdxl

@endpoint(families={"sdxl": Sdxl})
class Generate:
    def generate(self, ctx: RequestContext, p: In, sdxl: Sdxl) -> Out:
        steps = p.steps if p.steps is not None else sdxl.tuned.steps
        latents = sdxl.denoiser(
            resolution=1024,
            sample=sample, timestep=t,
            encoder_hidden_states=embeds,
            added_cond_kwargs={"text_embeds": pooled, "time_ids": ids},
        )
        return Out(image=ctx.save_image(sdxl.decoder(resolution=1024, latents=latents)).ref)
```

`Sdxl` is the **type**; the injected value is a fully resolved **instance** —
graph, bound weights, and the checkpoint's catalog-stamped tuned values. Two
parameters of one family type are two checkpoints with independent tuning
(`flux_a: Flux1Dev, flux_b: Flux1Dev`), sharing one compiled artifact and
paying VRAM per weight set.

## Three axes, and they do not cross

| axis | lives on | examples |
|---|---|---|
| class | the family class | buckets, graph specializations, signatures, loop, scheduler |
| checkpoint | the instance | weights binding, `inst.tuned`, the ref label |
| request | `ctx` | cancellation, logging, progress, seed, assets |

`inst.tuned` is checkpoint-level, so it rides the instance. The values are still
stamped by the catalog per release slot; only the delivery address changed.

## Getting an instance

* **Declared** — `@endpoint(families={...})`. The default, because it is what
  lets placement prefetch the weights and verify the VRAM fit before a request
  lands. The declaration is emitted into `endpoint.lock`.
* **Dynamic** — `Flux1Dev.instance(payload.checkpoint_ref)` inside a handler.
  The one parse-don't-validate boundary; a request naming its own checkpoint is
  parsed once, there.
* **Fake** — `Sdxl.fake()`. Shape-correct deterministic tensors derived from the
  declaration, with no hub, no GPU and no weights, so a handler's payload
  validation, bucket mapping, composition, streaming and asset saving are all
  testable in CI through the real code path.

## Backings: the handler cannot tell

A typed callable's contract is its ingress digest; its backing is an adopted
compiled runner, the family's own eager module, or the fake one. Handler code
never branches on which. First serve of a family on a trusted eager-capable pod
serves eagerly and mints as a side effect (DESIGN-RULINGS §4.28); a pod with no
artifact and no eager half refuses, and the hub routes — nothing requests a
compile and nothing waits on one.

## Declaring a family

```python
FLUX1_DEV = GraphModelSpec(
    name="flux1_dev",
    tuned=Flux1DevTuned,                       # the SHAPE; catalog owns values
    buckets=(Bucket("resolution", (768, 1024)),),
    runners=(
        Runner("decoder",  build=_decoder,  example=_decoder_example,  axes=("resolution",)),
        Runner("denoiser", build=_denoiser, example=_denoiser_example, axes=("resolution",)),
    ),
    loop=Loop(stages=(Stage("denoiser", repeat="steps"), Stage("decoder"))),
    parameters=(Parameter("steps", minimum=1, maximum=100),),
    scheduler=Scheduler("flow_match_euler_discrete", {"shift": 3.0}),
)
```

`build(layout)` returns the eager module — the same one the mint traces.
`example(bucket, layout)` returns the exact call it is exported against. Import
`diffusers` inside `build` only, never at module scope: an adopt-only serve role
must be able to import the declaration without acquiring model code.

Bucket coverage is total per layout, so a generated `Literal` is exhaustive and
every selection resolves. A name a generator would have to escape is refused at
declaration, not at generation.

## Build steps

```sh
gen-worker model export   gen_worker.model.catalog.sdxl:SDXL
gen-worker model generate src/gen_worker/model/catalog/_generated/sdxl.export.json \
    --spec gen_worker.model.catalog.sdxl:SDXL
```

`export` runs `torch.export` with **fake** tensors — no GPU, no weights, no
network, no compile — and writes `family_export_v1`. `generate` turns that
document into the typed module, and is a pure function of it: no torch, no
diffusers. Both halves are committed.

Bindings generate from the **declaration**, never from a mint-emitted recipe
(torchcg G16) — otherwise a new family could not type-check until somebody
rented a card. The recipe a mint emits is the drift assertion
(`gen_worker.model.assert_recipe`): same runners, same ingress digests, or
refuse.

## `Tuned[...]`: stop writing the precedence by hand

Every handler used to open with a line per field:

```python
steps = p.steps if p.steps is not None else ctx.defaults.steps
```

which is once per place the precedence can be written backwards. Annotate the
payload field instead:

```python
from gen_worker.model import Tuned, resolve

class In(msgspec.Struct):
    prompt: str = ""
    steps: Tuned[int] = None
    guidance: Tuned[float] = None

...
values = resolve(p, sdxl)      # {"steps": ..., "guidance": ...}
```

An explicit payload value wins; `None` means "no opinion" and falls through to
the checkpoint's tuned value. `Tuned[int]` is an `Annotated` alias of
`int | None`, so msgspec decodes it unchanged and the exported JSON Schema is
byte-identical to the plain optional's — the annotation adds derivability, not
a wire shape. A field added to the payload participates without anybody
updating a second list.

## Product grid ≠ family buckets

Aspect ratios and megapixel tiers are endpoint policy; a resolution graph specialization
is family truth. Declare the mapping and the build fails on a bucket the family
lacks, while the boot warm plan derives from the same mapping:

```python
GRID = BucketMap(SDXL, "resolution", {"square": 1024, "portrait": 768})
GRID.bucket_for("portrait")   # 768
GRID.warm_plan()              # (768, 1024)
```

## Autoregressive families

An LLM/VLM family declares `Loop(kind=LoopKind.HOST, session_state=...)`. The
recipe states the per-step classes and who owns the state threaded between
them, and says outright that the iteration is host code — so codegen emits the
typed callables and **no** driver, step count or termination condition.

```python
with llm.session() as decode:
    decode.state["kv"] = ...          # host-owned, torn down with the session
    logits = llm.decode(context=1024, tokens=tokens)
```

## A catalog family is two modules (pgw#1331)

`<family>.py` **declares**: its `build` callables construct diffusers and
transformers modules, so it is mint-side and the serve role may not import it.
`<family>_serve.py` is what the request path reads — the tuned schemas, the
shape arithmetic, the pipeline loop — and imports nothing above `torch`. The
direction is one-way: the declaration reads from the serving half, never the
reverse, so the family's shape arithmetic has one definition and an artifact
cannot be minted at a shape the loop will not ask for.

`role.MODEL_FREE_MODULES` is the surface, `role.FORBIDDEN_LIBRARIES` is what
it may not reach, and `serve.guard` blocks those names at run time.

## Scheduler as bare math

The recipe records a scheduler as a NAME and a block of scalars, and says the
host implements it. `gen_worker.model.scheduler` is that host half:

```python
schedule = flux.scheduler().schedule(steps=28, image_seq_len=4096)
for index, sigma in enumerate(schedule.sigmas[:-1]):
    velocity = flux.denoiser(resolution=1024, timestep=..., hidden_states=latents, ...)
    latents = schedule.step(index, velocity, latents)
```

`scheduler()` is GENERATED and its return type is the concrete class, so no
handler spells a scheduler name. A family whose declared scheduler the SDK does
not implement has **no** `scheduler()` method — an `AttributeError` your type
checker reports, not a fallback that puts a model library back on the path.

The constants come from the DECLARATION's `Scheduler(...)` block, which rides
the export digest: re-declaring a schedule re-identifies the family instead of
silently changing every request.

## Serving a family end to end

`gen_worker.model.catalog.flux1_dev_serve` is the worked example — tokenize,
encode both text branches, denoise, decode, with no model library imported:

```python
from gen_worker.model.catalog import Flux1Dev, flux1_dev_serve as flux

image = flux.generate(
    instance,                    # a resolved Flux1Dev
    resolution=1024,
    clip_ids=flux.clip_token_ids(clip_tokens, device=device),
    t5_ids=flux.t5_token_ids(t5_tokens, device=device),
    steps=28, guidance=3.5, seed=seed,
)
```

The same body runs against a compiled backing, an eager one, or `Flux1Dev.fake()`
— which is how it is tested in CI without a card.

## Minting a family

```
gen-worker model mint gen_worker.model.catalog.flux1_dev:FLUX1_DEV \
    --out-dir ./compiled graphs --runner decoder --runner clip --json minted.json
```

A real compile: **run it on a pod, never on a shared box.** It needs a GPU and a
toolchain; it needs no weights and no network, because compiled graph identity is
checkpoint-free (§4.27) and the constants arrive at arm time from the store.
The trace is the declaration's own, so a minted class's ingress digest is the
committed export's by construction — `family.mint.assert_matches_export` states
that as an assertion, in the direction torchcg G16 requires: the export is the
source and the mint is checked against it.

## Declaring what the model's code can execute

Three axes ride the DECLARATION, because they describe the model's own code and
two endpoints binding one model state one demand:

```python
SDXL = GraphModelSpec(
    name="sdxl", tuned=SdxlTuned, runners=(...),
    layouts={"*": ("plain.bf16@1",),
             "text_encoder": ("cozy.fp8-rowwise@1", "plain.bf16@1")},
    layout_requirements={"cozy.fp8-rowwise@1": "sm89+, vram24g"},
)
```

* `layouts` — per COMPONENT PATH, the set of tensor-layout contract handles this
  model's code can execute. `"*"` is the whole-tree default. The set is a
  compatibility FILTER and its order carries no preference, so handles are
  stored in canonical order.
* `layouts_undeclarable` — the explicit third rung, a REASON string, mutually
  exclusive with `layouts`. For bytes no registered handle names: a tokenizer
  tree with no tensors, a GGUF quant axis, a compressed-tensors checkpoint.
* `layout_requirements` — keyed by the HANDLE it guards, what EXECUTING that
  contract needs of the machine. The compact term list IS the minimum;
  `recommended` is additive and gates nothing. A key naming a handle `layouts`
  does not accept is refused — a requirement guarding nothing is never checked.

`Runner.layouts` is a different axis and keeps its own meaning: which layouts
that GRAPH CLASS has traced variants for. A model may accept fp8 bytes for a
component it has no fp8 graph specialization for.

## Declaring what only the ENDPOINT knows

Three axes cannot live on a shared model, because they name this endpoint's
payload and this deployment. They ride `Bind`:

```python
@endpoint(models={"pipeline": Sdxl})                        # the common case
@endpoint(models={"pipeline": Bind(Krea2, selected_by="model")})
```

* `selected_by` — the `str`-typed payload field that branches which checkpoint
  serves this parameter.
* `default_checkpoint` — the code-side bootstrap ref, and the only resolution
  source in hub-less mode. A live hub mapping always wins.
* `root` — which model answers the residual `ctx` questions that resolve against
  one. Marking none is normal: a handler names every model it binds, so there is
  nothing for a root to disambiguate. Marking two is a decoration-time error.

`optional` is still DERIVED, never passed: a model is optional exactly when its
handler parameter carries a default.
