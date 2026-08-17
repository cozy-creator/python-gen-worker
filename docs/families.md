# Families — the one authoring contract

A **family** is how a model is described, declared, bound and called. There is
no second way: the runtime never accepts a bare pipeline or module, so a
catalog family and an author-defined inline family are identical at the call
site and promoting one into the catalog is a copy rather than a rewrite.

```python
from gen_worker import RequestContext, endpoint
from gen_worker.family.catalog import Sdxl

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
| class | the family class | buckets, graph classes, signatures, loop, scheduler |
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
FLUX1_DEV = GraphFamily(
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
gen-worker family export   gen_worker.family.catalog.sdxl:SDXL
gen-worker family generate src/gen_worker/family/catalog/_generated/sdxl.export.json \
    --spec gen_worker.family.catalog.sdxl:SDXL
```

`export` runs `torch.export` with **fake** tensors — no GPU, no weights, no
network, no compile — and writes `family_export_v1`. `generate` turns that
document into the typed module, and is a pure function of it: no torch, no
diffusers. Both halves are committed, and
`scripts/check_family_bindings.py` (a required CI gate) proves they still agree.

Bindings generate from the **declaration**, never from a mint-emitted recipe
(torchcg G16) — otherwise a new family could not type-check until somebody
rented a card. The recipe a mint emits is the drift assertion
(`gen_worker.family.assert_recipe`): same runners, same ingress digests, or
refuse.

## `Tuned[...]`: stop writing the precedence by hand

Every handler used to open with a line per field:

```python
steps = p.steps if p.steps is not None else ctx.defaults.steps
```

which is once per place the precedence can be written backwards. Annotate the
payload field instead:

```python
from gen_worker.family import Tuned, resolve

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

Aspect ratios and megapixel tiers are endpoint policy; a resolution graph class
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
