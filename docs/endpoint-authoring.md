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

## The tensor-binding contract: `register_buffer`, never a plain attribute (pgw#857)

A compiled cell rebinds tensors **by name** at load, which is what lets one cell
serve every fine-tune of a family. That only holds if every tensor your model
needs is reachable through `state_dict()`. **The tensor-binding contract**
(formerly "weight-binding") is that rule — the artifact's LINKING rule for
tensors. A tensor your module holds is either:

- **bound by name at load** (DYNAMIC) — a named `state_dict` entry the CAS
  delivers and rebinds. It is an opaque slot the compiler must never
  value-specialize, and that opacity is exactly what makes a cell
  CHECKPOINT-AGNOSTIC;
- **a baked literal** (STATIC) — its value folds into the artifact's identity,
  so two checkpoints differing only in that value need different cells. This
  case is driven to zero;
- for GB-scale derived data, neither: a **named CAS component** (corollary
  below), bound by name exactly like a weight.

There is no fourth outcome, and you choose which one by how you assign it.

`tensor-` and not `weight-`, deliberately: the rule governs scales, buffers and
computed tables, not just trained weights. Its sibling is the **tensor-layout
contract** — how tensors exist ON DISK (byte packing, scale layout, swizzle,
key-naming convention, file topology), named by a descriptor handle
`<producer>.<format>@<major>`. Layout says what the bytes ARE; binding says how
they are ADDRESSED at load.

**Declare the layout you execute, not a quant** (th#1803). Your code should be
quant-generic over its declared layout, so the platform can answer "will this
checkpoint work with this code?" before renting a pod — and so the owner can
rebind fp8↔bf16 as config. See "Do NOT quantize weights in `setup()`" below.

**You configure the compiler by how you write the code.** The classification is
not a setting supplied out of band — it derives from `state_dict` membership at
trace time. The code IS the configuration, and the assignment style below is the
whole of it.

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
the compiled cell — so it becomes part of the artifact rather than part of the
checkpoint. Two checkpoints of your family that differ only in the config that
derives that table then need *different* cells. The cell key handles this
correctly today (pgw#857 folds literal VALUES into the identity), so nothing
breaks — but you have moved several MB out of the deduplicated weight store and
into every compiled artifact, and you have coupled your cell identity to a
number you probably meant to be a cache.

A registered buffer costs nothing and stays a weight.

### The DYNAMIC half is now ENFORCED at mint, not assumed (pgw#1097)

"An opaque slot the compiler must never value-specialize" was, until this
issue, a property nothing checked. Inductor does not read the contract: with
constant folding left at torch's default, `GraphLowering.get_attr`
renders a lifted tensor's VALUES straight into the kernel source whenever its
**shape** meets either rule, 0-dim or `len(shape) == 1 and shape[0] <= 8`. The
tensor then appears in no table anyone can rebind, so the cell carries the
minting checkpoint's copy and every other fine-tune of the family silently gets
the wrong numbers. It is a shape rule, not a value rule: small norms, group-norm
scales, `logit_scale`-style learned scalars and short conv biases are the whole
target set, and they are exactly the tensors a fine-tune changes.

Two things close it, and you need neither in your model code:

- every mint compiles under `aot_mint.CONSTANT_BINDING_CONFIGS`
  (`aot_inductor.use_runtime_constant_folding=True`), which defers the fold to
  load so nothing is inlined; and
- `aot_package.folded_weights` PROVES it per entry against the artifact's own
  constant table, and a mint that lifted a weight the package does not declare
  is **refused by name**. A cell minted before the fence is refused at adoption
  too — `constant_folding_fenced` is a declared axis, like
  `package_constants_in_so`.

What this means for you: the classification you choose by assignment style is
now the classification you get. A `state_dict` tensor is DYNAMIC all the way to
the kernel, whatever its shape.

### Corollary: a GB-scale derived tensor is a saved component, not a buffer (pgw#1056)

The buffer rule above assumes the tensor is small enough that computing it at
`__init__` is free. **Scale decides, not provenance.** A large tensor derived
from config — MiniMax-H3's precomputed step-count positional tables are GBs — is
neither an init-computed buffer nor a baked literal. It is **saved model data**:
a named CAS component with its own safetensors header, bound by name exactly
like a weight (the composition system already does this for fp8 overrides).
Precompute it once, publish it as a component, and delete the generator.

The test is size, not where the numbers came from: big ⇒ component; only
trivially recomputable state stays a `register_buffer`-computed tensor.

### Corollary: `__init__` must be meta-device safe (pgw#1056, coming)

The zero-download forge instantiates your module on the **meta device** — code
plus config plus the safetensors header, no tensor bytes. A module that does
REAL tensor computation in `__init__` breaks that silently. A gate is coming
that detects materialized real tensors at instantiation and refuses with
authoring guidance, so the violation is caught at mint time instead of after a
cell ships. Write `__init__` so it allocates shapes and dtypes, never values.

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

### Type-check against the INSTALLED SDK (pgw#942)

A mypy run with `gen_worker` not installed is **not supported**, and the SDK
cannot make it work: with the package absent, `--ignore-missing-imports`
resolves every SDK name to `Any`, and mypy then types the members of
`class AspectRatio(StringEnum)` as bare `str` — so every `AspectRatio`-keyed
table lights up with false `dict-item` errors. A `.pyi` inside the wheel
cannot fix that, because the wheel is exactly what is missing.

So: install `gen-worker` in the environment your type-check runs in, and add
`--follow-imports=silent` so the SDK's own diagnostics are not attributed to
your package. Do NOT reshape production imports to satisfy a bare checker —
a `if TYPE_CHECKING: from enum import StrEnum as StringEnum / else: from
gen_worker import StringEnum` shim is a report script editing your source.
Import the name once, normally:

```python
from gen_worker import StringEnum
```

(`StringEnum` is a direct alias for `enum.StrEnum`, not a subclass — an
empty enum base makes a tenant's members resolve as `str`. It exists as the
named, documented vocabulary for endpoint payload enums.)

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
formats are platform-produced stored artifacts (`#fp8`, `#nvfp4` on Blackwell)
— there is no runtime "quantize my model" kwarg, and which one a component
serves is deploy CONFIG, not a literal you pick here (th#980, th#1803).

> **`flavor` is being deleted (th#1803, DESIGN-RULINGS §1.33).** Paul: the
> flavor system was *"an arbitrary-string sub-selector within a tag-group…
> too imprecise."* Selection within a tag group becomes tensor-layout-contract
> compatibility — the endpoint declares a per-slot SET of accepted layouts and
> the platform grades each candidate COMPATIBLE / CONVERTIBLE / PRODUCIBLE /
> INCOMPATIBLE ahead of time. `#flavor` refs, flavored tag rows and the flavor
> columns go away with no alias. The replacement surfaces are being designed in
> th#1809 (hub) and pgw#1143 (SDK); the `flavor=`/`#fp8` spellings in this
> document describe what exists today, not what to build against.

The one exception is the
EMERGENCY rung (automatic on CUDA hosts): when
even the downloaded flavor cannot fit free VRAM, the loading layer
runtime-quantizes the denoiser to 4-bit nf4 with a loud warning (quality
below platform standards) rather than falling straight to CPU offload.
Fit ladder: bf16 → `#fp8` → `#nvfp4` (Blackwell) → emergency-nf4 → offload.

## Model selection

Checkpoint selection is a runtime PAYLOAD FIELD, not a build-time fan-out:
16 near-identical fine-tunes = ONE `generate(model=)`, not 16 functions. Use
`Slot` (below) — the model SET is CATALOG, not code.

**BYOM is the field TYPE.** The payload field named by `selected_by` is plain
`str` for curated-only, or `str | ModelRef` to additionally accept an
arbitrary client-supplied `ModelRef` (bring-your-own-model). No `@byom`
decorator, no `sources=` — architecture compatibility is derived from the
pipeline the endpoint's `models=` loads. Per-method policy falls out of
method=contract: a `generate` method can be BYOM-open while a
`generate_turbo` method (fixed distillation LoRA) stays curated.

Divergent WIRE contracts are separate METHODS, not `Optional` fields; only
weight-sharing forces one class. A distilled turbo that shares the base is a
`generate_turbo` method on the same class (shares the resident base); a
standalone distilled checkpoint is a separate class/endpoint.

> The older pgw#509 `ModelChoice` / `Model` / `ModelDefaults` enum API that
> baked the curated set into the endpoint image is **gone** — those names have
> no definition in `gen_worker` and importing them fails. `Slot` replaced it.

## `Slot`: hub-resolved model slots (SDK v2, pgw#647)

`Slot(pipeline_cls, selected_by=, family=, default_checkpoint=)` resolves the model set
from the hub catalog (th#767) rather than from code, so adding a checkpoint is
not a software release. Under SDK v2 the declaration shrinks to the ROOT of the
component tree:

```python
from gen_worker import HF, RequestContext, Slot, endpoint
from .defaults import SdxlDefaults   # YOUR family vocabulary — see below

@endpoint(models={
    "pipeline": Slot(
        StableDiffusionXLPipeline,
        selected_by="model",                                              # payload field that branches this slot
        default_checkpoint=HF("stabilityai/stable-diffusion-xl-base-1.0"), # hub-less / seed-publish ref
    ),
})
class Generate:
    def setup(self, pipeline: StableDiffusionXLPipeline) -> None:
        self.pipeline = pipeline

    def generate(self, ctx: RequestContext[SdxlDefaults], p: TextToImage) -> ImageOutput:
        d = ctx.defaults                     # typed SdxlDefaults — the catalog-resolved recipe
        steps = p.steps if p.steps is not None else d.steps
```

- **The component tree is DERIVED from the pipeline class.**
  `pipeline.unet` / `pipeline.vae` / `pipeline.text_encoder(_2)` /
  `pipeline.scheduler` are addressable automatically (diffusers pipelines
  self-describe), and the derived tree is published into the release
  manifest for per-path catalog policy (`pipeline` open to a checkpoint
  pick, `pipeline.vae` curated, `pipeline.unet` fixed) and
  component-level routing. Declaring a part as a SIBLING slot
  (`"vae": Slot(AutoencoderKL)`) or a `str` modifier slot next to a
  pipeline slot (`turbo_lora: Slot(str)`) is a decoration-time error —
  component overrides are CATALOG DATA (th#1116), adapters ride the model
  binding. Explicit multi-slot declaration survives only for runtimes the
  SDK cannot introspect (llama/gguf, custom engines).
- **The config SCHEMA derives from the context annotation**
  (`ctx: RequestContext[SdxlDefaults]`), never from a Slot kwarg
  (`default_config=` is deleted). The catalog owns recipe VALUES: the hub
  stamps one resolved recipe per slot and `ctx.defaults` hands it to the
  handler typed. With no catalog metadata (hub-less `gen-worker run`,
  tests), the neutral schema defaults (`SdxlDefaults()`) apply — identical
  to the hub's neutral stamp.
- `selected_by` names a payload field typed **plain `str`** (or
  `str | ModelRef`, the wire's BYOM-open shape) — validated at
  registration. The hub overlays the live allowed-value enum onto the
  field; the SDK never bakes a curated list.
- `default_checkpoint` seeds the hub mapping at first publish and is the
  ONLY resolution source in hub-less mode; a live hub mapping always wins.
- `family=None` keeps the compatibility inference: root/ref-bearing slots
  inherit the handler's family, while a non-root/defaultless slot is treated
  as family-agnostic. Set `family="qwen-image"` when such a non-root slot is a
  real deploy-bound model lane; set `family=""` when a ref-bearing auxiliary
  explicitly remains architecture-agnostic. Explicit intent wins over the
  inference and is published into the slot manifest.
- **Component sharing is AUTOMATIC by content address** — byte-identical
  components (the qwen text encoder, a shared VAE) load once and are
  refcounted across checkpoint picks; there is nothing to declare
  (`share_components=` is deleted).
- **`layouts=` declares WHAT BYTES THIS SLOT'S CODE CAN EXECUTE** (§1.33,
  pgw#1143) — the DEMAND half of the tensor-layout contract, per component
  path, a SET:

  ```python
  from gen_worker.models.tensor_layout_contract import (
      CONTRACT_HF_FP8_BLOCKWISE, CONTRACT_PLAIN_BF16)

  Slot(StableDiffusionXLPipeline, selected_by="model", layouts={
      "*":            (CONTRACT_PLAIN_BF16,),
      "text_encoder": (CONTRACT_HF_FP8_BLOCKWISE, CONTRACT_PLAIN_BF16),
  })
  ```

  `"*"` is the whole-tree default; a component key overrides it for that
  component. The hub compares an artifact's PROVEN layout against this set at
  rebind and refuses a mismatch with both sides named, before any pod is
  bought. **Omitting `layouts=` leaves the slot UNDECLARED** — the gate then
  falls back to the image-wide decoder census, and absence is never read as
  "accepts everything"; an empty mapping or an empty tuple is a
  decoration-time error. Handles must be registered
  (`KNOWN_CONTRACTS`, transcribed from tensorhub's `internal/tensorlayout`)
  and written as LITERALS or as constants imported from that module —
  `scripts/lint_layout_declarations.py` refuses anything the AST sweep cannot
  read. Declaring a handle no decoder in the image backs is NOT an error: it
  lands in the build log as `layouts_census_unbacked`, because plenty of
  layouts are decoded natively by `transformers`/`diffusers` with no cozy
  marker.

  **The set is a compatibility FILTER — its order carries no preference**
  (§1.33 pt 2 as amended by th#1803). Preference has exactly one authority:
  the owner-configured ordered ladder of (GPU, lane) pairs. The SDK
  canonicalizes what you write, so spelling the set in a different order is
  the same declaration, not a different one.

  **When an artifact does not match, §1.33's ladder decides** — COMPATIBLE
  (in the set) → **CONVERTIBLE** (a registered LOSSLESS mapping reaches an
  accepted layout) → **PRODUCIBLE** (only a re-quantization from a named
  higher-precision source does — a priced job, never automatic) →
  INCOMPATIBLE (refused, both sides named, before any pod). The CONVERTIBLE
  edge set is `gen_worker.convert.layout_converters`, declared by whoever
  owns the format:

  ```python
  from gen_worker.convert import (
      ConversionCase, CorpusTensor, TopologyConversion,
      register_layout_conversion)
  from gen_worker.convert.repack_spec import RenameRule

  register_layout_conversion(TopologyConversion(
      from_id=TOPOLOGY_COMFY_SPLITFILES,
      to_id=TOPOLOGY_DIFFUSERS_MULTIFILE,
      version=1,
      rules=(RenameRule(kind="prefix",
             pairs=(("model.diffusion_model.", "transformer."),)),),
      inverse_rules=(RenameRule(kind="prefix",
             pairs=(("transformer.", "model.diffusion_model."),)),),
      corpus=(ConversionCase(name="dit-block0", tensors={...}),),
  ))
  ```

  Registration RUNS the mapping's bit-exactness obligation over its corpus
  before the edge exists: key bijection, payload invariance, and `A → B → A`
  content recovery. **A re-quantization cannot pass that round trip, so it
  cannot be registered as a converter** — it is
  `register_layout_production()`, which carries a recipe name and a quality
  gate and no transform. Bump `version` when the mapping changes; the digest
  moves and every derived artifact gets a new identity, so bytes never
  silently change under a name.

**Per-family defaults vocabulary**: a typed, versioned,
JSON-Schema-exportable struct per architecture — the shape tensorhub validates
catalog recipe values against. `gen_worker.families` ships the REGISTRY and
nothing else: **you declare the vocabulary in your own endpoint** (pgw#740
moved `SdxlDefaults`/`WanDefaults` out of the SDK — a vocabulary in the library
would need a wheel release to change). Declare it anywhere imported before
discovery runs:

```python
from gen_worker.families import GenerationDefaults, family

@family("sdxl")
class SdxlDefaults(GenerationDefaults, frozen=True):
    scheduler: Literal["euler_a", "dpmpp_2m_karras", "dpmpp_2m_sde_karras"] = "euler_a"
    steps: int = 28
    guidance: float = 6.0
    max_guidance: float | None = None   # a CLAMP constraint, never a wire reshape
```

`gen-worker families export-schemas <dir>` writes one
`<family>[.lora].schema.json` per registered family (LoRA-kind families get the
`.lora` infix). Code owns this SCHEMA; the catalog owns the VALUES.

**Per-request views**: `ctx.for_request(self.pipeline, sampler=, seed=)`
returns a container copy sharing every module by reference (zero weight
VRAM; the compiled graph stays bound to the module objects) with its OWN
scheduler cloned from config — the SDK owns the sampler-name table
(`gen_worker.view.SAMPLERS`; recipes select among its names, endpoints
never define private sampler maps) and applies the resolved checkpoint's
OBJECTIVE (pgw#654: `epsilon` / `v_prediction` — prediction_type plus
zero-terminal-SNR for v-pred; `flow` — flow-match scheduler classes only)
automatically. Never assign `self.pipeline.scheduler` per request:
schedulers are stateful, shared, and part of the instance.

Per-function contract facts live ON the function (pgw#654):

```python
@worker_function(objectives=("epsilon", "v_prediction"), distilled=False)
def generate(self, ctx, payload: TextToImageInput) -> ImageOutput: ...
```

`objectives` = which checkpoint training objectives the handler's code
path serves (omit = unrestricted); `distilled` = True (only distilled) /
False (only non-distilled) / omit (either). The boot WARM PLAN is DERIVED
— defaulted fields keep their schema defaults, `CompileAxis` fields
cross-product their classes' `warm=` representatives, required fields
synthesize neutral values — so there is no `warmup=` payload dict to
write or to drift. Cheapen non-graph work on `ctx.boot_warmup`
(`steps = 1 if ctx.boot_warmup else steps`). A per-function
`@worker_function(warm={...}, warm_reason=...)` override exists only for
a non-axis field that genuinely changes tracing. When the serve path
modifies a requested value (clamps, substitutions), record it with
`ctx.adjusted(field, requested, applied, reason)` / `ctx.clamp(...)` —
the rows ride the result envelope to the caller.

Imports live at module top by convention — no function-body imports
unless breaking a genuine cycle or deferring an optional extra.

**Testing:** `gen_worker.testing` builds a `RequestContext` with stubbed
`ctx.slots` for handler unit tests, no hand-rolled fake context needed:

```python
from gen_worker.testing import fake_context
from gen_worker import HF
from .defaults import SdxlDefaults   # YOUR family vocabulary

ctx = fake_context(slots={
    "pipeline": (HF("stabilityai/stable-diffusion-xl-base-1.0"), SdxlDefaults(steps=28)),
})
out = Generate().generate(ctx, TextToImage(prompt="a cat"))
```

Add a `Recorder` to assert what the handler SAVED and LOGGED (pgw#942).
Outputs run the SDK's real encode / C2PA stamp / size-limit path and are
written into the recorder's own directory instead of uploaded, so there is
no hub and no network — and, unlike a `save_image` override, the encode
under test is the one production runs:

```python
from gen_worker.testing import Recorder, fake_context

rec = Recorder()
ctx = fake_context(slots={...}, recorder=rec)
Generate().generate(ctx, TextToImage(prompt="a cat"))

assert rec.refs == ["outputs/test-request/image.webp"]
assert rec.images[0].call["quality"] == 95        # what the handler asked for
assert rec.images[0].read_bytes()[:4] == b"RIFF"  # what it actually produced
assert rec.messages == ["scheduler: dpmpp_2m_karras"]   # ctx.log
assert rec.progress[-1].payload["step"] == 28           # ctx.progress
```

`rec.saved` holds every `save_*` in order (`images` / `audio` / `videos` /
`files` filter it); each entry carries the typed asset the handler received,
with its real `sha256` and `size_bytes`. `ctx.log` and `ctx.progress` are
captured at the emitter seam, so neither needs an override either.

**Do not assert the SDK's own shape.** A field the SDK deleted is the SDK's
fact: it is listed once in `gen_worker.api.sdk_shape.DELETED_FIELDS`
and asserted by the SDK's suite. An endpoint asserts what IT declares, and
gets the deleted-field walk for free:

```python
from gen_worker.testing import assert_declaration_shape

decl = Generate.__gen_worker_endpoint__
assert_declaration_shape(decl)          # decl + resources + compile + slots
assert decl.resources.gpu is True       # your declaration — keep asserting this
```

A hand-written `assert not hasattr(decl.resources, "vram_gb")` is a copy of
that list which no SDK change can update: `compute_capability` was deleted
at 0.60 and RESTORED at 0.75 (pgw#660), and every stale copy of the list
had to be found by hand.

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

Declare ONLY what the endpoint cannot run without:

```python
Resources(gpu=True, libraries=("nunchaku",))
```

Fields: `gpu`, `gpu_count`, `max_gpu_count`,
`max_gpus_per_execution_group`, `parallel`, `libraries`, `strict_vram`
(bindings that cannot tolerate CPU-resident weights), `vcpus`,
`compute_capability`, and the two hints `vram_gb_hint` / `ram_gb_hint`.
`max_gpus_per_execution_group` / `parallel` are the multi-GPU axis — see
[multi-gpu.md](multi-gpu.md).

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

The members you will actually reach for (the class has ~27 public members;
`request_context/__init__.py` is the full list):

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

The SDK reads `pyproject.toml` only:

```toml
[tool.gen_worker]
main = "my_endpoint.main"
# optional: discovery_heavy_deps = [...]  — extra heavy import roots
```

`endpoint.toml` is a different file at a different layer: the platform-side
build/deploy manifest that **tensorhub** owns and reads (`[[build.profiles]]`,
`[resources]`). gen-worker never parses it. See
[dockerfile.md](dockerfile.md) for the build-profile side.

## Errors

Raise `ValidationError` (bad input, don't retry), `RetryableError`,
`CanceledError`, or `FatalError`. Anything else is reported as an internal
error.

## The output-integrity floor (pgw#1094)

`ctx.save_image`, `io.write_image` and `io.write_video` look at the PIXELS
before they encode anything. A render whose frames are NOISE (median
adjacent-frame grey correlation below 0.6) or BLANK (a constant-fill frame), or
that carries NaN/Inf pixels, raises `OutputIntegrityError` and is never
uploaded — production once served VAE-decoded noise on billed, settled requests
because nothing on this side had ever looked (ie#634). It costs ~3 ms on a
121-frame 1344x768 clip and there is no way to turn it off.

Two consequences for endpoint authors: a legitimately constant-fill output is
not servable through these calls, and a green integrity result is **not** a
quality signal — a melted or over-smoothed render scores HIGHER on this
statistic than a clean one. See `gen_worker.output_integrity`.

## Do NOT quantize weights in `setup()` (th#1803)

**Component quant selection is BINDING CONFIG, not endpoint code.** fp8 vs
bf16 for a text encoder or a denoiser is a config record change — point the
component ref at the fp8 artifact — with no code change, no rebuild and no
redeploy, and it is overridable per request where the endpoint allows it.
Write `setup()` **quant-generically**: declare the tensor layout you execute
and serve whatever satisfies it. A `serve_recipe` that casts bf16 weights on
every cold boot, switchable only by shipping a new endpoint version, is the
rejected pattern (DESIGN-RULINGS §1.32).

**No inference-time quantization at all.** Quantization happens ahead of time —
a conversion endpoint produces the artifact — for two measured reasons: it
lengthens every cold boot, and it wastes transfer (download 30 GB of bf16 and
immediately discard 15 GB). So the recipe LOADS the bound pre-quantized
artifact; the boot-quant path is deleted, not kept as a fallback. (The fit
ladder's emergency nf4 rung below is a different thing — a last-resort OOM
degradation, not a selection mechanism.)

What stays in code: kernel selection, compile scope, allocator settings, the
warmup obligation. What leaves: the choice of which weights to run.

### If a recipe still converts weights, report the lane it applied (pgw#1104)

Two shipped endpoints predate the ruling. Until they are converted, the
reporting call below is mandatory for them — and it stays in the SDK
regardless, because runtime-gated *engagement* (a kernel or compile arm that
may or may not apply on this card) still has to be reported honestly.

A serve-time recipe — torchao `quantize_()`, an fp8 cast, anything that
converts the weights inside `setup()` — moves the lane the endpoint EXECUTES
away from the checkpoint the hub bound. The worker cannot see that (sniffing
tensor subclasses is deliberately not done), so the recipe says so:

```python
def setup(self, pipeline: Pipe) -> None:
    if not w8a8_capable():          # runtime gate: report only what APPLIED
        return
    quantize_(pipeline.transformer, _w8a8_config(), filter_fn=...)
    gen_worker.report_applied_lane(
        "transformer", "fp8-w8a8-dynamic", modules=300, kept_bf16=70)
```

`metrics.lane` / `ctx.lane` then report `fp8-w8a8-dynamic+compiled` instead of
the binding's `bf16-w16a16`. This matters because the lane id is a KEY: quality
verdicts, compile cells, serving floors and pricing all join on it, and
minimax-h3 spent four issues being priced as bf16 while serving a 21.7 GiB fp8
DiT.

Rules: the second argument must be one of
`gen_worker.models.execution_lanes.known_execution_lane_bodies()` (the same
vocabulary `handles=` uses) — anything else raises `ValueError`, because the
lane vocabulary is shared with the hub. Never name the execution axis
(`+compiled` / `+eager`): the platform owns it. Call it AFTER the conversion
returns, never on the strength of an intention — outside a `setup()` scope it
logs and returns `False` instead of raising, so it is safe under `cozy run`.

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

## Author CI: proving your compiled path is CORRECT and FASTER (pgw#1150)

Declaring `compile=` claims two things about your own code — that the compiled
output matches eager, and that it is faster. Both are the AUTHOR's claims to
prove (DESIGN-RULINGS §4.32, ie#664), because every parity failure caught so far
was an endpoint code or model config defect. One command, on a pod, in the
endpoint's own image:

```bash
python -m gen_worker.author_ci --payload '{"prompt":"a cat"}'   # add --write
```

It asserts the fleet line first (exits **90** off the line, **91** when the host
cannot run the wheel — `rigcheck`'s own numbers), arms through your real
`setup()`, takes the parity verdict from the mint-parent gate, then measures
steady-state compiled-vs-eager in ONE process: the compiled arm, then the same
pipeline under pgw#1142's eager-only order. N>=5 per arm, first request of each
discarded, median and p95 off `stage_ms.<stage>` — never a round trip
(th#1795). It writes the `[proof]` block of `<endpoint>/author-ci.toml` and
leaves your `[parity]` / `[speed]` declarations untouched; those are the bar it
judges against, and `min_speedup` defaults to the fleet's 1.10.

A family with open `Compile.blockers` reports `blocked-by-declaration` — a legal
state — and runs eager-only. A below-bar speedup is recorded as `failed` with
its evidence, never as a proof.

**It measures; it never gates publish.** Promotion runs on trusted hardware, on
the published code (th#1811). The standard and the record schema live in
`inference-endpoints/AUTHOR-CI.md`.
